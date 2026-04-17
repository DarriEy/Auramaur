"""On-chain redemption of winning Polymarket conditional tokens.

Polymarket holds your tokens inside a Gnosis Safe v1.3.0 proxy wallet owned
by your EOA. To redeem a resolved position to USDC, we:

  1. Build calldata for CTF.redeemPositions(collateralToken, parentCollectionId,
     conditionId, indexSets) — the Conditional Tokens contract burns winning
     ERC-1155 positions and transfers USDC to the holder (the Safe).
  2. Wrap that call in a Safe transaction via execTransaction(...), with an
     EIP-712 signature from the EOA (the sole owner, threshold=1).
  3. Broadcast from the EOA as a regular Polygon tx. Safe verifies the
     signature, checks the nonce, executes the call. USDC lands in the Safe.

Only the CTF (standard binary) path is implemented here. NegRisk markets go
through a separate adapter contract and raise NotImplementedError for now.

Safety:
  * All real submissions require triple-gate (AURAMAUR_LIVE + execution.live +
    dry_run=False) PLUS a distinct AURAMAUR_ENABLE_REDEMPTION gate. The
    redemption gate is independent so live-trading doesn't imply
    live-redemption.
  * Every submission is recorded in the `redemptions` table keyed by
    condition_id, which doubles as a replay guard — a condition marked
    `submitted` or `confirmed` won't be re-submitted.
  * Private keys never appear in logs. Only the EOA address is logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak
from web3 import Web3

from auramaur.broker.redeemer import RedeemablePosition
from auramaur.db.database import Database

log = structlog.get_logger()


# Polygon mainnet contract addresses (chainId 137).
POLYGON_CHAIN_ID = 137
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER_ADDRESS = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# Minimal ABIs — only the functions we actually call.
CTF_ABI: list[dict[str, Any]] = [
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    },
    {
        "name": "payoutNumerators",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "index", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]

SAFE_ABI: list[dict[str, Any]] = [
    {
        "name": "nonce",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "getOwners",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address[]"}],
    },
    {
        "name": "execTransaction",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "signatures", "type": "bytes"},
        ],
        "outputs": [{"name": "success", "type": "bool"}],
    },
]

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
ZERO_BYTES32 = b"\x00" * 32

# EIP-712 types for Safe v1.3.0 transaction signing.
_SAFE_TX_TYPES: dict[str, list[dict[str, str]]] = {
    "EIP712Domain": [
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "SafeTx": [
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "data", "type": "bytes"},
        {"name": "operation", "type": "uint8"},
        {"name": "safeTxGas", "type": "uint256"},
        {"name": "baseGas", "type": "uint256"},
        {"name": "gasPrice", "type": "uint256"},
        {"name": "gasToken", "type": "address"},
        {"name": "refundReceiver", "type": "address"},
        {"name": "nonce", "type": "uint256"},
    ],
}


@dataclass
class RedemptionResult:
    condition_id: str
    title: str
    status: str                # "built" | "submitted" | "confirmed" | "rejected" | "skipped"
    tx_hash: str = ""
    safe_nonce: int | None = None
    calldata_preview: str = ""
    error: str = ""

    @property
    def submitted(self) -> bool:
        return self.status in ("submitted", "confirmed")


class OnChainRedeemer:
    """Redeem resolved Polymarket positions held by a Gnosis Safe proxy."""

    def __init__(self, settings, db: Database):
        self._settings = settings
        self._db = db
        self._w3: Web3 | None = None
        self._eoa_address: str | None = None

    # ------------------------------------------------------------------
    # Gate checks
    # ------------------------------------------------------------------

    def _is_live_submission_allowed(self) -> bool:
        """All gates must be open to broadcast a real tx.

        Triple-gate covers live trading generally; AURAMAUR_ENABLE_REDEMPTION
        is a separate opt-in because redemption is a distinct money-moving
        operation and shouldn't ride on the trading toggle.
        """
        if Path("KILL_SWITCH").exists():
            return False
        if not self._settings.auramaur_live:
            return False
        if not self._settings.execution.live:
            return False
        if not self._settings.auramaur_enable_redemption:
            return False
        return True

    # ------------------------------------------------------------------
    # Lazy web3 / account init
    # ------------------------------------------------------------------

    def _init(self) -> None:
        if self._w3 is not None:
            return
        rpc = self._settings.polygon_rpc_url
        self._w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 15}))
        if not self._w3.is_connected():
            raise RuntimeError(f"polygon RPC unreachable: {rpc}")
        pk = self._settings.polygon_private_key
        if not pk:
            raise RuntimeError("polygon_private_key is not configured")
        # NOTE: derive address once and never log the key.
        self._eoa_address = Account.from_key(pk).address

    # ------------------------------------------------------------------
    # Calldata construction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_bytes32(value: str) -> bytes:
        """Parse a 0x-prefixed 32-byte hex string into raw bytes."""
        s = value[2:] if value.startswith("0x") else value
        b = bytes.fromhex(s)
        if len(b) != 32:
            raise ValueError(f"expected 32-byte value, got {len(b)}: {value}")
        return b

    def build_ctf_redeem_calldata(self, condition_id: str) -> bytes:
        """Build CTF.redeemPositions calldata for a binary market.

        index_sets=[1, 2] redeems both YES and NO positions held on this
        condition. Polymarket holds one side per market; the other index just
        redeems 0 tokens, which is a no-op on-chain (no cost beyond calldata).
        """
        self._init()
        ctf = self._w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)
        cond_bytes = self._normalize_bytes32(condition_id)
        # parentCollectionId is bytes32(0) for top-level conditions — all
        # Polymarket markets are top-level.
        encoded = ctf.encode_abi(
            "redeemPositions",
            args=[
                Web3.to_checksum_address(USDC_E_ADDRESS),
                ZERO_BYTES32,
                cond_bytes,
                [1, 2],
            ],
        )
        # web3.py returns a 0x-prefixed hex string; the rest of the pipeline
        # (EIP-712 bytes-arg hashing + raw tx construction) wants bytes.
        if isinstance(encoded, str):
            return bytes.fromhex(encoded[2:] if encoded.startswith("0x") else encoded)
        return bytes(encoded)

    # ------------------------------------------------------------------
    # Safe EIP-712 signing
    # ------------------------------------------------------------------

    def _safe_contract(self):
        self._init()
        return self._w3.eth.contract(
            address=Web3.to_checksum_address(self._settings.polymarket_proxy_address),
            abi=SAFE_ABI,
        )

    def _sign_safe_tx(
        self,
        to: str,
        value: int,
        data: bytes,
        operation: int,
        safe_tx_gas: int,
        base_gas: int,
        gas_price: int,
        gas_token: str,
        refund_receiver: str,
        nonce: int,
    ) -> bytes:
        """EIP-712 sign a SafeTx. Returns r||s||v packed (65 bytes)."""
        safe_addr = Web3.to_checksum_address(self._settings.polymarket_proxy_address)
        message = {
            "to": Web3.to_checksum_address(to),
            "value": value,
            "data": data,
            "operation": operation,
            "safeTxGas": safe_tx_gas,
            "baseGas": base_gas,
            "gasPrice": gas_price,
            "gasToken": Web3.to_checksum_address(gas_token),
            "refundReceiver": Web3.to_checksum_address(refund_receiver),
            "nonce": nonce,
        }
        typed = {
            "types": _SAFE_TX_TYPES,
            "primaryType": "SafeTx",
            "domain": {"chainId": POLYGON_CHAIN_ID, "verifyingContract": safe_addr},
            "message": message,
        }
        signable = encode_typed_data(full_message=typed)
        signed = Account.sign_message(signable, private_key=self._settings.polygon_private_key)
        # Safe expects r || s || v as 65 bytes. eth_account returns the same.
        return bytes(signed.signature)

    # ------------------------------------------------------------------
    # DB tracking — replay guard
    # ------------------------------------------------------------------

    async def _already_submitted(self, condition_id: str) -> bool:
        row = await self._db.fetchone(
            "SELECT status FROM redemptions WHERE condition_id = ?",
            (condition_id,),
        )
        if row is None:
            return False
        return row["status"] in ("submitted", "confirmed")

    async def _record_attempt(
        self,
        position: RedeemablePosition,
        safe_nonce: int,
        status: str,
        tx_hash: str = "",
        error: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO redemptions
               (condition_id, asset_id, title, neg_risk, size, expected_payout,
                safe_nonce, tx_hash, status, submitted_at, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(condition_id) DO UPDATE SET
                   safe_nonce = excluded.safe_nonce,
                   tx_hash = excluded.tx_hash,
                   status = excluded.status,
                   submitted_at = excluded.submitted_at,
                   error = excluded.error""",
            (
                position.condition_id,
                position.asset_id,
                position.title,
                1 if position.neg_risk else 0,
                position.size,
                position.payout,
                safe_nonce,
                tx_hash,
                status,
                now if status in ("submitted", "confirmed") else None,
                error,
            ),
        )
        await self._db.commit()

    async def _mark_confirmed(self, condition_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE redemptions SET status = 'confirmed', confirmed_at = ? WHERE condition_id = ?",
            (now, condition_id),
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def redeem(
        self,
        position: RedeemablePosition,
        *,
        dry_run: bool = True,
        wait_for_receipt: bool = True,
    ) -> RedemptionResult:
        """Redeem one resolved position.

        dry_run=True (default) builds and signs the Safe transaction but does
        NOT broadcast it. Use this to review the calldata before flipping the
        AURAMAUR_ENABLE_REDEMPTION gate.
        """
        if position.neg_risk:
            return RedemptionResult(
                condition_id=position.condition_id,
                title=position.title,
                status="rejected",
                error="NegRisk adapter redemption not implemented yet",
            )

        if await self._already_submitted(position.condition_id):
            return RedemptionResult(
                condition_id=position.condition_id,
                title=position.title,
                status="skipped",
                error="already submitted or confirmed in redemptions table",
            )

        self._init()
        safe = self._safe_contract()
        safe_nonce = safe.functions.nonce().call()

        # Build the inner call — CTF.redeemPositions(USDC, 0, conditionId, [1,2])
        inner_data = self.build_ctf_redeem_calldata(position.condition_id)

        # Wrap in Safe execTransaction. EOA pays Polygon gas directly, so all
        # internal gas parameters are zero — Safe forwards gasleft() to the
        # inner call and no refund logic kicks in.
        to = Web3.to_checksum_address(CTF_ADDRESS)
        value = 0
        operation = 0  # CALL
        signature = self._sign_safe_tx(
            to=to,
            value=value,
            data=inner_data,
            operation=operation,
            safe_tx_gas=0,
            base_gas=0,
            gas_price=0,
            gas_token=ZERO_ADDRESS,
            refund_receiver=ZERO_ADDRESS,
            nonce=safe_nonce,
        )

        exec_fn = safe.functions.execTransaction(
            to, value, inner_data, operation,
            0, 0, 0, ZERO_ADDRESS, ZERO_ADDRESS, signature,
        )
        preview = exec_fn._encode_transaction_data()

        log.info(
            "onchain.redeem_built",
            condition_id=position.condition_id[:10],
            title=position.title[:50],
            payout=round(position.payout, 2),
            safe_nonce=safe_nonce,
            calldata_bytes=len(preview) // 2 - 1,
            dry_run=dry_run,
            live_gates_open=self._is_live_submission_allowed(),
        )

        if dry_run or not self._is_live_submission_allowed():
            # Record the built (but unsubmitted) attempt so a follow-up live
            # run can see what was planned.
            await self._record_attempt(
                position,
                safe_nonce,
                status="built",
            )
            return RedemptionResult(
                condition_id=position.condition_id,
                title=position.title,
                status="built",
                safe_nonce=safe_nonce,
                calldata_preview=preview,
            )

        # === LIVE SUBMISSION ===
        tx = exec_fn.build_transaction({
            "from": self._eoa_address,
            "nonce": self._w3.eth.get_transaction_count(self._eoa_address),
            "gasPrice": int(self._w3.eth.gas_price * 1.25),  # slight bump
            "chainId": POLYGON_CHAIN_ID,
        })
        # Estimate gas — small buffer. Safe exec is ~150-250k gas for CTF redeem.
        try:
            tx["gas"] = int(self._w3.eth.estimate_gas(tx) * 1.2)
        except Exception as e:
            log.error("onchain.gas_estimate_failed", error=str(e)[:200])
            await self._record_attempt(
                position, safe_nonce, status="rejected", error=f"gas_estimate: {e}"[:200]
            )
            return RedemptionResult(
                condition_id=position.condition_id,
                title=position.title,
                status="rejected",
                safe_nonce=safe_nonce,
                error=f"gas_estimate failed: {e}"[:200],
            )

        signed_tx = Account.sign_transaction(tx, private_key=self._settings.polygon_private_key)
        tx_hash = self._w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = tx_hash.hex()

        log.warning(
            "onchain.redeem_submitted",
            condition_id=position.condition_id[:10],
            safe_nonce=safe_nonce,
            tx_hash=tx_hash_hex,
            polygonscan=f"https://polygonscan.com/tx/0x{tx_hash_hex.lstrip('0x')}",
        )

        await self._record_attempt(
            position, safe_nonce, status="submitted", tx_hash=tx_hash_hex
        )

        result = RedemptionResult(
            condition_id=position.condition_id,
            title=position.title,
            status="submitted",
            tx_hash=tx_hash_hex,
            safe_nonce=safe_nonce,
        )

        if wait_for_receipt:
            try:
                receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
                if receipt.status == 1:
                    await self._mark_confirmed(position.condition_id)
                    result.status = "confirmed"
                    log.info(
                        "onchain.redeem_confirmed",
                        condition_id=position.condition_id[:10],
                        tx_hash=tx_hash_hex,
                        gas_used=receipt.gasUsed,
                    )
                else:
                    await self._db.execute(
                        "UPDATE redemptions SET status='rejected', error='tx reverted' WHERE condition_id=?",
                        (position.condition_id,),
                    )
                    await self._db.commit()
                    result.status = "rejected"
                    result.error = "tx reverted on-chain"
                    log.error("onchain.redeem_reverted", tx_hash=tx_hash_hex)
            except Exception as e:
                log.warning("onchain.receipt_timeout", tx_hash=tx_hash_hex, error=str(e)[:200])

        return result


def _safe_tx_hash_for_testing(
    *,
    to: str,
    value: int,
    data: bytes,
    operation: int,
    safe_tx_gas: int,
    base_gas: int,
    gas_price: int,
    gas_token: str,
    refund_receiver: str,
    nonce: int,
    safe_address: str,
    chain_id: int = POLYGON_CHAIN_ID,
) -> bytes:
    """Compute the EIP-712 message hash a Safe would sign. Exposed for tests."""
    safe_tx_typehash = keccak(text=(
        "SafeTx(address to,uint256 value,bytes data,uint8 operation,"
        "uint256 safeTxGas,uint256 baseGas,uint256 gasPrice,address gasToken,"
        "address refundReceiver,uint256 nonce)"
    ))
    domain_typehash = keccak(text="EIP712Domain(uint256 chainId,address verifyingContract)")

    from eth_abi import encode as abi_encode

    data_hash = keccak(data)
    encoded = abi_encode(
        [
            "bytes32", "address", "uint256", "bytes32", "uint8",
            "uint256", "uint256", "uint256", "address", "address", "uint256",
        ],
        [
            safe_tx_typehash,
            Web3.to_checksum_address(to),
            value, data_hash, operation,
            safe_tx_gas, base_gas, gas_price,
            Web3.to_checksum_address(gas_token),
            Web3.to_checksum_address(refund_receiver),
            nonce,
        ],
    )
    tx_hash = keccak(encoded)

    domain_separator = keccak(abi_encode(
        ["bytes32", "uint256", "address"],
        [domain_typehash, chain_id, Web3.to_checksum_address(safe_address)],
    ))

    return keccak(b"\x19\x01" + domain_separator + tx_hash)
