"""Tests for OnChainRedeemer: gating, calldata, EIP-712, DB tracking."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from eth_account import Account
from eth_utils import keccak
from web3 import Web3

from auramaur.broker.onchain import (
    CTF_ADDRESS,
    OnChainRedeemer,
    POLYGON_CHAIN_ID,
    USDC_E_ADDRESS,
    ZERO_ADDRESS,
    ZERO_BYTES32,
    _safe_tx_hash_for_testing,
)
from auramaur.broker.redeemer import RedeemablePosition
from auramaur.db.database import Database


# Deterministic test key — NEVER used on mainnet. Address matches the well-
# known hardhat/anvil account #0, so if this string ever leaks no funds are
# at risk.
TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
TEST_EOA = Account.from_key(TEST_PRIVATE_KEY).address
TEST_SAFE = "0x377d1939B32FeFAD321190085c0B28337275DDDd"


def _make_settings(
    *,
    live: bool = False,
    execution_live: bool = True,
    enable_redemption: bool = False,
    private_key: str = TEST_PRIVATE_KEY,
    proxy_address: str = TEST_SAFE,
) -> MagicMock:
    s = MagicMock()
    s.auramaur_live = live
    s.execution.live = execution_live
    s.auramaur_enable_redemption = enable_redemption
    s.polygon_private_key = private_key
    s.polymarket_proxy_address = proxy_address
    s.polygon_rpc_url = "https://rpc.example/test"
    return s


def _make_position(
    *,
    condition_id: str = "0x" + "ab" * 32,
    neg_risk: bool = False,
    size: float = 10.0,
) -> RedeemablePosition:
    return RedeemablePosition(
        condition_id=condition_id,
        asset_id="asset-1",
        title="Test market",
        outcome="Yes",
        size=size,
        avg_price=0.5,
        cur_price=1.0,
        payout=size,
        is_winner=True,
        redeemable_now=True,
        status="redeemable",
        neg_risk=neg_risk,
        mergeable=False,
        end_date="",
        slug="",
    )


@pytest.fixture
async def db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    await db.connect()
    yield db
    await db.close()


# ---------------- Gating ----------------


def test_gate_requires_all_flags(tmp_path):
    """Live submission requires live, execution.live, enable_redemption, and no kill switch."""
    db = Database(":memory:")
    cases = [
        dict(live=False, execution_live=True, enable_redemption=True),
        dict(live=True, execution_live=False, enable_redemption=True),
        dict(live=True, execution_live=True, enable_redemption=False),
    ]
    for flags in cases:
        r = OnChainRedeemer(_make_settings(**flags), db)
        assert r._is_live_submission_allowed() is False, f"should block with {flags}"

    # All gates open
    r = OnChainRedeemer(
        _make_settings(live=True, execution_live=True, enable_redemption=True), db
    )
    assert r._is_live_submission_allowed() is True


def test_kill_switch_blocks_live_submission(tmp_path, monkeypatch):
    """Presence of KILL_SWITCH file blocks even with all gates open."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "KILL_SWITCH").write_text("halt")
    db = Database(":memory:")
    r = OnChainRedeemer(
        _make_settings(live=True, execution_live=True, enable_redemption=True), db
    )
    assert r._is_live_submission_allowed() is False


# ---------------- Calldata ----------------


def test_ctf_redeem_calldata_encodes_correctly():
    """Calldata should match the known CTF.redeemPositions ABI selector + args."""
    db = Database(":memory:")
    r = OnChainRedeemer(_make_settings(), db)
    # Skip network init — we only need the local ABI encoder.
    r._w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:0"))

    condition_id = "0x" + "cd" * 32
    data = r.build_ctf_redeem_calldata(condition_id)

    # redeemPositions(address,bytes32,bytes32,uint256[]) selector
    expected_selector = keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
    assert data[:4] == expected_selector

    # Verify the USDC.e address is embedded (first arg)
    usdc_word = bytes.fromhex(USDC_E_ADDRESS[2:].rjust(64, "0"))
    assert usdc_word in data

    # parentCollectionId is zero (second arg)
    assert b"\x00" * 32 in data

    # conditionId is embedded (third arg)
    assert bytes.fromhex("cd" * 32) in data


# ---------------- EIP-712 signing ----------------


def test_safe_tx_hash_matches_eth_account_signing():
    """Our test helper's hash should match the hash eth_account signs.

    This guards against mistakes in the EIP-712 types — if a Safe ever
    rejected our signature on mainnet, diagnosing it from contract-side
    would be painful. This test catches typo-level errors before any
    mainnet interaction.
    """
    db = Database(":memory:")
    r = OnChainRedeemer(_make_settings(), db)
    r._w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:0"))

    to = Web3.to_checksum_address(CTF_ADDRESS)
    data = b"\xde\xad\xbe\xef"
    nonce = 27

    signature = r._sign_safe_tx(
        to=to,
        value=0,
        data=data,
        operation=0,
        safe_tx_gas=0,
        base_gas=0,
        gas_price=0,
        gas_token=ZERO_ADDRESS,
        refund_receiver=ZERO_ADDRESS,
        nonce=nonce,
    )
    assert len(signature) == 65  # r || s || v

    # Recompute the hash manually and recover the signer — must match EOA.
    manual_hash = _safe_tx_hash_for_testing(
        to=to, value=0, data=data, operation=0,
        safe_tx_gas=0, base_gas=0, gas_price=0,
        gas_token=ZERO_ADDRESS, refund_receiver=ZERO_ADDRESS,
        nonce=nonce, safe_address=TEST_SAFE,
    )
    recovered = Account._recover_hash(manual_hash, signature=signature)
    assert Web3.to_checksum_address(recovered) == Web3.to_checksum_address(TEST_EOA)


# ---------------- DB tracking & replay guard ----------------


@pytest.mark.asyncio
async def test_already_submitted_prevents_replay(db):
    r = OnChainRedeemer(_make_settings(), db)
    pos = _make_position()
    await r._record_attempt(pos, safe_nonce=5, status="submitted", tx_hash="0xdead")
    assert await r._already_submitted(pos.condition_id) is True


@pytest.mark.asyncio
async def test_built_status_does_not_count_as_submitted(db):
    """Only 'submitted' and 'confirmed' should block replay — 'built' is a
    pre-submission record and the user should be free to try again."""
    r = OnChainRedeemer(_make_settings(), db)
    pos = _make_position()
    await r._record_attempt(pos, safe_nonce=5, status="built")
    assert await r._already_submitted(pos.condition_id) is False


# ---------------- End-to-end dry run ----------------


@pytest.mark.asyncio
async def test_neg_risk_position_is_rejected(db):
    r = OnChainRedeemer(_make_settings(), db)
    pos = _make_position(neg_risk=True)
    result = await r.redeem(pos, dry_run=True)
    assert result.status == "rejected"
    assert "NegRisk" in result.error


@pytest.mark.asyncio
async def test_dry_run_records_built_but_does_not_submit(db):
    """With dry_run=True, redeem should build + sign + record, never broadcast."""
    r = OnChainRedeemer(_make_settings(), db)
    pos = _make_position()

    # Mock web3 so we don't need a real RPC. The only on-chain read is Safe.nonce().
    mock_w3 = MagicMock()
    mock_w3.is_connected.return_value = True
    mock_contract = MagicMock()
    mock_contract.functions.nonce.return_value.call.return_value = 27
    # ABI encoding still needs to work — let Web3 do it via a live-ish contract.
    real_w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:0"))
    ctf = real_w3.eth.contract(
        address=Web3.to_checksum_address(CTF_ADDRESS),
        abi=[{
            "name": "redeemPositions", "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"},
            ],
            "outputs": [],
        }],
    )

    def eth_contract(address, abi):
        # Return a mock-with-methods for Safe, real contract for CTF
        if address.lower() == TEST_SAFE.lower():
            safe_mock = MagicMock()
            safe_mock.functions.nonce.return_value.call.return_value = 27
            safe_exec = MagicMock()
            safe_exec._encode_transaction_data.return_value = "0x" + "aa" * 100
            safe_mock.functions.execTransaction.return_value = safe_exec
            return safe_mock
        return ctf

    mock_w3.eth.contract.side_effect = eth_contract
    r._w3 = mock_w3
    r._eoa_address = TEST_EOA

    result = await r.redeem(pos, dry_run=True)

    assert result.status == "built"
    assert result.tx_hash == ""
    assert result.safe_nonce == 27
    assert result.calldata_preview.startswith("0x")

    # Verify DB recorded the attempt as 'built', not 'submitted'.
    row = await db.fetchone(
        "SELECT status, tx_hash FROM redemptions WHERE condition_id = ?",
        (pos.condition_id,),
    )
    assert row["status"] == "built"
    assert row["tx_hash"] == ""


@pytest.mark.asyncio
async def test_submission_blocked_when_gates_closed(db):
    """Even with dry_run=False, missing gates must force a dry-run outcome."""
    # All gates closed
    r = OnChainRedeemer(_make_settings(live=False), db)
    pos = _make_position()

    # Stub web3 enough to get through the build phase
    mock_w3 = MagicMock()
    mock_w3.is_connected.return_value = True
    safe_mock = MagicMock()
    safe_mock.functions.nonce.return_value.call.return_value = 27
    safe_exec = MagicMock()
    safe_exec._encode_transaction_data.return_value = "0x" + "aa" * 100
    safe_mock.functions.execTransaction.return_value = safe_exec

    real_w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:0"))
    ctf = real_w3.eth.contract(
        address=Web3.to_checksum_address(CTF_ADDRESS),
        abi=[{
            "name": "redeemPositions", "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"},
            ],
            "outputs": [],
        }],
    )

    def eth_contract(address, abi):
        if address.lower() == TEST_SAFE.lower():
            return safe_mock
        return ctf

    mock_w3.eth.contract.side_effect = eth_contract
    r._w3 = mock_w3
    r._eoa_address = TEST_EOA

    # Caller asks to submit, but gates are closed — should degrade to "built".
    result = await r.redeem(pos, dry_run=False)
    assert result.status == "built"
    # Broadcast methods must not have been called.
    mock_w3.eth.send_raw_transaction.assert_not_called()


# ---------------- Constants sanity ----------------


def test_polygon_constants_are_checksummed():
    """Every on-chain address constant should be a valid checksum address.
    A typo in any of these would silently mis-redeem to the wrong contract."""
    for addr in (USDC_E_ADDRESS, CTF_ADDRESS, ZERO_ADDRESS):
        # to_checksum_address raises if the input is malformed.
        Web3.to_checksum_address(addr)
    assert POLYGON_CHAIN_ID == 137
    assert ZERO_BYTES32 == b"\x00" * 32
