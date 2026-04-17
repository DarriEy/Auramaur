"""Detect Polymarket proxy wallet type (1155Proxy vs Gnosis Safe).

Read-only RPC queries — does NOT submit any transactions.

Determines the correct on-chain redemption code path. The two possibilities:

  1155Proxy (older, Polymarket-specific)
    - proxy.proxy([(typeCode, to, value, data)], signature)
    - Custom EIP-712 domain
    - masterCopy() call REVERTS (no such function)

  Gnosis Safe proxy (newer)
    - safe.execTransaction(to, value, data, operation, ..., signatures)
    - Standard Safe EIP-712 domain
    - masterCopy() returns the Safe singleton address

Run: .venv/bin/python scripts/detect_proxy_type.py
"""

from __future__ import annotations

import sys

from web3 import Web3

from config.settings import Settings


# Polygon public RPCs — read-only, no key needed for eth_call / eth_getCode.
# Tries them in order until one responds.
DEFAULT_RPCS = (
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon.drpc.org",
    "https://polygon-rpc.com",
)

# masterCopy() selector: keccak("masterCopy()")[:4] = 0xa619486e
# Present on Gnosis Safe proxies; reverts on 1155Proxy.
MASTER_COPY_SELECTOR = "0xa619486e"

# getImplementation() selector: 0xaaf10f42 — some newer Safe variants.
GET_IMPL_SELECTOR = "0xaaf10f42"

# Known Gnosis Safe singleton addresses on Polygon (any version)
KNOWN_SAFE_SINGLETONS = {
    # v1.3.0 L2 singleton
    "0xfb1bffc9d739b8d520daf37df666da4c687191ea",
    # v1.3.0 (non-L2) singleton
    "0xd9db270c1b5e3bd161e8c8503c55ceabee709552",
    # v1.1.1 singleton
    "0x34cfac646f301356faa8b21e94227e3583fe3f5f",
}


def main() -> int:
    s = Settings()
    proxy = s.polymarket_proxy_address
    if not proxy:
        print("ERROR: polymarket_proxy_address is not set in .env", file=sys.stderr)
        return 2

    w3 = None
    rpc_used = None
    for url in DEFAULT_RPCS:
        candidate = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 5}))
        try:
            if candidate.is_connected():
                w3 = candidate
                rpc_used = url
                break
        except Exception:
            continue
    if w3 is None:
        print("ERROR: cannot reach any Polygon public RPC", file=sys.stderr)
        return 2

    proxy = Web3.to_checksum_address(proxy)
    print(f"proxy:       {proxy}")
    print(f"rpc:         {rpc_used}")

    # Check that code exists at the address
    code = w3.eth.get_code(proxy)
    if code in (b"", b"\x00"):
        print("ERROR: no contract deployed at proxy address", file=sys.stderr)
        return 2
    print(f"code_size:   {len(code)} bytes")

    # Try masterCopy()
    def try_call(selector: str) -> str | None:
        try:
            raw = w3.eth.call({"to": proxy, "data": selector})
        except Exception:
            return None
        if not raw or len(raw) < 32:
            return None
        # address is last 20 bytes of 32-byte return
        addr_bytes = raw[-20:]
        if addr_bytes == b"\x00" * 20:
            return None
        return Web3.to_checksum_address("0x" + addr_bytes.hex())

    master_copy = try_call(MASTER_COPY_SELECTOR)
    get_impl = try_call(GET_IMPL_SELECTOR)

    print(f"masterCopy():      {master_copy or 'reverted/empty'}")
    print(f"getImplementation(): {get_impl or 'reverted/empty'}")

    singleton = master_copy or get_impl
    if singleton and singleton.lower() in KNOWN_SAFE_SINGLETONS:
        print()
        print("VERDICT: Gnosis Safe proxy")
        print(f"  singleton: {singleton}")
        print("  -> on-chain redemption uses Safe.execTransaction with Safe EIP-712 domain")
        return 0
    if singleton:
        print()
        print("VERDICT: unknown singleton behind a Safe-like proxy")
        print(f"  singleton: {singleton}")
        print("  -> probably Safe-compatible, but verify on Polygonscan before submitting")
        return 0

    # No masterCopy/getImplementation -> likely Polymarket's custom 1155Proxy
    print()
    print("VERDICT: Polymarket 1155Proxy (non-Safe)")
    print("  -> on-chain redemption uses proxy.proxy([...]) with Polymarket's custom EIP-712 domain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
