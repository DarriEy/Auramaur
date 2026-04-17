"""Probe the Polymarket proxy wallet's interface via the singleton.

Read-only. Tries known selectors to identify which proxy pattern this is.
"""

from __future__ import annotations

import sys

from eth_utils import keccak
from web3 import Web3

from config.settings import Settings


RPCS = (
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon.drpc.org",
)


# Selectors we want to probe. A selector is the first 4 bytes of keccak(signature).
PROBES: dict[str, str] = {
    # Polymarket-style "1155Proxy" / custom ProxyWallet
    "proxy((uint256,address,uint256,bytes)[])": "",
    "proxy(bytes)": "",
    # Gnosis Safe functions
    "execTransaction(address,uint256,bytes,uint8,uint256,uint256,uint256,address,address,bytes)": "",
    "getThreshold()": "",
    "getOwners()": "",
    "nonce()": "",
    "VERSION()": "",
    # Generic proxy / ownership
    "owner()": "",
    "implementation()": "",
    "masterCopy()": "",
    # Polymarket-specific
    "isValidSignature(bytes32,bytes)": "",
}


def sel(sig: str) -> str:
    return "0x" + keccak(text=sig).hex()[:8]


def main() -> int:
    s = Settings()
    proxy = s.polymarket_proxy_address
    if not proxy:
        print("ERROR: polymarket_proxy_address not set", file=sys.stderr)
        return 2

    w3 = None
    for url in RPCS:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 5}))
        if w3.is_connected():
            break
    if w3 is None:
        print("ERROR: no RPC reachable", file=sys.stderr)
        return 2

    proxy = Web3.to_checksum_address(proxy)

    # Fetch the singleton so we can also query it directly (avoids proxy
    # fallback shenanigans).
    try:
        raw = w3.eth.call({"to": proxy, "data": "0xa619486e"})  # masterCopy()
        singleton = Web3.to_checksum_address("0x" + raw[-20:].hex())
    except Exception:
        singleton = None

    print(f"proxy:     {proxy}")
    print(f"singleton: {singleton}")
    print()

    targets = [("proxy", proxy)]
    if singleton:
        targets.append(("singleton", singleton))

    for label, addr in targets:
        print(f"--- {label} {addr} ---")
        code = w3.eth.get_code(addr)
        print(f"code_size: {len(code)} bytes")

        for sig in PROBES:
            selector = sel(sig)
            try:
                result = w3.eth.call({"to": addr, "data": selector})
                # Non-empty result suggests selector is implemented. Returning
                # zero-length bytes can happen when fallback accepts anything.
                has = "yes" if result else "no"
                extra = ""
                if result and len(result) == 32:
                    # decode as address or uint
                    as_addr = "0x" + result[-20:].hex()
                    if int.from_bytes(result[-20:], "big") != 0:
                        extra = f" -> addr={Web3.to_checksum_address(as_addr)}"
                    else:
                        extra = f" -> uint={int.from_bytes(result, 'big')}"
                print(f"  [{has}] {selector} {sig}{extra}")
            except Exception as e:
                msg = str(e)[:80]
                print(f"  [revert] {selector} {sig}  ({msg})")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
