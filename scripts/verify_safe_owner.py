"""Verify that the configured EOA is an owner of the Polymarket Safe proxy.

Does NOT log the private key — only the derived EOA address.
"""

from __future__ import annotations

import sys

from eth_account import Account
from web3 import Web3

from config.settings import Settings


RPCS = (
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon.drpc.org",
)


def main() -> int:
    s = Settings()
    proxy = s.polymarket_proxy_address
    pk = s.polygon_private_key

    if not proxy or not pk:
        print("ERROR: missing polymarket_proxy_address or polygon_private_key", file=sys.stderr)
        return 2

    eoa = Account.from_key(pk).address

    w3 = None
    for url in RPCS:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 5}))
        if w3.is_connected():
            break
    if w3 is None:
        print("ERROR: no RPC", file=sys.stderr)
        return 2

    proxy = Web3.to_checksum_address(proxy)
    eoa = Web3.to_checksum_address(eoa)

    # ABI fragment for getOwners()
    safe = w3.eth.contract(
        address=proxy,
        abi=[
            {"name": "getOwners", "type": "function", "stateMutability": "view",
             "inputs": [], "outputs": [{"name": "", "type": "address[]"}]},
            {"name": "getThreshold", "type": "function", "stateMutability": "view",
             "inputs": [], "outputs": [{"name": "", "type": "uint256"}]},
            {"name": "nonce", "type": "function", "stateMutability": "view",
             "inputs": [], "outputs": [{"name": "", "type": "uint256"}]},
            {"name": "VERSION", "type": "function", "stateMutability": "view",
             "inputs": [], "outputs": [{"name": "", "type": "string"}]},
        ],
    )

    owners = safe.functions.getOwners().call()
    threshold = safe.functions.getThreshold().call()
    nonce = safe.functions.nonce().call()
    try:
        version = safe.functions.VERSION().call()
    except Exception:
        version = "unknown"

    owners = [Web3.to_checksum_address(o) for o in owners]

    print(f"safe:      {proxy}")
    print(f"eoa:       {eoa}")
    print(f"owners:    {owners}")
    print(f"threshold: {threshold}")
    print(f"nonce:     {nonce}")
    print(f"version:   {version}")
    print()

    if eoa in owners:
        print("OK — EOA is an owner of the Safe. Single-sig submission will work.")
        return 0
    print("FAIL — EOA is NOT an owner. execTransaction would revert.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
