"""Polymarket maker-rebate ingestion and accounting."""

from __future__ import annotations

import aiohttp


class MakerRebateSync:
    def __init__(self, db, maker_address: str) -> None:
        self.db = db
        self.maker_address = maker_address

    async def sync(self, date: str) -> float:
        if not self.maker_address:
            return 0.0
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                "https://clob.polymarket.com/rebates/current",
                params={"date": date, "maker_address": self.maker_address},
            ) as response:
                response.raise_for_status()
                rows = await response.json()
        total = 0.0
        for row in rows:
            amount = float(row.get("rebated_fees_usdc", 0) or 0)
            total += amount
            await self.db.execute(
                """INSERT OR REPLACE INTO maker_rebates
                   (date, condition_id, maker_address, rebate_usdc)
                   VALUES (?, ?, ?, ?)""",
                (date, str(row.get("condition_id", "")), self.maker_address, amount),
            )
        await self.db.commit()
        return total
