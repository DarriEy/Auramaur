"""Position reconciler — builds ground-truth positions from CLOB trade history.

Solves the ID mapping problem:
  Our DB uses numeric market IDs (e.g. "1339769")
  CLOB uses condition_ids (hex hashes) and asset_ids (76-digit token IDs)
  Gamma API bridges them via conditionId field

This module:
1. Fetches confirmed trades from CLOB API
2. Reconstructs net positions per token
3. Maps CLOB condition_ids to our market IDs via CLOB market lookup
4. Returns ground-truth positions with correct token_ids for selling
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from auramaur.exchange.models import LivePosition, TokenType

log = structlog.get_logger()


@dataclass
class ReconciledPosition:
    """A position with all three ID mappings resolved."""

    market_id: str          # Our numeric ID (for DB lookups)
    condition_id: str       # CLOB condition hash (for market queries)
    token_id: str           # CLOB asset_id (for placing sell orders)
    outcome: str            # "Yes" or "No"
    question: str           # Market question
    size: float             # Net token balance
    avg_cost: float = 0.0   # From cost_basis table
    current_price: float = 0.0


class PositionReconciler:
    """Reconciles positions between CLOB trade history and our DB."""

    def __init__(self, exchange, db):
        self._exchange = exchange
        self._db = db
        # Cache: condition_id -> CLOB market info
        self._market_cache: dict[str, dict] = {}

    async def reconcile(self) -> list[ReconciledPosition]:
        """Full reconciliation from CLOB trade history.

        1. Fetch all confirmed trades
        2. Reconstruct net positions per token
        3. Look up market info for each position
        4. Match to our DB market IDs
        """
        self._exchange._init_clob_client()
        client = self._exchange._clob_client
        proxy = self._exchange._settings.polymarket_proxy_address.lower()

        # Step 1: Get all trades
        try:
            trades = client.get_trades()
        except Exception as e:
            log.error("reconciler.trades_error", error=str(e))
            return []

        if not trades:
            return []

        # Step 2: Reconstruct positions from confirmed trades
        token_positions: dict[str, dict] = {}  # asset_id -> {net, total_cost, condition_id, outcome}

        for t in trades:
            if t.get("status") != "CONFIRMED":
                continue

            condition_id = t.get("market", "")
            asset_id = None
            side = None
            size = 0.0
            price = 0.0
            outcome = t.get("outcome", "")

            # Check if we're the maker
            for mo in t.get("maker_orders", []):
                if mo.get("maker_address", "").lower() == proxy:
                    asset_id = mo["asset_id"]
                    side = mo["side"]
                    size = float(mo["matched_amount"])
                    price = float(mo.get("price", 0))
                    outcome = mo.get("outcome", outcome)
                    break

            # Or the taker
            if not asset_id and t.get("trader_side") == "TAKER":
                asset_id = t["asset_id"]
                side = t["side"]
                size = float(t["size"])
                price = float(t.get("price", 0))

            if not asset_id or not side:
                continue

            if asset_id not in token_positions:
                token_positions[asset_id] = {
                    "net": 0.0,
                    "total_cost": 0.0,
                    "condition_id": condition_id,
                    "outcome": outcome,
                }
            if side == "BUY":
                token_positions[asset_id]["net"] += size
                token_positions[asset_id]["total_cost"] += size * price
            else:
                token_positions[asset_id]["net"] -= size
                token_positions[asset_id]["total_cost"] -= size * price

        # Filter to non-zero positions
        active = {
            k: v for k, v in token_positions.items() if v["net"] > 0.01
        }

        log.info("reconciler.positions_from_trades", total=len(active))

        # Step 3: Look up market info and match to our DB
        positions: list[ReconciledPosition] = []
        for asset_id, pos_data in active.items():
            condition_id = pos_data["condition_id"]

            # Get market info from CLOB (cached)
            market_info = await self._get_market_info(condition_id)
            if not market_info:
                continue

            question = market_info.get("question", "")
            slug = market_info.get("market_slug", "")

            # Find current price from tokens list
            current_price = 0.0
            tokens = market_info.get("tokens", [])
            for tok in tokens:
                if tok.get("token_id") == asset_id:
                    current_price = float(tok.get("price", 0))
                    break

            # Match to our DB market_id
            market_id = await self._find_market_id(condition_id, question, slug)

            # Compute avg cost from real trade data.
            # Cost basis can go weird (>1.0 or negative) for dust positions
            # where partial sells made the cost/size ratio meaningless.
            # Clamp to a sensible range so downstream exit logic doesn't fire
            # stop-loss on fictitious -98% losses built from rounding artifacts.
            net = pos_data["net"]
            total_cost = pos_data["total_cost"]
            avg_cost = total_cost / net if net > 0 else 0.0
            if avg_cost <= 0 or avg_cost > 1.0:
                # Fall back to current price — treats the dust as flat P&L.
                avg_cost = current_price if current_price > 0 else 0.5

            positions.append(ReconciledPosition(
                market_id=market_id or condition_id[:16],
                condition_id=condition_id,
                token_id=asset_id,
                outcome=pos_data["outcome"],
                question=question,
                size=pos_data["net"],
                avg_cost=avg_cost,
                current_price=current_price,
            ))

            # Register token mapping for sells
            self._exchange.register_market_tokens(
                market_id or condition_id[:16],
                # Map YES/NO tokens
                *self._extract_token_pair(tokens, asset_id, pos_data["outcome"]),
            )

        log.info(
            "reconciler.complete",
            positions=len(positions),
            total_tokens=sum(p.size for p in positions),
        )

        return positions

    async def _get_market_info(self, condition_id: str) -> dict | None:
        """Fetch market info from CLOB, with caching."""
        if condition_id in self._market_cache:
            return self._market_cache[condition_id]

        try:
            info = self._exchange._clob_client.get_market(condition_id)
            if info:
                self._market_cache[condition_id] = info
                return info
        except Exception as e:
            log.debug("reconciler.market_lookup_error",
                      condition_id=condition_id[:20], error=str(e))
        return None

    async def _find_market_id(
        self, condition_id: str, question: str, slug: str,
    ) -> str | None:
        """Match a CLOB condition_id to our numeric market_id via DB."""
        # Try matching by condition_id
        row = await self._db.fetchone(
            "SELECT id FROM markets WHERE condition_id = ?",
            (condition_id,),
        )
        if row:
            return row["id"]

        # Try matching by question text (fuzzy)
        if question:
            row = await self._db.fetchone(
                "SELECT id FROM markets WHERE question = ?",
                (question,),
            )
            if row:
                return row["id"]

        # Try matching by slug / ticker
        if slug:
            row = await self._db.fetchone(
                "SELECT id FROM markets WHERE ticker = ?",
                (slug,),
            )
            if row:
                return row["id"]

        # No match — insert a stub so exits and risk checks can find it
        if question and condition_id:
            from datetime import datetime, timezone
            stub_id = condition_id[:16]
            try:
                await self._db.execute(
                    """INSERT OR IGNORE INTO markets
                       (id, condition_id, question, last_updated)
                       VALUES (?, ?, ?, ?)""",
                    (stub_id, condition_id, question,
                     datetime.now(timezone.utc).isoformat()),
                )
                await self._db.commit()
                log.info("reconciler.stub_market_created",
                         market_id=stub_id, question=question[:60])
            except Exception:
                pass
            return stub_id

        return None

    @staticmethod
    def _extract_token_pair(
        tokens: list[dict], held_asset_id: str, held_outcome: str,
    ) -> tuple[str, str]:
        """Extract (clob_yes, clob_no) from CLOB token list."""
        yes_id = ""
        no_id = ""
        for tok in tokens:
            if tok.get("outcome") == "Yes":
                yes_id = tok.get("token_id", "")
            elif tok.get("outcome") == "No":
                no_id = tok.get("token_id", "")
        return yes_id, no_id

    async def repair_orphaned_ids(self, reconciled: list[ReconciledPosition]) -> int:
        """Fix cost_basis/portfolio entries that use truncated condition_ids.

        When the reconciler previously couldn't match a condition_id to a
        market, it stored condition_id[:16] as the market_id.  Now that we
        may have the real mapping, update those rows.

        Returns number of rows repaired.
        """
        repaired = 0
        for pos in reconciled:
            if not pos.market_id or pos.market_id == pos.condition_id[:16]:
                continue  # Still unresolved

            orphan_id = pos.condition_id[:16]
            # Check if cost_basis has the orphan ID
            row = await self._db.fetchone(
                "SELECT market_id FROM cost_basis WHERE market_id = ?",
                (orphan_id,),
            )
            if row:
                await self._db.execute(
                    "UPDATE cost_basis SET market_id = ? WHERE market_id = ?",
                    (pos.market_id, orphan_id),
                )
                await self._db.execute(
                    "UPDATE portfolio SET market_id = ? WHERE market_id = ?",
                    (pos.market_id, orphan_id),
                )
                await self._db.execute(
                    "UPDATE fills SET market_id = ? WHERE market_id = ?",
                    (pos.market_id, orphan_id),
                )
                repaired += 1
                log.info("reconciler.id_repaired",
                         orphan_id=orphan_id, real_id=pos.market_id)

        if repaired:
            await self._db.commit()
        return repaired

    def to_live_positions(
        self, reconciled: list[ReconciledPosition],
    ) -> list[LivePosition]:
        """Convert reconciled positions to LivePosition objects."""
        return [
            LivePosition(
                market_id=p.market_id,
                token_id=p.token_id,
                token=TokenType.YES if p.outcome == "Yes" else TokenType.NO,
                size=p.size,
                avg_cost=p.avg_cost,
                current_price=p.current_price,
                market_question=p.question,
            )
            for p in reconciled
        ]
