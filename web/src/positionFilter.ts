import type { Position } from "./types";

export function filterPositions(positions: Position[], query: string): Position[] {
  const needle = query.trim().toLowerCase();
  return positions
    .filter((p) => !needle || `${p.question} ${p.market_id} ${p.token} ${p.exchange}`
      .toLowerCase().includes(needle))
    .sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl));
}
