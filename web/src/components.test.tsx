import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PositionsPanel } from "./components";
import { filterPositions } from "./positionFilter";
import type { Position } from "./types";

function positions(count: number): Position[] {
  return Array.from({ length: count }, (_, i) => ({
    market_id: `market-${i}`,
    question: i === count - 1 ? "Needle market" : `Market ${i}`,
    token: i % 2 ? "NO" : "YES",
    exchange: "polymarket",
    side: "BUY",
    size: i + 1,
    avg_price: .4,
    current_price: .5,
    pnl: i - 5,
  }));
}

describe("position book", () => {
  it("keeps and renders every row in a book larger than the old cap", () => {
    const rows = positions(25);
    expect(filterPositions(rows, "")).toHaveLength(25);
    const html = renderToStaticMarkup(<PositionsPanel positions={rows} />);
    expect(html).toContain("Search all 25 positions");
    expect(html).toContain("Market 0");
    expect(html).toContain("Needle market");
    expect((html.match(/<tr>/g) ?? []).length).toBe(26); // header + 25 rows
  });

  it("searches question, market id, outcome and venue", () => {
    const rows = positions(25);
    expect(filterPositions(rows, "needle").map((p) => p.market_id))
      .toEqual(["market-24"]);
    expect(filterPositions(rows, "market-7").map((p) => p.market_id))
      .toEqual(["market-7"]);
    expect(filterPositions(rows, "NO")).toHaveLength(12);
    expect(filterPositions(rows, "POLYMARKET")).toHaveLength(25);
  });

  it("sorts matching rows by absolute P&L", () => {
    const rows = positions(10);
    const found = filterPositions(rows, "polymarket");
    expect(found[0].pnl).toBe(-5);
    expect(Math.abs(found[0].pnl)).toBeGreaterThanOrEqual(
      Math.abs(found.at(-1)?.pnl ?? 0));
  });
});
