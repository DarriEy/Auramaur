import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { DashboardState } from "./types";
import { OperatorWorkspace, WorkspaceNav } from "./workspace";

const state: DashboardState = {
  now: "2026-07-21T12:00:00Z", is_live: false, transfers_armed: false,
  kill_switch: false, venues: {}, ibkr_books: [], pillars: [], activity: [],
  health: { errors: 0, warnings: 0, top: [] }, positions: [], position_count: 0,
  position_value: 0, signals: [], trade_count: 0, total_pnl: 0, drawdown: 0,
  balance: 100, strategies: [], categories: [], performance_history: [],
};

describe("operator workspace", () => {
  it("renders task-based navigation with an active destination", () => {
    const html = renderToStaticMarkup(<WorkspaceNav view="execution" onView={() => undefined} />);
    expect(html).toContain("Dashboard sections");
    expect(html).toContain('aria-current="page"');
    expect(html).toContain('href="#execution"');
    expect(html).toContain("Opportunities");
  });

  it("falls back to the overview and presents an all-clear posture", () => {
    const html = renderToStaticMarkup(
      <OperatorWorkspace s={state} initialView={"not-a-view" as never} />
    );
    expect(html).toContain("All monitored systems are operational");
    expect(html).toContain("Baseline saved");
    expect(html).toContain("Portfolio");
  });

  it("does not report unavailable reconciliation as healthy", () => {
    const html = renderToStaticMarkup(
      <OperatorWorkspace s={{ ...state, reconciliation: undefined }} initialView="execution" />
    );
    expect(html).toContain("Reconciliation unavailable");
    expect(html).not.toContain("No venue/database discrepancies");
  });

  it("shows explicit empty states and no dead intelligence actions", () => {
    const html = renderToStaticMarkup(
      <OperatorWorkspace s={state} initialView="intelligence" />
    );
    expect(html).toContain("No resolved forecasts are available yet");
    expect(html).not.toContain("Open workspace");
  });
});
