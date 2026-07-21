import { useEffect, useMemo, useState } from "react";
import type { DashboardState, HealthTop, Position, Signal } from "./types";
import { ago, deltaClass, money, price } from "./format";

export type View = "overview" | "portfolio" | "opportunities" | "execution" | "intelligence" | "system";
type Inspectable =
  | { kind: "position"; value: Position }
  | { kind: "signal"; value: Signal }
  | { kind: "health"; value: HealthTop }
  | { kind: "reconciliation"; value: Record<string, unknown> };

const VIEWS: { id: View; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "portfolio", label: "Portfolio" },
  { id: "opportunities", label: "Opportunities" },
  { id: "execution", label: "Execution" },
  { id: "intelligence", label: "Intelligence" },
  { id: "system", label: "System" },
];

function useStored(key: string, fallback: string) {
  const [value, setValue] = useState(() => typeof localStorage === "undefined" ? fallback : localStorage.getItem(key) ?? fallback);
  useEffect(() => { if (typeof localStorage !== "undefined") localStorage.setItem(key, value); }, [key, value]);
  return [value, setValue] as const;
}

function SinceLastVisit({ s, scope }: { s: DashboardState; scope: string }) {
  const key = `auramaur.last-visit.${scope}`;
  const [previous] = useState<{ at: string; pnl: number; positions: number; fills: number } | null>(() => {
    if (typeof localStorage === "undefined") return null;
    try { return JSON.parse(localStorage.getItem(key) ?? "null"); } catch { return null; }
  });
  useEffect(() => {
    if (typeof localStorage !== "undefined") localStorage.setItem(key, JSON.stringify({
      at: new Date().toISOString(), pnl: s.total_pnl, positions: s.position_count, fills: s.trade_count,
    }));
  }, [key, s.total_pnl, s.position_count, s.trade_count]);
  if (!previous) return <p className="visit-summary">Baseline saved. Changes will appear on your next visit.</p>;
  return <p className="visit-summary">
    Since {isoAge(previous.at)}: <strong className={deltaClass(s.total_pnl - previous.pnl)}>{money(s.total_pnl - previous.pnl, true)} P&amp;L</strong>
    {" · "}{s.position_count - previous.positions >= 0 ? "+" : ""}{s.position_count - previous.positions} positions
    {" · "}{s.trade_count - previous.fills >= 0 ? "+" : ""}{s.trade_count - previous.fills} fills
  </p>;
}

function isoAge(value?: string | null) {
  if (!value) return "unknown";
  const stamp = new Date(value).getTime();
  return Number.isFinite(stamp) ? ago((Date.now() - stamp) / 1000) : value;
}

function safeJson(value: unknown) {
  return JSON.stringify(value, null, 2);
}

async function copyText(value: string) {
  await navigator.clipboard.writeText(value);
}

function activateRow(event: React.KeyboardEvent, action: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

function download(name: string, value: unknown) {
  const blob = new Blob([safeJson(value)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function WorkspaceNav({ view, onView }: { view: View; onView: (view: View) => void }) {
  return (
    <nav className="workspace-nav" aria-label="Dashboard sections">
      {VIEWS.map((item) => (
        <button
          className={view === item.id ? "active" : ""}
          key={item.id}
          aria-current={view === item.id ? "page" : undefined}
          onClick={() => onView(item.id)}
        >
          {item.label}
        </button>
      ))}
    </nav>
  );
}

function AttentionCenter({
  s,
  onView,
  onInspect,
}: {
  s: DashboardState;
  onView: (view: View) => void;
  onInspect: (item: Inspectable) => void;
}) {
  const stale = Object.values(s.venues).filter((venue) => (venue.age_seconds ?? Infinity) > 900).length;
  const dead = s.pillars.filter((pillar) => (pillar.age_seconds ?? Infinity) >= 600).length;
  const recon = s.reconciliation;
  const mismatches = recon?.available && !recon.in_sync
    ? recon.missing.length + recon.extra.length + recon.size_mismatches.length
    : 0;
  const cards = [
    { count: s.health.errors, label: "errors", tone: "critical", view: "system" as View },
    { count: s.health.warnings, label: "warnings", tone: "warning", view: "system" as View },
    { count: stale + dead, label: "stale sources", tone: "warning", view: "system" as View },
    { count: mismatches, label: "sync mismatches", tone: "serious", view: "execution" as View },
  ].filter((card) => card.count > 0);
  if (!cards.length) {
    return <div className="all-clear" role="status">✓ All monitored systems are operational</div>;
  }
  return (
    <section className="attention" aria-labelledby="attention-title">
      <div>
        <span className="eyebrow">Needs attention</span>
        <h2 id="attention-title">{cards.reduce((sum, card) => sum + card.count, 0)} items to review</h2>
      </div>
      <div className="attention-items">
        {cards.map((card) => (
          <button
            className={"attention-item " + card.tone}
            key={card.label}
            onClick={() => {
              onView(card.view);
              if (card.label === "sync mismatches" && recon) {
                onInspect({ kind: "reconciliation", value: recon as unknown as Record<string, unknown> });
              }
            }}
          >
            <strong>{card.count}</strong><span>{card.label}</span><span aria-hidden="true">→</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <span className="spark-empty">Trend builds as daily marks arrive</span>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const points = values.map((value, index) =>
    `${(index / (values.length - 1)) * 100},${28 - ((value - min) / spread) * 24}`
  ).join(" ");
  return (
    <svg className="spark" viewBox="0 0 100 32" role="img" aria-label="Recent P and L trend">
      <polyline points={points} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Overview({ s, scope, historyIsCurrent, onView, onInspect }: {
  s: DashboardState; scope: string; historyIsCurrent: boolean;
  onView: (v: View) => void; onInspect: (i: Inspectable) => void;
}) {
  const history = historyIsCurrent ? s.performance_history ?? [] : [];
  const today = history.at(-1);
  const prior = history.at(-2);
  const dailyDelta = today && prior ? today.total_pnl - prior.total_pnl : null;
  const largest = [...s.positions].sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl))[0];
  const newest = s.signals[0];
  return (
    <>
      <SinceLastVisit s={s} scope={scope} />
      <AttentionCenter s={s} onView={onView} onInspect={onInspect} />
      <section className="hero-metrics" aria-label="Account summary">
        <article><span>Capital</span><strong>{money(s.balance)}</strong><small>{money(s.position_value)} deployed</small></article>
        <article><span>Total P&amp;L</span><strong className={deltaClass(s.total_pnl)}>{money(s.total_pnl, true)}</strong>
          <small className={deltaClass(dailyDelta)}>{dailyDelta == null ? "Daily delta unavailable" : `${money(dailyDelta, true)} today`}</small>
        </article>
        <article className="metric-trend"><span>14-day performance</span><Sparkline values={history.map((row) => row.total_pnl)} />
          <small>{history.length} daily marks</small></article>
        <article><span>Open risk</span><strong>{s.position_count} positions</strong><small>Max drawdown {s.drawdown.toFixed(1)}%</small></article>
      </section>
      <div className="summary-grid">
        <SummaryCard title="Portfolio" status={largest ? "Review largest mover" : "No open exposure"} onOpen={() => onView("portfolio")}>
          {largest && <button className="summary-row" onClick={() => onInspect({ kind: "position", value: largest })}>
            <span>{largest.question}</span><strong className={deltaClass(largest.pnl)}>{money(largest.pnl, true)}</strong>
          </button>}
          <p>{s.categories.slice(0, 3).map((c) => `${c.category} ${money(c.value)}`).join(" · ") || "No category exposure"}</p>
        </SummaryCard>
        <SummaryCard title="Opportunities" status={`${s.signals.length} recent signals`} onOpen={() => onView("opportunities")}>
          {newest && <button className="summary-row" onClick={() => onInspect({ kind: "signal", value: newest })}>
            <span>{newest.question}</span><strong>{newest.edge == null ? "—" : `${newest.edge.toFixed(1)}% edge`}</strong>
          </button>}
          <p>{newest ? `${newest.strategy_source} · ${isoAge(newest.timestamp)}` : "No recent signals"}</p>
        </SummaryCard>
        <SummaryCard title="Execution" status={s.reconciliation?.in_sync ? "In sync" : "Review required"} onOpen={() => onView("execution")}>
          <p>{s.trade_count} fills · {Object.keys(s.venues).length} venues</p>
          <p>Last venue snapshot {isoAge(s.reconciliation?.fetched_at)}</p>
        </SummaryCard>
        <SummaryCard title="System" status={s.health.errors ? `${s.health.errors} errors` : "Healthy"} onOpen={() => onView("system")}>
          <p>{s.pillars.filter((p) => (p.age_seconds ?? Infinity) < 90).length}/{s.pillars.length} pillars fresh</p>
          <p>{s.health.warnings} warnings in current log window</p>
        </SummaryCard>
      </div>
    </>
  );
}

function SummaryCard({ title, status, onOpen, children }: {
  title: string; status: string; onOpen: () => void; children: React.ReactNode;
}) {
  return (
    <article className="summary-card">
      <header><div><span className="eyebrow">{title}</span><h2>{status}</h2></div>
        <button className="text-button" onClick={onOpen}>Open workspace →</button></header>
      {children}
    </article>
  );
}

function Portfolio({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  const [query, setQuery] = useStored("auramaur.positions.query", "");
  const [venue, setVenue] = useStored("auramaur.positions.venue", "all");
  const [preset, setPreset] = useStored("auramaur.positions.preset", "all");
  const [sort, setSort] = useStored("auramaur.positions.sort", "pnl");
  const venues = [...new Set(s.positions.map((p) => p.exchange))].sort();
  const rows = useMemo(() => {
    const q = query.toLowerCase();
    const now = new Date(s.now).getTime();
    return s.positions.filter((p) => {
      if (venue !== "all" && p.exchange !== venue) return false;
      if (q && ![p.question, p.market_id, p.token, p.category, p.exchange].some((v) => v?.toLowerCase().includes(q))) return false;
      if (preset === "review" && Math.abs(p.pnl) < 1) return false;
      if (preset === "movers" && Math.abs(p.pnl) < 0.5) return false;
      if (preset === "stale" && (!p.updated_at || now - new Date(p.updated_at).getTime() < 15 * 60_000)) return false;
      if (preset === "resolution" && (!p.end_date || new Date(p.end_date).getTime() - now > 7 * 86400_000)) return false;
      return true;
    }).sort((a, b) => sort === "value"
      ? (b.current_value ?? 0) - (a.current_value ?? 0)
      : sort === "age" ? new Date(a.updated_at ?? 0).getTime() - new Date(b.updated_at ?? 0).getTime()
      : Math.abs(b.pnl) - Math.abs(a.pnl));
  }, [s.positions, s.now, query, venue, preset, sort]);
  return (
    <Workspace title="Portfolio" subtitle="Find exposure, stale marks, concentration, and positions requiring review."
      action={<button onClick={() => download("auramaur-positions.json", rows)}>Export JSON</button>}>
      <div className="toolbar">
        <input aria-label="Search positions" placeholder="Search market, ID, token, category…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <select aria-label="Filter venue" value={venue} onChange={(e) => setVenue(e.target.value)}>
          <option value="all">All venues</option>{venues.map((v) => <option key={v}>{v}</option>)}
        </select>
        <select aria-label="Position preset" value={preset} onChange={(e) => setPreset(e.target.value)}>
          <option value="all">All positions</option><option value="review">Needs review</option>
          <option value="movers">Largest movers</option><option value="stale">Stale marks</option>
          <option value="resolution">Resolving soon</option>
        </select>
        <select aria-label="Sort positions" value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="pnl">Sort: absolute P&amp;L</option><option value="value">Sort: exposure</option><option value="age">Sort: stalest</option>
        </select>
      </div>
      <p className="result-count">{rows.length} of {s.positions.length} positions · filters save automatically</p>
      <DataTable headers={["Market", "Venue", "Category", "Outcome", "Exposure", "Mark", "P&L", "Updated"]}>
        {rows.map((p) => {
          const open = () => onInspect({ kind: "position", value: p } as Inspectable);
          return <tr key={`${p.market_id}-${p.token}`} tabIndex={0} onClick={open} onKeyDown={(event) => activateRow(event, open)}>
          <td className="market">{p.question}</td><td>{p.exchange}</td><td>{p.category ?? "—"}</td>
          <td>{p.token}</td><td className="num">{money(p.current_value)}</td><td className="num">{price(p.current_price)}</td>
          <td className={`num ${deltaClass(p.pnl)}`}>{money(p.pnl, true)}</td><td>{isoAge(p.updated_at)}</td>
        </tr>;
        })}
      </DataTable>
      <div className="summary-grid workspace-breakdowns">
        <section className="section-block"><h2>Category concentration</h2>
          <DataTable headers={["Category", "Positions", "Exposure"]}>
            {s.categories.map((row) => <tr key={row.category}><td>{row.category}</td>
              <td className="num">{row.positions}</td><td className="num">{money(row.value)}</td></tr>)}
          </DataTable>
        </section>
        <section className="section-block"><h2>Strategy realized performance</h2>
          <DataTable headers={["Strategy", "Entries", "Fees", "Realized"]}>
            {s.strategies.map((row) => <tr key={row.strategy}><td>{row.strategy}</td>
              <td className="num">{row.entries}</td><td className="num">{money(row.fees)}</td>
              <td className={`num ${deltaClass(row.pnl)}`}>{money(row.pnl, true)}</td></tr>)}
          </DataTable>
        </section>
      </div>
    </Workspace>
  );
}

function Opportunities({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  const grouped = useMemo(() => {
    const map = new Map<string, Signal[]>();
    s.signals.forEach((signal) => map.set(signal.market_id, [...(map.get(signal.market_id) ?? []), signal]));
    return [...map.values()].sort((a, b) => Math.abs(b[0].edge ?? 0) - Math.abs(a[0].edge ?? 0));
  }, [s.signals]);
  return (
    <Workspace title="Opportunities" subtitle="Signals grouped by market so repeated evaluations read as one decision timeline."
      action={<button onClick={() => download("auramaur-signals.json", s.signals)}>Export JSON</button>}>
      <DataTable headers={["Market", "Evaluations", "Strategy", "Model", "Market", "Edge", "Action", "Observed"]}>
        {grouped.map((signals) => {
          const sig = signals[0];
          const open = () => onInspect({ kind: "signal", value: sig });
          return <tr key={sig.market_id} tabIndex={0} onClick={open} onKeyDown={(event) => activateRow(event, open)}>
            <td className="market">{sig.question}</td><td className="num">{signals.length}</td><td>{sig.strategy_source}</td>
            <td className="num">{price(sig.claude_prob)}</td><td className="num">{price(sig.market_prob)}</td>
            <td className={`num ${deltaClass(sig.edge)}`}>{sig.edge == null ? "—" : `${sig.edge.toFixed(1)}%`}</td>
            <td><span className="action-chip">{sig.action}</span></td><td>{isoAge(sig.timestamp)}</td>
          </tr>;
        })}
      </DataTable>
    </Workspace>
  );
}

function Execution({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  const recon = s.reconciliation;
  const discrepancies = recon ? [
    ...recon.missing.map((v) => ({ type: "Missing locally", ...v as object })),
    ...recon.extra.map((v) => ({ type: "Extra locally", ...v as object })),
    ...recon.size_mismatches.map((v) => ({ type: "Quantity mismatch", ...v as object })),
  ] as Record<string, unknown>[] : [];
  return (
    <Workspace title="Execution" subtitle="Venue posture, fills, and exact reconciliation differences."
      action={<button disabled={!recon} onClick={() => recon && download("auramaur-reconciliation.json", recon)}>Export reconciliation</button>}>
      <div className="venue-grid">{Object.entries(s.venues).map(([name, venue]) =>
        <article className="venue-card" key={name}><header><strong>{name}</strong><Status age={venue.age_seconds} /></header>
          <p>{venue.detail}</p><small>Recorded {ago(venue.age_seconds)}</small></article>)}</div>
      <section className="section-block"><header className="section-heading"><div><span className="eyebrow">Reconciliation</span>
        <h2>{recon?.in_sync ? "Venue and ledger agree" : `${discrepancies.length} discrepancies`}</h2></div>
        {recon && <small>Snapshot {isoAge(recon.fetched_at)}</small>}</header>
        {discrepancies.length ? <DataTable headers={["Type", "Market / asset", "Outcome", "Venue qty", "Ledger qty"]}>
          {discrepancies.map((row, index) => {
            const open = () => onInspect({ kind: "reconciliation", value: row });
            return <tr key={index} tabIndex={0} onClick={open} onKeyDown={(event) => activateRow(event, open)}>
            <td>{String(row.type)}</td><td className="market">{String(row.title ?? row.market_id ?? row.asset_id ?? "—")}</td>
            <td>{String(row.outcome ?? row.token ?? "—")}</td><td className="num">{String(row.venue_size ?? row.size ?? "—")}</td>
            <td className="num">{String(row.db_size ?? "—")}</td>
          </tr>;
          })}</DataTable> : <div className="all-clear">✓ No venue/database discrepancies</div>}
      </section>
      {(s.kraken_paper ?? []).length > 0 && <section className="section-block"><h2>Kraken paper positions</h2>
        <DataTable headers={["Pair", "Strategy", "Quantity", "Entry", "Peak gain", "Opened"]}>
          {(s.kraken_paper ?? []).map((row) => <tr key={`${row.strategy}-${row.pair}`}><td>{row.pair}</td>
            <td>{row.strategy}</td><td className="num">{row.quantity.toFixed(4)}</td>
            <td className="num">{row.entry_price.toFixed(2)}</td><td className="num">{row.peak_gain_pct.toFixed(1)}%</td>
            <td>{isoAge(row.opened_at)}</td></tr>)}
        </DataTable>
      </section>}
    </Workspace>
  );
}

function Intelligence({ s }: { s: DashboardState }) {
  const llm = s.local_llm;
  return <Workspace title="Intelligence" subtitle="Model throughput, reliability, calibration, and evidence output.">
    <div className="summary-grid">
      <SummaryCard title="Evidence distiller" status={`${llm?.claims_24h ?? 0} claims in 24h`} onOpen={() => undefined}>
        <p>Last claim {isoAge(llm?.last_claim_at)}</p>
      </SummaryCard>
      {llm && Object.entries(llm.purposes).map(([purpose, stat]) =>
        <SummaryCard key={purpose} title={purpose} status={`${stat.ok}/${stat.calls} successful`} onOpen={() => undefined}>
          <p>{stat.errors} errors · {stat.avg_ms ?? "—"}ms average</p><p>{stat.prompt_tokens + stat.output_tokens} tokens</p>
        </SummaryCard>)}
    </div>
    <section className="section-block"><h2>Resolved forecast scorecard</h2>
      <DataTable headers={["Arm", "Model", "Forecasts", "Brier", "Market Brier", "Abstains"]}>
        {(s.intelligence_eval ?? []).map((row) => <tr key={row.arm}><td>{row.arm}</td><td>{row.model}</td>
          <td className="num">{row.forecasts}</td><td className="num">{row.brier ?? "—"}</td>
          <td className="num">{row.market_brier ?? "—"}</td><td className="num">{row.abstains}</td></tr>)}
      </DataTable>
    </section>
  </Workspace>;
}

function System({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  return <Workspace title="System" subtitle="Source liveness and diagnostics ordered for investigation."
    action={<button onClick={() => download("auramaur-diagnostics.json", { pillars: s.pillars, health: s.health, activity: s.activity })}>Export diagnostics</button>}>
    <div className="pillar-grid">{s.pillars.map((pillar) =>
      <article className="pillar-card" key={pillar.name}><Status age={pillar.age_seconds} /><div><strong>{pillar.name}</strong>
        <small>Last event {ago(pillar.age_seconds)}</small></div></article>)}</div>
    <section className="section-block"><header className="section-heading"><div><span className="eyebrow">Diagnostics</span>
      <h2>{s.health.errors} errors · {s.health.warnings} warnings</h2></div></header>
      <DataTable headers={["Occurrences", "Event", "Latest detail", "Last seen"]}>
        {s.health.top.map((event) => {
          const open = () => onInspect({ kind: "health", value: event });
          return <tr key={event.event} tabIndex={0} onClick={open} onKeyDown={(keyEvent) => activateRow(keyEvent, open)}>
          <td className="num">{event.count}</td><td>{event.event}</td><td className="market">{event.last_msg || "No message recorded"}</td>
          <td>{isoAge(event.last_ts)}</td></tr>;
        })}
      </DataTable>
    </section>
    <section className="section-block"><h2>Recent operational activity</h2>
      <div className="timeline">{[...s.activity].reverse().map((item, index) =>
        <div key={index}><time>{item.time}</time><span>{item.text}</span></div>)}</div>
    </section>
  </Workspace>;
}

function Status({ age }: { age: number | null }) {
  const state = age == null ? "unknown" : age < 90 ? "fresh" : age < 600 ? "stale" : "dead";
  return <span className={`status ${state}`}><span aria-hidden="true" />{state}</span>;
}

function Workspace({ title, subtitle, action, children }: {
  title: string; subtitle: string; action?: React.ReactNode; children: React.ReactNode;
}) {
  return <main id="main-content" className="workspace"><header className="workspace-heading"><div><h1>{title}</h1><p>{subtitle}</p></div>{action}</header>{children}</main>;
}

function DataTable({ headers, children }: { headers: string[]; children: React.ReactNode }) {
  return <div className="data-table-wrap"><table className="data-table"><thead><tr>{headers.map((header) =>
    <th key={header} className={["Exposure", "Mark", "P&L", "Edge", "Model", "Market", "Occurrences", "Forecasts", "Brier", "Market Brier", "Abstains", "Venue qty", "Ledger qty", "Evaluations", "Positions", "Entries", "Fees", "Realized", "Quantity", "Entry", "Peak gain"].includes(header) ? "num" : ""}>{header}</th>)}
  </tr></thead><tbody>{children}</tbody></table></div>;
}

function Inspector({ item, onClose }: { item: Inspectable | null; onClose: () => void }) {
  useEffect(() => {
    const close = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);
  if (!item) return null;
  const value = item.value as unknown as Record<string, unknown>;
  const title = item.kind === "position" || item.kind === "signal"
    ? String(value.question) : item.kind === "health" ? String(value.event) : "Reconciliation detail";
  const checklist = item.kind === "health"
    ? ["Confirm the latest occurrence time", "Review the event message and related activity", "Check the affected pillar or venue", "Copy diagnostics before remediation"]
    : item.kind === "reconciliation"
      ? ["Confirm snapshot freshness", "Compare venue and ledger quantities", "Copy the asset or market ID", "Run the position sync/reconciliation workflow"]
      : ["Confirm source freshness", "Review exposure and related signal", "Copy the market ID for deeper investigation"];
  return <div className="inspector-backdrop" onMouseDown={onClose}>
    <aside className="inspector" role="dialog" aria-modal="true" aria-labelledby="inspector-title" onMouseDown={(e) => e.stopPropagation()}>
      <header><div><span className="eyebrow">{item.kind}</span><h2 id="inspector-title">{title}</h2></div>
        <button className="icon-button" aria-label="Close details" onClick={onClose}>×</button></header>
      <dl>{Object.entries(value).filter(([, v]) => v !== "" && v != null && typeof v !== "object").map(([key, val]) =>
        <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{String(val)}</dd></div>)}</dl>
      <section><h3>Investigation checklist</h3><ol>{checklist.map((step) => <li key={step}>{step}</li>)}</ol></section>
      <div className="inspector-actions">
        {"market_id" in value && <button onClick={() => copyText(String(value.market_id))}>Copy market ID</button>}
        {"asset_id" in value && <button onClick={() => copyText(String(value.asset_id))}>Copy asset ID</button>}
        <button onClick={() => copyText(safeJson(value))}>Copy details</button>
        <button onClick={() => download(`auramaur-${item.kind}.json`, value)}>Export JSON</button>
      </div>
    </aside>
  </div>;
}

export function OperatorWorkspace({ s, scope = "default", historyIsCurrent = true, initialView = "overview" }: {
  s: DashboardState; scope?: string; historyIsCurrent?: boolean; initialView?: View;
}) {
  const safeInitial = VIEWS.some((candidate) => candidate.id === initialView) ? initialView : "overview";
  const [view, setView] = useState<View>(safeInitial);
  const [inspect, setInspect] = useState<Inspectable | null>(null);
  const changeView = (next: View) => {
    setView(next);
    history.replaceState(null, "", `#${next}`);
    document.getElementById("main-content")?.focus();
  };
  return <>
    <a className="skip-link" href="#main-content">Skip to dashboard content</a>
    <WorkspaceNav view={view} onView={changeView} />
    {view === "overview" && <main id="main-content" tabIndex={-1}><Overview s={s} scope={scope} historyIsCurrent={historyIsCurrent} onView={changeView} onInspect={setInspect} /></main>}
    {view === "portfolio" && <Portfolio s={s} onInspect={setInspect} />}
    {view === "opportunities" && <Opportunities s={s} onInspect={setInspect} />}
    {view === "execution" && <Execution s={s} onInspect={setInspect} />}
    {view === "intelligence" && <Intelligence s={s} />}
    {view === "system" && <System s={s} onInspect={setInspect} />}
    <Inspector item={inspect} onClose={() => setInspect(null)} />
  </>;
}
