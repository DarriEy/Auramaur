import { useEffect, useMemo, useRef, useState } from "react";
import type { DashboardState, HealthTop, Position, Signal } from "./types";
import { ago, compactNumber, deltaClass, exactTime, humanize, money, price } from "./format";

export type View = "overview" | "portfolio" | "opportunities" | "compare" | "execution" | "intelligence" | "system";
type Inspectable =
  | { kind: "position"; value: Position }
  | { kind: "signal"; value: Signal; related?: Signal[] }
  | { kind: "health"; value: HealthTop }
  | { kind: "reconciliation"; value: Record<string, unknown> };

const VIEWS: { id: View; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "portfolio", label: "Portfolio" },
  { id: "opportunities", label: "Opportunities" },
  { id: "compare", label: "Compare" },
  { id: "execution", label: "Execution" },
  { id: "intelligence", label: "Intelligence" },
  { id: "system", label: "System" },
];

const STRATEGY_HELP: Record<string, { family: string; description: string }> = {
  llm: { family: "Forecast", description: "General model-assisted probability estimate." },
  bias_harvest: { family: "Relative value", description: "Looks for repeatable market-pricing bias." },
  long_horizon: { family: "Forecast", description: "Targets slower-moving, longer-dated mispricing." },
  market_maker: { family: "Liquidity", description: "Two-sided quoting intended to capture spread." },
  cross_exchange_arb: { family: "Arbitrage", description: "Compares equivalent outcomes across venues." },
  entailment_arb: { family: "Arbitrage", description: "Checks logically related markets for inconsistent prices." },
  resolution_lens: { family: "Event-driven", description: "Focuses on resolution mechanics and near-expiry evidence." },
};

function strategyHelp(name: string) {
  const base = name.replace(/_(kalshi|polymarket|haiku|opus|gpro|wide)$/, "");
  return STRATEGY_HELP[name] ?? STRATEGY_HELP[base] ?? {
    family: humanize(base), description: `Signals emitted by the ${humanize(name)} strategy.`,
  };
}

function StrategyName({ name }: { name: string }) {
  const help = strategyHelp(name);
  return <span className="strategy-name" title={help.description}><strong>{humanize(name)}</strong><small>{help.family}</small></span>;
}

function utcTime(value?: string | null) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? `${date.toISOString().slice(0, 19).replace("T", " ")} UTC` : value;
}

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
  if (!previous) return <p className="visit-summary" title="Stored only in this browser">Baseline saved in this browser. Changes will appear on your next visit.</p>;
  return <p className="visit-summary" title="Compared with a baseline stored only in this browser">
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
        <a
          className={view === item.id ? "active" : ""}
          key={item.id}
          href={`#${item.id}`}
          aria-current={view === item.id ? "page" : undefined}
          onClick={(event) => { event.preventDefault(); onView(item.id); }}
        >
          {item.label}
        </a>
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
  const trend = values.at(-1)! - values[0];
  return (
    <svg className={`spark ${deltaClass(trend)}`} viewBox="0 0 100 32" role="img"
      aria-label={`Recent P and L trend, ${trend >= 0 ? "up" : "down"} ${money(Math.abs(trend))}`}>
      <polyline points={points} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function BotLoopFeed({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  const signals = s.signals.slice(0, 7);
  const repeated = signals.length >= 3 && new Set(signals.slice(0, 3).map((item) => item.market_id)).size === 1;
  return <section className="loop-feed" aria-labelledby="loop-title">
    <header className="section-heading"><div><span className="eyebrow">Live market loop</span>
      <h2 id="loop-title">{signals.length ? "Latest market evaluations" : "Waiting for an evaluation"}</h2></div>
      <span className="live-indicator"><span /> Streaming</span></header>
    {repeated && <p className="loop-warning">The latest evaluations repeat one market. Review the decision timeline for possible lack of progress.</p>}
    {signals.length ? <ol>{signals.map((signal, index) => <li key={`${signal.market_id}-${signal.timestamp}-${index}`}>
      <button onClick={() => onInspect({ kind: "signal", value: signal,
        related: s.signals.filter((item) => item.market_id === signal.market_id) })}>
        <time title={isoAge(signal.timestamp)}>{utcTime(signal.timestamp)}</time>
        <span className="loop-market"><strong>{signal.question}</strong><small>{signal.exchange ?? "Unknown venue"} · <StrategyName name={signal.strategy_source} /></small></span>
        <span className="action-chip">{signal.action}</span><span className={deltaClass(signal.edge)}>{signal.edge == null ? "—" : `${signal.edge.toFixed(1)}%`}</span>
      </button>
    </li>)}</ol> : <div className="empty-state">No market evaluations are present in the current snapshot.</div>}
  </section>;
}

function Overview({ s, scope, historyIsCurrent, onView, onInspect }: {
  s: DashboardState; scope: string; historyIsCurrent: boolean;
  onView: (v: View) => void; onInspect: (i: Inspectable) => void;
}) {
  const history = historyIsCurrent ? (s.performance_history ?? []).slice(-14) : [];
  const today = history.at(-1);
  const prior = history.at(-2);
  const dailyDelta = today && prior ? today.total_pnl - prior.total_pnl : null;
  const largest = [...s.positions].sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl))[0];
  const newest = s.signals[0];
  return (
    <>
      <header className="workspace-heading overview-heading"><div><h1>Overview</h1>
        <p>Current posture, changes since your last browser visit, and work requiring attention.</p></div></header>
      <SinceLastVisit s={s} scope={scope} />
      <AttentionCenter s={s} onView={onView} onInspect={onInspect} />
      <BotLoopFeed s={s} onInspect={onInspect} />
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
          <p>{newest ? <><StrategyName name={newest.strategy_source} /> · {utcTime(newest.timestamp)}</> : "No recent signals"}</p>
        </SummaryCard>
        <SummaryCard title="Execution" status={s.reconciliation?.in_sync ? "In sync" : "Review required"} onOpen={() => onView("execution")}>
          <p>{s.trade_count} fills · {Object.keys(s.venues).length} venues</p>
          <p>Last venue snapshot {isoAge(s.reconciliation?.fetched_at)}</p>
        </SummaryCard>
        <SummaryCard title="System" status={s.health.errors ? `${s.health.errors} errors` : s.health.warnings ? `${s.health.warnings} warnings` : "Healthy"} onOpen={() => onView("system")}>
          <p>{s.pillars.filter((p) => (p.age_seconds ?? Infinity) < 90).length}/{s.pillars.length} pillars fresh</p>
          <p>{s.health.warnings} warnings in current log window</p>
        </SummaryCard>
      </div>
    </>
  );
}

function SummaryCard({ title, status, onOpen, children }: {
  title: string; status: string; onOpen?: () => void; children: React.ReactNode;
}) {
  return (
    <article className="summary-card">
      <header><div><span className="eyebrow">{title}</span><h2>{status}</h2></div>
        {onOpen && <button className="text-button" onClick={onOpen}>Open workspace →</button>}</header>
      {children}
    </article>
  );
}

function Portfolio({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  const [query, setQuery] = useStored("auramaur.positions.query", "");
  const [venue, setVenue] = useStored("auramaur.positions.venue", "all");
  const [preset, setPreset] = useStored("auramaur.positions.preset", "all");
  const [sort, setSort] = useStored("auramaur.positions.sort", "pnl");
  const [page, setPage] = useState(1);
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
  const pageSize = 50;
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const visibleRows = rows.slice((currentPage - 1) * pageSize, currentPage * pageSize);
  return (
    <Workspace title="Portfolio" subtitle="Find exposure, stale marks, concentration, and positions requiring review."
      action={<button onClick={() => download("auramaur-positions.json", rows)}>Export JSON</button>}>
      <div className="toolbar">
        <input aria-label="Search positions" placeholder="Search market, ID, token, category…" value={query}
          onChange={(e) => { setQuery(e.target.value); setPage(1); }} />
        <select aria-label="Filter venue" value={venue} onChange={(e) => { setVenue(e.target.value); setPage(1); }}>
          <option value="all">All venues</option>{venues.map((v) => <option key={v}>{v}</option>)}
        </select>
        <select aria-label="Position preset" value={preset} onChange={(e) => { setPreset(e.target.value); setPage(1); }}>
          <option value="all">All positions</option><option value="review">Needs review</option>
          <option value="movers">Largest movers</option><option value="stale">Stale marks</option>
          <option value="resolution">Resolving soon</option>
        </select>
        <select aria-label="Sort positions" value={sort} onChange={(e) => { setSort(e.target.value); setPage(1); }}>
          <option value="pnl">Sort: absolute P&amp;L</option><option value="value">Sort: exposure</option><option value="age">Sort: stalest</option>
        </select>
        <button className="secondary-button" disabled={!query && venue === "all" && preset === "all" && sort === "pnl"}
          onClick={() => { setQuery(""); setVenue("all"); setPreset("all"); setSort("pnl"); }}>Clear filters</button>
      </div>
      <div className="result-line"><p className="result-count">{rows.length} of {s.positions.length} positions · filters save automatically</p>
        {rows.length > pageSize && <nav className="pagination" aria-label="Position pages">
          <button disabled={currentPage === 1} onClick={() => setPage((value) => value - 1)}>Previous</button>
          <span>Page {currentPage} of {pageCount}</span>
          <button disabled={currentPage === pageCount} onClick={() => setPage((value) => value + 1)}>Next</button>
        </nav>}</div>
      <DataTable headers={["Market", "Venue", "Category", "Outcome", "Exposure", "Mark", "P&L", "Updated"]}
        empty={rows.length === 0 ? "No positions match these filters." : undefined}>
        {visibleRows.map((p) => {
          const open = () => onInspect({ kind: "position", value: p } as Inspectable);
          return <tr key={`${p.market_id}-${p.token}`} tabIndex={0} role="button" aria-label={`Inspect ${p.question}`}
            onClick={open} onKeyDown={(event) => activateRow(event, open)}>
          <td data-label="Market" className="market">{p.question}</td><td data-label="Venue">{p.exchange}</td><td data-label="Category">{p.category ?? "—"}</td>
          <td data-label="Outcome">{p.token}</td><td data-label="Exposure" className="num">{money(p.current_value)}</td><td data-label="Mark" className="num">{price(p.current_price)}</td>
          <td data-label="P&L" className={`num ${deltaClass(p.pnl)}`}>{money(p.pnl, true)}</td>
          <td data-label="Updated" title={exactTime(p.updated_at)}>{isoAge(p.updated_at)}</td>
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
            {s.strategies.map((row) => <tr key={row.strategy}><td><StrategyName name={row.strategy} /></td>
              <td className="num">{row.entries}</td><td className="num">{money(row.fees)}</td>
              <td className={`num ${deltaClass(row.pnl)}`}>{money(row.pnl, true)}</td></tr>)}
          </DataTable>
        </section>
      </div>
    </Workspace>
  );
}

function Opportunities({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  const [query, setQuery] = useStored("auramaur.signals.query", "");
  const [strategy, setStrategy] = useStored("auramaur.signals.strategy", "all");
  const [action, setAction] = useStored("auramaur.signals.action", "all");
  const [venue, setVenue] = useStored("auramaur.signals.venue", "all");
  const [sort, setSort] = useStored("auramaur.signals.sort", "edge");
  const [edgeFloor, setEdgeFloor] = useStored("auramaur.signals.edge", "0");
  const [selected, setSelected] = useState<string[]>([]);
  const strategies = [...new Set(s.signals.map((signal) => signal.strategy_source))].sort();
  const actions = [...new Set(s.signals.map((signal) => signal.action))].sort();
  const venues = [...new Set(s.signals.map((signal) => signal.exchange).filter(Boolean))].sort() as string[];
  const grouped = useMemo(() => {
    const map = new Map<string, Signal[]>();
    s.signals.filter((signal) => {
      if (strategy !== "all" && signal.strategy_source !== strategy) return false;
      if (action === "__actionable" && ["hold", "skip", "abstain", "none"].includes(signal.action.toLowerCase())) return false;
      if (action !== "all" && action !== "__actionable" && signal.action !== action) return false;
      if (venue !== "all" && signal.exchange !== venue) return false;
      if (Math.abs(signal.edge ?? 0) < Number(edgeFloor)) return false;
      return !query || [signal.question, signal.market_id, signal.exchange, signal.strategy_source]
        .some((value) => value?.toLowerCase().includes(query.toLowerCase()));
    }).forEach((signal) => map.set(signal.market_id, [...(map.get(signal.market_id) ?? []), signal]));
    return [...map.values()].sort((a, b) => sort === "recent"
      ? new Date(b[0].timestamp ?? 0).getTime() - new Date(a[0].timestamp ?? 0).getTime()
      : sort === "agreement" ? b.length - a.length
      : Math.abs(b[0].edge ?? 0) - Math.abs(a[0].edge ?? 0));
  }, [s.signals, query, strategy, action, venue, edgeFloor, sort]);
  const latestById = new Map(grouped.map((signals) => [signals[0].market_id, signals[0]]));
  const compared = selected.map((id) => latestById.get(id) ?? s.signals.find((signal) => signal.market_id === id)).filter(Boolean) as Signal[];
  const actionable = grouped.filter((signals) => !["hold", "skip", "abstain", "none"].includes(signals[0].action.toLowerCase())).length;
  const averageEdge = grouped.length ? grouped.reduce((sum, signals) => sum + Math.abs(signals[0].edge ?? 0), 0) / grouped.length : 0;
  const toggleCompare = (id: string) => setSelected((current) => current.includes(id)
    ? current.filter((item) => item !== id) : current.length < 3 ? [...current, id] : current);
  return (
    <Workspace title="Opportunities" subtitle="Signals grouped by market so repeated evaluations read as one decision timeline."
      action={<button onClick={() => download("auramaur-signals.json", s.signals)}>Export JSON</button>}>
      <div className="opportunity-metrics" aria-label="Opportunity summary">
        <button onClick={() => { setAction("all"); setEdgeFloor("0"); }}><span>Markets</span><strong>{grouped.length}</strong><small>{s.signals.length} evaluations</small></button>
        <button onClick={() => setEdgeFloor("5")}><span>Meaningful edge</span><strong>{grouped.filter((rows) => Math.abs(rows[0].edge ?? 0) >= 5).length}</strong><small>At least 5 percentage points</small></button>
        <button onClick={() => setAction("__actionable")}><span>Actionable</span><strong>{actionable}</strong><small>Excludes hold and abstain</small></button>
        <div><span>Average absolute edge</span><strong>{averageEdge.toFixed(1)}%</strong><small>Current filtered set</small></div>
      </div>
      <div className="toolbar">
        <input aria-label="Search opportunities" placeholder="Search market, ID, venue, strategy…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <select aria-label="Filter signal strategy" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          <option value="all">All strategies</option>{strategies.map((value) => <option key={value}>{value}</option>)}
        </select>
        <select aria-label="Filter signal action" value={action} onChange={(e) => setAction(e.target.value)}>
          <option value="all">All actions</option><option value="__actionable">Actionable only</option>{actions.map((value) => <option key={value}>{value}</option>)}
        </select>
        <select aria-label="Filter signal venue" value={venue} onChange={(e) => setVenue(e.target.value)}>
          <option value="all">All venues</option>{venues.map((value) => <option key={value}>{value}</option>)}
        </select>
        <select aria-label="Minimum absolute edge" value={edgeFloor} onChange={(e) => setEdgeFloor(e.target.value)}>
          <option value="0">Any edge</option><option value="2">Edge ≥ 2%</option><option value="5">Edge ≥ 5%</option><option value="10">Edge ≥ 10%</option>
        </select>
        <select aria-label="Sort opportunities" value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="edge">Sort: strongest edge</option><option value="recent">Sort: most recent</option><option value="agreement">Sort: most evaluated</option>
        </select>
        <button className="secondary-button" disabled={!query && strategy === "all" && action === "all" && venue === "all" && edgeFloor === "0" && sort === "edge"}
          onClick={() => { setQuery(""); setStrategy("all"); setAction("all"); setVenue("all"); setEdgeFloor("0"); setSort("edge"); }}>Clear filters</button>
      </div>
      <p className="result-count">{grouped.length} markets · select up to three to compare · click a row for its timeline and evidence</p>
      {compared.length > 0 && <section className="compare-tray" aria-label="Selected opportunity comparison"><header><div><span className="eyebrow">Side-by-side</span><h2>{compared.length} selected</h2></div><button className="text-button" onClick={() => setSelected([])}>Clear comparison</button></header>
        <div>{compared.map((sig) => <article key={sig.market_id}><button className="remove-compare" aria-label={`Remove ${sig.question} from comparison`} onClick={() => toggleCompare(sig.market_id)}>×</button>
          <strong>{sig.question}</strong><small>{sig.exchange ?? "Unknown venue"} · {utcTime(sig.timestamp)}</small>
          <dl><div><dt>Model</dt><dd>{price(sig.claude_prob)}</dd></div><div><dt>Market</dt><dd>{price(sig.market_prob)}</dd></div>
            <div><dt>Edge</dt><dd className={deltaClass(sig.edge)}>{sig.edge == null ? "—" : `${sig.edge.toFixed(1)}%`}</dd></div><div><dt>Action</dt><dd>{sig.action}</dd></div></dl>
          <p>{sig.evidence_summary || "No evidence summary was recorded."}</p></article>)}</div>
      </section>}
      <DataTable headers={["Compare", "Market", "Venue", "Evaluations", "Strategy", "Confidence", "Model probability", "Market probability", "Edge", "Action", "Observed"]}
        empty={grouped.length === 0 ? "No opportunities match these filters." : undefined}>
        {grouped.map((signals) => {
          const sig = signals[0];
          const open = () => onInspect({ kind: "signal", value: sig, related: signals });
          return <tr key={sig.market_id} tabIndex={0} role="button" aria-label={`Inspect signal timeline for ${sig.question}`}
            onClick={open} onKeyDown={(event) => activateRow(event, open)}>
            <td data-label="Compare"><input type="checkbox" aria-label={`Compare ${sig.question}`} checked={selected.includes(sig.market_id)} disabled={!selected.includes(sig.market_id) && selected.length >= 3}
              onClick={(event) => event.stopPropagation()} onChange={() => toggleCompare(sig.market_id)} /></td>
            <td data-label="Market" className="market"><strong>{sig.question}</strong>{sig.evidence_summary && <small className="evidence-preview">{sig.evidence_summary}</small>}</td>
            <td data-label="Venue">{sig.exchange ?? "—"}</td><td data-label="Evaluations" className="num">{signals.length}</td><td data-label="Strategy"><StrategyName name={sig.strategy_source} /></td>
            <td data-label="Confidence">{sig.confidence ? humanize(sig.confidence) : "—"}</td>
            <td data-label="Model probability" className="num">{price(sig.claude_prob)}</td><td data-label="Market probability" className="num">{price(sig.market_prob)}</td>
            <td data-label="Edge" className={`num ${deltaClass(sig.edge)}`}>{sig.edge == null ? "—" : `${sig.edge.toFixed(1)}%`}</td>
            <td data-label="Action"><span className="action-chip">{sig.action}</span></td>
            <td data-label="Observed" title={isoAge(sig.timestamp)}>{utcTime(sig.timestamp)}</td>
          </tr>;
        })}
      </DataTable>
    </Workspace>
  );
}

function Compare({ s, onView }: { s: DashboardState; onView: (v: View) => void }) {
  const [mode, setMode] = useStored("auramaur.compare.mode", "strategies");
  const strategyRows = useMemo(() => {
    const names = new Set([...s.strategies.map((row) => row.strategy), ...s.signals.map((row) => row.strategy_source)]);
    return [...names].map((name) => {
      const realized = s.strategies.find((row) => row.strategy === name);
      const signals = s.signals.filter((row) => row.strategy_source === name);
      const latest = signals.map((row) => row.timestamp).filter(Boolean).sort().at(-1);
      const venues = [...new Set(signals.map((row) => row.exchange).filter(Boolean))];
      return { name, realized, signals, latest, venues };
    }).sort((a, b) => (b.realized?.pnl ?? 0) - (a.realized?.pnl ?? 0));
  }, [s.strategies, s.signals]);
  const venueRows = Object.entries(s.venues).sort(([a], [b]) => a.localeCompare(b));
  return <Workspace title="Compare" subtitle="Put strategies or venues side by side using the same current snapshot.">
    <div className="segmented" role="group" aria-label="Comparison type">
      <button className={mode === "strategies" ? "active" : ""} onClick={() => setMode("strategies")}>Strategies</button>
      <button className={mode === "venues" ? "active" : ""} onClick={() => setMode("venues")}>Venues</button>
    </div>
    {mode === "strategies" ? <DataTable headers={["Strategy", "Health", "Evaluations", "Venues", "Entries", "Fees", "Realized", "Latest observed"]}
      empty={!strategyRows.length ? "No strategy data is available to compare." : undefined}>
      {strategyRows.map((row) => {
        const age = row.latest ? (new Date(s.now).getTime() - new Date(row.latest).getTime()) / 1000 : null;
        return <tr key={row.name}><td><StrategyName name={row.name} /></td><td><Status age={age} /></td>
          <td className="num">{row.signals.length}</td><td>{row.venues.join(", ") || "—"}</td>
          <td className="num">{row.realized?.entries ?? 0}</td><td className="num">{money(row.realized?.fees ?? 0)}</td>
          <td className={`num ${deltaClass(row.realized?.pnl ?? null)}`}>{money(row.realized?.pnl ?? 0, true)}</td>
          <td title={row.latest ? isoAge(row.latest) : undefined}>{utcTime(row.latest)}</td></tr>;
      })}</DataTable> : <DataTable headers={["Venue", "Health", "Balance detail", "Available", "Equity", "Snapshot UTC"]}
        empty={!venueRows.length ? "No venue snapshots are available to compare." : undefined}>
        {venueRows.map(([name, venue]) => <tr key={name}><td><strong>{humanize(name)}</strong></td><td><Status age={venue.age_seconds} /></td>
          <td>{venue.detail}</td><td className="num">{venue.available == null ? "—" : money(venue.available)}</td>
          <td className="num">{venue.equity == null ? "—" : money(venue.equity)}</td><td title={ago(venue.age_seconds)}>{utcTime(venue.fetched_at)}</td></tr>)}</DataTable>}
    <p className="comparison-link">Need detail? <button className="text-button" onClick={() => onView(mode === "strategies" ? "opportunities" : "execution")}>Open the {mode === "strategies" ? "signal explorer" : "execution workspace"} →</button></p>
  </Workspace>;
}

function Execution({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  const recon = s.reconciliation;
  const venues = Object.entries(s.venues).sort(([, a], [, b]) => (b.age_seconds ?? Infinity) - (a.age_seconds ?? Infinity));
  const discrepancies = recon ? [
    ...recon.missing.map((v) => ({ type: "Missing locally", ...v as object })),
    ...recon.extra.map((v) => ({ type: "Extra locally", ...v as object })),
    ...recon.size_mismatches.map((v) => ({ type: "Quantity mismatch", ...v as object })),
  ] as Record<string, unknown>[] : [];
  return (
    <Workspace title="Execution" subtitle="Venue posture, fills, and exact reconciliation differences."
      action={<button disabled={!recon} onClick={() => recon && download("auramaur-reconciliation.json", recon)}>Export reconciliation</button>}>
      <div className="venue-grid">{venues.map(([name, venue]) =>
        <article className="venue-card" key={name}><header><strong>{name}</strong><Status age={venue.age_seconds} /></header>
          <p>{venue.detail}</p><small title={ago(venue.age_seconds)}>Recorded {utcTime(venue.fetched_at)}</small></article>)}</div>
      <section className="section-block"><header className="section-heading"><div><span className="eyebrow">Reconciliation</span>
        <h2>{!recon?.available ? "Reconciliation unavailable" : recon.in_sync ? "Venue and ledger agree" : `${discrepancies.length} discrepancies`}</h2></div>
        {recon && <small title={exactTime(recon.fetched_at)}>Snapshot {isoAge(recon.fetched_at)}</small>}</header>
        {!recon?.available ? <div className="unavailable">No venue snapshot is available, so sync has not been verified.</div>
          : discrepancies.length ? <DataTable headers={["Severity", "Type", "Market / asset", "Outcome", "Venue qty", "Ledger qty"]}>
          {discrepancies.map((row, index) => {
            const open = () => onInspect({ kind: "reconciliation", value: row });
            return <tr key={index} tabIndex={0} role="button" aria-label={`Inspect ${String(row.type)} discrepancy`}
              onClick={open} onKeyDown={(event) => activateRow(event, open)}>
            <td><span className={`severity ${row.type === "Missing locally" ? "critical" : "warning"}`}>
              {row.type === "Missing locally" ? "Critical" : "Review"}</span></td>
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
      <SummaryCard title="Evidence distiller" status={`${llm?.claims_24h ?? 0} claims in 24h`}>
        <p>Last claim {isoAge(llm?.last_claim_at)}</p>
      </SummaryCard>
      {llm && Object.entries(llm.purposes).map(([purpose, stat]) =>
        <SummaryCard key={purpose} title={humanize(purpose)} status={`${stat.ok}/${stat.calls} successful`}>
          <p>{stat.errors} errors · {stat.avg_ms ?? "—"}ms average</p><p>{compactNumber(stat.prompt_tokens + stat.output_tokens)} tokens</p>
        </SummaryCard>)}
    </div>
    <section className="section-block"><h2>Resolved forecast scorecard</h2>
      <p className="section-help">Lower Brier scores indicate better calibrated forecasts.</p>
      <DataTable headers={["Arm", "Model", "Forecasts", "Brier", "Market Brier", "Abstains"]}
        empty={(s.intelligence_eval ?? []).length === 0 ? "No resolved forecasts are available yet. Calibration appears after outcomes resolve." : undefined}>
        {(s.intelligence_eval ?? []).map((row) => <tr key={row.arm}><td>{row.arm}</td><td>{row.model}</td>
          <td className="num">{row.forecasts}</td><td className="num">{row.brier ?? "—"}</td>
          <td className="num">{row.market_brier ?? "—"}</td><td className="num">{row.abstains}</td></tr>)}
      </DataTable>
    </section>
  </Workspace>;
}

function System({ s, onInspect }: { s: DashboardState; onInspect: (i: Inspectable) => void }) {
  const pillars = [...s.pillars].sort((a, b) => (b.age_seconds ?? Infinity) - (a.age_seconds ?? Infinity));
  const diagnostics = [...s.health.top].sort((a, b) => {
    const rank = (level?: string) => level === "error" || level === "critical" ? 2 : level === "warning" ? 1 : 0;
    return rank(b.level) - rank(a.level) || b.count - a.count;
  });
  return <Workspace title="System" subtitle="Source liveness and diagnostics ordered for investigation."
    action={<button onClick={() => download("auramaur-diagnostics.json", { pillars: s.pillars, health: s.health, activity: s.activity })}>Export diagnostics</button>}>
    <p className="section-help">Fresh &lt; 90 seconds · stale 90 seconds–10 minutes · dead after 10 minutes.</p>
    <div className="pillar-grid">{pillars.map((pillar) =>
      <article className="pillar-card" key={pillar.name}><Status age={pillar.age_seconds} /><div><strong>{pillar.name}</strong>
        <small>Last event {ago(pillar.age_seconds)}</small></div></article>)}</div>
    <section className="section-block"><header className="section-heading"><div><span className="eyebrow">Diagnostics</span>
      <h2>{s.health.errors} errors · {s.health.warnings} warnings</h2></div></header>
      <DataTable headers={["Occurrences", "Event", "Latest detail", "Last seen"]}
        empty={diagnostics.length === 0 ? "No diagnostics in the current log window." : undefined}>
        {diagnostics.map((event) => {
          const open = () => onInspect({ kind: "health", value: event });
          return <tr key={event.event} tabIndex={0} role="button" aria-label={`Inspect diagnostic ${event.event}`}
            onClick={open} onKeyDown={(keyEvent) => activateRow(keyEvent, open)}>
          <td className="num">{event.count}</td><td>{event.event}</td><td className="market">{event.last_msg || "No message recorded"}</td>
          <td title={exactTime(event.last_ts)}>{isoAge(event.last_ts)}</td></tr>;
        })}
      </DataTable>
    </section>
    <section className="section-block"><h2>Recent operational activity</h2>
      {s.activity.length ? <div className="timeline">{[...s.activity].reverse().map((item, index) =>
        <div key={index}><time>{item.time}</time><span>{humanize(item.text)}</span></div>)}</div>
        : <div className="empty-state">No recent operational activity.</div>}
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
  return <main id="main-content" className="workspace" tabIndex={-1}><header className="workspace-heading"><div><h1>{title}</h1><p>{subtitle}</p></div>{action}</header>{children}</main>;
}

function DataTable({ headers, children, empty }: { headers: string[]; children: React.ReactNode; empty?: string }) {
  return <div className="data-table-wrap"><table className="data-table"><thead><tr>{headers.map((header) =>
    <th key={header} className={["Exposure", "Mark", "P&L", "Edge", "Model probability", "Market probability", "Occurrences", "Forecasts", "Brier", "Market Brier", "Abstains", "Venue qty", "Ledger qty", "Evaluations", "Positions", "Entries", "Fees", "Realized", "Quantity", "Entry", "Peak gain"].includes(header) ? "num" : ""}>{header}</th>)}
  </tr></thead><tbody>{empty ? <tr className="empty-row"><td colSpan={headers.length}>{empty}</td></tr> : children}</tbody></table></div>;
}

function Inspector({ item, onClose }: { item: Inspectable | null; onClose: () => void }) {
  const dialogRef = useRef<HTMLElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);
  const [notice, setNotice] = useState("");
  useEffect(() => {
    if (!item) return;
    returnFocus.current = document.activeElement as HTMLElement;
    const dialog = dialogRef.current;
    const focusable = () => [...(dialog?.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    ) ?? [])];
    focusable()[0]?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab") {
        const nodes = focusable();
        if (!nodes.length) return;
        const first = nodes[0], last = nodes[nodes.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault(); last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault(); first.focus();
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("keydown", handleKey);
      returnFocus.current?.focus();
    };
  }, [item, onClose]);
  if (!item) return null;
  const copy = async (label: string, text: string) => {
    try {
      await copyText(text);
      setNotice(`${label} copied`);
    } catch {
      setNotice(`Could not copy ${label.toLowerCase()}`);
    }
  };
  const value = item.value as unknown as Record<string, unknown>;
  const title = item.kind === "position" || item.kind === "signal"
    ? String(value.question) : item.kind === "health" ? String(value.event) : "Reconciliation detail";
  const checklist = item.kind === "health"
    ? ["Confirm the latest occurrence time", "Review the event message and related activity", "Check the affected pillar or venue", "Copy diagnostics before remediation"]
    : item.kind === "reconciliation"
      ? ["Confirm snapshot freshness", "Compare venue and ledger quantities", "Copy the asset or market ID", "Run the position sync/reconciliation workflow"]
      : ["Confirm source freshness", "Review exposure and related signal", "Copy the market ID for deeper investigation"];
  return <div className="inspector-backdrop" onMouseDown={onClose}>
    <aside ref={dialogRef} className="inspector" role="dialog" aria-modal="true" aria-labelledby="inspector-title" onMouseDown={(e) => e.stopPropagation()}>
      <header><div><span className="eyebrow">{item.kind}</span><h2 id="inspector-title">{title}</h2></div>
        <button className="icon-button" aria-label="Close details" onClick={onClose}>×</button></header>
      <dl>{Object.entries(value).filter(([, v]) => v !== "" && v != null && typeof v !== "object").map(([key, val]) =>
        <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{String(val)}</dd></div>)}</dl>
      {item.kind === "signal" && item.related && item.related.length > 1 && <section>
        <h3>Decision timeline</h3>
        <div className="decision-timeline">{item.related.map((signal, index) =>
          <div key={`${signal.timestamp}-${index}`}><time title={exactTime(signal.timestamp)}>{isoAge(signal.timestamp)}</time>
            <span>{signal.action} · {signal.edge == null ? "no edge" : `${signal.edge.toFixed(1)}% edge`}</span></div>)}</div>
      </section>}
      <section><h3>Investigation checklist</h3><ol>{checklist.map((step) => <li key={step}>{step}</li>)}</ol></section>
      <div className="inspector-actions">
        {"market_id" in value && <button onClick={() => void copy("Market ID", String(value.market_id))}>Copy market ID</button>}
        {"asset_id" in value && <button onClick={() => void copy("Asset ID", String(value.asset_id))}>Copy asset ID</button>}
        <button onClick={() => void copy("Details", safeJson(value))}>Copy details</button>
        <button onClick={() => download(`auramaur-${item.kind}.json`, value)}>Export JSON</button>
      </div>
      <p className="copy-status" role="status" aria-live="polite">{notice}</p>
    </aside>
  </div>;
}

export function OperatorWorkspace({ s, scope = "default", historyIsCurrent = true, initialView = "overview" }: {
  s: DashboardState; scope?: string; historyIsCurrent?: boolean; initialView?: View;
}) {
  const safeInitial = VIEWS.some((candidate) => candidate.id === initialView) ? initialView : "overview";
  const [view, setView] = useState<View>(safeInitial);
  const [inspect, setInspect] = useState<Inspectable | null>(null);
  const changeView = (next: View, replace = false) => {
    setView(next);
    history[replace ? "replaceState" : "pushState"](null, "", `#${next}`);
    document.title = `${VIEWS.find((candidate) => candidate.id === next)?.label ?? "Overview"} · Auramaur`;
    requestAnimationFrame(() => document.getElementById("main-content")?.focus());
  };
  useEffect(() => {
    const sync = () => {
      const hash = location.hash.slice(1) as View;
      const next = VIEWS.some((candidate) => candidate.id === hash) ? hash : "overview";
      setView(next);
      document.title = `${VIEWS.find((candidate) => candidate.id === next)?.label ?? "Overview"} · Auramaur`;
    };
    window.addEventListener("hashchange", sync);
    window.addEventListener("popstate", sync);
    sync();
    return () => {
      window.removeEventListener("hashchange", sync);
      window.removeEventListener("popstate", sync);
    };
  }, []);
  return <>
    <a className="skip-link" href="#main-content">Skip to dashboard content</a>
    <WorkspaceNav view={view} onView={changeView} />
    {view === "overview" && <main id="main-content" tabIndex={-1}><Overview s={s} scope={scope} historyIsCurrent={historyIsCurrent} onView={changeView} onInspect={setInspect} /></main>}
    {view === "portfolio" && <Portfolio s={s} onInspect={setInspect} />}
    {view === "opportunities" && <Opportunities s={s} onInspect={setInspect} />}
    {view === "compare" && <Compare s={s} onView={changeView} />}
    {view === "execution" && <Execution s={s} onInspect={setInspect} />}
    {view === "intelligence" && <Intelligence s={s} />}
    {view === "system" && <System s={s} onInspect={setInspect} />}
    <Inspector item={inspect} onClose={() => setInspect(null)} />
  </>;
}
