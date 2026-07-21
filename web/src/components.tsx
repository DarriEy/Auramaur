import { useState } from "react";
import type {
  Book,
  CategoryStat,
  DashboardState,
  Health,
  KrakenPaperPosition,
  Pillar,
  Position,
  Signal,
  StrategyStat,
} from "./types";
import type { Phase } from "./useDashboardState";
import { ago, deltaClass, liveness, money, price, utcClock } from "./format";
import { filterPositions } from "./positionFilter";

export function TopBar({
  s,
  botMode,
  book,
  onBook,
  phase,
  receivedAgo,
}: {
  s: DashboardState;
  botMode: Book;
  book: Book;
  onBook: (b: Book) => void;
  phase: Phase;
  receivedAgo: number | null;
}) {
  const freshness =
    receivedAgo === null ? "" : receivedAgo < 3 ? "updated just now" : `updated ${ago(receivedAgo)}`;
  return (
    <header className="topbar">
      <h1>AURAMAUR</h1>
      {botMode === "live" ? (
        <span className="badge badge-live"><span className="dot" />BOT: LIVE</span>
      ) : (
        <span className="badge badge-paper"><span className="dot" />BOT: PAPER</span>
      )}
      {s.transfers_armed && (
        <span className="badge badge-warn"><span className="dot" />transfers armed</span>
      )}
      <div className="book-toggle" role="group" aria-label="Book to view">
        <span className="dim">viewing</span>
        {(["paper", "live"] as Book[]).map((b) => (
          <button
            key={b}
            className={`seg${book === b ? " active" : ""}`}
            onClick={() => onBook(b)}
          >
            {b}
          </button>
        ))}
        {book !== botMode && <span className="dim">(bot trades {botMode})</span>}
      </div>
      <span className="clock">{utcClock(s.now)}</span>
      <span className={`link-status${phase === "live" ? "" : " down"}`}>{freshness}</span>
    </header>
  );
}

/** One line that always tells the truth about the data on screen. Silent when
 *  everything is fresh and healthy. */
export function ConnectionBanner({
  phase,
  error,
  receivedAgo,
}: {
  phase: Phase;
  error: string | null;
  receivedAgo: number | null;
}) {
  if (phase === "live" || phase === "connecting") return null;
  return (
    <div className={`conn-banner ${phase}`} role="alert">
      {phase === "disconnected" ? (
        <>
          ⚠ No fresh data for {receivedAgo === null ? "a while" : ago(receivedAgo)} — showing the
          last snapshot. Is <code>auramaur web</code> still running?
        </>
      ) : (
        <>⚠ Live updates failing — showing the last good snapshot. {error}</>
      )}
    </div>
  );
}

export function KillSwitchBanner() {
  return (
    <div className="killswitch-banner" role="alert">
      ⛔ KILL SWITCH ACTIVE — all trading halted
    </div>
  );
}

export function StatTiles({ s }: { s: DashboardState }) {
  const pnlDir = deltaClass(s.total_pnl);
  return (
    <section className="tiles">
      <div className="tile">
        <div className="label">Balance</div>
        <div className="value">{money(s.balance)}</div>
        {s.balance === null && <div className="sub">live cash via bot syncer</div>}
      </div>
      <div className="tile">
        <div className="label">Total P&L</div>
        <div className={`value ${pnlDir}`}>{money(s.total_pnl, true)}</div>
      </div>
      <div className="tile">
        <div className="label">Open positions</div>
        <div className="value">{s.position_count}</div>
        <div className="sub">{money(s.position_value)} at mark</div>
      </div>
      <div className="tile">
        <div className="label">Fills</div>
        <div className="value">{s.trade_count}</div>
      </div>
      <div className="tile">
        <div className="label">Max drawdown</div>
        <div className="value">{s.drawdown.toFixed(1)}%</div>
        <div className="sub">latest daily stat</div>
      </div>
    </section>
  );
}

// The recorder writes every 60s (IBKR every 300s); past this, the bot or its
// recorder has stopped and the number should read as suspect.
const BALANCE_STALE_S = 900;

export function VenuesPanel({ s }: { s: DashboardState }) {
  // ONE row per venue: positions from the book being viewed, merged with the
  // bot-recorded balance and its age. IBKR paper books get their own lines
  // beneath the IBKR balance row (positions live in ibkr_paper_positions,
  // not the shared portfolio table).
  const byExchange = new Map<string, { n: number; value: number }>();
  for (const p of s.positions) {
    const cur = byExchange.get(p.exchange) ?? { n: 0, value: 0 };
    cur.n += 1;
    cur.value += (p.current_price || 0) * (p.size || 0);
    byExchange.set(p.exchange, cur);
  }
  const names = [...new Set([...byExchange.keys(), ...Object.keys(s.venues)])].sort();
  return (
    <section className="panel">
      <h2>Venues</h2>
      {s.reconciliation?.available && !s.reconciliation.in_sync && (
        <div className="conn-banner degraded" role="alert">
          Venue mismatch: Polymarket reports {s.reconciliation.venue_count} positions,
          Auramaur has {s.reconciliation.db_count}. Missing {s.reconciliation.missing.length},
          extra {s.reconciliation.extra.length}, quantity mismatches
          {` ${s.reconciliation.size_mismatches.length}`}.
        </div>
      )}
      {s.reconciliation?.available && s.reconciliation.in_sync && (
        <div className="clean">✓ Venue and database are in sync ({s.reconciliation.venue_count})</div>
      )}
      <div className="kv">
        {names.map((name) => {
          const pos = byExchange.get(name);
          const bal = s.venues[name];
          const parts = [
            pos ? `${pos.n} pos, ${money(pos.value)}` : null,
            bal ? bal.detail : null,
          ].filter(Boolean);
          return (
            <div className="row" key={name}>
              <span className="k">{name}</span>
              <span className="v">{parts.length ? parts.join(" \u00b7 ") : "\u2014"}</span>
              {bal && (
                <span className={`age${(bal.age_seconds ?? Infinity) > BALANCE_STALE_S ? " stale" : ""}`}>
                  {ago(bal.age_seconds)}
                </span>
              )}
            </div>
          );
        })}
        {names.length === 0 && (
          <div className="row"><span className="k">book</span><span className="v">empty</span></div>
        )}
        {(s.ibkr_books ?? []).map((b) => (
          <div className="row" key={`ibkr-${b.book}`}>
            <span className="k">ibkr:{b.book}</span>
            <span className="v">
              {b.positions} pos, {money(b.unrealized)} unrl
              {b.equity != null ? ` \u00b7 eq ${money(b.equity)}` : ""}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function StrategiesPanel({ strategies }: { strategies: StrategyStat[] }) {
  return (
    <section className="panel">
      <h2>Strategies — realized P&L</h2>
      {strategies.length === 0 ? (
        <div className="empty">Nothing in the ledger for this book</div>
      ) : (
        <div className="tbl-wrap">
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th className="num">Entries</th>
                <th className="num">Fees</th>
                <th className="num">Realized</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((st) => (
                <tr key={st.strategy}>
                  <td>{st.strategy}</td>
                  <td className="num">{st.entries}</td>
                  <td className="num dim">{money(st.fees)}</td>
                  <td className={`num ${deltaClass(st.pnl)}`}>
                    {money(st.pnl, true)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function CategoriesPanel({ categories }: { categories: CategoryStat[] }) {
  return (
    <section className="panel">
      <h2>Category exposure</h2>
      {categories.length === 0 ? (
        <div className="empty">No open positions</div>
      ) : (
        <div className="kv">
          {categories.map((c) => (
            <div className="row" key={c.category}>
              <span className="k">{c.category}</span>
              <span className="v">{c.positions} pos, {money(c.value)}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export function KrakenPaperPanel({ rows }: { rows: KrakenPaperPosition[] }) {
  if (rows.length === 0) return null;
  return (
    <section className="panel">
      <h2>Kraken paper book</h2>
      <div className="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Pair</th>
              <th>Strategy</th>
              <th className="num">Qty</th>
              <th className="num">Entry</th>
              <th className="num">Peak</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.strategy}-${r.pair}`}>
                <td>{r.pair}</td>
                <td className="dim">{r.strategy}</td>
                <td className="num">{r.quantity.toFixed(4)}</td>
                <td className="num">{r.entry_price.toFixed(2)}</td>
                <td className={`num ${deltaClass(r.peak_gain_pct)}`}>
                  {r.peak_gain_pct.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function PillarsPanel({ pillars }: { pillars: Pillar[] }) {
  return (
    <section className="panel">
      <h2>Pillars</h2>
      <div className="pillars">
        {pillars.map((p) => (
          <div className="pillar-row" key={p.name}>
            <span className={`dot ${liveness(p.age_seconds)}`} />
            <span className="name">{p.name}</span>
            <span className="age">{ago(p.age_seconds)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function HealthPanel({ health }: { health: Health }) {
  return (
    <section className="panel">
      <h2>Health</h2>
      <div className="health-counts">
        <span className={health.errors ? "err" : "dim"}>
          <span className="n">{health.errors}</span> err
        </span>
        <span className={health.warnings ? "warn" : "dim"}>
          <span className="n">{health.warnings}</span> warn
        </span>
        {!health.errors && !health.warnings && <span className="clean">✓ clean</span>}
      </div>
      <div className="health-top">
        {health.top.map((t) => (
          <div className="row" key={t.event}>
            <span className="count">{t.count}</span>
            <span className="event" title={t.message}>{t.event}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
  // The complete book remains available; search narrows it without truncation.
export function PositionsPanel({ positions }: { positions: Position[] }) {
  const [query, setQuery] = useState("");
  const shown = filterPositions(positions, query);
  return (
    <section className="panel">
      <h2>Positions</h2>
      <input
        aria-label="Search positions"
        className="position-search"
        placeholder={`Search all ${positions.length} positions`}
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      {shown.length === 0 ? (
        <div className="empty">No open positions</div>
      ) : (
        <div className="tbl-wrap">
          <table>
            <thead>
              <tr>
                <th>Market</th>
                <th>Venue</th>
                <th>Token</th>
                <th>Side</th>
                <th className="num">Size</th>
                <th className="num">Avg</th>
                <th className="num">Mark</th>
                <th className="num">P&L</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((p) => (
                <tr key={`${p.market_id}-${p.token}-${p.side}`}>
                  <td className="market" title={p.question}>{p.question}</td>
                  <td className="dim">{p.exchange}</td>
                  <td>{p.token}</td>
                  <td>{p.side}</td>
                  <td className="num">{p.size.toFixed(0)}</td>
                  <td className="num">{price(p.avg_price)}</td>
                  <td className="num">{price(p.current_price)}</td>
                  <td className={`num ${deltaClass(p.pnl)}`}>
                    {money(p.pnl, true)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function SignalsPanel({ signals }: { signals: Signal[] }) {
  return (
    <section className="panel">
      <h2>Recent signals</h2>
      {signals.length === 0 ? (
        <div className="empty">No signals yet</div>
      ) : (
        <div className="tbl-wrap">
          <table>
            <thead>
              <tr>
                <th>Market</th>
                <th>Strategy</th>
                <th className="num">Claude</th>
                <th className="num">Market</th>
                <th className="num">Edge</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((sig, i) => (
                <tr key={`${sig.market_id}-${i}`}>
                  <td className="market" title={sig.question}>{sig.question}</td>
                  <td className="dim">{sig.strategy_source}</td>
                  <td className="num">{price(sig.claude_prob)}</td>
                  <td className="num">{price(sig.market_prob)}</td>
                  <td className={`num ${deltaClass(sig.edge)}`}>
                    {sig.edge === null ? "—" : `${sig.edge > 0 ? "+" : ""}${sig.edge.toFixed(1)}%`}
                  </td>
                  <td>{sig.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function ActivityPanel({ s }: { s: DashboardState }) {
  return (
    <section className="panel">
      <h2>Activity</h2>
      {s.activity.length === 0 ? (
        <div className="empty">Nothing recent in the log tail</div>
      ) : (
        <div className="feed">
          {[...s.activity].reverse().map((a, i) => (
            <div className="row" key={`${a.time}-${i}`}>
              <span className="time">{a.time}</span>
              <span className="text">{a.text}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export function IntelligencePanel({ s }: { s: DashboardState }) {
  // Local Ollama tier + shadow evaluation scorecard. Both sources are
  // evidence-side (no order paths); the panel omits itself entirely until
  // the tier has produced any rows, so pre-v35 deployments render nothing.
  const llm = s.local_llm;
  const evalRows = s.intelligence_eval ?? [];
  const purposes = llm ? Object.entries(llm.purposes) : [];
  if (!llm && evalRows.length === 0) return null;
  if (llm && purposes.length === 0 && llm.claims_24h === 0 && evalRows.length === 0) return null;
  return (
    <section className="panel">
      <h2>Intelligence tier</h2>
      <div className="kv">
        {llm && (
          <div className="row">
            <span className="k">distiller</span>
            <span className="v">
              {llm.claims_24h} claims 24h
              {llm.last_claim_at ? ` \u00b7 last ${llm.last_claim_at.slice(11, 16)}Z` : ""}
            </span>
          </div>
        )}
        {purposes.map(([p, st]) => (
          <div className="row" key={p}>
            <span className="k">{p}</span>
            <span className="v">
              {st.calls} calls
              {st.errors ? ` \u00b7 ${st.errors} err` : ""}
              {st.avg_ms != null ? ` \u00b7 ${st.avg_ms}ms` : ""}
            </span>
          </div>
        ))}
        {evalRows.map((r) => {
          const delta =
            r.brier != null && r.market_brier != null
              ? r.market_brier - r.brier
              : null;
          return (
            <div className="row" key={`${r.arm}-${r.model}`}>
              <span className="k">eval:{r.arm}</span>
              <span className="v">
                n={r.forecasts} \u00b7 Brier {r.brier?.toFixed(3) ?? "\u2014"} vs mkt{" "}
                {r.market_brier?.toFixed(3) ?? "\u2014"}
                {delta != null ? ` (${delta >= 0 ? "+" : ""}${delta.toFixed(3)})` : ""}
              </span>
            </div>
          );
        })}
        {evalRows.length === 0 && (
          <div className="row">
            <span className="k">eval</span>
            <span className="v">no resolved episodes yet</span>
          </div>
        )}
      </div>
    </section>
  );
}

