import { useEffect, useState } from "react";
import {
  ActivityPanel,
  CategoriesPanel,
  ConnectionBanner,
  HealthPanel,
  IntelligencePanel,
  KillSwitchBanner,
  KrakenPaperPanel,
  PillarsPanel,
  PositionsPanel,
  SignalsPanel,
  StatTiles,
  StrategiesPanel,
  TopBar,
  VenuesPanel,
} from "./components";
import type { Book } from "./types";
import { useDashboardState } from "./useDashboardState";

const TROUBLESHOOT_AFTER_S = 4;

export default function App() {
  const { envelope, phase, receivedAgo } = useDashboardState();
  const [waitingFor, setWaitingFor] = useState(0);

  useEffect(() => {
    const mountedAt = Date.now();
    const timer = setInterval(() => setWaitingFor((Date.now() - mountedAt) / 1000), 1000);
    return () => clearInterval(timer);
  }, []);

  // Which BOOK is on screen — a view choice, defaulting to what the bot is
  // armed to trade. Entirely independent of any trading gate.
  const [bookChoice, setBookChoice] = useState<Book | null>(null);
  const book: Book = bookChoice ?? envelope?.bot_mode ?? "paper";
  const state = envelope?.books?.[book] ?? null;

  // Nothing at all yet: keep the wait honest — say what is being tried, and
  // after a few seconds say what to check instead of spinning forever.
  if (!state) {
    return (
      <div className="shell">
        <div className="loading">
          <p>
            {phase === "degraded"
              ? "Connected to the auramaur web service"
              : "Connecting to the auramaur web service…"}
          </p>
          {phase === "degraded" && envelope?.error && (
            <div className="trouble" role="alert">
              <p>The service is up but has no data to show:</p>
              <p className="detail">{envelope.error}</p>
            </div>
          )}
          {phase !== "degraded" && waitingFor > TROUBLESHOOT_AFTER_S && (
            <div className="trouble">
              <p>No response from <code>/api/state</code> yet. Check that:</p>
              <ul>
                <li>the API is running: <code>auramaur web</code> (or the <code>auramaur-web</code> container)</li>
                <li>it's on the port this page proxies to (default <code>127.0.0.1:8484</code>)</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      <TopBar
        s={state}
        botMode={envelope!.bot_mode}
        book={book}
        onBook={setBookChoice}
        phase={phase}
        receivedAgo={receivedAgo}
      />
      <ConnectionBanner phase={phase} error={envelope?.error ?? null} receivedAgo={receivedAgo} />
      {state.kill_switch && <KillSwitchBanner />}
      <StatTiles s={state} />
      <div className="grid">
        <div className="col">
          <VenuesPanel s={state} />
          <PillarsPanel pillars={state.pillars} />
          <CategoriesPanel categories={state.categories} />
          <HealthPanel health={state.health} />
          <IntelligencePanel s={state} />
        </div>
        <div className="col">
          <PositionsPanel positions={state.positions} />
          <StrategiesPanel strategies={state.strategies} />
          <SignalsPanel signals={state.signals} />
        </div>
        <div className="col col-side">
          <ActivityPanel s={state} />
          {book === "paper" && <KrakenPaperPanel rows={state.kraken_paper ?? []} />}
        </div>
      </div>
    </div>
  );
}
