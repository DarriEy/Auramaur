// Mirrors auramaur/web/app.py serialize_state() — the JSON shape of
// GET /api/state and the SSE `state` events.

export interface Pillar {
  name: string;
  last_seen: string | null;
  age_seconds: number | null;
}

export interface ActivityItem {
  time: string;
  text: string;
}

export interface HealthTop {
  event: string;
  count: number;
  level?: string;
  last_msg?: string;
  last_ts?: string;
}

export interface Health {
  errors: number;
  warnings: number;
  top: HealthTop[];
}

export interface Position {
  market_id: string;
  question: string;
  /** YES/NO — with market_id the row's identity: a market can hold both. */
  token: string;
  exchange: string;
  side: string;
  size: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  updated_at?: string;
  category?: string;
  end_date?: string | null;
  initial_value?: number;
  current_value?: number;
  to_win?: number;
}

export interface StrategyStat {
  strategy: string;
  entries: number;
  pnl: number;
  fees: number;
  /** Open-book mark-to-market, attributed to the position's entry strategy. */
  unrealized?: number;
  open_positions?: number;
}

export interface StrategyHeartbeat {
  last_beat_at: string;
  status: string;
  entries: number | null;
  cycles: number;
  interval_seconds: number | null;
  age_seconds: number | null;
  detail: string;
}

export interface CategoryStat {
  category: string;
  positions: number;
  value: number;
}

export interface KrakenPaperPosition {
  strategy: string;
  pair: string;
  quantity: number;
  entry_price: number;
  peak_gain_pct: number;
  opened_at: string;
}

export interface Signal {
  market_id: string;
  question: string;
  timestamp?: string;
  exchange?: string;
  claude_prob: number | null;
  market_prob: number | null;
  edge: number | null;
  action: string;
  strategy_source: string;
  confidence?: string;
  evidence_summary?: string;
}

/** Venue cash as recorded bot-side into the DB; age is how old the row is —
 *  the dashboard holds no venue credentials, so staleness is shown, not hidden. */
export interface VenueBalance {
  detail: string;
  age_seconds: number | null;
  available?: number | null;
  equity?: number | null;
  fetched_at?: string;
}

export interface DashboardState {
  now: string;
  is_live: boolean;
  transfers_armed: boolean;
  kill_switch: boolean;
  venues: Record<string, VenueBalance>;
  ibkr_books: { book: string; positions: number; unrealized: number; equity: number | null }[];
  reconciliation?: {
    available: boolean;
    in_sync: boolean;
    venue_count: number;
    db_count: number;
    venue_value?: number;
    db_value?: number;
    fetched_at: string | null;
    missing: unknown[];
    extra: unknown[];
    size_mismatches: unknown[];
  };
  pillars: Pillar[];
  activity: ActivityItem[];
  health: Health;
  positions: Position[];
  position_count: number;
  position_value: number;
  signals: Signal[];
  trade_count: number;
  total_pnl: number;
  drawdown: number;
  balance: number | null;
  strategies: StrategyStat[];
  /** Per-strategy liveness written by the bot every pillar cycle. */
  heartbeats?: Record<string, StrategyHeartbeat>;
  categories: CategoryStat[];
  /** Present on the paper book only — Kraken's isolated directional book. */
  kraken_paper?: KrakenPaperPosition[];
  local_llm?: {
    purposes: Record<string, {
      calls: number; ok: number; errors: number; avg_ms: number | null;
      prompt_tokens: number; output_tokens: number;
    }>;
    claims_24h: number;
    last_claim_at: string | null;
  };
  intelligence_eval?: {
    arm: string; model: string; forecasts: number;
    brier: number | null; market_brier: number | null; abstains: number;
  }[];
  performance_history?: {
    date: string; total_pnl: number; trades_count: number; wins: number;
    losses: number; max_drawdown: number; peak_balance: number;
  }[];
}

export type Book = "paper" | "live";

/** Every /api/state response and SSE event. `books` keeps the last GOOD
 *  snapshots (both books) even while `error` is set, so the UI never goes
 *  blank. `bot_mode` is what the bot is armed to trade — the view toggle is
 *  independent of it. */
export interface Envelope {
  ok: boolean;
  error: string | null;
  updated_at: string | null;
  bot_mode: Book;
  books: Record<Book, DashboardState> | null;
}
