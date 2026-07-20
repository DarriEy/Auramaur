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
  message?: string;
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
}

export interface StrategyStat {
  strategy: string;
  entries: number;
  pnl: number;
  fees: number;
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
  claude_prob: number | null;
  market_prob: number | null;
  edge: number | null;
  action: string;
  strategy_source: string;
}

export interface DashboardState {
  now: string;
  is_live: boolean;
  transfers_armed: boolean;
  kill_switch: boolean;
  venues: Record<string, string>;
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
  categories: CategoryStat[];
  /** Present on the paper book only — Kraken's isolated directional book. */
  kraken_paper?: KrakenPaperPosition[];
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
