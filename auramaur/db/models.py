"""SQLite table schemas as SQL strings."""

SCHEMA_VERSION = 9

TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    exchange TEXT DEFAULT 'polymarket',
    condition_id TEXT DEFAULT '',
    ticker TEXT DEFAULT '',
    question TEXT NOT NULL,
    description TEXT,
    category TEXT,
    end_date TEXT,
    active INTEGER DEFAULT 1,
    outcome_yes_price REAL,
    outcome_no_price REAL,
    volume REAL DEFAULT 0,
    liquidity REAL DEFAULT 0,
    last_updated TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS news_items (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    published_at TEXT,
    relevance_score REAL DEFAULT 0,
    market_ids TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    exchange TEXT DEFAULT 'polymarket',
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    claude_prob REAL NOT NULL,
    claude_confidence TEXT NOT NULL,
    market_prob REAL NOT NULL,
    edge REAL NOT NULL,
    second_opinion_prob REAL,
    divergence REAL,
    evidence_summary TEXT,
    action TEXT,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    exchange TEXT DEFAULT 'polymarket',
    signal_id INTEGER,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    side TEXT NOT NULL,
    size REAL NOT NULL,
    price REAL NOT NULL,
    is_paper INTEGER NOT NULL DEFAULT 1,
    order_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    pnl REAL,
    kelly_fraction REAL,
    risk_checks_passed TEXT,
    FOREIGN KEY (market_id) REFERENCES markets(id),
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS portfolio (
    market_id TEXT PRIMARY KEY,
    exchange TEXT DEFAULT 'polymarket',
    side TEXT NOT NULL,
    size REAL NOT NULL,
    avg_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL DEFAULT 0,
    category TEXT,
    token TEXT NOT NULL DEFAULT 'YES',
    token_id TEXT DEFAULT '',
    is_paper INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    total_pnl REAL DEFAULT 0,
    trades_count INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    max_drawdown REAL DEFAULT 0,
    peak_balance REAL DEFAULT 0,
    api_calls_claude INTEGER DEFAULT 0,
    api_cost_estimate REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS nlp_cache (
    cache_key TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    response TEXT NOT NULL,
    probability REAL NOT NULL,
    confidence TEXT NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    market_price REAL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    predicted_prob REAL NOT NULL,
    actual_outcome INTEGER,
    resolved_at TEXT,
    category TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calibration_params (
    category TEXT PRIMARY KEY,
    a REAL NOT NULL,
    b REAL NOT NULL,
    n INTEGER NOT NULL,
    brier_score REAL,
    fitted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS category_stats (
    category TEXT PRIMARY KEY,
    total_pnl REAL DEFAULT 0,
    trade_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    avg_edge REAL DEFAULT 0,
    brier_score REAL,
    kelly_multiplier REAL DEFAULT 1.0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS market_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id_a TEXT NOT NULL,
    market_id_b TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    strength REAL DEFAULT 0,
    description TEXT,
    detected_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(market_id_a, market_id_b)
);

CREATE TABLE IF NOT EXISTS source_accuracy (
    source TEXT PRIMARY KEY,
    brier_score REAL,
    prediction_count INTEGER DEFAULT 0,
    weight REAL DEFAULT 1.0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    exchange TEXT DEFAULT 'polymarket',
    price REAL NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    token_id TEXT DEFAULT '',
    side TEXT NOT NULL,
    token TEXT NOT NULL DEFAULT 'YES',
    size REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL DEFAULT 0,
    is_paper INTEGER NOT NULL DEFAULT 1,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cost_basis (
    market_id TEXT PRIMARY KEY,
    token TEXT NOT NULL DEFAULT 'YES',
    token_id TEXT DEFAULT '',
    size REAL NOT NULL,
    avg_cost REAL NOT NULL,
    total_cost REAL NOT NULL,
    realized_pnl REAL DEFAULT 0,
    is_paper INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_price_history_market ON price_history(market_id);
CREATE INDEX IF NOT EXISTS idx_price_history_time ON price_history(recorded_at);

CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_id);
CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_nlp_cache_created ON nlp_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_items(published_at);
CREATE INDEX IF NOT EXISTS idx_fills_market ON fills(market_id);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_market_rel_a ON market_relationships(market_id_a);
CREATE INDEX IF NOT EXISTS idx_market_rel_b ON market_relationships(market_id_b);
CREATE INDEX IF NOT EXISTS idx_market_rel_type ON market_relationships(relationship_type);

CREATE TABLE IF NOT EXISTS ensemble_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    model TEXT NOT NULL,
    category TEXT DEFAULT '',
    probability REAL NOT NULL,
    actual_outcome INTEGER,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ensemble_model ON ensemble_predictions(model);
CREATE INDEX IF NOT EXISTS idx_ensemble_market ON ensemble_predictions(market_id);

CREATE TABLE IF NOT EXISTS position_peaks (
    market_id TEXT PRIMARY KEY,
    peak_pnl_pct REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rebalance_blocks (
    event_key TEXT PRIMARY KEY,
    blocked_until TEXT NOT NULL,
    reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS order_build_drops (
    market_id TEXT PRIMARY KEY,
    blocked_until TEXT NOT NULL,
    reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS slippage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    exchange TEXT DEFAULT '',
    side TEXT NOT NULL,
    expected_price REAL NOT NULL,
    filled_price REAL NOT NULL,
    slippage_bps REAL NOT NULL,
    size REAL NOT NULL,
    order_type TEXT DEFAULT 'limit',
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS redemptions (
    condition_id TEXT PRIMARY KEY,
    asset_id TEXT DEFAULT '',
    title TEXT DEFAULT '',
    neg_risk INTEGER DEFAULT 0,
    size REAL NOT NULL,
    expected_payout REAL NOT NULL,
    safe_nonce INTEGER,
    tx_hash TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    submitted_at TEXT,
    confirmed_at TEXT,
    error TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_redemptions_status ON redemptions(status);
"""
