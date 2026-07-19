"""SQLite table schemas as SQL strings."""

SCHEMA_VERSION = 31

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
    clob_token_yes TEXT DEFAULT '',
    clob_token_no TEXT DEFAULT '',
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
    strategy_source TEXT DEFAULT 'llm',
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
    strategy_source TEXT DEFAULT 'llm',
    FOREIGN KEY (market_id) REFERENCES markets(id),
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS portfolio (
    market_id TEXT NOT NULL,
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
    PRIMARY KEY (market_id, is_paper, token),
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

-- One outstanding directional "bet" per Kraken pair, used to close the
-- calibration feedback loop: snapshot the LLM P(up) + a reference price at
-- open, then at horizon (due_at) compare spot vs ref_price to resolve the
-- prediction (record_resolution). At most one unresolved row per pair so the
-- "most-recent-unresolved" resolution semantics stay correct.
CREATE TABLE IF NOT EXISTS kraken_dir_signals (
    pair TEXT PRIMARY KEY,
    prob REAL NOT NULL,
    ref_price REAL NOT NULL,
    opened_at TEXT NOT NULL DEFAULT (datetime('now')),
    due_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id TEXT PRIMARY KEY, query TEXT NOT NULL, category TEXT DEFAULT '',
    market_id TEXT DEFAULT '', started_at TEXT NOT NULL, completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running', active_sources INTEGER NOT NULL DEFAULT 0,
    raw_items INTEGER NOT NULL DEFAULT 0, unique_items INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS source_fetches (
    run_id TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL,
    item_count INTEGER NOT NULL DEFAULT 0, latency_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT DEFAULT '', observed_at TEXT NOT NULL,
    information_mode TEXT NOT NULL DEFAULT 'production', PRIMARY KEY (run_id, source)
);

CREATE TABLE IF NOT EXISTS evidence_observations (
    run_id TEXT NOT NULL, item_id TEXT NOT NULL, source TEXT NOT NULL,
    title TEXT NOT NULL, url TEXT DEFAULT '', content_hash TEXT NOT NULL,
    excerpt TEXT DEFAULT '', published_at TEXT, observed_at TEXT NOT NULL,
    timestamp_quality TEXT NOT NULL DEFAULT 'exact', relevance_score REAL NOT NULL DEFAULT 0,
    rank_position INTEGER, market_id TEXT DEFAULT '',
    information_mode TEXT NOT NULL DEFAULT 'production', PRIMARY KEY (run_id, item_id)
);

CREATE TABLE IF NOT EXISTS forecast_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, market_id TEXT NOT NULL,
    exchange TEXT NOT NULL, category TEXT DEFAULT '',
    forecast_purpose TEXT NOT NULL DEFAULT 'analysis', forecast_horizon TEXT DEFAULT '',
    raw_probability REAL NOT NULL CHECK(raw_probability BETWEEN 0 AND 1),
    calibrated_probability REAL CHECK(calibrated_probability BETWEEN 0 AND 1),
    market_yes_price REAL NOT NULL CHECK(market_yes_price BETWEEN 0 AND 1),
    market_no_price REAL CHECK(market_no_price BETWEEN 0 AND 1),
    observed_at TEXT NOT NULL, evidence_run_ids TEXT NOT NULL DEFAULT '[]',
    model TEXT DEFAULT '', strategy_source TEXT DEFAULT 'llm',
    config_fingerprint TEXT DEFAULT '', actual_outcome INTEGER CHECK(actual_outcome IN (0, 1)),
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS information_strategies (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, category TEXT NOT NULL DEFAULT '',
    horizon TEXT NOT NULL DEFAULT '', event_type TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT 'shadow', created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source, category, horizon, event_type)
);

CREATE TABLE IF NOT EXISTS information_trials (
    id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL, market_id TEXT NOT NULL,
    observed_at TEXT NOT NULL, assignment TEXT NOT NULL CHECK(assignment IN ('control','treatment')),
    assignment_hash TEXT NOT NULL, market_price REAL NOT NULL,
    resolved_outcome INTEGER, resolved_at TEXT,
    UNIQUE(strategy_id, market_id, observed_at)
);

CREATE TABLE IF NOT EXISTS paired_forecasts (
    trial_id TEXT NOT NULL, arm TEXT NOT NULL CHECK(arm IN ('control','treatment')),
    probability REAL NOT NULL CHECK(probability BETWEEN 0 AND 1), forecast_id INTEGER,
    net_paper_pnl REAL, created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY(trial_id, arm)
);

CREATE TABLE IF NOT EXISTS source_contributions (
    trial_id TEXT PRIMARY KEY, source TEXT NOT NULL, control_brier REAL,
    treatment_brier REAL, control_log_loss REAL, treatment_log_loss REAL,
    incremental_brier REAL, incremental_log_loss REAL, incremental_pnl REAL,
    computed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS information_graduation_state (
    strategy_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'registered',
    influence_multiplier REAL NOT NULL DEFAULT 0, resolved_trials INTEGER NOT NULL DEFAULT 0,
    paired_forecasts INTEGER NOT NULL DEFAULT 0, incremental_brier REAL,
    incremental_log_loss REAL, incremental_pnl REAL, source_success_rate REAL,
    reason TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ingestion_started ON ingestion_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_source_fetches_source_time ON source_fetches(source, observed_at);
CREATE INDEX IF NOT EXISTS idx_evidence_market_time ON evidence_observations(market_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_forecast_market_time ON forecast_snapshots(market_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_forecast_resolved ON forecast_snapshots(actual_outcome, resolved_at);

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
    market_id TEXT NOT NULL,
    token TEXT NOT NULL DEFAULT 'YES',
    token_id TEXT DEFAULT '',
    size REAL NOT NULL,
    avg_cost REAL NOT NULL,
    total_cost REAL NOT NULL,
    realized_pnl REAL DEFAULT 0,
    is_paper INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (market_id, is_paper, token)
);

-- Realized dollar P&L per market, written when a market resolves. Lets us
-- measure edge in $ (not just calibration) and allocate by actual profit.
CREATE TABLE IF NOT EXISTS resolution_pnl (
    market_id TEXT PRIMARY KEY,
    category TEXT DEFAULT '',
    pnl REAL NOT NULL,
    resolved_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pnl_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    venue TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    strategy_source TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    token TEXT NOT NULL DEFAULT 'YES',
    qty REAL NOT NULL DEFAULT 0,
    pnl REAL NOT NULL,
    fees REAL NOT NULL DEFAULT 0,
    is_paper INTEGER NOT NULL DEFAULT 1,
    source_ref TEXT NOT NULL UNIQUE,
    realized_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pnl_ledger_market ON pnl_ledger(market_id, is_paper);
CREATE INDEX IF NOT EXISTS idx_pnl_ledger_realized ON pnl_ledger(realized_at);

CREATE TABLE IF NOT EXISTS oddlot_filings (
    accession TEXT PRIMARY KEY,
    cik TEXT NOT NULL DEFAULT '',
    ticker TEXT NOT NULL DEFAULT '',
    company TEXT NOT NULL DEFAULT '',
    form TEXT NOT NULL DEFAULT '',
    filed_at TEXT NOT NULL DEFAULT '',
    odd_lot_priority INTEGER NOT NULL DEFAULT 0,
    tender_price REAL NOT NULL DEFAULT 0,
    tender_price_high REAL NOT NULL DEFAULT 0,
    expiration TEXT NOT NULL DEFAULT '',
    conditions TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'detected',
    checked_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS gap_audits (
    market_id TEXT PRIMARY KEY,
    claude_prob REAL NOT NULL,
    market_prob REAL NOT NULL,
    mechanism TEXT NOT NULL DEFAULT 'none',
    reason TEXT NOT NULL DEFAULT '',
    audited_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS lens_verdicts (
    market_id TEXT PRIMARY KEY,
    fair_prob REAL NOT NULL,
    gap_score REAL NOT NULL DEFAULT 0,
    mechanism TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '',
    checked_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- Adversarial mechanism check: -1 not yet verified, 0 refuted, 1 confirmed.
    verified INTEGER NOT NULL DEFAULT -1
);

CREATE TABLE IF NOT EXISTS entailment_verdicts (
    market_id_a TEXT NOT NULL,
    market_id_b TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'llm',
    reasoning TEXT NOT NULL DEFAULT '',
    traded_at TEXT,
    checked_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (market_id_a, market_id_b)
);

CREATE INDEX IF NOT EXISTS idx_price_history_market ON price_history(market_id);
CREATE INDEX IF NOT EXISTS idx_price_history_time ON price_history(recorded_at);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    token_id TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT 'polymarket',
    best_bid REAL,
    best_ask REAL,
    bid_size REAL,
    ask_size REAL,
    mid REAL,
    bid2 REAL,
    ask2 REAL,
    bid2_size REAL,
    ask2_size REAL,
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_orderbook_market_time ON orderbook_snapshots(market_id, recorded_at);

-- Immutable decision-time observations used for executable-price and
-- closing-line-value evaluation.  These are separate from mutable signals so
-- later analysis cannot rewrite the research record.
CREATE TABLE IF NOT EXISTS decision_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    strategy_source TEXT NOT NULL,
    signal_id INTEGER,
    side TEXT NOT NULL,
    fair_probability REAL NOT NULL,
    reference_price REAL NOT NULL,
    executable_price REAL,
    best_bid REAL,
    best_ask REAL,
    requested_size REAL NOT NULL DEFAULT 0,
    fee_estimate REAL NOT NULL DEFAULT 0,
    filled INTEGER NOT NULL DEFAULT 0,
    observed_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(signal_id, strategy_source)
);
CREATE INDEX IF NOT EXISTS idx_decision_market_time
    ON decision_snapshots(market_id, observed_at);

CREATE TABLE IF NOT EXISTS decision_marks (
    decision_id INTEGER NOT NULL,
    horizon_seconds INTEGER NOT NULL,
    bid REAL,
    ask REAL,
    mid REAL,
    marked_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (decision_id, horizon_seconds)
);

CREATE TABLE IF NOT EXISTS maker_rebates (
    date TEXT NOT NULL,
    condition_id TEXT NOT NULL,
    maker_address TEXT NOT NULL,
    rebate_usdc REAL NOT NULL,
    strategy_source TEXT NOT NULL DEFAULT 'market_maker',
    recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (date, condition_id, maker_address)
);

CREATE TABLE IF NOT EXISTS strategy_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_source TEXT NOT NULL,
    market_id TEXT NOT NULL,
    mechanism TEXT NOT NULL,
    score REAL NOT NULL,
    expected_edge REAL NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    is_paper INTEGER NOT NULL DEFAULT 1,
    observed_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(strategy_source, market_id, mechanism, observed_at)
);

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

CREATE TABLE IF NOT EXISTS ibkr_etf_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_alias TEXT NOT NULL,
    model TEXT NOT NULL,
    symbol TEXT NOT NULL,
    probability REAL NOT NULL,
    confidence TEXT NOT NULL,
    thesis TEXT NOT NULL DEFAULT '',
    risks_json TEXT NOT NULL DEFAULT '[]',
    reference_price REAL NOT NULL,
    final_price REAL,
    actual_outcome INTEGER,
    intelligence_cost_usd REAL NOT NULL DEFAULT 0,
    opened_at TEXT NOT NULL DEFAULT (datetime('now')),
    opened_session_date TEXT NOT NULL,
    horizon_sessions INTEGER NOT NULL,
    sessions_elapsed INTEGER NOT NULL DEFAULT 0,
    last_session_date TEXT,
    due_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ibkr_etf_forecast_arm
    ON ibkr_etf_forecasts(model_alias, symbol, due_at);

CREATE TABLE IF NOT EXISTS ibkr_etf_state (
    model_alias TEXT PRIMARY KEY,
    refresh_cursor INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ibkr_etf_cooldowns (
    model_alias TEXT NOT NULL,
    symbol TEXT NOT NULL,
    until_epoch REAL NOT NULL,
    PRIMARY KEY (model_alias, symbol)
);

CREATE TABLE IF NOT EXISTS ibkr_etf_openai_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_alias TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'started',
    response_id TEXT DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT DEFAULT '',
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ibkr_etf_attempt_arm_day
    ON ibkr_etf_openai_attempts(model_alias, started_at);

CREATE TABLE IF NOT EXISTS ibkr_etf_positions (
    model_alias TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_cost REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL NOT NULL DEFAULT 0,
    peak_pnl_pct REAL NOT NULL DEFAULT 0,
    stop_price REAL NOT NULL DEFAULT 0,
    initial_risk_usd REAL NOT NULL DEFAULT 0,
    opened_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (model_alias, symbol)
);

CREATE TABLE IF NOT EXISTS ibkr_etf_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_alias TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission_usd REAL NOT NULL DEFAULT 0,
    fill_ref TEXT NOT NULL UNIQUE,
    filled_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ibkr_etf_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_alias TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('trade', 'commission', 'intelligence')),
    pnl REAL NOT NULL,
    source_ref TEXT NOT NULL UNIQUE,
    realized_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ibkr_etf_ledger_arm_day
    ON ibkr_etf_ledger(model_alias, realized_at);

-- Isolated accounting for the six typed IBKR multi-asset paper books. These
-- tables never feed the shared prediction-market paper wallet.
CREATE TABLE IF NOT EXISTS ibkr_paper_positions (
    book TEXT NOT NULL,
    instrument_key TEXT NOT NULL,
    con_id INTEGER NOT NULL DEFAULT 0,
    currency TEXT NOT NULL,
    quantity REAL NOT NULL,
    multiplier REAL NOT NULL DEFAULT 1,
    fx_to_usd REAL NOT NULL DEFAULT 1,
    avg_cost REAL NOT NULL,
    current_price REAL,
    unrealized_pnl_usd REAL NOT NULL DEFAULT 0,
    stop_price REAL NOT NULL DEFAULT 0,
    initial_risk_usd REAL NOT NULL DEFAULT 0,
    entry_commission_usd REAL NOT NULL DEFAULT 0,
    entry_fill_ref TEXT NOT NULL DEFAULT '',
    price_source TEXT NOT NULL DEFAULT 'ibkr_unknown',
    instrument_spec_json TEXT NOT NULL DEFAULT '',
    opened_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (book, instrument_key)
);

CREATE TABLE IF NOT EXISTS ibkr_paper_fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book TEXT NOT NULL,
    instrument_key TEXT NOT NULL,
    con_id INTEGER NOT NULL DEFAULT 0,
    side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL,
    multiplier REAL NOT NULL DEFAULT 1,
    price REAL NOT NULL,
    currency TEXT NOT NULL,
    fx_to_usd REAL NOT NULL DEFAULT 1,
    commission_usd REAL NOT NULL DEFAULT 0,
    price_source TEXT NOT NULL DEFAULT 'ibkr_unknown',
    fill_ref TEXT NOT NULL UNIQUE,
    filled_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ibkr_paper_fills_book_time
    ON ibkr_paper_fills(book, filled_at);

CREATE TABLE IF NOT EXISTS ibkr_paper_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('trade', 'commission', 'financing')),
    pnl_usd REAL NOT NULL,
    source_ref TEXT NOT NULL UNIQUE,
    realized_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ibkr_paper_ledger_book_day
    ON ibkr_paper_ledger(book, realized_at);

CREATE TABLE IF NOT EXISTS ibkr_paper_round_trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book TEXT NOT NULL, instrument_key TEXT NOT NULL,
    entry_fill_ref TEXT NOT NULL DEFAULT '',
    exit_fill_ref TEXT NOT NULL UNIQUE,
    gross_pnl_usd REAL NOT NULL,
    entry_commission_usd REAL NOT NULL DEFAULT 0,
    exit_commission_usd REAL NOT NULL DEFAULT 0,
    financing_usd REAL NOT NULL DEFAULT 0,
    borrow_usd REAL NOT NULL DEFAULT 0,
    roll_cost_usd REAL NOT NULL DEFAULT 0,
    intelligence_cost_usd REAL NOT NULL DEFAULT 0,
    net_pnl_usd REAL NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ibkr_round_trips_book_closed
    ON ibkr_paper_round_trips(book, closed_at);

CREATE TABLE IF NOT EXISTS ibkr_paper_state (
    book TEXT PRIMARY KEY,
    refresh_cursor INTEGER NOT NULL DEFAULT 0,
    last_cycle_at TEXT,
    last_success_at TEXT,
    last_error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Authoritative shadow book for Kraken validate-only strategies. Paper orders
-- do not alter the real wallet, so wallet reconciliation must not own this state.
CREATE TABLE IF NOT EXISTS kraken_paper_positions (
    strategy TEXT NOT NULL DEFAULT 'llm',
    pair TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    peak_gain_pct REAL NOT NULL DEFAULT 0,
    opened_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (strategy, pair)
);
CREATE INDEX IF NOT EXISTS idx_kraken_paper_positions_pair
    ON kraken_paper_positions(pair);

-- Broker-qualified identities for the deterministic IBKR manifest. Discovery
-- may refresh these rows but cannot introduce an undeclared instrument.
CREATE TABLE IF NOT EXISTS ibkr_contract_registry (
    instrument_key TEXT PRIMARY KEY,
    book TEXT NOT NULL,
    kind TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    con_id INTEGER NOT NULL,
    local_symbol TEXT NOT NULL DEFAULT '',
    trading_class TEXT NOT NULL DEFAULT '',
    exchange TEXT NOT NULL DEFAULT '',
    currency TEXT NOT NULL DEFAULT '',
    multiplier REAL NOT NULL DEFAULT 1,
    status TEXT NOT NULL CHECK(status IN
        ('eligible', 'qualified_no_live_data', 'pending_approval',
         'quarantined', 'drifted')),
    approved INTEGER NOT NULL DEFAULT 0,
    approval_reason TEXT NOT NULL DEFAULT '',
    quote_source TEXT NOT NULL DEFAULT 'ibkr_unknown',
    has_history INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    qualified_at TEXT NOT NULL DEFAULT (datetime('now')),
    validated_at TEXT NOT NULL DEFAULT (datetime('now')),
    approved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ibkr_contract_registry_status
    ON ibkr_contract_registry(book, status, approved);

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

CREATE TABLE IF NOT EXISTS signal_rejections (
    market_id TEXT PRIMARY KEY,
    exchange TEXT DEFAULT '',
    rejected_at TEXT NOT NULL DEFAULT (datetime('now')),
    yes_price REAL NOT NULL DEFAULT 0,
    reason TEXT DEFAULT '',
    streak INTEGER NOT NULL DEFAULT 1
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

-- Decision-time Kalshi book snapshots. These separate forecasting edge from
-- execution edge and make paper graduation auditable against live liquidity.
CREATE TABLE IF NOT EXISTS kalshi_execution_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    strategy_source TEXT DEFAULT '',
    token TEXT NOT NULL,
    side TEXT NOT NULL,
    requested_size REAL NOT NULL,
    fillable_size REAL NOT NULL,
    best_bid REAL,
    best_ask REAL,
    vwap REAL,
    marginal_price REAL,
    fair_probability REAL,
    market_probability REAL,
    is_live INTEGER NOT NULL DEFAULT 0,
    observed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_kalshi_execution_samples_market_time
    ON kalshi_execution_samples(market_id, observed_at);

-- Interim-manager proposal queue: operator-proposed entries awaiting the
-- pillar's charter/risk/ladder gauntlet. Terminal rows are the audit log.
CREATE TABLE IF NOT EXISTS manager_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    venue TEXT NOT NULL,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,
    fair_prob REAL NOT NULL,
    stake_usd REAL NOT NULL,
    thesis TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_manager_proposals_status
    ON manager_proposals(status, created_at);

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
