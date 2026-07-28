# One IBKR strategy, designed from the cost arithmetic

Written 2026-07-27, after the previous book was measured and rejected. Capital: **$778.69 USD** — the account holds 1,099.29 CAD and USDCAD measured
1.4117 on 2026-07-28. The first draft of this document said $1,100 by reading
the CAD balance as USD, a 41% overstatement of a book that buys USD-denominated
ETFs. Worse than the size error: $1,100 sits ABOVE the $1,000 point where
IBKR's $1 commission minimum and its 0.1% marginal rate cross, and $779 sits
below it, so the draft assumed the marginal rate (27bps) when the floor
actually binds (32.7bps). Every conviction threshold below inherits that.

Every other number here is measured, not assumed.

## The constraint nobody had priced

Expected excess return over the benchmark is

    E[excess] = 2 * (p - base_rate) * E|move|,   E|move| = sigma*sqrt(h/252)*sqrt(2/pi)

Round-trip cost at $1,100 is **27bps**: the IBKR equity schedule is
`max(1.0, min(notional*0.001, 10.0))`, so a $1,100 position pays the $1 floor
(20bps round trip) plus ~7bps of spread and slippage.

The LLM arms' **measured** conviction is |p-0.5| = 0.019-0.025 typical, 0.060
maximum ever. Against that, at a 5-session horizon:

| | ann vol | conviction needed (2x margin) | verdict |
|---|---|---|---|
| SLV | 64.4% | 0.045 | tradeable |
| GLD | 28.4% | 0.102 | out |
| XLK | 24.9% | 0.097 | out |
| QQQ | 19.0% | 0.126 | out |
| SPY | 12.7% | 0.189 | out |
| TLT | 9.3% | 0.258 | out |

**SPY needs 19pp of conviction at this size.** The model has never produced more
than 6. That is not a threshold to tune — it is an instrument to remove.

More capital does not fix it. Between $1,000 and $10,000 the commission is a
flat 10bps per leg either way (the $1 floor and the 0.1% marginal rate cross at
$1,000), so cost in bps is constant across that whole range. Only above $10,000,
where the $10 cap binds, does it improve.

## What the arithmetic says to do instead

`E|move|` grows with sqrt(h) while cost does not, so **horizon is the lever**:

Recomputed at the CORRECTED $775 of capital, where the $1 commission floor
binds and a round trip costs 32.8bps:

| horizon | tradeable at peak (0.06) | at typical (0.02) | names |
|---|---|---|---|
| 5 sessions | 1 | 0 | SLV |
| 10 | 1 | 0 | SLV |
| 21 | 2 | 0 | SLV, GLD |
| **42** | **7** | **1** | SLV, GLD, XLE, QQQ, DBC, IWM, VWO |
| 63 | 7 | 1 | same seven |

**42 sessions is the setting.** It is where the universe stops being a single
instrument and where SLV clears at TYPICAL conviction (0.016) rather than
needing a peak call. 63 adds no names and doubles the wait, so the curve is
flat past two months — there is nothing further to buy by holding longer.

Longer holds also cut the number of round trips, which compounds because
commission is charged per trip rather than per unit of time.

**The cost: clearance moves out.** A 42-session forecast takes 42 sessions to
resolve, so the traded horizon cannot be proven for roughly two months. The
314 five-session forecasts already in flight are not wasted — they resolve over
the next fortnight and serve as an EARLY KILL SIGNAL through
`auramaur ibkr-calibration`. If the model shows no skill there, abandon before
waiting out the long clock. They cannot open the gate, though: clearance is
scoped to the traded horizon, because skill at five sessions is not evidence of
skill at forty-two.

## The design

**Forecast on the short horizon, trade on the long one.** This is what makes
"validate in days" compatible with "trade economically", and the two goals are
otherwise in direct conflict.

1. **Forecast every eligible instrument daily at 5 AND 21 sessions.** Forecasts
   are free. The 5-session leg resolves weekly and is what proves the model has
   skill; ~125 resolutions a week accumulate.
2. **Trading stays locked** until `clearance()` shows the arm's Brier edge over
   its own trailing base rate has a 95% lower bound clear of zero. A genuine
   10pp edge needs ~370 resolutions to clear that, so expect **~3 weeks** — not
   six months, and nothing is risked while waiting.
3. **Universe = `viable_universe(...)` at the traded horizon**, recomputed from
   live volatility. Instruments whose required conviction exceeds the model's
   demonstrated maximum are excluded by construction.
4. **Entry = `clears_costs(...)`**: expected edge must beat 2x round-trip cost.
   One position, full capital, because splitting $1,100 doubles the fee burden
   for no diversification benefit at these sizes.
5. **Term structure as a filter**: act only when the 5- and 21-session forecasts
   agree in sign. Disagreement means the model is not expressing a view, it is
   producing noise at two horizons.

## What this design refuses to do

- **No threshold tuning.** The exit study searched 48 geometries on the old
  book: the deployed one was best on train, none had a positive train LCB, and
  the winner landed below the median out of sample. Parameter surfaces here are
  noise.
- **No trading to learn.** The previous book paid $926 in commission to
  discover its gross P&L was -$42.
- **No shorting the observed U-shape.** The signal study found Q1 (most
  negative momentum) has the best forward excess return at every horizon. That
  is one universe over one window with a plausible compositional explanation,
  and acting on it without a pre-registered out-of-sample test would be the
  same mistake in a new direction.

## Open, and genuinely an operator decision

The Polymarket engine works on `claude_prob` vs `market_prob` — a real
market-implied probability. Equities have none; **options do**, and options
would also make the graduation ladder's Brier-edge bar directly applicable
instead of needing a base-rate substitute. Options market data is
`qualified_no_live_data` — entitlement-blocked. Whether to buy that
subscription is now a strategy question, not an ops one.
