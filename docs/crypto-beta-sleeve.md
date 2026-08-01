# Crypto beta sleeve — specification

Status: **proposed**, not implemented. Written 2026-08-01.

## Purpose

Hold crypto beta on Kraken, with a *scored* steering layer that must earn the
right to move money before it is allowed to.

This exists because everything else we tried on Kraken failed for the same
reason: intelligence was spent on decisions nobody could score. The directional
desk went 0-for-26, never issued a view above 0.540 against a break-even bar of
0.524-0.541, and its best theoretical cell needed 182 years to verify. The
design constraint here is therefore **measurability first, returns second**.

## The constraint that shapes everything

Kraken's fee floor at our tier is **0.52% round trip taker / 0.32% maker**.
That kills every high-turnover shape — market making, per-cycle funding
capture, and typical cross-venue arb gaps all come out negative. What survives
is *low turnover*: pay the fee once, hold.

But low turnover also means few observations, which is what made the
directional book unmeasurable.

**The resolution is to decouple scoring frequency from trading frequency.**
Forecast weekly and score every forecast; act monthly. Fees constrain trading,
not forecasting. This gives ~52 scored observations a year while paying the
spread 12 times.

## 1. Weights

**85% BTC / 15% ETH. No SOL.**

Measured over 721 daily bars split into two independent halves:

| weights | full 2y | H1 | H2 |
|---|---|---|---|
| 100% BTC | +6.7% / 53.1% DD | +94.3% | -45.5% |
| 80/15/5 | -1.3% / 55.5% | +82.5% | -46.8% |
| 70/20/10 | -5.8% / 56.8% | +75.9% | -47.6% |
| equal 1/3 | -23.6% / 62.7% | +50.6% | -51.0% |

The ranking is monotone and identical in both halves — more BTC gave higher
return *and* lower drawdown, in an up regime and a down regime. The mechanism
is not subtle: pairwise correlations are 0.79-0.82 while annualised vols are
BTC 45%, ETH 69%, SOL 77%. At those correlations ETH and SOL are levered BTC,
not diversifiers, so they import volatility without importing independence.

SOL is dropped outright: highest vol, largest drag, no diversification.

**The 15% ETH is a deliberate, priced deviation from what the data says.** On
this sample it costs roughly 4pp per two years versus 100% BTC. It buys a
hedge against BTC-specific failure (protocol, regulatory, or a single-asset
shock the correlation table cannot see). That is a risk-management choice, not
a return-maximising one, and it should be labelled as such rather than
justified after the fact.

**Rebalancing: none below $10k.** Measured benefit of the best band was 1.9pp
over two years (~1%/yr), and the fee model used did not compound drag, so that
is a ceiling. At a $200-500 sleeve that is $2-5/yr and not worth the
complexity. Above ~$10k, revisit with a 5pt drift band.

## 2. Sizing

Sleeve target: **10-15% of total book**, capped at what a 60% drawdown can be
absorbed without affecting other pillars.

Total book is ~$2,800 (Polymarket $1,164 equity, IBKR ~$795, Kalshi $221,
Kraken ~$145). A $200-420 sleeve at a 60% drawdown costs $120-250, i.e. 4-9%
of book. That is tolerable; more is not.

Kraken already holds ~200 CAD idle, so the sleeve is fundable today without
moving capital away from anything that is working.

## 3. The scored forecast (fast track — measurement)

**Weekly**, per asset, the local tier (qwen3:8b — free, so cadence is not cost
constrained) emits:

```
{ "asset": "BTC",
  "horizon_days": 30,
  "prob_higher": 0.0-1.0,      # P(close in 30d > close today)
  "confidence": "LOW|MEDIUM|HIGH",
  "rationale": "<=200 chars" }
```

Scored on resolution with the **existing** evaluation machinery — this is a
reuse, not a new harness:

- `score_forecast()` in `auramaur/evaluation/scoring.py`, market-relative
- materialised into `forecast_score_facts`
- `event_weighted_summary()` in `auramaur/evaluation/evidence.py`
- reported via `auramaur intelligence-eval`

Two rules carried over from the intelligence-audit null result:

1. **The prompt must never contain the quantity being scored.** The v1 harness
   fed the model the market price and then scored it against that price; every
   arm returned brier skill +0.000 for a week. Snapshot must be price-blind
   beyond what is needed for context.
2. **Never pool prompt versions.** Partition by `prompt_version`, or a v1 row
   silently wins the dedup window and v2 disappears from the scorecard.

Baseline for skill: **the unconditional base rate** of a 30-day-higher close
over the sample, not 0.5. A forecaster that only reproduces the base rate has
demonstrated nothing.

## 4. The monthly allocation decision (slow track — acting)

**Monthly**, one call decides a single number: target sleeve exposure.

```
{ "target_exposure": 0.0-1.0,   # fraction of the sleeve budget held in crypto
  "rationale": "<=300 chars" }
```

Inputs: realized vol (30/90d), current drawdown from sleeve peak, the running
forecast scorecard, and portfolio context. Explicitly NOT news sentiment —
that is where directional prediction crept in last time.

### Shrinkage — the steering cannot hurt much until it has earned the right

The executed exposure is never the raw model output:

```
w_exec = w_base + k * (w_model - w_base)
```

`k` starts at **0.25** and moves only on demonstrated forecast skill:

| forecast record (v2, price-blind, vs base rate) | k |
|---|---|
| < 30 scored observations | 0.00 (baseline only — observe, do not act) |
| >= 30 obs, brier skill <= 0 | 0.00 |
| >= 30 obs, brier skill > 0, not significant | 0.25 |
| >= 50 obs, brier skill > 0 at t >= 2 | 0.50 |
| >= 100 obs, sustained | 0.75 |

`k = 0` for the first 30 observations is the important line. The sleeve runs
as pure baseline beta while the forecaster builds a record. Nothing about the
steering is trusted until it has been measured, which is precisely the
discipline every previous Kraken attempt lacked.

## 5. The baseline it is measured against

**`w_base = 1.0` — fully invested in the 85/15 basket, always.**

Chosen because it is the honest null: "just hold" is the strategy the steering
must beat, and it is what we would do absent any intelligence.

Every month, record both paths:

| field | meaning |
|---|---|
| `w_base` | 1.0 |
| `w_exec` | what was actually held |
| `r_basket` | realized basket return that month |
| `delta` | `(w_exec - w_base) * r_basket` — steering's contribution |

Cumulative `sum(delta)` is the steering's P&L attribution, with the baseline
path tracked in parallel as a shadow portfolio. This is a **paired
difference**, which is the point: differences have far lower variance than
levels, and it is the only reason this is measurable at all.

Expected time to significance on the P&L track: monthly basket sigma is ~13%
(45% annualised / sqrt(12)). At a +/-0.25 exposure shift the per-decision
difference has sd ~3.25%. Detecting a genuine 1%/month contribution at t=2
needs ~42 months; 2%/month needs ~11. **State this up front** — the P&L track
is a multi-year verdict, which is exactly why the Brier fast track exists to
give a read years earlier.

## 6. Kill criteria — decided now, not after we are attached to it

Retire the steering layer (drop to permanent `k = 0`, keep the beta) when any
of these holds:

- 50+ scored forecasts with brier skill <= 0 versus base rate
- 24 months with cumulative `sum(delta)` < 0
- the sleeve's own drawdown exceeds 70% (the beta thesis itself is wrong)

Retire the whole sleeve if crypto beta is no longer wanted in the book —
that is an allocation decision, unrelated to whether the steering works.

## 7. What this deliberately does not do

- **No entry/exit timing.** Ruled out: the desk's 249 views had a 0.490 median
  and never exceeded 0.540 against a 0.524-0.541 break-even bar.
- **No rebalancing below $10k.** Worth ~$2-5/yr at this size.
- **No paid LLM calls.** The local tier is free, so scoring cadence is not a
  cost decision. The previous desk burned 4,380 reserved Opus calls a year at
  83 calls per signal opened.
- **No leverage, no perps, no margin.** The Kraken client is spot-only and
  should stay that way for this sleeve.

## 8. Open questions before implementation

1. Does the `forecast_score_facts` schema accommodate a continuous asset
   forecast, or does it assume a binary prediction-market outcome? Likely
   needs a resolution adapter, not a schema change.
2. ~~Kraken minimum order sizes vs a $200-420 sleeve at 85/15.~~ **RESOLVED
   2026-08-01**: `ordermin` is 0.001 ETH ($1.84) and 5e-05 XBT ($3.14). The
   15% ETH leg of even a $200 sleeve is $30 — 16x the floor. Not a constraint
   at any sleeve size we would consider.
3. Where does the shadow baseline portfolio live? Probably alongside
   `kraken_paper_positions` rather than in `portfolio`, to keep it out of
   real-exposure accounting.
