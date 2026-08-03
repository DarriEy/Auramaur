# Event/surprise data — Phase 1 spike findings

**Gate G1 of `ibkr-event-driven-design.md`. Spike started 2026-08-03.
Verdict so far: no blocker found — two verifications open before G1
sign-off.** Everything below was checked against the live web or probed
empirically on the date above; re-verify links before relying on them
months later.

## Per event family

### FOMC (US) — GREEN

**Bauer–Swanson Monetary Policy Surprises**, published and periodically
updated by the San Francisco Fed
(frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises).
High-frequency market-implied surprises around every FOMC announcement,
raw and orthogonalized variants. Lookahead-clean *by construction* —
derived from asset-price changes in tight windows, no survey consensus
involved. The academic standard (Bauer & Swanson 2023). The related
USMPD event-study database covers announcements, press conferences and
minutes. Remaining work: download, validate coverage span, freeze a
copy with a retrieval date.

### ECB — GREEN

**EA-MPD** (Altavilla, Brugnolini, Gürkaynak, Motto, Ragusa 2019),
public and periodically updated: intraday price changes (OIS, sovereign
yields, stocks, FX) around every Governing Council announcement, with
the press-release vs press-conference windows separated. Same
lookahead-clean-by-construction property as Bauer–Swanson. Relevant for
the European names on the equity roster and any future EUR fx work.

### US CPI — PROBABLE (one manual verification)

No free historical *survey consensus* exists without scraping someone's
calendar. The clean proxy: **Cleveland Fed daily inflation nowcasts** —
produced before each release, archived, and per the Fed's own 2023
Economic Commentary often *more* accurate than professional survey
consensus. Surprise := actual − final pre-release nowcast. This is a
documented, public, reproducible surprise series. OPEN: the site blocks
automated fetching (403), so the vintage archive's depth needs one
manual browser download to confirm (operator task, minutes). If the
vintage archive disappoints, fallback is restricting the macro leg to
FOMC/ECB where the event-study databases already suffice.

### Earnings, 16-name international roster — GREEN with caveats

Probed empirically 2026-08-03 via the ADR listings (yfinance):
ASML/SAP/AZN return **25 quarterly events each spanning 2020→2026, all
carrying EPS estimates**; HSBC is sparse (6 rows) and will need a
secondary source for gaps (Alpha Vantage's EARNINGS endpoint, free key,
is the candidate). Pooled across the roster this is ~300 events over
~5 years — enough for a **pooled** skeleton replay, not per-name
verdicts. Caveats that must be resolved before G1 sign-off:

1. **Estimate provenance.** Yahoo's historical "EPS Estimate" is
   believed to be the consensus frozen at report time; cross-validate a
   sample (~20 events) against a second source before trusting it as
   lookahead-clean.
2. **Timing.** Each event needs a report-time column (before/after
   which market's open, ADR vs home listing) or entry timing in the
   replay is fiction.
3. **Survivorship.** The roster is today's; a 2021 replay over it
   carries mild survivorship. Acceptable for a skeleton read if
   documented; do not silently forget it.

### BoE — DEFERRED

Analogous UK event-study data exists in the literature; not probed.
Start without it — FOMC/ECB/CPI/earnings is already more surface than
the skeleton needs.

## Price side

Daily bars for the roster and macro-sensitive instruments already flow
through the paths the momentum replay used (IBKR history + Alpaca IEX).
The daily-horizon design needs nothing faster.

## Remaining Phase 1 work

1. Manual download: Cleveland nowcast vintage archive (operator, browser).
2. Download + freeze Bauer–Swanson and EA-MPD files with retrieval dates.
3. Cross-validate the Yahoo estimate sample; fill HSBC-class gaps.
4. Assemble the frozen dataset + per-event timing column; then G1
   sign-off and Phase 2 (replay harness) begins.
