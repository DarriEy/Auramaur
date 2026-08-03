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

### US CPI — GREEN (dataset frozen 2026-08-03)

No free historical *survey consensus* exists without scraping someone's
calendar. The clean proxy: **Cleveland Fed daily inflation nowcasts** —
produced before each release, archived, and per the Fed's own 2023
Economic Commentary often *more* accurate than professional survey
consensus. Surprise := actual − final pre-release nowcast.

RESOLVED same day the spike started. The chart's own data feed is a
plain JSON behind the page (the CSV button is a client-side slice, which
is why it exposes no URL):

    https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json?sc_lang=en
    (siblings: nowcast_quarter.json, nowcast_year.json)

The page 403s generic fetchers but serves a normal browser user-agent.
`nowcast_month.json` holds **157 monthly vintage panels, 2013-07 →
2026-07**: daily nowcast values for CPI / Core CPI / PCE / Core PCE plus
the ACTUAL releases as separate series, with each panel's date axis
running past month-end to the release date. Surprise construction —
release date, final pre-release nowcast, actual — needs nothing outside
this one file. 13 years ≈ 156 monthly CPI events, 2.5× the design's
five-year requirement.

**Frozen copy:** `data/events/cleveland-nowcast-month-retrieved-2026-08-03.json.gz`
(sha256 of uncompressed:
409d0f2121c5292217a780d7934ce5adbdf1ef5e5b8ed18d6de13b435e61740d).
Caveat to carry into Phase 2: the nowcasting model itself was revised
over the years (the Fed documents this); vintages are as-published,
which is the honest real-time series — do not "fix" old vintages with
the current model.

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

1. ~~Cleveland nowcast vintage archive~~ — DONE 2026-08-03, frozen in
   `data/events/` (see above).
2. Download + freeze Bauer–Swanson and EA-MPD files with retrieval dates.
3. Cross-validate the Yahoo estimate sample; fill HSBC-class gaps.
4. Assemble the frozen dataset + per-event timing column; then G1
   sign-off and Phase 2 (replay harness) begins.
