# Event/surprise data — Phase 1 spike findings

**Gate G1 of `ibkr-event-driven-design.md`. Spike started 2026-08-03.
Verdict so far: no blocker found — two verifications open before G1
sign-off.** Everything below was checked against the live web or probed
empirically on the date above; re-verify links before relying on them
months later.

## Per event family

### FOMC (US) — GREEN (dataset frozen 2026-08-03)

**Bauer–Swanson Monetary Policy Surprises**, published by the San
Francisco Fed. High-frequency market-implied surprises around every FOMC
announcement, raw and orthogonalized variants. Lookahead-clean *by
construction* — derived from asset-price changes in tight windows, no
survey consensus involved. The academic standard (Bauer & Swanson 2023).

Frozen: `data/events/bauer-swanson-mps-retrieved-2026-08-03.xlsx`
(sha256 708606624de050cdaf6fd850a320b38d45099ab7c2398e01ba12dcece5743668)
plus the page's chart CSV
(`bauer-swanson-chart-retrieved-2026-08-03.csv`, sha256
55aabf5b0742447614e6de91299822cc81de6be210475957fc3db7e4f4807508).
Coverage measured on inspection: **361 FOMC events, 1988-02-04 →
2023-12-13** ("update 2023" sheet; the chart CSV confirms the public
series ends Dec 2023). The 2013–2023 overlap with the CPI vintages is
the skeleton's natural macro window; extending 2024+ (USMPD or fed-funds
futures construction) is optional Phase-2 work, not a G1 blocker — the
forward paper phase covers the present by definition.

### ECB — GREEN (dataset frozen 2026-08-03)

**EA-MPD** (Altavilla, Brugnolini, Gürkaynak, Motto, Ragusa 2019),
downloaded from the ECB (ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx):
intraday price changes (OIS, sovereign yields, stocks, FX) around every
Governing Council announcement, press-release vs press-conference
windows separated. Same lookahead-clean-by-construction property.

Frozen: `data/events/ea-mpd-retrieved-2026-08-03.xlsx` (sha256
f417fb861a305e3cbb871a7571c4899ba6bca003d9e087024d2f3f0744426cef).
Coverage measured on inspection: **315 ECB events, 1999-01-07 →
2025-10-30** — near-current.

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
### (dataset assembled and frozen 2026-08-03)

Pulled via the ADR listings (yfinance) for all 16 names: **347 events**,
most names 24–25 estimated quarterly events spanning 2020→2026. Frozen:
`data/events/earnings-yfinance-adr-retrieved-2026-08-03.csv` (sha256
c054849f83f3306942aef902149b01087e2376b4be13cc9aefb7a9f5fe519e35),
columns home_listing / adr_ticker / event_datetime_et / eps_estimate /
eps_actual / surprise_pct. Coverage notes from the pull: BHP (14), RIO
(12) and UL (24 over two decades) are **semi-annual reporters** — their
low counts are complete histories, not gaps; HSBC (6) is genuinely thin
and needs a secondary source (Alpha Vantage EARNINGS, free key, is the
candidate); TCEHY/SIEGY/ALIZY came back healthier than feared (16–25
estimated events each). Caveats that must be resolved before G1
sign-off:

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

## G1 sign-off gates — both closed 2026-08-03

**(a) Estimate cross-validation (Yahoo vs Nasdaq/Zacks consensus).**
33 joined events across the 10 Nasdaq-covered names. EPS *levels*
disagree systematically — and the pattern explains itself: AZN differs by
exactly 2× every quarter (ADR share basis), ASML by ~1.17× (EUR vs USD),
RY by ~1.41× (CAD vs USD). Yahoo quotes home-currency / ordinary-share
bases; Nasdaq quotes USD-per-ADS. The basis-invariant test is the
*surprise percentage*, where each vendor pairs its own estimate with its
own actual: **32/33 events agree on surprise SIGN** (one flip: BP
2026-02-10), while magnitudes scatter (independent analyst panels
genuinely disagree about what consensus was). Validation evidence
frozen: `nasdaq-surprise-validation-2026-08-03.json.gz`.

**DESIGN CONSTRAINT carried into Phase 2 (binding):** the earnings
skeleton must be **sign-based, or rank-based within one vendor's own
series**. A magnitude threshold like "|surprise| ≥ X%" imports vendor
noise as if it were signal, and cross-vendor magnitude is not a real
quantity. HSBC stays thin (Nasdaq adds only ~4 recent quarters);
excluding it entirely would cost ~2% of pooled events — Phase 2's
skeleton spec decides, either way is documented.

**(b) Report-timing column.** Built from the pull's own ET timestamps
and frozen as `earnings-events-timing-2026-08-03.csv`: 265 BMO (<09:30
ET → enter at the same day's US close), 47 AMC (≥16:00 → next day's
close), 10 MID and 25 UNKNOWN (both → next day's close, conservative).
76% BMO matches the roster's European-morning reporting reality.

## G1: SIGNED 2026-08-03

All four data legs frozen with hashes; the one binding constraint above
travels with the earnings leg. Phase 2 (the event replay harness and the
frozen Stage-1 rule set) may begin. Per the design brief, Stage-1
parameters must be fixed before scoring and the harness must never write
to the forward-evidence tables.
