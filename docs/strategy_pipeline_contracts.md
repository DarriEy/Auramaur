# Strategy and exposure pipeline contracts

Auramaur treats strategy identity and money movement as two related but separate
perimeters.

## Strategy registry

`auramaur.strategy.registry.STRATEGY_SPECS` is the source of truth for every
bot-scheduled trading strategy or venue lane. A strategy specification declares:

- its implementation class and bot task;
- the real cycle runner (`run_once`, `run_cycle`, `scan`, or `check_for_news`);
- its execution mode and paper/live eligibility;
- supported venues and required lifecycle capabilities.

Every `_task_*` defined in the bot scheduling modules must be either registered
as a strategy or listed in `NON_STRATEGY_TASKS` as operational/research-only.
Adding a task without classifying it fails `test_strategy_protocol.py`.

## Exposure registry

`auramaur.strategy.exposure_registry` owns the complete application exposure
perimeter. Each path documents its decision source, data dependencies, risk
authority, execution boundary, modes, booking, monitoring, exit, reconciliation,
settlement, and attribution.

`REGISTERED_CALLSITES` is an exact multiset. Counts are intentional: paired legs
and two-sided quotes contain more than one adapter call. A sensitive callsite
must have exactly one exposure-path owner.

The tests also inventory the final live SDK/HTTP boundary behind each adapter:
Polymarket, Kalshi, Crypto.com, Kraken, IBKR, and on-chain Safe submission.
The web package is asserted to contain no exposure mutation calls; the agent MCP
booking surface is asserted paper-only.

## Adding or changing a trading path

1. Add or update its `StrategySpec` and choose the real runner.
2. Classify its deployment as `paper_only`, `graduatable`, or `structural_live`.
3. Add every new sensitive callsite to exactly one exposure path.
4. If a new broker SDK or HTTP order primitive is introduced, extend the adapter
   boundary inventory in `tests/test_exposure_registry.py`.
5. Preserve the global live gates, per-order dry-run gate, and kill switch.
6. Verify booking, monitoring/cancellation, exit/unwind, reconciliation,
   settlement, and attribution—not just order acceptance.
7. Run:

   ```powershell
   python -m pytest tests/test_strategy_protocol.py tests/test_exposure_registry.py -q
   python -m pytest -q
   python -m ruff check .
   ```

A registry entry is documentation and a test contract; it is not authorization
to enable a strategy or deploy live capital.