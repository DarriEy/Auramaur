# Live authority cutover

Directional live trading is authorized only by
`graduation.live_authority`. The tracked default is empty and therefore
paper-forces directional cells until an operator supplies grants in the
gitignored runtime override.

Before deploying a revision that removes directional exemptions:

1. Add each intended strategy, venue, and category to
   `runtime/config/defaults.local.yaml`.
2. Set a per-order stake cap, aggregate open-notional cap, grant timestamp,
   review date, realized-loss stop, and realized-exit review count.
3. Remove those strategies from `graduation.exempt_strategies`; only
   `arbitrage`, `market_maker`, and `order_monitor` are valid exemptions.
4. Load the actual override and run the startup cross-check:

   ```powershell
   $env:AURAMAUR_LOCAL_CONFIG='runtime/config/defaults.local.yaml'
   python -c "from config.settings import Settings; from auramaur.risk.graduation import GraduationLadder; s=Settings(); print(GraduationLadder(None, s).authority_crosscheck())"
   ```

   Deployment is ready only when this prints `[]`.
5. Restart. A live process logs and prints the number of verified grants before
   starting tasks; any unknown or ineligible scope aborts startup.

Removing a grant or setting the owning pillar to paper is the rollback.
Never restore a directional exemption. Blank-label realized losses count
conservatively against every matching strategy/venue grant, and both sells and
settlements advance the registered review boundary.
