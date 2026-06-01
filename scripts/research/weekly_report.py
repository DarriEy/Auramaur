"""Weekly edge report — runs all the read-only edge research in one shot.

Re-run this weekly as resolution_pnl / price_history accumulate, to see whether
the reallocation flips dollars positive and whether the coupling becomes
tradeable.

    python scripts/research/weekly_report.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("DOLLAR EDGE (realized $ by category)", "dollar_edge.py"),
    ("CALIBRATION -> $ GAP (adverse selection)", "edge_gap.py"),
    ("PREDICTION EDGE (calibration/Brier)", "edge_audit.py"),
    ("LEAD-LAG COUPLING (spot -> prediction)", "coupling_discovery.py"),
    ("COUPLING TRADEABILITY", "coupling_tradeability.py"),
]


def main() -> None:
    here = Path(__file__).parent
    for title, script in SCRIPTS:
        print("\n" + "#" * 78 + f"\n# {title}\n" + "#" * 78)
        try:
            out = subprocess.run([sys.executable, str(here / script)],
                                 capture_output=True, text=True, timeout=180)
            print(out.stdout.rstrip() or out.stderr[:500])
        except Exception as e:  # noqa: BLE001
            print(f"  [failed: {e}]")


if __name__ == "__main__":
    main()
