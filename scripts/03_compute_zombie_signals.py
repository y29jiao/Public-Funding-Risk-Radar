"""Step 3: compute zombie-recipient screening signals."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.zombie_detector import compute_zombie_signals
from src.utils import configure_logging, read_csv_safe, write_csv


def main() -> None:
    configure_logging()
    entity = read_csv_safe("entity_funding_summary.csv")
    if entity is None:
        print("Missing outputs/entity_funding_summary.csv; writing empty zombie_signals.csv")
    df = compute_zombie_signals(entity)
    path = write_csv(df, "zombie_signals.csv")
    print(f"Wrote {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
