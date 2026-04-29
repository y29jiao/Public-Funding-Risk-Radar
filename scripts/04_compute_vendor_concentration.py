"""Step 4: compute vendor concentration metrics."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.concentration import compute_vendor_concentration, load_vendor_records
from src.utils import configure_logging, write_csv


def main() -> None:
    configure_logging()
    records = load_vendor_records()
    market_shares, concentration = compute_vendor_concentration(records)
    shares_path = write_csv(market_shares, "vendor_market_shares.csv")
    concentration_path = write_csv(concentration, "vendor_concentration.csv")
    print(f"Wrote {len(market_shares)} rows to {shares_path}")
    print(f"Wrote {len(concentration)} rows to {concentration_path}")


if __name__ == "__main__":
    main()
