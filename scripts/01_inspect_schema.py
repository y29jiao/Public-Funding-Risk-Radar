"""Step 1: inspect database schema and write discovery outputs."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SCHEMA_KEYWORDS
from db.connection import get_engine
from db.schema_inspector import categorize_tables, find_keyword_matches, list_columns, list_tables, sample_table
from src.utils import configure_logging, write_csv


def main() -> None:
    configure_logging()
    engine = get_engine()
    tables = list_tables(engine)
    columns = list_columns(engine)
    write_csv(tables, "schema_tables.csv")
    write_csv(columns, "schema_columns.csv")

    matched = find_keyword_matches(columns, SCHEMA_KEYWORDS)
    buckets = categorize_tables(matched)
    _write_overview(engine, tables, columns, buckets)


def _write_overview(engine, tables, columns, buckets) -> None:
    path = PROJECT_ROOT / "outputs" / "schema_overview.md"
    lines = [
        "# Schema Overview",
        "",
        f"Discovered {len(tables)} tables/views and {len(columns)} columns.",
        "",
        "Keyword-based discovery only identifies candidates. Review samples before relying on inferred tables.",
        "",
    ]
    sampled = set()
    for title, df in buckets.items():
        lines.extend([f"## {title}", ""])
        if df.empty:
            lines.extend(["No candidates found.", ""])
            continue
        lines.append(df.head(50).to_markdown(index=False))
        lines.append("")
        if "table_schema" in df.columns and "table_name" in df.columns:
            for row in df[["table_schema", "table_name"]].drop_duplicates().head(10).itertuples(index=False):
                key = (row.table_schema, row.table_name)
                if key in sampled:
                    continue
                sampled.add(key)
                sample = sample_table(engine, row.table_schema, row.table_name, limit=5)
                lines.extend([f"### Sample: `{row.table_schema}.{row.table_name}`", ""])
                if sample is None or sample.empty:
                    lines.extend(["No sample rows available or sampling failed.", ""])
                else:
                    lines.append(sample.to_markdown(index=False))
                    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
