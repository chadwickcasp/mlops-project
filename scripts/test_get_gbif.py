#!/usr/bin/env python3
"""Evaluation runner for the SF March 2026 GBIF workflow."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sourcing.gbif import get_gbif

SF_POLYGON_WKT = (
    "POLYGON(("
    "-122.55 37.70, "
    "-122.35 37.70, "
    "-122.35 37.83, "
    "-122.55 37.83, "
    "-122.55 37.70"
    "))"
)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Expected output file not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Output file is empty: {path}")


def _evaluate_normalized_csv(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("Normalized CSV has no rows.")

    required_columns = {
        "scientificName",
        "speciesKey",
        "vernacularName",
        "imageUrl",
        "year",
        "month",
        "decimalLatitude",
        "decimalLongitude",
    }
    missing_columns = required_columns - set(rows[0].keys())
    if missing_columns:
        raise RuntimeError(f"Normalized CSV missing columns: {sorted(missing_columns)}")

    filled_vernacular = sum(1 for row in rows if str(row.get("vernacularName") or "").strip())
    if filled_vernacular == 0:
        raise RuntimeError("No vernacularName values were populated.")

    print(f"Normalized CSV row count: {len(rows)}")
    print(f"Rows with vernacularName: {filled_vernacular}")


def main() -> int:
    query = get_gbif.GBIFQuery(
        geometry_wkt=SF_POLYGON_WKT,
        year=2026,
        month=3,
        occurrence_status="PRESENT",
        has_coordinate=True,
    )
    outputs = get_gbif.build_run_outputs(run_label="sf_2026_03")

    print("Running GBIF SF March 2026 workflow from test script...")
    get_gbif.run_pipeline(query=query, outputs=outputs)

    print("Validating generated artifacts...")
    _require_file(outputs.occurrence_cache_path)
    _require_file(outputs.occurrence_id_index_path)
    _require_file(outputs.vernacular_index_path)
    _require_file(outputs.normalized_csv_path)
    _require_file(outputs.class_counts_csv_path)
    _require_file(outputs.map_png_path)
    _evaluate_normalized_csv(outputs.normalized_csv_path)

    print("GBIF evaluation checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        print(f"GBIF evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
