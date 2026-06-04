#!/usr/bin/env python3
"""Focused checks for GBIF map plotting helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sourcing.gbif import get_gbif


def _sample_rows() -> list[dict[str, object]]:
    return [
        {"decimalLatitude": 37.77, "decimalLongitude": -122.42, "class": "Aves"},
        {"decimalLatitude": 37.78, "decimalLongitude": -122.41, "class": "Insecta"},
        {"decimalLatitude": 37.76, "decimalLongitude": -122.43, "class": "Aves"},
    ]


def test_class_key_labels_use_common_names_without_numeric_legend_entries() -> None:
    labels = get_gbif.class_key_labels(["Aves", "Insecta"])

    if labels != ["Birds", "Insects"]:
        raise AssertionError(f"Unexpected class key labels: {labels}")
    if any("(" in label or "=" in label for label in labels):
        raise AssertionError(f"Class key labels should not include numeric legend details: {labels}")


def test_plot_distribution_writes_png() -> None:
    original_add_basemap = get_gbif.ctx.add_basemap
    get_gbif.ctx.add_basemap = lambda *args, **kwargs: None
    try:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gbif_plot.png"
            get_gbif.plot_distribution(_sample_rows(), output_path)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise AssertionError(f"Expected non-empty plot PNG at {output_path}")
    finally:
        get_gbif.ctx.add_basemap = original_add_basemap


def main() -> int:
    test_class_key_labels_use_common_names_without_numeric_legend_entries()
    test_plot_distribution_writes_png()
    print("GBIF plot checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
