#!/usr/bin/env python3
"""GBIF-specific data pull, enrichment, caching, and plotting primitives."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import contextily as ctx  # type: ignore[import-not-found]  # pylint: disable=import-error
    import matplotlib
    import matplotlib.colors as mcolors
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:
    raise ImportError(
        "Plotting dependencies are missing. Install with: "
        "python -m pip install -r requirements-gbif.txt"
    ) from error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_sourcing.scrape_apis import (  # noqa: E402
    create_session,
    fetch_paginated_results,
    request_json_with_backoff,
)

OCCURRENCE_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
SPECIES_API_URL = "https://api.gbif.org/v1/species"
USER_AGENT = "mlops-zoomcamp-gbif/0.1 (contact: casperchadwick@gmail.com)"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "species_dist" / "gbif"
DEFAULT_VIZ_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gbif_viz"
OCCURRENCE_CACHE_FILENAME = "occurrences_cache.jsonl"
OCCURRENCE_ID_INDEX_FILENAME = "occurrence_id_index.json"
VERNACULAR_INDEX_FILENAME = "species_vernacular_index.json"

MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_PAGE_LIMIT = 300

ALLOWED_BASIS_OF_RECORD = {
    "OBSERVATION",
    "HUMAN_OBSERVATION",
    "MACHINE_OBSERVATION",
    "MATERIAL_CITATION",
    "OCCURRENCE",
}
WILD_EXCLUSION_TOKENS = {"CAPTIVE", "CULTIVATED", "DOMESTIC", "MANAGED"}

# Layman short labels for GBIF `class` values (plot legend / keys only; CSV stays scientific).
CLASS_SHORT_LABELS: dict[str, str] = {
    "Aves": "Birds",
    "Magnoliopsida": "Dicot plants",
    "Insecta": "Insects",
    "Liliopsida": "Monocot plants",
    "Mammalia": "Mammals",
    "Malacostraca": "Crustaceans",
    "Arachnida": "Arachnids",
    "Gastropoda": "Snails/slugs",
    "Squamata": "Lizards/snakes",
    "Agaricomycetes": "Mushrooms",
    "Testudines": "Turtles",
    "UNKNOWN": "Unknown",
    "Amphibia": "Amphibians",
    "Phaeophyceae": "Brown algae",
    "Anthozoa": "Corals/anemones",
    "Maxillopoda": "Barnacles/copepods",
    "Lecanoromycetes": "Lichens",
    "Myxomycetes": "Slime molds",
    "Bivalvia": "Bivalves",
    "Echinoidea": "Sea urchins",
    "Polypodiopsida": "Ferns",
    "Megaviricetes": "Giant viruses",
    "Sordariomycetes": "Sac fungi",
    "Pinopsida": "Conifers",
    "Polyplacophora": "Chitons",
    "Marchantiopsida": "Liverworts",
    "Lycopodiopsida": "Clubmosses",
    "Elasmobranchii": "Sharks/rays",
    "Asteroidea": "Sea stars",
    "Hydrozoa": "Hydroids",
    "Clitellata": "Worms/leeches",
    "Diplopoda": "Millipedes",
    "Tentaculata": "Bryozoans",
    "Ascidiacea": "Sea squirts",
    "Demospongiae": "Sponges",
}

# Naturalistic marker colors (hex) keyed by scientific class name.
NATURAL_CLASS_COLORS: dict[str, str] = {
    "Mammalia": "#8B4513",
    "Magnoliopsida": "#6B8E23",
    "Liliopsida": "#9ACD32",
    "Polypodiopsida": "#2E8B57",
    "Pinopsida": "#556B2F",
    "Marchantiopsida": "#8FBC8F",
    "Lycopodiopsida": "#808000",
    "Aves": "#87AFC7",
    "Insecta": "#7B559C",
    "Arachnida": "#8B0000",
    "Gastropoda": "#C4A57B",
    "Squamata": "#2F4F4F",
    "Agaricomycetes": "#E67E22",
    "Lecanoromycetes": "#D35400",
    "Testudines": "#3D6B47",
    "UNKNOWN": "#888888",
    "Amphibia": "#3CB371",
    "Phaeophyceae": "#6B5344",
    "Anthozoa": "#FF7F50",
    "Maxillopoda": "#5F8A8B",
    "Myxomycetes": "#ADFF2F",
    "Sordariomycetes": "#7DCEA0",
    "Malacostraca": "#CD853F",
    "Bivalvia": "#BC8F8F",
    "Echinoidea": "#DEB887",
    "Megaviricetes": "#DDA0DD",
    "Polyplacophora": "#A0522D",
    "Elasmobranchii": "#4682B4",
    "Asteroidea": "#FFD700",
    "Hydrozoa": "#B0C4DE",
    "Clitellata": "#FF69B4",
    "Diplopoda": "#5C3D6E",
    "Tentaculata": "#87CEEB",
    "Ascidiacea": "#BA55D3",
    "Demospongiae": "#C17A5F",
}


def class_short_label(scientific_class: str) -> str:
    key = str(scientific_class or "").strip() or "UNKNOWN"
    return CLASS_SHORT_LABELS.get(key, key)


def class_key_labels(scientific_classes: list[str]) -> list[str]:
    return [class_short_label(name) for name in scientific_classes]


def natural_class_color_rgba(scientific_class: str, alpha: float = 0.75) -> tuple[float, float, float, float]:
    key = str(scientific_class or "").strip() or "UNKNOWN"
    hex_color = NATURAL_CLASS_COLORS.get(key, "#888888").lstrip("#")
    red = int(hex_color[0:2], 16) / 255.0
    green = int(hex_color[2:4], 16) / 255.0
    blue = int(hex_color[4:6], 16) / 255.0
    return (red, green, blue, alpha)


@dataclass(frozen=True)
class GBIFQuery:
    geometry_wkt: str
    year: int
    month: int
    occurrence_status: str = "PRESENT"
    has_coordinate: bool = True
    page_limit: int = DEFAULT_PAGE_LIMIT


@dataclass(frozen=True)
class GBIFRunOutputs:
    output_dir: Path
    occurrence_cache_path: Path
    occurrence_id_index_path: Path
    vernacular_index_path: Path
    normalized_csv_path: Path
    class_counts_csv_path: Path
    map_png_path: Path


def build_run_outputs(
    run_label: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    viz_output_dir: Path | None = DEFAULT_VIZ_OUTPUT_DIR,
) -> GBIFRunOutputs:
    viz_dir = viz_output_dir if viz_output_dir is not None else output_dir
    return GBIFRunOutputs(
        output_dir=output_dir,
        occurrence_cache_path=output_dir / OCCURRENCE_CACHE_FILENAME,
        occurrence_id_index_path=output_dir / OCCURRENCE_ID_INDEX_FILENAME,
        vernacular_index_path=output_dir / VERNACULAR_INDEX_FILENAME,
        normalized_csv_path=output_dir / f"occurrences_normalized_{run_label}.csv",
        class_counts_csv_path=output_dir / f"class_counts_{run_label}.csv",
        map_png_path=viz_dir / f"{run_label}_occurrence_map.png",
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_upper(value: Any) -> str:
    return "" if value is None else str(value).strip().upper()


def get_occurrence_key(record: dict[str, Any]) -> str | None:
    occurrence_id = str(record.get("occurrenceID") or "").strip()
    if occurrence_id:
        return occurrence_id
    gbif_id = str(record.get("gbifID") or record.get("key") or "").strip()
    return gbif_id or None


def load_id_index(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return {str(item).strip() for item in payload if str(item).strip()}
    if isinstance(payload, dict):
        keys = payload.get("ids", [])
        if isinstance(keys, list):
            return {str(item).strip() for item in keys if str(item).strip()}
    return set()


def save_id_index(path: Path, ids: set[str]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(sorted(ids), file, ensure_ascii=True, indent=2)


def fetch_all_occurrences(query: GBIFQuery) -> list[dict[str, Any]]:
    session = create_session(USER_AGENT)

    def _on_first_page(payload: dict[str, Any]) -> None:
        print(f"Estimated matches reported by GBIF: {payload.get('count')}")

    def _on_page(offset: int, page_results: int, accumulated: int) -> None:
        print(
            f"Fetched page offset={offset}, page_results={page_results}, "
            f"accumulated={accumulated}"
        )

    return fetch_paginated_results(
        session=session,
        url=OCCURRENCE_SEARCH_URL,
        base_params={
            "geometry": query.geometry_wkt,
            "hasCoordinate": str(query.has_coordinate).lower(),
            "occurrenceStatus": query.occurrence_status,
            "year": query.year,
            "month": query.month,
        },
        page_limit=query.page_limit,
        max_retries=MAX_RETRIES,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        page_sleep_seconds=0.2,
        on_first_page=_on_first_page,
        on_page=_on_page,
    )


def _looks_wild(record: dict[str, Any]) -> bool:
    joined = " ".join(
        str(item)
        for item in (
            record.get("establishmentMeans"),
            record.get("degreeOfEstablishment"),
            record.get("occurrenceRemarks"),
        )
        if item
    ).upper()
    return not any(token in joined for token in WILD_EXCLUSION_TOKENS)


def _basis_allowed(record: dict[str, Any]) -> bool:
    return _safe_upper(record.get("basisOfRecord")) in ALLOWED_BASIS_OF_RECORD


def _extract_image_url(record: dict[str, Any]) -> str:
    media = record.get("media")
    if isinstance(media, list):
        for item in media:
            if isinstance(item, dict):
                identifier = str(item.get("identifier") or "").strip()
                media_type = _safe_upper(item.get("type"))
                if identifier and ("IMAGE" in media_type or media_type == ""):
                    return identifier
    associated_media = str(record.get("associatedMedia") or "").strip()
    if "|" in associated_media:
        return associated_media.split("|")[0].strip()
    if ";" in associated_media:
        return associated_media.split(";")[0].strip()
    return associated_media


def _extract_description(record: dict[str, Any]) -> str:
    for field in ("occurrenceRemarks", "eventRemarks", "fieldNotes", "dynamicProperties"):
        value = str(record.get(field) or "").strip()
        if value:
            return value
    return ""


def _to_float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return None if value in (None, "") else int(value)
    except (TypeError, ValueError):
        return None


def load_vernacular_index(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in payload.items():
        clean_key = str(key).strip()
        clean_value = str(value).strip()
        if clean_key:
            cleaned[clean_key] = clean_value
    return cleaned


def save_vernacular_index(path: Path, index: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=True, indent=2, sort_keys=True)


def _select_best_vernacular_name(rows: list[dict[str, Any]]) -> str:
    best_score = -1
    best_name = ""
    for row in rows:
        name = str(row.get("vernacularName") or "").strip()
        if not name:
            continue
        lang = str(row.get("language") or "").strip().lower()
        country = str(row.get("country") or "").strip().upper()
        preferred = bool(row.get("preferred", False))
        score = (100 if lang in {"eng", "en"} else 0) + (50 if country == "US" else 0) + (
            10 if preferred else 0
        )
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def fetch_vernacular_name(session, species_key: str) -> str:
    offset = 0
    limit = 100
    all_rows: list[dict[str, Any]] = []
    while True:
        payload = request_json_with_backoff(
            session=session,
            url=f"{SPECIES_API_URL}/{species_key}/vernacularNames",
            params={"offset": offset, "limit": limit},
            max_retries=MAX_RETRIES,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        results = payload.get("results", [])
        if not results:
            break
        all_rows.extend(results)
        if payload.get("endOfRecords", False):
            break
        offset += limit
    return _select_best_vernacular_name(all_rows)


def _mercator_xy(lon: float, lat: float) -> tuple[float, float]:
    x = lon * 20037508.34 / 180.0
    clamped_lat = max(min(lat, 85.05112878), -85.05112878)
    y = math.log(math.tan((90.0 + clamped_lat) * math.pi / 360.0)) * 20037508.34 / math.pi
    return x, y


def is_target_record(record: dict[str, Any], query: GBIFQuery) -> bool:
    return (
        int(record.get("year") or -1) == query.year
        and int(record.get("month") or -1) == query.month
        and _safe_upper(record.get("occurrenceStatus")) == _safe_upper(query.occurrence_status)
        and _basis_allowed(record)
        and _looks_wild(record)
    )


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "occurrence_key": get_occurrence_key(record) or "",
        "gbifID": str(record.get("gbifID") or ""),
        "key": str(record.get("key") or ""),
        "scientificName": str(record.get("scientificName") or ""),
        "speciesKey": _to_int(record.get("speciesKey")),
        "species": str(record.get("species") or ""),
        "genus": str(record.get("genus") or ""),
        "family": str(record.get("family") or ""),
        "order": str(record.get("order") or ""),
        "class": str(record.get("class") or ""),
        "kingdom": str(record.get("kingdom") or ""),
        "taxonRank": str(record.get("taxonRank") or ""),
        "vernacularName": str(record.get("vernacularName") or ""),
        "imageUrl": _extract_image_url(record),
        "description": _extract_description(record),
        "basisOfRecord": str(record.get("basisOfRecord") or ""),
        "occurrenceStatus": str(record.get("occurrenceStatus") or ""),
        "year": int(record.get("year") or 0),
        "month": int(record.get("month") or 0),
        "day": int(record.get("day") or 0),
        "eventDate": str(record.get("eventDate") or ""),
        "decimalLatitude": _to_float(record.get("decimalLatitude")),
        "decimalLongitude": _to_float(record.get("decimalLongitude")),
        "countryCode": str(record.get("countryCode") or ""),
        "stateProvince": str(record.get("stateProvince") or ""),
        "locality": str(record.get("locality") or ""),
    }


def append_records_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True))
            file.write("\n")


def iter_cached_records(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_normalized_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "occurrence_key",
        "gbifID",
        "key",
        "scientificName",
        "speciesKey",
        "species",
        "genus",
        "family",
        "order",
        "class",
        "kingdom",
        "taxonRank",
        "vernacularName",
        "imageUrl",
        "description",
        "basisOfRecord",
        "occurrenceStatus",
        "year",
        "month",
        "day",
        "eventDate",
        "decimalLatitude",
        "decimalLongitude",
        "countryCode",
        "stateProvince",
        "locality",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_class_counts_csv(rows: list[dict[str, Any]], path: Path) -> None:
    class_counts: dict[str, int] = {}
    for row in rows:
        class_name = str(row.get("class") or "UNKNOWN").strip() or "UNKNOWN"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "count"])
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
            writer.writerow([class_name, count])


def _strip_spines(axis: Any) -> None:
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_distribution(rows: list[dict[str, Any]], output_path: Path) -> None:
    valid_rows = [
        row
        for row in rows
        if row.get("decimalLatitude") is not None and row.get("decimalLongitude") is not None
    ]
    if not valid_rows:
        print("No coordinate rows available for map plot.")
        return

    projected = [
        _mercator_xy(float(row["decimalLongitude"]), float(row["decimalLatitude"]))
        for row in valid_rows
    ]
    xs = [item[0] for item in projected]
    ys = [item[1] for item in projected]
    classes = [str(row.get("class") or "UNKNOWN") for row in valid_rows]

    unique_classes = sorted(set(classes))
    class_counts = {class_name: 0 for class_name in unique_classes}
    for class_name in classes:
        class_counts[class_name] += 1

    marker_colors = [natural_class_color_rgba(cn, alpha=0.72) for cn in classes]
    n_classes = len(unique_classes)
    bar_colors = [NATURAL_CLASS_COLORS.get(name, "#888888") for name in unique_classes]

    fig = plt.figure(figsize=(14, 9), layout="constrained")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.42],
        wspace=0.06,
    )
    ax = fig.add_subplot(grid[0, 0])
    right_grid = grid[0, 1].subgridspec(1, 3, width_ratios=[2.9, 0.36, 5.8], wspace=0.0)
    lax = fig.add_subplot(right_grid[0, 0])
    cax = fig.add_subplot(right_grid[0, 1], sharey=lax)
    hax = fig.add_subplot(right_grid[0, 2], sharey=cax)

    ax.scatter(
        xs,
        ys,
        c=marker_colors,
        s=16,
        edgecolors="none",
    )
    x_pad = (max(xs) - min(xs)) * 0.08 if len(xs) > 1 else 800
    y_pad = (max(ys) - min(ys)) * 0.08 if len(ys) > 1 else 800
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron)
    ax.set_title("GBIF Occurrences")
    ax.set_xlabel("Web Mercator X")
    ax.set_ylabel("Web Mercator Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    _strip_spines(ax)

    # Class color key (RGB rows match natural colors).
    strip_rgb = np.zeros((n_classes, 1, 3), dtype=float)
    for idx, name in enumerate(unique_classes):
        strip_rgb[idx, 0, :] = mcolors.to_rgb(NATURAL_CLASS_COLORS.get(name, "#888888"))
    lax.set_xlim(0, 1)
    lax.set_ylim(-0.5, n_classes - 0.5)
    lax.set_xticks([])
    lax.set_yticks([])
    lax.tick_params(
        left=False,
        right=False,
        bottom=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
        length=0,
    )
    for idx, label in enumerate(class_key_labels(unique_classes)):
        lax.text(
            0.98,
            idx,
            label,
            va="center",
            ha="right",
            fontsize=8,
        )
    lax.set_title("Organism Class", fontsize=8, pad=4)
    _strip_spines(lax)

    cax.imshow(strip_rgb, aspect="auto", origin="lower", interpolation="nearest")
    cax.set_xticks([])
    cax.set_yticks(range(n_classes))
    cax.set_yticklabels([])
    cax.tick_params(
        left=False,
        right=False,
        bottom=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
        length=0,
    )
    _strip_spines(cax)

    y_positions = list(range(n_classes))
    counts_by_index = [class_counts[name] for name in unique_classes]
    bars = hax.barh(
        y_positions,
        counts_by_index,
        color=bar_colors,
        alpha=0.92,
        linewidth=0,
        edgecolor="none",
    )
    hax.set_xlabel("Count", fontsize=9)
    hax.set_yticks(range(n_classes))
    hax.set_yticklabels([])
    hax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False, length=0)
    hax.set_title("Per class", fontsize=9, pad=4)
    max_count = max(counts_by_index) if counts_by_index else 0
    hax.set_xlim(0, max_count * 1.22 + 1)
    hax.tick_params(axis="x", labelsize=8)
    _strip_spines(hax)

    for idx, bar in enumerate(bars):
        value = counts_by_index[idx]
        hax.text(
            bar.get_width() + max_count * 0.015 + 0.25,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            ha="left",
            fontsize=7,
            color=bar_colors[idx],
            fontweight="bold",
        )

    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def update_cache_and_outputs(
    fetched_rows: list[dict[str, Any]],
    query: GBIFQuery,
    outputs: GBIFRunOutputs,
) -> None:
    ensure_output_dir(outputs.output_dir)
    ensure_output_dir(outputs.map_png_path.parent)

    seen_ids = load_id_index(outputs.occurrence_id_index_path)
    print(f"Loaded {len(seen_ids)} IDs from occurrence index cache.")

    new_rows: list[dict[str, Any]] = []
    for row in fetched_rows:
        cache_key = get_occurrence_key(row)
        if not cache_key or cache_key in seen_ids:
            continue
        seen_ids.add(cache_key)
        new_rows.append(row)

    append_records_jsonl(outputs.occurrence_cache_path, new_rows)
    save_id_index(outputs.occurrence_id_index_path, seen_ids)
    print(f"New records appended: {len(new_rows)}")
    print(f"Total indexed records: {len(seen_ids)}")

    normalized_rows = [
        normalize_record(cached_row)
        for cached_row in iter_cached_records(outputs.occurrence_cache_path)
        if is_target_record(cached_row, query)
    ]

    vernacular_index = load_vernacular_index(outputs.vernacular_index_path)
    species_keys_needing_lookup = {
        str(row["speciesKey"])
        for row in normalized_rows
        if not str(row.get("vernacularName") or "").strip() and row.get("speciesKey") is not None
    }

    if species_keys_needing_lookup:
        lookup_session = create_session(USER_AGENT)
        fetched_vernacular_count = 0
        for species_key in sorted(species_keys_needing_lookup):
            if species_key in vernacular_index:
                continue
            vernacular_index[species_key] = fetch_vernacular_name(lookup_session, species_key)
            fetched_vernacular_count += 1
            time.sleep(0.05)
        save_vernacular_index(outputs.vernacular_index_path, vernacular_index)
        print(f"Fetched vernacular names for {fetched_vernacular_count} species keys.")

    filled_names = 0
    for row in normalized_rows:
        if str(row.get("vernacularName") or "").strip() or row.get("speciesKey") is None:
            continue
        fallback = vernacular_index.get(str(row["speciesKey"]), "")
        if fallback:
            row["vernacularName"] = fallback
            filled_names += 1
    print(f"Filled vernacularName from species index for {filled_names} rows.")

    write_normalized_csv(normalized_rows, outputs.normalized_csv_path)
    write_class_counts_csv(normalized_rows, outputs.class_counts_csv_path)
    plot_distribution(normalized_rows, outputs.map_png_path)

    print(f"Normalized rows written: {len(normalized_rows)}")
    print(f"CSV: {outputs.normalized_csv_path}")
    print(f"Class counts: {outputs.class_counts_csv_path}")
    print(f"Map: {outputs.map_png_path}")


def run_pipeline(query: GBIFQuery, outputs: GBIFRunOutputs) -> None:
    fetched_rows = fetch_all_occurrences(query)
    print(f"Fetched total rows this run: {len(fetched_rows)}")
    update_cache_and_outputs(fetched_rows, query, outputs)


def main() -> None:
    raise SystemExit(
        "get_gbif.py is a library module, not the SF March 2026 runner.\n"
        "Install deps: python -m pip install -r requirements-gbif.txt\n"
        "Run workflow: python scripts/test_get_gbif.py"
    )


if __name__ == "__main__":
    main()
