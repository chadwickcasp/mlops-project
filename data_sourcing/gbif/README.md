# GBIF Data Sourcing (Sandbox)

This sandboxed workflow pulls GBIF occurrence data for a fixed baseline target:

- area: San Francisco bounding polygon
- time: March 2026

It writes cache and analysis artifacts under `data/species_dist/gbif/`.

## Dependencies

Keep this baseline lightweight:

- `requests`
- `matplotlib`
- `contextily` (for background map tiles)

Install with:

```bash
python -m pip install requests matplotlib contextily
```

## Run

From repo root:

```bash
python scripts/test_get_gbif.py
```

Evaluation runner for the SF March 2026 baseline:

```bash
python scripts/test_get_gbif.py
```

Legacy wrapper (same exploratory run):

```bash
python scripts/pull_gbif_data.py
```

## Cache Design

Cache files live in `data/species_dist/gbif/`:

- `occurrences_cache.jsonl`: append-only raw occurrence payloads
- `occurrence_id_index.json`: unique occurrence ID index used for fast lookup
- `species_vernacular_index.json`: per-species vernacular name lookup cache

Lookup key precedence:

1. `occurrenceID`
2. fallback to `gbifID` (or `key` if needed)

Runtime cache flow:

1. load `occurrence_id_index.json` into an in-memory `set`
2. check each fetched record key against the set
3. append only unseen records to `occurrences_cache.jsonl`
4. write the updated sorted ID index back to disk

For common names:

1. use occurrence-level `vernacularName` if present
2. fallback to GBIF species endpoint using `speciesKey`:
   - `GET /v1/species/{speciesKey}/vernacularNames`
3. cache selected vernacular names in `species_vernacular_index.json`

Expected behavior on rerun for same scope: `New records appended: 0`.

## Output Artifacts

Script outputs in `data/species_dist/gbif/`:

- `occurrences_normalized_sf_2026_03.csv`
- `class_counts_sf_2026_03.csv`
- `sf_2026_03_occurrence_map.png`

Normalized CSV includes taxonomy/time/location and feature fields used in this milestone:

- taxonomy: `kingdom`, `class`, `order`, `family`, `genus`, `species`
- name fields: `scientificName`, `vernacularName`
- media/description: `imageUrl`, `description`
- occurrence metadata: `basisOfRecord`, `occurrenceStatus`
- location/time: `decimalLatitude`, `decimalLongitude`, `year`, `month`, `day`, `eventDate`

## Rate Limit and Failure Notes

GBIF usage/rate limits are not fixed in this repo and may vary by request volume.
The script includes retry/backoff for transient failures:

- retries on request exceptions, `429`, and `5xx` responses
- exponential backoff (capped) between retries
- same retry strategy is used for species vernacular-name lookups

For larger pulls in future milestones, validate behavior with longer runs and monitor API responses.
