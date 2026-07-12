# Cleaning decisions and reproducibility

## Fixed decisions

- Explicit ALS label is the user rating in `[1,10]`.
- `rating=-1` means watched but not rated. It is deduplicated and retained separately, never converted to zero.
- Invalid/null/non-positive IDs and ratings outside `{-1,1..10}` are rejected and audited.
- Exact and conflicting explicit duplicates are reduced to one pair using the unrounded arithmetic mean. No timestamp exists, so no “latest” value can be inferred.
- Explicit and watched-unrated pairs whose `anime_id` is absent from the normalized catalogue are rejected; fake metadata is never created.
- Duplicate catalogue IDs are resolved deterministically: the row with the most populated normalized fields wins, then lexical serialization breaks ties.
- Missing genre/type display values become `Unknown`; `Unknown` is excluded from genre feature tokens.
- Episodes must be positive integer text. Unknown, decimal, zero, negative, or malformed values become null with `episodes_missing=1`.
- Members must be non-negative integer text; invalid values become null with `members_missing=1`.
- Community rating outside `[1,10]` becomes null. It is retained only for catalogue/EDA and is prohibited from ALS training/evaluation.
- Catalogue rows without explicit interactions are retained for serving/fallback and identified by `has_explicit_interaction=false`.
- Seed: `42`. Positive relevance threshold: `8.0`. Ranking K values: `5`, `10`.
- No sampling is used in canonical cleaning. EDA transfers only bounded aggregates to pandas.

## Raw dataset fingerprint (observed 2026-07-12)

| File | Rows excluding header | Bytes | SHA-256 |
|---|---:|---:|---|
| `database/rating.csv` | 7,813,737 | 111,404,899 | `F2CF790539CFF9C7AC462D546381493A5748FC22BEDAAF3FBF32D2E10294AB62` |
| `database/anime.csv` | 12,294 | 936,463 | `26C27B66120342544F19C9ACB575E3DCBF37D46D1898AE57BE3167EB0284A845` |

The generated `run_manifest.json` records file size, modified time, SHA-256, runtime versions, output schemas, row counts, and validation result for every run.

## Observed full-data results

- Raw interactions: 7,813,737; watched-unrated raw rows: 1,476,496; explicit-valid raw rows: 6,337,241.
- Explicit duplicates: 7 affected pairs, including 6 conflicting-rating pairs and 1 exact duplicate row. Aggregation removed 7 physical rows.
- Orphan catalogue IDs: `20261`, `30913`, `30924`. They account for 2 deduplicated explicit pairs and 8 watched-unrated pairs; all were excluded from canonical interaction tables.
- Canonical outputs: 6,337,232 explicit pairs, 1,476,488 watched-unrated pairs, and 12,294 catalogue/genre-feature rows.
- The plan's approximate 6.34M explicit rows and three orphan item IDs therefore match this dataset version. The exact clean count is lower than explicit-valid raw rows by 7 aggregation removals and 2 orphan explicit pairs.

## Pending Member 2

`split_quality_summary.csv`, `cold_start_excluded.parquet`, and `popularity_baseline.parquet` are intentionally not created before a real train/validation/test split exists. Member 1 provides and tests `validate_split`, `classify_cold_start`, and train-only popularity helpers. Status: **pending Member 2 split**.
