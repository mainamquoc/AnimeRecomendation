# Implementation plan — Member 1: Data & Analysis Lead

**Project:** Anime Recommendation System — Group 14  
**In charge:** Truong Nhat Truong (Member 1)  
**Delivery goal:** turn the two raw files in `database/` into a quality controlled dataset that can be used directly to train/evaluate Spark ALS (explicit feedback), with metadata/features for popularity baseline and content-based fallback.

---

## 1. Data scope and principles

| Category | Member 1 does | Output for Member 2 |
|---|---|---|
| Data loading | Read CSV with clear schema, check size and data type | Raw profile/audit table |
| Cleaning | Inappropriate sentinel type, handle duplicate/orphan/null | `clean_ratings`, `clean_anime` |
| EDA | Quality statistics, sparsity, long tail, genre/type | 4–6 charts + insights |
| Data preparation | Prepare interaction for ALS, candidate category, features genre | Parquet + feature tables + data dictionary |
| References | Summary of APA 7 documents for datasets/methods | `references_member1.md` or the References section in report |

### Mandatory principles

1. **The main model is ALS explicit feedback.** Label is just the user's `rating` in the range **1–10**.
2. `rating = -1` means *watched but not rated yet*, **not rating 0**. Do not include this line in train, RMSE, MAE, Precision@K or Recall@K of ALS explicit.
3. Do not use the `anime.rating` column (average community score) as a feature to predict/evaluate user ratings: this is aggregate target-like information and has the risk of leakage. This column is only described in the EDA if needed.
4. There is no timestamp, so you cannot infer the "latest rating". If a pair `(user_id, anime_id)` has many different explicit ratings, use **mean rating by pair**.
5. Every transform step must save the number of lines before/after, the reason for removal and the seed (if any) to reproduce the result.

---

## 2. Inventory existing data

| Source file | Grain | Role |
|---|---|---|
| `database/rating.csv` | An interaction `(user_id, anime_id)` | Primary source for ALS and ranking evaluation |
| `database/anime.csv` | An anime by `anime_id` | Anime name, genre, type, episodes, members; use catalog, EDA and cold-start fallback |

The points that need to be addressed are known from the project profile:

- `rating.csv`: 7,813,737 interactions; There are about 18.9% of `rating = -1` lines.
- After removing `-1`, there are about 6.34M explicit ratings on about 69.6K users and 9.9K anime; The data is very sparse and long-tail.
- There is duplicate `(user_id, anime_id)` and there are 3 `anime_id` in ratings that do not exist in the catalog.
- Metadata has missing values ​​in `genre`, `type`, `rating`; `episodes` has a text value like `Unknown` so it cannot be cast directly.

---

## 3. Recommended output structure

```text
data/
├── processed/
│   ├── clean_ratings.parquet              # explicit interaction data for ALS
│   ├── clean_anime.parquet                # normalized catalogue
│   ├── watched_unrated.parquet            # rating=-1, stored separately; not used to train ALS
│   ├── genre_features.parquet             # anime_id, genre tokens / multi-hot representation
│   ├── data_quality_summary.csv
│   ├── data_dictionary.md
│   └── cleaning_decisions.md
│   # Created during handoff, only after Member 2 has split the data:
│   ├── popularity_baseline.parquet        # Member 2 creates it from train only
│   ├── split_quality_summary.csv           # Member 1 runs the validator on the split
│   └── cold_start_excluded.parquet         # evaluation row that cannot be scored + reason
notebooks/
├── 01_data_preparation.ipynb  # understand raw data → define data contract → clean → validate → export
└── 02_eda_clean_data.ipynb    # reads clean data only, creates EDA charts, and derives insights
outputs/
└── figures/
    ├── rating_distribution.png
    ├── interactions_per_user.png
    ├── interactions_per_anime.png
    ├── genres_and_types.png
    └── data_quality_before_after.png
```
Prioritize **Parquet** for processed tables because Spark reads more efficiently than CSV. Do not commit Parquet files that are too large; add this path to `.gitignore` and provide command to regenerate from raw CSV.

---

## 4. Detailed execution pipeline

### Step 1 — Set up schema and load data

- Read `rating.csv` with schema: `user_id: int`, `anime_id: int`, `rating: float`.
- Read `anime.csv` with schema: `anime_id: int`, `name: string`, `genre: string`, `type: string`, `episodes: string`, `rating: float`, `members: int`.
- Set fail-fast check: key ID is not null; `user_id > 0`, `anime_id > 0`; raw rating only belongs to `{-1, 1..10}`.
- Save the initial audit: row count, distinct users/items, schema, missing values, min/max rating and number of records for each rating value.

**Acceptance:** notebook can run from raw CSV; `data_quality_summary.csv` has a `raw` row.

### Step 2 — Separating watched-unrated and explicit ratings

- Create `watched_unrated` from `rating == -1`, store separately with columns `user_id`, `anime_id`.
- Create `explicit_ratings_raw` from `rating BETWEEN 1 AND 10`.
- Eliminate or flag all ratings outside of the above two groups (if present), and record them in audit.

**Rules of use:**

- `watched_unrated`: only use if you need to filter watched anime later when serving; **not** used as a negative label.
- `explicit_ratings_raw`: unique input for ALS rating/ranking metrics.

### Step 3 — Handling duplicate interactions

- Count exact duplicate rows and the number of `(user_id, anime_id)` pairs that have more than one explicit rating.
- Delete identical duplicates.
- With conflicting ratings in the same pair, group by `(user_id, anime_id)` and get `avg(rating)`; output must ensure that each pair has only one line left.
- Store the number of merged pairs and describe the decision: there is no timestamp so mean is a neutral, reproducible choice.

### Step 4 — Synchronize interaction with catalog

- Check anti-join from explicit ratings to `anime.csv` by `anime_id`.
- Remove 3 orphan items from `clean_ratings` so that the training catalog and recommendation catalog are consistent.
- Save the number of lines/pairs eliminated in audit. Do not manually create fake metadata for orphan items.

**ALS-ready result:** `clean_ratings` includes only `user_id`, `anime_id`, `rating`; rating 1–10; unique `(user_id, anime_id)`; any `anime_id` exists in `clean_anime`.

> `clean_ratings` is a **standard model-input**, not just a “clean” table. Each column must be non-null and have a compatible Spark type: `user_id`/`anime_id` is `IntegerType` or `LongType`, `rating` is `FloatType` or `DoubleType`. ALS does not need a one-hot genre, standardization rating or normalization ID; genre/type is just data for fallback.

### Step 5 — Clean the anime catalog

- Keep `anime_id` unique and verify `name` is not empty for items still in interaction.
- Standardize text: trim whitespace; replace null/empty with `Unknown` for `genre` and `type`.
- Convert `episodes` to numeric (`episodes_num`); Unparsable values ​​(`Unknown`, `N/A`, …) become `null`, also creating `episodes_missing` (0/1).
- Normalize `members` to non-negative numeric; create `log_members = log1p(members)` when doing fallback/baseline.
- Keep `community_rating` (renamed from `anime.rating`) for catalog/EDA description only; attach warning **do not use in training or ranking evaluation**.
- Keep/not keep anime without explicit rating depending on the serving catalog, but clearly separate `has_explicit_interaction` so as not to mistake them as trainable items.

### Step 6 — Prepare features for content-based fallback

- Separate `genre` by comma, trim, standardize token names and empty token types.
- Create one of two reproducible representations:
  - `genre_tokens`: `anime_id`, `genres: array<string>` — favor Spark `CountVectorizer`; or
  - multi-hot genre matrix if done using pandas/scikit-learn.
- Prepare categorical `type` for one-hot in fallback pipeline when needed.
- For cold-start users: take the anime users have scored `>= 8`, find anime with high cosine genre similarity, then filter all anime users have interacted with (explicit **and**, if serving reality, watched-unrated).
- With cold-start items or users without positive history: fallback to popularity list. Do not mix content and ALS scores in the primary metric unless the team specifically defines a hybrid protocol.

### 4.1. Execution contract for Codex to deploy without needing to speculate

#### Environment and configuration

- The default implementation uses the **PySpark DataFrame API** for loading, cleaning, auditing and exporting. Only transfer small aggregate data to pandas/seaborn in EDA notebooks; Do not call `toPandas()` on the full interaction table.
- Python, Java and PySpark versions must be stated in the project's README/requirements. If the repository is not yet versioned, Codex must check the existing environment first, select a compatible set of versions, and record the selection; Don't silently change the framework to pandas.
- Do not hard-code paths and parameters throughout the notebook. Create a single cell/config (or `src/config.py`) with at least:

```python
RAW_RATINGS_PATH = "database/rating.csv"
RAW_ANIME_PATH = "database/anime.csv"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "outputs/figures"
SEED = 42
POSITIVE_THRESHOLD = 8.0
TOP_K_VALUES = [5, 10]
UNKNOWN_TOKEN = "Unknown"
```
- Notebook must run from **repository root**. Before processing, check two raw files exist; create output directory if missing; The error must clearly state which path is missing.
- Spark session must have a clear app name, fixed timezone (recommended `UTC`) and appropriate log level. Writing Parquet uses `mode("overwrite")` only with the correct subpaths defined in `PROCESSED_DIR`; Do not delete the parent folder as well.

#### Modules/functions should be separated so that the notebook can only be orchestrated

If the project allows creating source code, prioritize the following structure so that the logic can be unit tested and reused:

```text
src/
├── config.py
├── schemas.py          # StructTypes for the two CSV files and output schemas
├── data_io.py          # load_raw_data(), write_parquet_safe()
├── cleaning.py         # clean_ratings(), clean_anime(), build_genre_features()
├── audit.py            # profile_stage(), validate_*(), append_audit_row()
└── split_validation.py # validate_split(), classify_cold_start()
tests/
├── test_cleaning.py
├── test_data_contracts.py
└── test_split_validation.py
```
Minimum signature/behavior:

```python
load_raw_data(spark, ratings_path, anime_path) -> tuple[DataFrame, DataFrame]
clean_ratings(raw_ratings, clean_anime_ids) -> tuple[DataFrame, DataFrame, DataFrame]
# returns: clean_ratings, watched_unrated, rejected_or_orphan_rows

clean_anime(raw_anime, explicit_item_ids) -> DataFrame
build_genre_features(clean_anime) -> DataFrame
profile_stage(stage_name, ratings_df, anime_df=None) -> DataFrame
validate_clean_ratings(ratings_df, anime_df) -> dict[str, bool | int | float]
validate_split(train_df, validation_df, test_df) -> tuple[DataFrame, DataFrame]
# returns: split_quality_summary, cold_start_excluded
```
Codex can rename functions according to the existing convention of the repository, but must keep the same input/output contract and not embed all business logic into notebook cells.

#### Required dependency order

```text
raw CSV
  → schema validation + raw audit
  → clean catalogue keys/text
  → split watched-unrated / explicit / invalid-rating
  → deduplicate explicit pairs
  → anti-join orphan items
  → clean_ratings + clean_anime
  → genre_features + quality validation
  → atomic export + read-back verification
  → EDA notebook (read-only consumers)
  → Member 2 split
  → split validation + train-only popularity baseline
```
Note: a `clean_anime` set/valid key list is required before orphaning, but the `has_explicit_interaction` flag is only counted after `clean_ratings` is complete. Avoid circular dependencies by cleaning the catalog in two phases: standardize key/text first, add interaction statistics flags later.

### 4.2. Transform rules at the code level

#### Ratings/interactions

1. Parse CSV according to the declared schema. A malformed or unparsable record cannot automatically become null and then disappear; must be counted in the `invalid_schema`/`invalid_id`/`invalid_rating` group.
2. Classify records using mutual exclusion conditions:
   - `watched_unrated`: `rating == -1`;
   - `explicit_valid`: `1 <= rating <= 10`;
   - `rejected_invalid_rating`: all remaining cases, including null/NaN if any.
3. For `watched_unrated`, apply `dropDuplicates(["user_id", "anime_id"])` and anti-join catalog like explicit data. Specify the exact duplicate and orphan numbers separately; Do not let the same pair repeat in the serving filter file.
4. With explicit data, calculate audit metrics **before aggregation**: number of exact duplicate rows, number of duplicate pairs, number of pairs with `countDistinct(rating) > 1`. Then group by pair and get `avg(rating)` cast to `double`. No round mean; range must still be within `[1, 10]`.
5. Orphan check using `left_anti` join on the catalog's distinct `anime_id` list. Save rejected table with `reason` if capacity is small; If not saving details then at least save count and orphan list `anime_id` in `cleaning_decisions.md`.
6. Sort is not part of the data contract. Do not write tests that depend on row order in Parquet; test with key/count/set or clear order when displayed.

#### Catalog

1. `anime_id`: positive, non-null, unique. If ID matches:
   - exact duplicate: keep one row;
   - inconsistent metadata: no random selection; Prioritize the row with the most non-null fields, tie-break with deterministic rules (eg lexical), and record count and policy in audit.
2. `name`: trim; empty string to null. Anime that has explicit interaction but lacks a name is a quality gate error; anime without interaction can use `Unknown` if the group wants to keep in the serving catalog and must record the decision.
3. `genre`: holds a normalized text column for display and a `genres` array for features. Tokens are trimmed, empty, whitespace/case normalized deterministically, `array_distinct`, then sorted for stable output. `Unknown` cannot be a genre signal when calculating cosine similarity; represented by an empty array or this type of token when vectorize.
4. `type`: trim, empty/null to `Unknown`; Do not combine different types without approved mapping.
5. `episodes_num`: parse positive integer; `Unknown`, `N/A`, empty, zero or negative to null and `episodes_missing = 1`. If there are abnormal decimals, reject/flag instead of truncate silently.
6. `members`: parse non-negative integer/long. Invalid to null and has the `members_missing` flag; Only count `log_members` when members are valid. Do not use `members` or `log_members` in the main ALS metric.
7. `community_rating`: numeric in `[1, 10]` or null; Values outside the range become null/flag. This column is not exported to `clean_ratings` or the main model's feature vector.

### 4.3. Fixed output schema

| Output | Recommended Schema/nullable | Partition/order | Consumer |
|---|---|---|---|
| `clean_ratings.parquet` | `user_id: long not null`, `anime_id: long not null`, `rating: double not null` | no partition required; unique pair | Member 2's ALS |
| `watched_unrated.parquet` | `user_id: long not null`, `anime_id: long not null` | unique pair | filter serving |
| `clean_anime.parquet` | `anime_id: long not null`, `name: string not null`, `genre: string`, `genres: array<string>`, `type: string not null`, `episodes_num: int`, `episodes_missing: int not null`, `members: long`, `members_missing: int not null`, `log_members: double`, `community_rating: double`, `has_explicit_interaction: boolean not null` | unique `anime_id` | catalog/EDA/fallback |
| `genre_features.parquet` | `anime_id: long not null`, `genres: array<string> not null` | unique `anime_id`; does not contain empty tokens/Unknown | content fallback |
| `data_quality_summary.csv` | See schema audit below | sort by `stage`, `metric` when export | report + reproducibility |

Do not save Spark ML `VectorUDT` as the only contract of `genre_features`, because the vocabulary/index can be difficult to read and depends on the fit step. Keep `genres: array<string>` as canonical source; Member 2 fit `CountVectorizer` on train/candidate accordingly and save vocabulary/model if vector is needed.

### 4.4. Audit contract and reconciliation

`data_quality_summary.csv` should be long for easy addition of metrics:

| Column | Meaning |
|---|---|
| `run_id` | The deterministic ID or timestamp of the run; the same run uses the same ID |
| `stage` | `raw`, `parsed`, `explicit_before_dedup`, `explicit_after_dedup`, `clean`, `export_readback` |
| `entity` | `ratings`, `watched_unrated`, `anime`, `genre_features` |
| `metric` | example `row_count`, `distinct_users`, `null_rating`, `duplicate_pairs`, `orphan_rows` |
| `value` | numeric value |
| `unit` | `rows`, `pairs`, `users`, `items`, `percent` |
| `rule_or_note` | condition/policy create metric |

Reconciliation equations must pass and be asserted in the code:

```text
raw_rating_rows
= watched_unrated_raw_rows
 + explicit_valid_raw_rows
 + rejected_invalid_id_or_rating_rows

explicit_valid_raw_rows
= rows_removed_as_duplicate_or_aggregated
 + explicit_after_dedup_rows

explicit_after_dedup_rows
= orphan_explicit_rows
 + clean_ratings_rows
```
Because multiple rows are aggregated into a pair, the variable `rows_removed_as_duplicate_or_aggregated` must be calculated using the row count difference, not naively adding exact-duplicate count to conflicting-pair count. Similarly, audit must distinguish between `row_count` and `distinct_pair_count`.

After exporting, read each Parquet again and compare the schema, row count, distinct key count with the DataFrame before export. Only mark the pipeline as successful after the read-back validation pass.

### 4.5. Test requirements and Definition of Done for code

#### Unit tests using small synthetic fixture

Fixture must contain at least: a rating `-1`, marginal rating 1 and 10, rating invalid/null, exact duplicate, conflicting duplicate, orphan anime, null genre/type, `Unknown` episodes, negative members, duplicate anime metadata and genre with duplicate whitespace/token.

Required tests:

- classifier `-1` does not enter explicit and does not turn into 0;
- duplicate conflicts are aggregated with the correct meaning, each pair has one row;
- invalid IDs/ratings and orphans are counted/eliminated for the correct reason;
- catalog normalization for deterministic results;
- genre tokens are not empty, not duplicated and do not contain `Unknown` like signal;
- Validator fails when rating is out of range, null key, duplicate pair or orphan still exists;
- split validator correctly classifies `user_unseen`, `item_unseen`, `both`;
- read/write round-trip holds schema and count.

#### Integration/smoke test on raw data

- Run end-to-end pipeline on raw CSV (or limited sample for smoke test only), generating enough output and no quality gate failures.
- Compare the known numbers in Section 2 with reasonable tolerance; If it's wrong, clearly state whether it's due to the dataset version or the transform. Don't edit the assertion to force a pass.
- Notebook restart-and-run-all successfully from clean kernel. Does not depend on variables created manually in the previous cell in the wrong order.
- Write the command to recreate and test in the README, for example `pytest -q` and the command to run the actual notebook/script of the repository. Codex must use the exact runner that already exists in the repo; If not, create a clear entry point.

#### Definition of Done

A step is only considered complete when it simultaneously has: code transform, pre/post audit, assertion quality, correct output schema, corresponding test and decision description in the document. Do not consider it enough that the notebook "runs without errors" if reconciliation or read-back has not passed.

### 4.6. Error handling, logging and reproducibility

- Fail pipeline with understandable errors such as missing files, wrong schema header, contract violation ID/rating beyond the defined rejected group, duplicate catalog cannot be deterministically resolved, or read-back output does not match.
- Warning but still run for missing optional metadata (`genre`, `type`, `episodes`, `members`, `community_rating`) if the above standardization policy can handle it; count warning must go into audit.
- Do not log all records/user data. Show only small samples, aggregates, and up to a list of error IDs that need investigation.
- Every sampling must have a seed and record sample fraction/limit. Every chart aggregate must have a rebuild command from clean data.
- Record a minimum dataset fingerprint including file size, modified time and (if cost is acceptable) SHA-256 of raw files in `cleaning_decisions.md` or a manifest to detect raw-data changes.

### 4.7. Responsibility boundaries to avoid creating artifacts at the wrong time

- Member 1 creates four canonical tables: `clean_ratings`, `clean_anime`, `watched_unrated`, `genre_features`, and audit/docs/figures.
- `popularity_baseline.parquet` **not generated from full clean data**. Member 2 creates it after split only from train; Member 1 only provides helper/contract and review leakage. Therefore, this file is not included in the initial "four tables" handed over.
- `split_quality_summary.csv` and `cold_start_excluded` only have real data after Member 2 provides train/validation/test. Member 1 prepares the function/schema/test first, then runs validation in the handoff step; Do not create fake files or placeholder data.
- If Codex assigns task Member 1 separately but split does not exist, the valid deliverable is helper + test + template schema, and record the status `pending Member 2 split` in the handoff note.

---

## 5. Prepare data for model train/evaluate properly

### Data contract handed over to Member 2

| Table | Required column | Conditions |
|---|---|---|
| `clean_ratings` | `user_id`, `anime_id`, `rating` | rating 1–10, one line/pair, item in catalog |
| `clean_anime` | `anime_id`, `name`, `genre`, `type`, `episodes_num`, `members` | one line/anime; null normalized according to data dictionary |
| `genre_features` | `anime_id`, `genres` or vector genre | only serves content fallback |
| `watched_unrated` | `user_id`, `anime_id` | Do not use as label/metric explicit |

### Required conditions of the data set before training ALS

These are the conditions Member 1 must check and record in the handoff note. If a condition is not met, do not call the model "correctly trained/evaluated".

| Group | Dataset must meet | How to handle if not achieved |
|---|---|---|
| Grain | A single line for a pair `(user_id, anime_id)` | Delete exact duplicate, aggregate rating conflicts with mean |
| Label | `rating` numeric, non-null, only in `[1, 10]` | Remove `-1` and any invalid values ​​from explicit input; no impute rating |
| IDs | `user_id`, `anime_id` numeric, positive, non-null | Type/flag record invalid; don't reencode ID if not needed |
| Referential integrity | Every item in the interaction has a name/candidate in catalog | Type orphan interaction or separate it from model-ready table |
| Train support | Each user and item appears in **train** at least once; Recommended at least 2 interactions/user to maintain test ranking | Include insufficiently supported interactions in `cold_start_excluded` and report count/rate; Don't let Spark drop without recording |
| Candidate set | Recommended items must belong to the item catalog and have a factor in train | Generate candidates from `train_items`, not from the entire raw/validation/test |
| Leakage | Do not use `community_rating`, popularity statistics or genre vectors fit from validation/test to score/evaluate ALS | Fit every statistic/vectorizer on train only; static metadata is only used for fallback, separate from ALS metrics |

**Note on support filtering:** Do not aggressively filter all low-frequency users/items directly from `clean_ratings`, because it would bias the data toward active users/popular anime. Keep the full `clean_ratings` after cleaning, then create a `cold_start_excluded` report and repair the split to evaluate the scoreable subset. If the group wants to filter (for example, users with fewer than 2 explicit ratings), it must clearly report the number of records/users/items removed and apply the filter consistently for the ranking protocol.

### Checklist before train and handover table after split

Member 1 prepares test functions or notebook cells for Member 2 to run again after creating the split:

```text
clean_ratings
  ├── ratings_als_input                 # user_id, anime_id, rating; passes all model-input rules
  ├── train / validation / test          # Member 2 creates these with a fixed seed
  ├── eval_warm_validation / eval_warm_test
  │     # only rows whose user_id and anime_id both exist in train
  └── cold_start_excluded
        # test/validation interactions ALS cannot score; reason=user_unseen/item_unseen/both
```
Metrics that must appear in `split_quality_summary.csv`: number of rows, distinct users/items, rating mean/std, duplicate count, number of users/items with only one interaction, number of warm/cold rows for each split, and coverage `warm_rows / total_eval_rows`. This file is evidence to explain why the number of records for calculating RMSE/MAE may be smaller than the number of original test records.

### Requires split to avoid incorrect metrics

Member 1 prepares/tests the split helper or reviews Member 2's code according to the following rules:

1. Only split **after cleaning**, on `clean_ratings`.
2. Use a fixed seed for the entire group (recommended `42`) and save the seed in config/README.
3. With RMSE/MAE: random 80/10/10 train/validation/test is acceptable, but validation/test only counts pairs where **both user and item have appeared in train**. Cold-start lines must be reported separately as coverage/excluded count, not covered by Spark `coldStartStrategy='drop'`.
4. Before random split, prioritize a **repair split**: if a user/item only falls into validation/test and there are no interactions left in the train, transfer at least one interaction of that user/item to the train. If not repaired, this interaction must be removed from the metric and recorded in `cold_start_excluded`. Do not drop silently.
5. With Top-N: split according to users (for example, keep 20% positive or leave-one-out), but only hold out positives of users with enough history to still have at least one interaction train. Fixed relevant definition: `rating >= 8`; report minimum P@10 and R@10, additional recommended @5.
6. Recommendation candidate must be `train_items` and must exclude anime users who interacted in the train. When demoing/serving, you should also remove `watched_unrated` to avoid suggesting anime you've already watched.
7. Popularity baseline must be calculated **only from the train split**, not from the entire dataset/test; Use count explicit ratings and mean ratings with minimum-count/shrinkage to avoid prioritizing items with too few ratings.

---

## 6. EDA needs to be done and insights need to be written

All charts have titles, axes, units, and short captions; Use a seed if sampling is available. With large `rating.csv`, you can aggregate on Spark first and then transfer the small table to pandas/seaborn for drawing.

| Chart/table | How to calculate | Meaning needs to be stated |
|---|---|---|
| Distribution of ratings | Count/percent by `-1`, 1…10 | Prove why `-1` must be separated from the explicit label |
| Quality before/after cleaning | Rows, users, items, duplicate pairs, orphan rows, missing metadata | Transformation transparency and reproducibility |
| Interactions per user (log scale) | Number explicit rating/user, histogram/percentiles | Indicates user long-tail and cold-start |
| Interactions per anime (log scale) | Number of explicit ratings/anime, histogram/percentiles | Indicates popularity bias/item long-tail |
| Genre/type distribution | Anime count and/or explicit interaction count by genre/type | Description of catalog, motive for using genre fallback |
| Top anime | Top by explicit rating count and top by members | Distinguishing popularity in interaction with community metadata |

Minimum insight included in report/PPT:

- The problem is a sparse user–item matrix, so ALS matrix factorization is more suitable than a dense/simple lookup model.
- Sentinel `-1` means missing explicit feedback, not dislike.
- Long-tail alone as RMSE is not enough; Need Precision@K/Recall@K and coverage.
- Genre is useful for fallback/cold-start, but does not replace the fairness of the ALS test protocol.

---

## 7. Quality gates before handover

- [ ] `clean_ratings.rating` is in `[1, 10]`; no more `-1`.
- [ ] The three ALS columns (`user_id`, `anime_id`, `rating`) are all non-null, correct numeric type and positive ID.
- [ ] `clean_ratings` no longer duplicates `(user_id, anime_id)`.
- [ ] All `clean_ratings.anime_id` can be joined with `clean_anime.anime_id`.
- [ ] Schema, previous/next line numbers, duplicate/orphan/null numbers are all in `data_quality_summary.csv`.
- [ ] `clean_anime` has 1 line/anime; genre/type already has a valid value (`Unknown` when missing); `episodes_num` true numeric/null.
- [ ] `genre_features` joins 1:1 with catalog and has no empty tokens.
- [ ] No feature/model input using `community_rating` or test/full-dataset popularity causes leakage.
- [ ] After split, `split_quality_summary.csv` has warm/cold coverage and reason excluded; Every row that is calculated RMSE/MAE has user/item in the train.
- [ ] Ranking's candidate set contains only `train_items`; Top-N holdout does not remove all positive history of a user from the train.
- [ ] Test a sample user: get top candidates, confirm interaction train type filter (and watched-unrated when serving).
- [ ] Notebook successfully rerun from raw CSV and generates the same schema/output.

---

## 8. Member Deliverables 1

1. `notebooks/01_data_preparation.ipynb`: main processing notebook runs sequentially from load → explore raw data and column meanings → define data contract for ALS/Top-N → audit → cleaning → check quality gate → create feature → export Parquet. Do not perform EDA/report charting in this notebook other than the necessary checklists for making cleaning decisions.
2. `notebooks/02_eda_clean_data.ipynb`: only read cleaned Parquet tables; Create 4–6 EDA charts, captions and insights to include in the report/PPT. This notebook does not change processed data.
3. `data/processed/` (or link/save outside Git): four canonical tables in Section 3 (`clean_ratings`, `clean_anime`, `watched_unrated`, `genre_features`).
4. `data_quality_summary.csv`, `data_dictionary.md`, `cleaning_decisions.md`; add `split_quality_summary.csv` and `cold_start_excluded.parquet` after receiving the actual split from Member 2.
5. 4–6 EDA images in `outputs/figures/` for Member 2 to include in PPT/report.
6. `references_member1.md`: APA 7 document about datasets, Spark MLlib/ALS and recommender evaluation.
7. Handoff note for Member 2: table path, schema, last record number, seed, duplicate policy, relevance threshold, and data restrictions.

---

## 9. Suggested time plan

| Session | Work | Completion criteria |
|---|---|---|
| 1 | Load, schema, raw audit, confirm rules | There are raw quality profiles and cleaning decisions recorded |
| 2 | Separate `-1`, deduplicate, handle orphans, clean metadata | Export Parquet and all quality gates pass data |
| 3 | Genre/type features, EDA, captions/insights | There are 4–6 figures ready to be included in the report/PPT |
| 4 | Handoff + review split/leakage with Member 2 | Member 2 trains ALS from `clean_ratings`; Reproducible README |

---

## 10. Initial APA 7 references

- CooperUnion. (n.d.). *Anime recommendations database* [Data set]. Kaggle. https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer, 42*(8), 30–37. https://doi.org/10.1109/MC.2009.263
- Meng, X., Bradley, J., Yavuz, B., Sparks, E., Venkataraman, S., Liu, D., Freeman, J., Tsai, D. B., Amde, M., Owen, S., Xin, D., Xin, R., Franklin, M. J., Zadeh, R., Zaharia, M., & Talwalkar, A. (2016). MLLib: Machine learning in Apache Spark. *Journal of Machine Learning Research, 17*(34), 1–7. http://jmlr.org/papers/v17/15-237.html
- Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating collaborative filtering recommender systems. *ACM Transactions on Information Systems, 22*(1), 5–53. https://doi.org/10.1145/963770.963772

> Note: Member 1 standardizes APA in alphabetical order in the final report/PPT; Check the URL and italic format when laying out the LaTeX page.

---

## 11. Implementation Checklist for Codex

When asked to code according to this plan, Codex executes it in the following order and does not skip the verification step:

1. Read `README`, `AGENTS.md` (if available), dependency file and entire plan; Check the repository structure and status of existing files before editing.
2. Profile header/schema/sample of the two raw CSVs and compare with Section 2; Don't load the entire thing with pandas.
3. Finalize entry point, config, schema and output contract; specify any new assumptions in `cleaning_decisions.md`.
4. Write synthetic tests before or in parallel with each important transform, especially sentinel, duplicate, orphan and metadata parse.
5. Implement pipeline according to dependency in Section 4.1; The notebook only coordinates and presents audit results.
6. Run unit tests, smoke/integration tests, quality assertions and reconciliation.
7. Export Parquet/CSV/docs, read the output again to verify schema/count/key; Don't just check the file exists.
8. Restart-and-run-all two notebooks; Confirm that the EDA notebook only reads processed data and does not mutate output canonical.
9. Compare final data with known baseline, explain any deviations; check `git diff` to not overwrite unrelated changes.
10. Handoff with file/path list, recreated command, run tests, last row/schema, seed, cleaning decision, known limitations and pending Member 2.

### Codex criteria to stop and ask instead of deciding on your own

- Raw CSV/header is significantly different from the contract or has a different dataset version.
- There is a duplicate `anime_id` with conflicting metadata which the deterministic rule suggests loses important information.
- The Repository already has a pipeline/split protocol that is contrary to the plan and the change may affect the work of other members.
- Need to download new dependencies/datasets, change large environment versions, or write/commit large artifacts.
- A quality gate failure on real data that cannot be explained by the rejected group allowed by the plan.

In addition to the above cases, Codex is allowed to choose small implementation details consistent with existing conventions, but must record the choice and prove it by testing/audit.
