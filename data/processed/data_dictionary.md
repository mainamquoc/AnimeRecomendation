# Data dictionary

## `clean_ratings.parquet`

| Column | Spark type | Nullable | Meaning |
|---|---|---:|---|
| `user_id` | long | no | Positive anonymous user identifier. |
| `anime_id` | long | no | Positive catalogue identifier; foreign key to `clean_anime`. |
| `rating` | double | no | Explicit user rating in `[1,10]`; conflicting pair values are averaged without rounding. |

Grain: exactly one row per `(user_id, anime_id)`. This is the only canonical ALS input.

## `watched_unrated.parquet`

| Column | Spark type | Nullable | Meaning |
|---|---|---:|---|
| `user_id` | long | no | User identifier. |
| `anime_id` | long | no | Catalogue identifier. |

Grain: one unique watched pair. Source rating was `-1`; this is a serving-time exclusion list, not an explicit label.

## `clean_anime.parquet`

| Column | Spark type | Nullable | Meaning |
|---|---|---:|---|
| `anime_id` | long | no | Unique catalogue key. |
| `name` | string | no | Trimmed display name; `Unknown` only for non-interacted catalogue rows. |
| `genre` | string | yes | Trimmed display genre or `Unknown`. |
| `genres` | array<string> | no | Sorted, distinct, normalized signal tokens; no `Unknown` token. |
| `type` | string | no | Trimmed source type or `Unknown`. |
| `episodes_num` | integer | yes | Positive integer episode count; otherwise null. |
| `episodes_missing` | integer | no | `1` when episodes cannot be validly parsed. |
| `members` | long | yes | Non-negative source membership count. |
| `members_missing` | integer | no | `1` for invalid/missing members. |
| `log_members` | double | yes | `log1p(members)`; fallback description only. |
| `community_rating` | double | yes | Source aggregate score in `[1,10]`; EDA/catalogue only, never ALS input/evaluation feature. |
| `has_explicit_interaction` | boolean | no | Whether the anime occurs in `clean_ratings`. |

## `genre_features.parquet`

| Column | Spark type | Nullable | Meaning |
|---|---|---:|---|
| `anime_id` | long | no | Unique catalogue key. |
| `genres` | array<string> | no | Canonical input for a train/candidate-fitted `CountVectorizer`; empty array is allowed. |

## `data_quality_summary.csv`

Long-format audit with `run_id`, `stage`, `entity`, `metric`, numeric `value`, `unit`, and `rule_or_note`. Stages include `raw`, `clean`, and `export_readback`.

