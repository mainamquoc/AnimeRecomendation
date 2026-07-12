# Member 1 → Member 2 handoff

| Artifact | Consumer and restriction |
|---|---|
| `data/processed/clean_ratings.parquet` | Sole explicit ALS input. Columns: long `user_id`, long `anime_id`, double `rating`; unique pair, `[1,10]`. |
| `data/processed/clean_anime.parquet` | Catalogue and fallback metadata. Never use `community_rating` as an ALS feature or to evaluate predictions. |
| `data/processed/watched_unrated.parquet` | Exclude watched items at serving time. Never treat as label/negative. |
| `data/processed/genre_features.parquet` | Canonical normalized genre tokens for content fallback. Fit any vectorizer under the agreed train/candidate protocol. |
| `data/processed/data_quality_summary.csv` | Reconciliation and before/after audit evidence. |
| `data/processed/run_manifest.json` | Exact runtime, input fingerprints, schemas, row counts, and quality status. |

Group constants: seed `42`, relevance threshold `rating >= 8.0`, required `K={5,10}`.

After splitting only `clean_ratings`, call `src.split_validation.validate_split(train, validation, test)`. Evaluate RMSE/MAE only on warm rows whose user and item occur in train; every excluded row must be recorded with `user_unseen`, `item_unseen`, or `both`. Do not rely on Spark's silent `coldStartStrategy='drop'` as reporting.

For ranking, hold out positives only from users retaining positive train history. Candidate items must come from `train_items`; exclude the user's train interactions, and also `watched_unrated` for serving demonstrations. Build popularity only from train, using `build_train_only_popularity`; never from full clean data.

The output counts and schemas are populated in `run_manifest.json` after the verified full-data run. Split-specific artifacts remain pending until Member 2 supplies real splits.
