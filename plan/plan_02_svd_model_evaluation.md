# Implementation Plan: `02_svd_model_evaluation.ipynb`

## 1. Goal, scope, and language requirement

Notebook 02 will build an anime recommender using **collaborative filtering with Matrix Factorization (SVD)** on explicit ratings. It must produce a Top-10 list of unseen anime for each selected user and report all four required metrics: **RMSE, MAE, Precision@10, and Recall@10**.

This is a modeling and evaluation notebook. It must not repeat the EDA or cleaning work from Notebook 01. New steps such as sampling, minimum-interaction filtering, and train/test splitting create a model-specific dataset only; they must be audited clearly and must not overwrite the cleaned source files.

### Mandatory notebook language

The implementation of `notebooks/02_svd_model_evaluation.ipynb` must be entirely in **English**:

- All Markdown headings, explanations, conclusions, and `tl;dr` text.
- Code comments, function names, variable names, assertion messages, and printed messages.
- Table column names, chart titles, axis labels, legends, and CSV column names.
- Output filenames remain the English names defined in this plan.

Project decisions inherited from the earlier plans:

- Use only explicit ratings on the 1--10 scale. `rating = -1` is not a score and must never be used for training or evaluation.
- The main model is `surprise.SVD`, suitable for the sparse user--item matrix (Notebook 01 reports approximately 0.9172% density).
- Use `SEED = 42`, `K = 10`, and `POSITIVE_RATING = 8` throughout the notebook.
- Do not use Spark, Parquet, large cross-validation/grid searches, or z-score rating normalization.

## 2. Inputs, outputs, and path convention

### Required inputs

| File | Role | Required schema |
|---|---|---|
| `cleaned_data/ratings_clean.csv` | SVD training and evaluation signal | `user_id`, `anime_id`, `rating` |
| `cleaned_data/anime_clean.csv` | Metadata used to display recommendations | `anime_id`, `name`, `genre`, `type`, `episodes`, `anime_average_rating`, `members` |

At the time of planning, both files exist in `cleaned_data/`; `ratings_clean.csv` is about 102 MB and `anime_clean.csv` is about 0.96 MB. In Colab, Notebook 02 first uses files uploaded directly to `/content/` (`/content/ratings_clean.csv` and `/content/anime_clean.csv`) when both are present. Otherwise it falls back explicitly to the project's local `cleaned_data/` directory, checking the current working directory and its parent so the notebook runs from either the project root or `notebooks/`. It reports the selected source and fails clearly if neither location contains both files. It must never read the raw dataset or `data/`.

### Proposed outputs

| Artifact | Contents |
|---|---|
| `outputs/top_10_recommendations.csv` | Ten unseen recommendations for each demonstration user |
| `outputs/model_evaluation_metrics.csv` | Metrics, evaluation scope, and eligible-user counts |
| `outputs/figures/svd_error_metrics.png` | RMSE and MAE chart |
| `outputs/figures/svd_ranking_metrics_at_10.png` | Precision@10 and Recall@10 chart |
| `outputs/figures/svd_actual_vs_predicted.png` | Actual-versus-predicted rating chart from a deterministic test sample |

Create `outputs/` and `outputs/figures/` with `mkdir(parents=True, exist_ok=True)`. Export all CSV files as UTF-8 with `index=False`.

## 3. Model-data design

### 3.1 Sampling and k-core filtering

The complete cleaned dataset contains more than 6.3 million interactions. The default configuration for a personal computer is:

```python
MAX_RATINGS = 1_000_000  # Set to None to use the full dataset when resources allow.
MIN_USER_RATINGS = 5
MIN_ITEM_RATINGS = 5
MAX_CORE_FILTER_PASSES = 10
```

Process:

1. If `MAX_RATINGS` is not `None` and is smaller than the input row count, use `ratings_clean.sample(n=MAX_RATINGS, random_state=SEED)`. Otherwise use `.copy()` and explicitly state that full-data mode is active.
2. Repeatedly remove users with fewer than `MIN_USER_RATINGS` interactions and anime with fewer than `MIN_ITEM_RATINGS` interactions. Stop when the row count no longer changes or `MAX_CORE_FILTER_PASSES` is reached.
3. After sampling, after every filtering pass, and at the end, report interactions, users, anime, density, and minimum/median/P90 interactions per user and item.
4. Assert that the result is non-empty, ratings remain within `[1, 10]`, `(user_id, anime_id)` is unique, and every remaining user/item meets its threshold. Raise an error if the filter does not converge within ten passes.

Sampling and k-core filtering are resource and evaluation decisions. They must never be presented in the report or slides as the complete cleaned dataset.

### 3.2 Split without leakage

- Convert `user_id` and `anime_id` to `str` before passing data to Surprise, ensuring consistent raw IDs during training, prediction, and export.
- Use `Reader(rating_scale=(1, 10))` and `Dataset.load_from_df(ratings_model[["user_id", "anime_id", "rating"]], reader)`.
- Create one **outer 80/20 holdout split** with `surprise.model_selection.train_test_split(..., test_size=0.2, random_state=SEED)`.
- Never use the outer `testset` to select hyperparameters. If a second configuration is tried, compare candidates only on a validation split drawn from the outer training data, then refit the chosen configuration on the complete outer trainset and evaluate once on the outer testset.
- Report trainset users/items, test-prediction count, and the actual test proportion.

## 4. Notebook structure, section by section

Every code cell must have a concise English Markdown explanation immediately above it. No metric values may be hard-coded in Markdown; all values must come from the current execution.

### 4.0 Required explanatory Markdown narrative

In addition to the short Markdown introduction immediately above each code cell, add the following **standalone Markdown cells**. Their purpose is to help a reader understand the reasoning and mechanics of the experiment before seeing the implementation. Keep them in English, use the same terms consistently, and write observed counts/metric values only through the runtime-generated summary cells.

| Markdown cell | Placement | Required explanation |
|---|---|---|
| **M0 -- `## How to Read This Notebook`** | After `tl;dr` and before Setup | Give the reader the end-to-end path: prepared explicit ratings -> model-specific sample/filter -> leakage-safe holdout -> SVD fitting -> rating/ranking evaluation -> refit on all model data -> Top-10 export. State that evaluation and serving use separate fitted models for different purposes. |
| **M1 -- `## Recommendation Task and Data Boundaries`** | Before Cell 3 | Explain the unit of one row: one user's explicit 1--10 score for one anime. Clarify that `rating = -1` means watched but not scored and is excluded, and that anime metadata is used only to make output readable, not to train SVD. Include a short distinction between the cleaned source dataset and the sampled/filtered `ratings_model`. |
| **M2 -- `## Why Collaborative Filtering and SVD?`** | Before Cell 5 | Explain that SVD learns from patterns shared among users and anime, so it can estimate a rating even when the user has not rated that anime. State why it fits this sparse matrix better than treating missing ratings as zeros. Include a plain-language warning that SVD needs previous ratings and therefore cannot fully solve cold-start cases. |
| **M3 -- `## What the SVD Model Learns`** | Immediately before Cell 8 | Present the prediction equation in a display formula: `r_hat(u, i) = mu + b_u + b_i + p_u^T q_i`. Define `mu` (global mean), `b_u`/`b_i` (user and anime bias), and `p_u`/`q_i` (latent-factor vectors). Explain that a latent factor is an inferred preference pattern, not a manually assigned genre label; use an illustrative phrase such as “preference for action-oriented series” only as intuition, not as a claim about a particular factor. Add one short numbered example describing how user factors and anime factors combine into a predicted score. |
| **M4 -- `## How SVD Is Trained`** | Immediately before Cell 8, after M3 | Explain that the model compares predictions with known training ratings, adjusts the biases and latent factors over multiple epochs, and uses regularization to avoid fitting noise. Define the role of `n_factors`, `n_epochs`, `lr_all`, `reg_all`, and `random_state` in one compact table. State that the fixed configuration is a reproducible baseline, not evidence of exhaustive hyperparameter optimization. |
| **M5 -- `## Why the Split Must Come Before Training`** | Before Cell 7 | Explain the 80/20 outer holdout in terms of a mock exam: the model can learn only from training ratings; the test ratings remain hidden until final scoring. Explicitly identify leakage examples to avoid: fitting on test ratings, selecting hyperparameters by test RMSE, and deciding relevance from training data. Note that the random split evaluates this dataset protocol and is not a temporal forecasting test. |
| **M6 -- `## Two Complementary Views of Model Quality`** | Before Cell 10 | Distinguish rating prediction from recommendation ranking in a two-row table: RMSE/MAE ask “how close is the predicted score to the held-out score?”; Precision@10/Recall@10 ask “among highly predicted held-out items, how many are actually liked and how many of the user's liked test items were retrieved?”. State that lower RMSE/MAE is better, while higher Precision@10/Recall@10 is better. |
| **M7 -- `## How Precision@10 and Recall@10 Are Counted`** | Before Cell 11 | Restate the eligibility and relevance rules with a small worked symbolic example (for example, 3 relevant items in the test set and 2 relevant items in the top 10 gives Precision@10 = 2/10 and Recall@10 = 2/3). Explain why users with no relevant test item are excluded from recall rather than assigned zero, and why this is ranking over held-out observed items rather than the complete unseen catalog. |
| **M8 -- `## From Evaluation Model to Recommendations`** | Before Cell 13 | Explain why the outer-split model is retained only for honest metrics, while a new model is fitted on all `ratings_model` interactions for the final Top-10 list. Describe candidates as anime represented in the model data that the selected user has not rated; explain the exclusion prevents recommending an already-rated anime. |
| **M9 -- `## How to Interpret the Recommendation Table`** | Before Cell 14 | Define `rank`, `predicted_rating`, and the metadata columns. State that `predicted_rating` is the model's estimated preference on the 1--10 scale, not a verified future rating, objective quality score, or the public average rating. Mention the deterministic user-selection rule so the demonstration can be reproduced. |
| **M10 -- `## Reading the Results Responsibly`** | Before Cell 16 | Give a short interpretation checklist: compare metrics only under the same split and data scope; report the eligible-user count with ranking metrics; do not call the sampled/k-core data the whole dataset; and do not infer causal user preferences from latent factors. |

Use one simple workflow diagram made with Markdown/Unicode arrows in M0 (not an image that could become stale):

```text
Cleaned explicit ratings
  -> sample and k-core filter
  -> outer train/test split
  -> evaluate SVD on held-out ratings
  -> refit SVD on all model data
  -> rank unseen candidates and export Top-10
```

The equations and worked example must be explanatory only. They must not introduce new computations or conflict with the exact evaluation protocol in Section 5.2.

### 0. `# Anime Recommendation System - SVD Model and Evaluation`

**`## tl;dr`**

Leave the quantitative content blank when creating the notebook. After a successful full run, update it with two or three English sentences that state the post-filter model-data size, selected configuration, and the four observed metrics.

**`## Context, Inputs, and Outputs`**

- State that Notebook 01 prepares explicit ratings and Notebook 02 consumes the cleaned files from `cleaned_data/`.
- State that `-1` is invalid for SVD, sampling is not the full dataset, and metadata is display-only.
- List the input and output paths from Section 2.
- Add M0 immediately after this context so the notebook's modelling and recommendation flow is clear before implementation details begin.

### 1. `## Setup`

**Cell 1 -- Imports, Constants, and Paths**

Import `Path`, `defaultdict`, `time`, `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `surprise.Dataset`, `Reader`, `SVD`, `accuracy`, and `train_test_split`.

Declare:

```python
SEED = 42
RATING_MIN, RATING_MAX = 1, 10
MAX_RATINGS = 1_000_000
MIN_USER_RATINGS = MIN_ITEM_RATINGS = 5
MAX_CORE_FILTER_PASSES = 10
TEST_SIZE = 0.20
K = 10
POSITIVE_RATING = 8
N_EXAMPLE_USERS = 7
MIN_RATINGS_FOR_EXAMPLE_USER = 20
```

Resolve the input pair before declaring output paths: prefer `/content/ratings_clean.csv` and `/content/anime_clean.csv` when both Colab files exist; otherwise check `cleaned_data/` from the current directory and its parent. Record `CLEANED_DATA_DIR`, `RATINGS_PATH`, `ANIME_PATH`, the selected source, `PROJECT_ROOT`, `OUTPUTS_DIR`, and `FIGURES_DIR`. Fail fast with an English error that lists the attempted locations if a complete pair is missing.

**Cell 2 -- Helper Functions**

Create small, pure functions with English names and docstrings:

- `require_columns(df, required_columns, dataset_name)` validates the schema.
- `matrix_summary(ratings, stage)` returns one DataFrame row with interactions, users, items, density, rating mean/median, and user/item count statistics.
- `iterative_k_core_filter(ratings, min_user_ratings, min_item_ratings, max_passes)` returns the filtered frame and a per-pass audit table.
- `precision_recall_at_k(predictions, k, positive_rating)` follows the exact rules in Section 5.2.
- `save_figure(filename)` applies `tight_layout`, saves a 300-dpi PNG, and calls `show()`.
- `metadata_for_display(anime_df)` prepares title/genre fallbacks without mutating input metadata.

### 2. `## Load and Validate Prepared Data`

Place M1 before Cell 3.

**Cell 3 -- Read Cleaned CSV Files**

- Read ratings with `user_id` and `anime_id` as strings and `rating` as numeric. Read metadata with `anime_id` as string and text fields as `string`.
- Validate the required schema, print file sizes/shapes, and show a small preview only.

**Cell 4 -- Input Integrity Checks**

Display an English PASS/FAIL table and assert all of the following:

- Ratings have no missing `user_id`, `anime_id`, or `rating`.
- Ratings are within `[1, 10]` and no `-1` remains.
- No duplicate `(user_id, anime_id)` pairs remain.
- `anime_clean.anime_id` is unique.
- Separately report anime IDs rated by users but missing from metadata. Keep these ratings for SVD and use a display fallback on export.

End the cell with `matrix_summary(ratings_clean, "Clean input")` to compare against Notebook 01.

### 3. `## Resource-Aware Model Dataset`

Place M2 before Cell 5. Its explanation should explicitly connect sparsity and cold-start to the sampling/filtering decisions that follow.

**Cell 5 -- Reproducible Sampling**

Sample according to Section 3.1 and display the `Clean input -> Sampled input` or `Clean input -> Full input` audit. Do not sample metadata.

**Cell 6 -- Iterative Minimum-Interaction Filtering**

Run the k-core filter, display the per-pass audit, and display the final `ratings_model` summary. Keep `ratings_model` in memory; do not export it unless the group deliberately needs a separate audit artifact.

The preceding English Markdown must explain that this reduces artificial cold-start cases and ensures enough signal per user/item, while changing coverage relative to the original cleaned data.

### 4. `## Train/Test Split and SVD Training`

Place M5 before Cell 7, then M3 and M4 consecutively before Cell 8. This order lets the reader first understand what data the model may see, then what the model is learning, and finally how it is trained.

**Cell 7 -- Build the Surprise Dataset and Outer Split**

Create the `Reader`, Surprise `Dataset`, `trainset`, and `testset`. Display a split summary in English. Retain `ratings_model` for later full-data recommendation generation.

**Cell 8 -- Train the Fixed SVD Baseline**

Use this mandatory, reproducible configuration:

```python
SVD_CONFIG = {
    "n_factors": 50,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": SEED,
}
```

Initialize `SVD(**SVD_CONFIG)`, fit it on `trainset`, predict `testset`, and store `predictions`. Print training duration using `time.perf_counter()` and the prediction count.

**Cell 9 -- Optional Lightweight Tuning (Explicit Opt-In)**

Set `RUN_LIGHT_TUNING = False` by default. If enabled, create a validation split from the outer training data, compare exactly two configurations (50 and 100 factors), choose the lowest validation RMSE, refit on the complete outer `trainset`, and regenerate predictions for the outer `testset`. Do not run tuning on the testset and do not use exhaustive `GridSearchCV`.

When it remains `False`, Markdown must describe the result as a reproducible baseline and must not claim hyperparameter tuning was performed.

### 5. `## Evaluation`

Place M6 before Cell 10 and M7 before Cell 11. The Markdown immediately above Cell 11 must also link back to M7 rather than repeating the complete protocol word-for-word.

**Cell 10 -- Rating-Prediction Metrics**

- Calculate `rmse = accuracy.rmse(predictions, verbose=False)` and `mae = accuracy.mae(predictions, verbose=False)`.
- Build `prediction_df` with `user_id`, `anime_id`, `true_rating`, `predicted_rating`, and `error = predicted_rating - true_rating`.
- Assert that every predicted rating is finite and within `[1, 10]` (allowing a small floating-point tolerance if required).
- Build a metrics DataFrame with metric name, value, split, seed, model-rating count, prediction count, and serialized SVD configuration.

**Cell 11 -- Precision@10 and Recall@10**

To keep ranking metrics meaningful and reproducible, implement these exact rules:

1. Group `predictions` by user and sort by descending `predicted_rating`; break ties with ascending `anime_id`.
2. Evaluate only users with **at least 10** testset items and at least one relevant item in their complete testset. An item is relevant when `true_rating >= 8`.
3. For every eligible user, take the ten highest-predicted items. `hits@10` is the number of those items with `true_rating >= 8`.
4. Calculate `precision@10_user = hits@10 / 10` and `recall@10_user = hits@10 / total_relevant_items_in_that_user_testset`.
5. Macro-average both values across eligible users. Report `eligible_users`, `users_with_fewer_than_k_test_items`, `users_without_relevant_test_item`, and `total_test_users`.

Do not assign Recall = 0 to users without a relevant test item. Clearly state that this is **held-out observed-item ranking evaluation**, not ranking over every anime the user has never interacted with; this is a limitation of the explicit-rating dataset.

Append RMSE, MAE, Precision@10, Recall@10, and ranking-coverage counts to `model_evaluation_metrics.csv`.

**Cell 12 -- Evaluation Charts**

- Produce a labeled RMSE/MAE bar chart: `svd_error_metrics.png`.
- Produce a labeled Precision@10/Recall@10 bar chart; the title must include the eligible-user count: `svd_ranking_metrics_at_10.png`.
- Produce an actual-versus-predicted scatter or hexbin plot from at most 50,000 deterministically sampled predictions, with `random_state=SEED` and a `y = x` reference line: `svd_actual_vs_predicted.png`.

Do not draw all predictions when doing so makes the notebook slow or the result unreadable.

### 6. `## Generate Top-10 Recommendations`

Place M8 before Cell 13 and M9 before Cell 14.

**Cell 13 -- Refit a Separate Model for Recommendation Serving**

Metrics use the outer train/test split. To produce the final recommendations without discarding 20% of known ratings or recommending already rated anime, create a **separate final model**:

1. Build a Surprise dataset from all post-sample/post-filter `ratings_model` data.
2. Fit `final_model = SVD(**selected_config)` on that complete trainset.
3. Candidate items are anime that appear in the complete trainset and have not been rated by the user anywhere in `ratings_model`.

Never replace the outer-test metric results with this final model; it is only the model used to export recommendations.

**Cell 14 -- Select Example Users, Rank Candidates, and Export**

- If `EXAMPLE_USER_IDS` is declared, validate those IDs. If it is `None`, select seven distinct, reproducible demonstration scenarios from observed rating history: limited history (5--9 ratings), developing history (10--19 ratings), typical history, very active, highly positive, more critical, and broad genre exposure. The latter five require at least `MIN_RATINGS_FOR_EXAMPLE_USER` ratings. Use deterministic sorting and a deterministic unused-user fallback so the seven examples are always distinct. State clearly that scenario labels describe observed ratings only, not SVD latent factors or guaranteed future preferences.
- For every user, predict all candidates with `final_model.predict(uid, iid).est`, sort by descending predicted rating with ascending `anime_id` as the tie-breaker, and take `K = 10`.
- Left join `anime_clean`. Use `"Unknown title (ID: <anime_id>)"` for missing names and `"Unknown"` for missing genres. Do not use `anime_average_rating` as a training label or predicted score.
- Export this exact column order: `user_id`, `rank`, `anime_id`, `name`, `genre`, `type`, `episodes`, `anime_average_rating`, `members`, `predicted_rating`.
- Assert that every exported user has ten rows, ranks 1--10, contains no anime rated in `ratings_model`, and has no duplicate `(user_id, anime_id)` pair.

Display a compact scenario-profile table and a readable English Top-10 table for every selected user, so the demo can compare recommendations across different user contexts. Keep the exported CSV schema unchanged.

### 7. `## Conclusion, Limitations, and Reproducibility`

Place M10 before Cell 16, after the runtime-generated summary, so caveats are read in the context of the actual results.

**Cell 15 -- Auto-Generated Result Summary**

Print English bullets from runtime variables: `ratings_model` interactions/users/items, SVD configuration, RMSE, MAE, Precision@10, Recall@10, eligible ranking users, and output paths. This becomes the source for the Results and Evaluation section in the report and presentation.

**Cell 16 -- Limitations and Future Work**

State concisely in English:

- SVD does not solve cold-start users/anime with no or very few ratings; a genre-based metadata fallback could help.
- The data contains only explicit ratings, excludes watched-but-unrated `-1` interactions, and sampling/k-core filtering can bias coverage toward active users/anime.
- Precision/Recall are measured on observed holdout ratings and may not fully represent full-catalog recommendation ranking.
- Future work includes lightweight validation/tuning, a temporal split if timestamps become available, hybrid genre-based recommendation, and catalog evaluation with clearly documented negative sampling.

**Cell 17 -- Final Checklist and Rerun Command**

Display an English PASS/FAIL checklist that confirms input files, valid `ratings_model`, finite metrics, recommendation CSV schema, ten recommendations per user, metrics CSV, and all three PNG files. Print:

```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/02_svd_model_evaluation.ipynb
```

## 5. Reporting rules and mapping to course requirements

| Requirement | Notebook 02 evidence |
|---|---|
| Collaborative filtering | `surprise.SVD` factorizes the user--item explicit-rating matrix |
| Top-N per user | `outputs/top_10_recommendations.csv` contains ten unseen anime per demonstration user |
| RMSE / MAE | Outer holdout testset, Cell 10, and metrics CSV |
| Precision@K / Recall@K | `K = 10`, held-out ranking protocol in Cell 11, and eligible-user count |
| Result charts | Three evaluation PNG files in `outputs/figures/` |
| Runnable code | Explicit paths, constants, assertions, seed, and rerun command |

Do not overstate the metrics: RMSE/MAE measure rating-prediction error, while Precision@10/Recall@10 measure whether highly rated held-out items appear near the top of the observed-item ranking. Rating count/popularity is not anime quality.

## 6. Acceptance checklist before committing Notebook 02

- [ ] The notebook runs top-to-bottom with `MAX_RATINGS = 1_000_000` and does not depend on manually created intermediate artifacts.
- [ ] The notebook first uses both cleaned files uploaded to the Colab server root when present, otherwise explicitly falls back to the project's local `cleaned_data/` directory; it does not read the raw dataset.
- [ ] Every Notebook 02 Markdown cell, code comment, function/variable name, printed message, table, chart, and exported column name is in English.
- [ ] The notebook includes M0--M10 from Section 4.0, including the SVD prediction equation, parameter-role table, leakage explanation, ranking worked example, and separate evaluation-versus-serving explanation.
- [ ] Conceptual Markdown does not hard-code run-dependent counts or metrics and does not describe latent factors as known genre labels.
- [ ] Input validation confirms ratings in `[1, 10]`, no `-1`, no duplicate key, and unique metadata IDs.
- [ ] Every sample/filter/split step reports interactions, users, anime, and density for auditability.
- [ ] SVD uses a fixed seed and its exact configuration is printed and saved with the metrics.
- [ ] RMSE, MAE, Precision@10, and Recall@10 are calculated from holdout predictions and are not hard-coded.
- [ ] The ranking report includes eligible-user counts and does not assign zero recall to users with no relevant test item.
- [ ] The recommendation export contains only anime unseen in `ratings_model`, ten items per user, and metadata fallbacks.
- [ ] The metrics CSV and three charts exist and can be used directly in the report and presentation.
- [ ] `requirements.txt` keeps `numpy==1.26.4` with `scikit-surprise==1.1.4`, because Surprise in this project requires the NumPy 1.x C API.
