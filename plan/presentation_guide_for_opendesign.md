# OpenDesign Guide - Anime Recommendation System (12 Slides)

## Purpose and communication goal

Build a content-first academic presentation for **DS 423 SA - Machine Learning with Large Datasets** at **Duy Tan University, Da Nang**. The audience is the course lecturer. By the end, the lecturer should see that Group 14 built a reproducible anime recommender with a defensible explicit-rating signal, a leakage-aware SVD evaluation, and a clear boundary around what the reported metrics do and do not prove.

The deck must follow this argument:

> Correct data meaning -> sparse-data evidence -> appropriate collaborative-filtering baseline -> honest evaluation -> personalized Top-10 serving output -> limitations and responsible next steps.

Use these sources as ground truth:

- `overleaf-report/Group14_Report.pdf`
- `plan/Instructions_Group_14.docx.md`
- `notebooks/01_eda_data_preparation.ipynb`
- `notebooks/02_svd_model_evaluation.ipynb`
- `outputs/model_evaluation_metrics.csv`
- `outputs/top_10_recommendations.csv`
- Charts in `outputs/figures/`

## Mandatory content rules

- Use exactly **12 slides**, including title, contribution, and references.
- Make the content detailed, explicit, and readable. Prefer clear evidence, definitions, and interpretation over decorative visuals.
- Keep one primary claim per slide and use takeaway-style titles.
- The final reported metrics must use `outputs/model_evaluation_metrics.csv` and the report, not stale Notebook 02 cell outputs.
- Always distinguish the **cleaned dataset** from the **sampled and k-core-filtered model dataset**.
- Always describe Precision@10 and Recall@10 as **held-out observed-item ranking** metrics, not full-catalogue recommendation performance.
- Use English for the deck; preserve Vietnamese diacritics in member names.

## Slide specification

### 1. Title - Anime Recommendation System

Include:

- Anime Recommendation System
- Group 14
- DS 423 SA - Machine Learning with Large Datasets
- Duy Tan University, Da Nang
- Trương Nhật Trường - 29211164915
- Mai Nam Quốc - 29211145181

Keep the title slide minimal. It identifies the project; it should not contain an agenda or technical claims.

### 2. The project ranks unseen anime from explicit user preferences

State the real-world problem and exact task:

- Anime catalogues are too large for viewers to inspect title by title.
- Input: historical explicit user ratings on the original 1-10 scale.
- Output: a personalized Top-10 list of anime the user has not already rated.
- Missing user-anime interactions are unknown preferences, not zero ratings or dislikes.

State three project objectives:

1. Prepare an auditable explicit-rating dataset.
2. Predict held-out ratings using RMSE and MAE.
3. Retrieve highly rated held-out items near the top using Precision@10 and Recall@10.

### 3. The data is large, sparse, and split between learning signals and display metadata

Include a compact two-source-table explanation:

- Kaggle source: **Anime Recommendations Database**, CooperUnion (2016).
- `rating.csv`: 7,813,737 user-anime interactions with `user_id`, `anime_id`, and `rating`.
- `anime.csv`: 12,294 anime records with title, genre, type, episodes, public average rating, and membership count.
- Ratings are the learning signal. Metadata is joined after ranking to make Top-10 output readable.
- `anime_average_rating` is not an SVD target and must never be confused with a predicted personal rating.

### 4. Cleaning preserves the meaning of an explicit preference signal

Show the cleaning audit as a data funnel or concise table:

| Stage | Count | Meaning |
| --- | ---: | --- |
| Raw interactions | 7,813,737 | Source user-anime rows |
| `rating = -1` removed | 1,476,496 (18.90%) | Watched but not explicitly scored; not a dislike |
| Invalid or missing ratings removed | 0 | Required rating values are complete and parseable |
| Duplicate user-anime keys merged | 7 | Mean rating retains one target per matrix cell |
| Clean explicit interactions | 6,337,234 | Reproducible collaborative-filtering baseline |

Add the final coverage: 69,600 users, 9,927 rated anime, and 0.9172% matrix density.

Explain why `-1` cannot become 0 or 1: doing so would incorrectly train the model that “watched but unrated” means negative preference. Note that duplicate averaging may create valid 6.5 or 8.5 ratings.

### 5. EDA explains the modeling constraints, not model quality

Use the actual project charts:

- `outputs/figures/eda_rating_distribution.png`
- `outputs/figures/eda_user_rating_count_distribution.png`

State the findings and their implications:

- Mean explicit rating = 7.81; mode = 8; 82.55% of explicit ratings are at least 7.
- User-history distribution is long-tailed: median = 45 ratings, P90 = 230, P99 = 640.
- Popularity is concentrated: Death Note has 34,226 explicit ratings.
- The sparse 0.9172% matrix supports collaborative filtering but signals cold-start, unequal-evidence, and popularity-bias risks.

Do not say that EDA evaluates the model or that rating count measures anime quality.

### 6. Two notebooks create an auditable path from raw CSV files to Top-10 output

Show the implementation pipeline in sequence:

1. **Notebook 01:** load raw files, validate schema, remove `-1`, merge duplicate interactions, save cleaned data, and export EDA figures.
2. **Notebook 02:** reload and validate cleaned files, sample and k-core filter, split outer train/test data, tune and evaluate SVD, refit a serving model, and export recommendations.
3. **Saved artifacts:** `ratings_clean.csv`, `anime_clean.csv`, metric CSV, figures, and `top_10_recommendations.csv`.

Show only short readable code excerpts or pseudocode for: removing `-1`, the seeded split, and excluding already-rated anime. Mention deterministic seed 42 and validation assertions. Do not show full notebook screenshots.

### 7. Biased SVD estimates personalized ratings from shared user-anime patterns

Include the model equation:

`r_hat_ui = mu + b_u + b_i + q_i^T p_u`

Explain:

- `mu`: global mean rating.
- `b_u` and `b_i`: user and anime bias terms.
- `p_u` and `q_i`: learned latent preference vectors.
- The model learns regularized squared-error estimates with stochastic gradient descent.

Explain why this is appropriate:

- It exploits shared patterns among observed ratings without replacing unknown pairs with zeros.
- It is a reproducible baseline for sparse explicit-rating data.
- Latent factors are inferred patterns, not automatically interpretable genre labels.
- SVD does not solve pure new-user or new-anime cold start.

### 8. The experiment makes resource limits and leakage protection visible

Show the scope progression and tuning protocol:

| Scope | Interactions | Users | Anime |
| --- | ---: | ---: | ---: |
| Cleaned baseline | 6,337,234 | 69,600 | 9,927 |
| Fixed-seed sample | 1,000,000 | 60,743 | 8,218 |
| After iterative min-5 k-core | 952,497 | 41,370 | 5,985 |
| Outer training split | 761,997 | - | - |
| Outer test split | 190,500 | - | - |

Then explain the leakage-safe order:

- Create the outer 80/20 random holdout first.
- Inside the outer training data only, compare 30, 50, and 100 latent factors.
- Validation RMSE: 30 = 1.2649, 50 = 1.2723, 100 = 1.2817.
- Select 30 factors; then evaluate once on the untouched outer test.
- Use 20 epochs, learning rate 0.005, regularization 0.02, and random state 42.

State that sampling and k-core filtering are model-scope choices, not extra data cleaning, and that the random split is not temporal forecasting.

### 9. Four metrics answer two different questions under a stated ranking protocol

Define the metrics before giving results:

- **RMSE / MAE:** error between a predicted and true held-out 1-10 rating; lower is better.
- **Relevance:** a true held-out rating of at least 8.
- **Precision@10:** fraction of the 10 highest-predicted held-out items that are relevant.
- **Recall@10:** fraction of an eligible user's relevant held-out items recovered in the top 10.

Define eligibility and scope:

- A user needs at least 10 test items and at least one relevant test item.
- 4,802 of 37,748 test users are eligible; 32,911 have fewer than 10 test items and 35 have no relevant test item.
- This is **held-out observed-item ranking**. It is not ranking against every unseen anime in the full catalogue.

### 10. The selected SVD model produces scoped ranking evidence but only marginally improves rating error

Use these final metrics exactly:

| Metric | Bias-only benchmark | Selected 30-factor SVD | Interpretation |
| --- | ---: | ---: | --- |
| RMSE | 1.2552 | 1.2551 | Lower is better; improvement is marginal |
| MAE | 0.9604 | 0.9560 | Average absolute rating error is just under one point |
| Precision@10 | - | 0.6397 | Higher is better, under held-out observed-item ranking |
| Recall@10 | - | 0.8273 | Higher is better, under the same ranking protocol |

Use `outputs/figures/svd_error_metrics.png` and `outputs/figures/svd_ranking_metrics_at_10.png`.

Interpret carefully: the SVD result is reproducible evidence, but the tiny RMSE/MAE gain means it must not be presented as a dramatic improvement. Precision@10 and Recall@10 are promising only under the documented eligible-user and candidate-set rules.

### 11. Serving refits after evaluation and returns only unseen anime

Explain the serving protocol:

1. Keep outer-test metrics fixed.
2. Refit the chosen 30-factor configuration on all 952,497 model-scope interactions.
3. Remove anime already rated by the selected user.
4. Score remaining model-scope candidates, sort by predicted personal rating, break ties by anime ID, and return the first 10.
5. Join title, genre, and type metadata after ranking.

Show a short Top-5 excerpt for the **Typical history** demonstration profile (user 25321; 33 retained ratings; mean observed rating 7.79):

| Rank | Anime | Predicted rating |
| ---: | --- | ---: |
| 1 | Ginga Eiyuu Densetsu | 9.6250 |
| 2 | Gintama° | 9.4408 |
| 3 | Neon Genesis Evangelion | 9.4265 |
| 4 | Fullmetal Alchemist: Brotherhood | 9.3124 |
| 5 | Byousoku 5 Centimeter | 9.2897 |

State that a predicted rating is not a verified future rating, public average rating, or claim of objective quality. The seven user profiles are serving demonstrations, not additional test folds.

### 12. Conclusion, limitations, contribution, and references

Lead with the conclusion:

> The project delivers a reproducible biased-SVD baseline that turns explicit user ratings into unseen-anime Top-10 recommendations, while keeping the data, evaluation, and serving boundaries explicit.

List limitations and linked next steps:

- Cold start -> add genre/content fallback or hybrid recommendation.
- Sampling and k-core coverage bias -> report coverage/diversity and compare on broader data.
- Observed-item ranking -> evaluate the full catalogue with documented negative sampling.
- Random split -> add temporal validation if timestamps become available.
- Marginal gain over the bias-only model -> compare stronger baselines and tune more parameters.

Show contribution as 50% / 50%:

- **Mai Nam Quốc - 29211145181 (50%)**: data loading, cleaning, preprocessing, EDA, data preparation, APA references, and report review.
- **Trương Nhật Trường - 29211164915 (50%)**: SVD modeling, evaluation, Top-10 export, presentation preparation, report integration, and conclusion.
- **Shared:** integrate, test, and cross-check code, figures, metrics, and report claims.

Use readable APA references:

- CooperUnion. (2016). *Anime recommendations database* [Data set]. Kaggle. https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
- Hug, N. (n.d.). *Matrix factorization-based algorithms*. Surprise. https://surprise.readthedocs.io/en/stable/matrix_factorization.html
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer, 42*(8), 30-37. https://doi.org/10.1109/MC.2009.263

## Source consistency warning

Notebook 02 contains stale Markdown and partially inconsistent saved outputs from earlier runs. Do **not** use its 50-factor description or older values such as RMSE 1.2632, MAE 0.9625, Precision@10 0.6386, or Recall@10 0.8255. The report and `outputs/model_evaluation_metrics.csv` agree on the authoritative 30-factor values used in this guide.

The instructions file's predefined responsibility table conflicts with the completed report's role assignment. This guide uses the report's 50/50 contribution wording so that the deck matches the completed report and notebooks; confirm the roles before submission.

## Design direction

Use a restrained academic layout: white background, black/gray text, limited blue and green only for evidence and charts. Use the project's own charts, tables, and code excerpts before decorative imagery. Keep type large enough for a classroom. Do not use dashboard-style card grids, full notebook screenshots, or visual elements that reduce space for the evidence.
