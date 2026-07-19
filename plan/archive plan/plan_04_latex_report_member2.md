# Plan: Complete the LaTeX Report — Member 2 (Modeling & Evaluation Lead)

## Goal and delivery scope

Complete Member 2's modeling, evaluation, report-integration, and conclusion work, then deliver one self-contained Overleaf project that compiles with **XeLaTeX + Biber**.

The integrated report entry point already exists at [`overleaf-report/main.tex`](../overleaf-report/main.tex). It contains both Member 1's Dataset/EDA work and Member 2's model/evaluation work, together with the required bibliography and four report figures. Do **not** create a duplicate `member2.tex` or a second standalone report; review and edit this single entry point so that both members' sections remain consistent.

## Ownership and current report mapping

| Report component | Primary owner | Current location in `main.tex` |
|---|---|---|
| Dataset, cleaning, EDA, and APA source inputs | Member 1 | `Dataset and data preparation` |
| SVD choice, training, evaluation, charts, and Top-10 export | Member 2 | `Methodology`, Notebook 02 part of `Implementation`, and `Results and evaluation` |
| Abstract, introduction, report integration, conclusion, and presentation | Member 2 integrates; Member 1 verifies data facts | Drafted in the corresponding report sections |
| Contribution percentages | Both members approve | Currently 50% / 50%; must match the presentation |

## Verified Notebook 02 evidence

Use the executed [`notebooks/02_svd_model_evaluation.ipynb`](../notebooks/02_svd_model_evaluation.ipynb) and [`outputs/model_evaluation_metrics.csv`](../outputs/model_evaluation_metrics.csv) as the authoritative sources. If Notebook 02 is re-run with changed data or configuration, update every affected report number, table, chart, and conclusion together.

| Item | Verified value for the current run |
|---|---:|
| Sampling / seed | 1,000,000 clean ratings / 42 |
| K-core rule | At least 5 ratings per retained user and anime |
| Final model data | 952,497 interactions; 41,370 users; 5,985 anime |
| Outer holdout | 80/20: 761,997 training ratings; 190,500 test predictions |
| Fixed SVD baseline | 50 factors; 20 epochs; `lr_all=0.005`; `reg_all=0.02`; seed 42 |
| Hyperparameter tuning | Disabled; this is a reproducible baseline, not exhaustive optimisation |
| RMSE / MAE | 1.2632 / 0.9625 |
| Precision@10 / Recall@10 | 0.6386 / 0.8255 |
| Ranking population | Held-out observed-item ranking; 4,802 eligible users of 37,748 test users |
| Relevance definition | True held-out rating ≥ 8 |

## Work plan

### 1. Preserve the Member 1 data boundary

- Fact-check the Dataset/EDA claims against Notebook 01 before changing related wording: 6,337,234 cleaned explicit ratings; 1,476,496 `-1` records excluded; 7 duplicate pairs merged; 69,600 users; 9,927 rated anime; and 0.9172% matrix density.
- Keep a strict distinction between `ratings_clean` (the audited cleaning output) and `ratings_model` (the sampled and k-core-filtered dataset used by SVD).
- Keep `anime_average_rating` as display metadata only. It is neither an SVD target nor a personalised prediction.
- Retain the two validated EDA figures with project-relative paths under `figures/`.

**Exit criterion:** the data section and methodology use the same definitions and never present the sampled/k-core subset as the entire cleaned dataset.

### 2. Finalise Member 2's methodology

Review the `Methodology` section in this order:

1. Explain why collaborative filtering is appropriate for sparse explicit ratings and why missing ratings are not treated as zero.
2. Present the biased SVD equation, `\hat{r}_{ui}=\mu+b_u+b_i+q_i^Tp_u`, and define the global mean, user/item biases, and latent factors. Do not equate latent factors with genre labels.
3. State the resource-aware sampling, iterative five-interaction k-core filtering, and the resulting coverage limitation.
4. State the fixed SVD configuration and make clear that test data was not used to fit the model or select hyperparameters.
5. Define RMSE and MAE as outer-holdout rating-error metrics. Define Precision@10 and Recall@10 using the relevance threshold, eligibility rule, macro average, and held-out observed-item candidate set.
6. Distinguish the evaluation model from the separate serving model refitted on all `ratings_model` interactions to score only unseen anime.

**Exit criterion:** each reported metric is accompanied by its data scope, split, direction of improvement, `K`, relevance threshold, and eligibility rule.

### 3. Complete implementation and result presentation

- Keep the implementation section concise: Notebook 01 produces validated clean inputs and EDA figures; Notebook 02 samples, filters, splits, trains, evaluates, refits the serving model, and exports recommendations.
- Retain the four-metric table and the two model-result figures: `svd_error_metrics.png` and `svd_ranking_metrics_at_10.png`.
- State explicitly that the ranking chart covers 4,802 eligible users and does not measure full-catalog recommendation quality.
- If page space permits, add a compact Top-5 recommendation table for user `10004` (the `Limited history` scenario) from `outputs/top_10_recommendations.csv`. Include only title and predicted rating, and label the values as estimated preferences for unseen items—not future verified ratings or objective quality.
- If the report is constrained by pages, omit the optional Top-5 table before removing evaluation-protocol details or limitations.

**Exit criterion:** the reader can connect the protocol, metric table, charts, and Top-N output without seeing a notebook code dump.

### 4. Finish the integrated sections

- **Abstract (150–220 words):** state the problem, cleaned-data scope, SVD approach, all four metrics, and one important limitation. Do not cite sources here.
- **Introduction:** frame the streaming recommendation problem and state three measurable objectives: defensible cleaning, holdout rating prediction, and held-out ranking retrieval.
- **Conclusion and future work:** limit conclusions to the sampled/k-core outer-holdout experiment. Address cold start, explicit-rating selection bias, metadata gaps, the non-temporal split, and observed-item ranking. Recommend validation-based tuning, temporal evaluation where timestamps exist, documented negative sampling for catalogue ranking, and a hybrid genre fallback.
- **Individual contribution:** confirm names, IDs, and percentages with the presentation. Percentages must sum to 100%.
- **References:** retain APA/Biber; check that every `\parencite` has an entry in `references.bib` and that all cited sources are appropriate for the claims made.

**Exit criterion:** the report makes bounded claims and the abstract, conclusion, contribution table, notebook, and presentation agree.

### 5. Validate the Overleaf package and submit

1. Upload the complete [`overleaf-report/`](../overleaf-report/) directory, including `main.tex`, `references.bib`, and `figures/`.
2. Select **XeLaTeX** as the Overleaf compiler and **Biber** as the bibliography tool.
3. Recompile until bibliography and cross-references resolve, then inspect the PDF and build log for `??`, missing citations, missing figures, clipped tables, overfull text, or font problems.
4. Confirm the title page states Duy Tan University, Da Nang; DS 423; Group 14; and both members' correct names and IDs.
5. Cross-check report, notebook, and PPT metrics and contribution percentages. Submit both the `.tex` source and compiled PDF as required.

## Non-negotiable reporting rules

- Do not describe Precision@10 or Recall@10 as full-catalog recommendation performance.
- Do not use outer-test ratings to fit the model or choose hyperparameters.
- Do not call the sampled/k-core subset the whole 6.337 million-rating cleaned dataset.
- Do not conflate public anime average ratings with SVD predicted ratings.
- Do not change values only to match prose. Re-run Notebook 02 first, then synchronise all downstream report evidence.

## Definition of done

- [ ] `main.tex` contains every required assignment section and remains the sole report entry point.
- [ ] All figures and bibliography files are inside `overleaf-report/` and use project-relative paths.
- [ ] Model and evaluation claims are traceable to the executed notebook and generated artifacts.
- [ ] The four metrics include full protocol context and appropriate limitations.
- [ ] Abstract, conclusion, references, and contribution table have been reviewed by both members.
- [ ] The XeLaTeX + Biber Overleaf build is clean and the resulting PDF has been visually checked.
