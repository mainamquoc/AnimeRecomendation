# Plan: write the Member 1 EDA/Data Preparation part of the LaTeX report

## Goal and boundary

Produce a report-ready, evidence-based draft for the data portions of the Anime Recommendation System report.  This is Member 1's substantive contribution: dataset source/structure, cleaning and preparation, EDA, data limitations, implementation handoff, and APA reference inputs.

This plan does **not** write SVD training, model tuning, final metrics, Top-10 evaluation, overall abstract, or conclusion as if their results were already known.  Those belong to the Model & Evaluation work, although the data draft must connect cleanly to them.

## Required context to read before acting

Read in this exact order before drafting or modifying LaTeX:

1. [Instructions_Group_14.docx.md](Instructions_Group_14.docx.md)
2. [context_report_latex.md](context_report_latex.md)
3. [plan_01_eda_data_preparation.md](plan_01_eda_data_preparation.md)
4. Executed [01_eda_data_preparation.ipynb](../notebooks/01_eda_data_preparation.ipynb)
5. Actual EDA figures and clean CSV schemas under `outputs/figures/` and `data/`.
6. [plan_02_svd_model_evaluation.md](plan_02_svd_model_evaluation.md), only to write a correct handoff boundary.
7. `report_template.tex`, once available; preserve its section names and commands.

If the notebook output, clean CSV, or figure differs from the numbers below, update the draft to the executed artifact.  Do not use the plan as a substitute for verification.

## Deliverables from Member 1

1. LaTeX-ready text for the Dataset and Data Preparation section (target: about 650--900 words before template constraints).
2. A concise EDA subsection (about 350--550 words) with a data-summary table and two selected figures.
3. A short Notebook 01 implementation paragraph for the Implementation section (about 120--180 words).
4. A data limitations paragraph (about 120--180 words) for Results/Conclusion integration.
5. APA entries/source details for dataset, any EDA/data-cleaning references actually cited, and a handoff note to Member 2.

Member 2 remains responsible for integrating all report sections, the model/evaluation prose, final results, conclusion, and final LaTeX compilation.  Member 1 should provide reviewed source text and artifacts, not an isolated narrative that conflicts with the model protocol.

## Work sequence

### Phase 0 -- inspect the report template and establish a workspace

- Locate `report_template.tex`.  It is absent from the workspace at planning time; do not create a replacement unless instructed.
- When it arrives, record: compiler, bibliography approach, title fields, prescribed section names/order, figure paths, page/page-limit constraints, and existing macros.
- Build for Overleaf from the start: keep the template entry point unchanged; upload every cited figure and bibliography/style file into the same Overleaf project; use only forward-slash relative paths such as `figures/eda_rating_distribution.png`.
- In Overleaf, set the compiler to **XeLaTeX** before adding Vietnamese names.  Test a minimal title page and one EDA image immediately, then inspect the compile log.  Do not reference the local workspace with `C:\\...`, `../outputs/...`, or any absolute path.
- Follow the template's bibliography engine exactly.  Do not introduce a second document class, local-only font, `minted`, or external conversion package unless it has been tested successfully on Overleaf.
- Create an evidence ledger while drafting: every numeric statement maps to a notebook cell/output, every external claim maps to an APA source, and every figure maps to its generated artifact.

**Exit criterion:** the template constraints are known; a self-contained Overleaf project compiles with XeLaTeX; and there is no guessed bibliography command, missing asset, or local-only figure path.

### Phase 1 -- verify Member 1 evidence

Run/inspect Notebook 01 from top to bottom if artifacts may be stale.  Record the exact output date/commit if the group uses one.  Verify:

| Evidence | Current value to verify | Report use |
|---|---:|---|
| Raw interactions | 7,813,737 | Establish scale. |
| `-1` rows removed | 1,476,496 (18.90%) | Explain why invalid-for-rating-model records are excluded. |
| Duplicate interactions merged | 7 | Explain unique user-item observation rule. |
| Clean explicit interactions | 6,337,234 | State modeling input baseline. |
| Users / rated anime / metadata anime | 69,600 / 9,927 / 12,294 | Describe coverage. |
| Matrix density | 0.9172% | Motivate sparse collaborative filtering. |
| Rating mean/mode | 7.81 / 8 | Interpret rating distribution carefully. |
| Activity median/P90/P99 | 45 / 230 / 640 | Demonstrate long-tail user history. |

Also verify that `data/ratings_clean.csv` has only `user_id`, `anime_id`, `rating`; ratings lie in [1, 10]; no duplicate key remains; and `data/anime_clean.csv` uses `anime_average_rating` only as display metadata.

**Exit criterion:** every value used in prose/table has an executable source and matches it.

### Phase 2 -- draft Dataset and Data Preparation

Use subsections if the template permits; otherwise use compact paragraphs in this logical order.

#### 2.1 Dataset source and unit of observation

- Name the Anime Recommendations Database and cite Kaggle in APA format.
- State the two source tables: `rating.csv` (`user_id`, `anime_id`, `rating`) and `anime.csv` (name, genre, type, episodes, average rating, members).
- Define one interaction: an explicit score assigned by one user to one anime.  Do not describe missing/unrated items as zero ratings.
- State that metadata makes recommendations interpretable but does not supply SVD training labels.

#### 2.2 Cleaning decisions and their rationale

Use a compact table rather than a long procedural list.

| Decision | Observed issue | Action | Reason/impact |
|---|---|---|---|
| Sentinel ratings | `rating = -1` | Remove from explicit-rating dataset | “Watched but not rated” is neither dislike nor a 1--10 score. |
| Invalid/missing IDs or ratings | Validation check | Remove if present and audit | A collaborative-filtering row needs a valid user, item, and score. |
| Repeated `(user_id, anime_id)` keys | 7 duplicate rows | Average valid ratings into one pair | Avoid arbitrary row selection and enforce one user-item signal. |
| Metadata gaps | Missing/unparseable display fields | Preserve missing values; use output fallbacks | Do not manufacture genre/title/average-rating information. |
| Long-tail activity | Unequal user/item counts | Preserve in Notebook 01 | It is signal structure, not a numeric outlier; later model filters must be separately reported. |

Write explicitly that duplicate averaging can yield fractional valid ratings (observed 6.5 and 8.5), while remaining within [1, 10].  This shows the decision was understood, not mechanically copied.

#### 2.3 Final prepared dataset and reproducibility

- Present the before/after counts in Table 1.
- State outputs and strict boundary: Notebook 01 exports clean ratings and display metadata; it does not sample, k-core filter, split, scale, or train SVD.
- State that this separation preserves the cleaned source baseline and makes later model-scope changes auditable.
- Mention deterministic design/validation assertions and CSV paths only if they are useful in the template's implementation section; avoid excessive path detail in the Dataset section.

**Exit criterion:** the reader understands not only what was done, but why each non-trivial cleaning decision is methodologically safe.

### Phase 3 -- draft EDA with decision-oriented captions

Include a brief opening statement: the purpose is to characterize the clean explicit-rating matrix for modeling, not to claim model quality.

#### Table 1 -- Data preparation summary

Suggested caption: **“Data preparation audit for the explicit-rating matrix.”**  Include raw interactions, `-1` removed, invalid/missing removed, duplicates merged, clean interactions, unique users, unique rated anime, metadata anime, and matrix density.  Add a note defining density as interactions divided by users times rated anime.

#### Figure 1 -- explicit rating distribution

Use [eda_rating_distribution.png](../outputs/figures/eda_rating_distribution.png).

Suggested caption: **“Distribution of explicit user ratings after excluding watched-but-unrated (`-1`) interactions.”**

Interpret in 2--3 sentences: ratings skew positive (mean 7.81; mode 8; 82.55% at least 7 according to Notebook 01).  A high predicted score should be interpreted in the context of this positively skewed scale; it is not evidence of universal anime quality.

#### Figure 2 -- user rating activity

Use [eda_user_rating_count_distribution.png](../outputs/figures/eda_user_rating_count_distribution.png).

Suggested caption: **“Long-tail distribution of explicit-rating counts per user (logarithmic frequency axis).”**

Interpret in 2--3 sentences: median history is 45 ratings, while P90/P99 are 230/640.  This motivates reporting model coverage and later minimum-interaction filtering separately; it does not justify deleting active users as outliers.

#### Optional Figure 3 -- popularity concentration

Use [eda_top_10_anime_by_rating_count.png](../outputs/figures/eda_top_10_anime_by_rating_count.png) only if the template has room or the lecturer expects three EDA visuals.  Explain that it indicates interaction concentration/popularity bias and weaker signal for less-rated titles.  Never label it an anime-quality ranking.

**Exit criterion:** each selected chart has a numbered caption, is cited in prose, and leads to one modeling/evaluation implication.

### Phase 4 -- provide the implementation handoff

Write one compact paragraph for the report's Implementation section:

> Notebook 01 loads and validates the raw CSV files, audits missingness/duplicates/sentinel ratings, creates the two stable clean CSV outputs, and saves the EDA figures.  Notebook 02 consumes only those outputs, then independently records any sampling and minimum-interaction filtering before the train/test split.  This separation prevents model-specific resource choices from being misrepresented as data cleaning.

Adjust wording to the actual final pipeline.  Supply Member 2 with the verified data counts and the non-negotiable definitions: `-1` exclusion, metadata display-only role, unique pair rule, and clean-vs-model-data distinction.

**Exit criterion:** Member 2 can write methodology without contradicting the data section.

### Phase 5 -- limitations and references

Draft a data-limitation paragraph with only defensible points:

- explicit ratings omit much viewing behavior and exclude watched-but-unrated interactions;
- sparse, long-tail user/item histories imply uneven signal and possible popularity bias;
- metadata gaps are preserved rather than inferred, so some displayed recommendations need fallbacks;
- results after model sampling/k-core filtering cover that model subset, not automatically the full clean dataset.

Compile APA 7 input entries alphabetically.  At minimum, verify the dataset author/organization, publication/update year, title, `[Data set]`, platform, and URL from the dataset page.  Add a reference only when cited in the report; do not pad the bibliography.

**Exit criterion:** every citation resolves to an APA entry and every limitation is tied to the actual data/protocol.

### Phase 6 -- integration review

Before release to Member 2, check:

- all figures/tables have labels and text references;
- no result metrics from Notebook 02 are invented or copied from an unfinalized run;
- no sentence conflates public `anime_average_rating` with a user rating or SVD prediction;
- all counts align with the final Notebook 01 execution;
- contribution statement says Member 1 owns data collection/loading, cleaning/preprocessing, EDA, feature/data preparation, and APA compilation, while both members integrate/review;
- percentages are deliberately agreed with Member 2 and sum to 100% in both report and PPT.

## Suggested LaTeX asset mapping

First copy/upload selected artifact files into the Overleaf project's `figures/` folder and confirm that the template entry file can see them.  Confirm actual paths against the template before inserting.  A likely pattern is:

```latex
\includegraphics[width=0.88\linewidth]{figures/eda_rating_distribution.png}
\label{fig:eda-rating-distribution}
```

Use `\caption{...}` before `\label{...}` and refer to it as `Figure~\ref{fig:eda-rating-distribution}`.  If a section file is loaded using `\input`, paths are normally resolved from `main.tex`, not the section file; verify this with one image rather than guessing.

## Definition of done

- [ ] The mandatory source files were read in order and the template was inspected when available.
- [ ] Dataset/data-preparation draft explains decisions and their reasons, not just code steps.
- [ ] Table 1 and two figures are legible, labeled, referenced, and fact-checked.
- [ ] Claims about sparsity, positive skew, long tail, and popularity bias are bounded and evidence-led.
- [ ] The Notebook 01 -> Notebook 02 handoff is explicit and does not blur cleaning with model filtering.
- [ ] APA source data is ready, citations are traceable, and the contribution text is consistent with the assignment.
- [ ] Member 2 has received a concise evidence handoff for final report integration.
- [ ] The uploaded Overleaf project uses XeLaTeX, has no local/parent-directory paths, contains all required figures and bibliography/style files, and recompiles without errors or unresolved references.
