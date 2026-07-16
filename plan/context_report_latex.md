# Context: LaTeX report for the Anime Recommendation System

## Purpose and operating rule

This file is the shared source of truth for planning and writing the final report.  The report should make a lecturer able to answer, quickly and confidently:

1. What practical recommendation problem is being solved, for whom, and why?
2. What data was used, what does each field mean, and which limitations matter?
3. Which decisions were made from evidence rather than by habit (especially the handling of `rating = -1`, duplicates, sparsity, and evaluation)?
4. How was information leakage prevented and how can the experiment be reproduced?
5. What do the metrics and examples actually support -- and what must not be claimed?
6. What did each student contribute?

The report is **not** a chronological notebook dump.  It is an evidence-led argument: problem -> data reality -> justified method -> reproducible implementation -> honest results -> limitations and next steps.

## Mandatory reading order for Codex and writers

Before editing any `.tex`, drafting report prose, or changing a report figure/table, read these files in this order:

1. [Instructions_Group_14.docx.md](Instructions_Group_14.docx.md) -- binding deliverables, required report sections, roles, and APA requirement.
2. This file -- narrative structure, evidence standards, and ownership boundaries.
3. [plan_01_eda_data_preparation.md](plan_01_eda_data_preparation.md) and the executed [01_eda_data_preparation.ipynb](../notebooks/01_eda_data_preparation.ipynb) -- authoritative EDA decisions and observed values.
4. [plan_02_svd_model_evaluation.md](plan_02_svd_model_evaluation.md), [plan_simple_svd.md](plan_simple_svd.md), and the executed [02_svd_model_evaluation.ipynb](../notebooks/02_svd_model_evaluation.ipynb) -- authoritative model and evaluation protocol/results.
5. `report_template.tex`, when it is supplied -- preserve its required order, title-page fields, packages, and compile instructions.  It is not currently present in this workspace, so do not invent a replacement template yet.
6. The actual generated artifacts referenced below.  Never copy run-dependent numbers from a plan if they disagree with the executed notebook or CSV outputs.

If any source conflicts, use this priority: assignment instructions/template -> executed code and its artifacts -> current plan -> this contextual guidance.  Flag the conflict instead of silently reconciling it.

## Overleaf-first compatibility contract

The final LaTeX source will be written and compiled on **Overleaf**, not on a local TeX installation.  Treat the uploaded Overleaf project as self-contained: it must compile after a collaborator opens the project and presses Recompile, without any files outside that project.

Use this portable project layout when the template does not prescribe another one:

```text
overleaf-report/
  main.tex                 <- template entry point; do not rename if template says otherwise
  references.bib           <- bibliography database, if the template uses BibTeX/Biber
  figures/
    eda_rating_distribution.png
    eda_user_rating_count_distribution.png
    ...
  sections/                <- optional; only if the template structure permits \input
    dataset_eda.tex
    methodology.tex
```

Rules that prevent upload/build failures:

- Use only project-relative paths such as `figures/eda_rating_distribution.png`; never use Windows paths, absolute paths, `C:`, backslashes, or `../outputs/...` in the submitted `.tex`.
- Upload every referenced figure, logo, `.bib`, and custom `.sty`/`.cls` file.  Use stable ASCII filenames with no spaces; do not rely on a local `data/` or `outputs/` directory being present in Overleaf.
- Follow the supplied template's root filename and bibliography system.  Do not mix `biblatex`/Biber with `natbib`/BibTeX, and do not add a second `\documentclass` or duplicate package imports.
- Select **XeLaTeX** in Overleaf because the assignment requires it.  Confirm that all Vietnamese names render correctly after upload.
- Prefer default TeX Live fonts (or the exact font mechanism already in the template).  A local font file or `fontspec` declaration that is not available in Overleaf is a build risk; test the title page immediately.
- Use `graphicx` with PNG/PDF/JPG figures.  Do not use SVG directly unless it has been converted or the template explicitly supports it.
- Avoid `minted` unless shell-escape is enabled and the template requires it; use `listings` or short verbatim/pseudocode instead.  Avoid packages requiring external executables or local scripts.
- Escape LaTeX special characters in normal text/URLs when not handled by `\url{}`: `%`, `&`, `_`, `#`, `{`, `}`, and `\\`.  Let a bibliography package/DOI field handle URLs where possible.
- Recompile from a clean state after each integration.  A green local preview is not sufficient if new assets or `.bib` changes have not been uploaded.

## What a strong report demonstrates

A strong undergraduate ML report makes a chain of reasoning visible.  Each claim should have one of four roles:

| Claim type | What the reader needs | Example in this project |
|---|---|---|
| Fact | Source, table, figure, or runtime output | 6,337,234 explicit ratings remain after cleaning. |
| Decision | Rule plus a domain/method reason | Exclude `-1` because it represents watched-but-unrated, not a low score. |
| Result | Exact protocol and metric scope | RMSE/MAE are from the held-out split; Precision@10/Recall@10 use the documented observed-item protocol. |
| Interpretation | Bounded implication, plus caveat | Sparse, long-tail interactions motivate collaborative filtering, but do not prove that a recommendation will be liked. |

Avoid two common weak patterns:

- A figure with no message: every selected visual needs a caption that states the takeaway and a nearby paragraph explaining why it changes the next decision.
- A metric with no protocol: every reported score needs its split, data scope, `K`/relevance threshold where relevant, and direction of “better”.

## Required report architecture

Use the assignment's required headings even if the template uses slightly different names.  The suggested allocation is a guide, not a substitute for the template's page limit.

| Required section | Reader question answered | Core contents and evidence | Main owner / draft input |
|---|---|---|---|
| Title page | Who, what, and under which course? | Duy Tan University, Da Nang; DS 423; project title; Group 14; both names and IDs. | Member 2 integrates; both verify. |
| Abstract | What was built and what was learned? | 150--220 words: problem, data, cleaning scope, SVD approach, four evaluated metrics (only after final run), and one limitation. No citations or unexplained detail. | Member 2 drafts after results; Member 1 verifies data facts. |
| 1. Introduction and problem statement | Why is Top-N anime recommendation useful and what is the task? | Streaming use case; predict/rank unseen anime from explicit ratings; project objective; 2--3 measurable success criteria. | Shared; Member 2 integrates. |
| 2. Dataset and data preparation | What data is trustworthy enough to model? | Kaggle source and table schemas; cleaning audit; `-1` meaning; duplicates; metadata boundary; sparsity and EDA evidence. | **Member 1**. |
| 3. Methodology | Why this approach and how is it evaluated fairly? | Collaborative filtering/SVD rationale and prediction equation; resource-aware sample/filter if used; train/test split; fixed configuration/tuning status; metrics and Top-10 protocol. | **Member 2**, using Member 1's data-boundary text. |
| 4. Implementation | Can another student run the system? | Notebook responsibilities, inputs/outputs, libraries, seed, validation checks, pipeline diagram/pseudocode, recommendation export. Summarize code; do not paste long code listings. | Shared; Member 1 owns Notebook 01 paragraph; Member 2 owns Notebook 02 paragraph. |
| 5. Results and evaluation | What happened under the stated protocol? | Cleaning/data summary table; selected EDA visual(s); RMSE/MAE and Precision@10/Recall@10 table/chart; sample Top-10 output; concise interpretation and limitations. | Member 2 integrates; Member 1 supplies EDA subsection. |
| 6. Individual contribution | Was the split fair and explicit? | Named tasks and percentages totaling 100%; note shared integration/review. Must match the PPT. | Both approve. |
| 7. Conclusion and future work | What can be claimed and what should improve next? | Answer objective, summarize evidence, limitations, feasible next steps (e.g., cold-start/hybrid genre fallback, validation, temporal/catalog evaluation). | Member 2; both review. |
| References | Can all external claims/data/code be traced? | Alphabetical APA 7 entries: dataset, Surprise/SVD documentation or paper, relevant recommender reference, other cited materials. | Member 1 compiles; Member 2 checks citations. |

## Recommended narrative flow

```text
Real streaming recommendation problem
  -> Explicit ratings + metadata dataset
  -> Cleaning decisions preserve a valid 1--10 signal
  -> EDA reveals sparse, skewed, long-tail data
  -> SVD learns user-item preference patterns
  -> Leakage-safe evaluation measures prediction and ranking quality
  -> Refit model ranks unseen anime for Top-10 examples
  -> Results are useful within stated data and cold-start limitations
```

This flow should be apparent from headings and transitions.  It shows process of thinking without narrating every cell execution.

## Evidence map and reporting rules

### Dataset/EDA facts currently available from Notebook 01

Use the executed notebook and generated files as the final source.  Its current runtime summary reports:

- raw interactions: 7,813,737;
- watched-but-unrated (`-1`) rows removed: 1,476,496 (18.90%);
- clean explicit interactions: 6,337,234;
- duplicate user-anime rows merged: 7;
- users: 69,600; rated anime: 9,927; metadata anime: 12,294;
- matrix density: 0.9172%; rating mean: 7.81; mode: 8;
- activity median/P90/P99: 45 / 230 / 640 ratings;
- one rated anime lacks metadata; incomplete metadata remains transparent rather than imputed.

These values should be verified after any re-run.  Describe `anime_average_rating` as public metadata only; it is **not** the user-rating target used by SVD.

### Report-ready EDA artifacts

| Artifact | Report use | Caption must communicate |
|---|---|---|
| [eda_rating_distribution.png](../outputs/figures/eda_rating_distribution.png) | Dataset section or results subsection | Explicit scores are concentrated at the high end after excluding `-1`; this affects interpretation of a high-score threshold. |
| [eda_top_10_anime_by_rating_count.png](../outputs/figures/eda_top_10_anime_by_rating_count.png) | Dataset section, optional if space permits | Interactions concentrate around popular titles; rating count is activity/popularity, not quality. |
| [eda_user_rating_count_distribution.png](../outputs/figures/eda_user_rating_count_distribution.png) | Dataset section | User histories are long-tailed; data support varies materially across users. |
| Cleaning summary/audit from Notebook 01 | Table 1 | Every discarded/merged record type, its count, and the defensible rule. |

Do not use all three EDA charts merely because they exist.  Prefer one cleaning/data-summary table plus two charts (rating distribution and user activity) in the report; move the popularity chart to the PPT or appendix if the template is short.  Every figure must be introduced in prose before it appears, cited as “Figure X”, legible in the PDF, and referred to after it appears.

### Model/evaluation evidence to add after Notebook 02 is final

Only report numbers produced by the final executed Notebook 02.  Preserve the exact data scope: cleaning -> sampling (if any) -> k-core filtering (if any) -> outer holdout.  Required evidence:

- exact `SVD_CONFIG`, seed, rating scale, data size after each filter, and train/test split;
- RMSE and MAE with “lower is better”;
- Precision@10 and Recall@10 with “higher is better”, relevance rule and eligible-user count;
- one error-metric/ranking chart and optionally actual-vs-predicted chart;
- a compact Top-10 example that identifies the chosen user rule and states that scores are predicted preferences, not verified ratings or objective quality.

Never compare metrics from different data scopes as if they were directly comparable.  Never claim exhaustive tuning when the fixed baseline was used.  Never call held-out observed-item ranking “full-catalog recommendation evaluation”.

## Tables, figures, equations, and code style

1. Number every table and figure consistently through LaTeX labels/references (`\label` and `\ref`).  Avoid manual numbers.
2. Each table needs a descriptive caption, units/definitions, and readable decimal precision (normally 2--4 decimals for model metrics; counts with thousands separators where template/style permits).
3. Keep visuals vector/PDF when practical; otherwise use the existing 300-dpi PNGs.  Do not stretch them or use screenshots of notebook output.
4. Include at most one concise SVD equation and define every symbol in plain language.  Equations support explanation, not mathematical decoration.
5. Replace code dumps with a pipeline figure or short pseudocode.  If the template has an appendix and the lecturer permits it, put only essential snippets there.
6. Use one language consistently in the body (recommended: English, because notebooks/figures and course deliverables already use English).  Vietnamese names must render correctly with XeLaTeX.
7. Use past tense for completed experiment facts, present tense for general method statements, and conditional/modal wording for limitations.

## LaTeX quality and submission gate

When `report_template.tex` becomes available, first inspect its compiler, required packages, placeholders, bibliography method, page constraints, and existing labels.  Fill it rather than replacing it.  Create/upload a self-contained Overleaf project using the compatibility contract above, set its compiler to XeLaTeX, then visually inspect the PDF for:

- exact university/course/group/member details on the title page;
- no overfull lines, clipped figures, broken references, `??`, or missing citations;
- captions, tables, figures, and font sizes readable at normal zoom;
- figures cited in the text and results consistent across report, notebook, and PPT;
- APA 7 references alphabetized and every in-text citation represented;
- contribution percentages totaling 100% and matching the PPT;
- no missing-file error, unsupported font, package conflict, or bibliography-engine warning in the Overleaf build log;
- both `.tex` and compiled `.pdf` included in the final submission.

## Drafting checklist for every paragraph

Before keeping a paragraph, ask:

- Does its first sentence make a claim rather than announce a section?
- Is the claim supported by a cited source, an artifact, or a stated method?
- Does it explain why the evidence matters for the next decision?
- Does it avoid an unsupported claim of accuracy, causality, personalization, or generalization?
- Would a reader who has not opened the notebooks understand the key decision?
