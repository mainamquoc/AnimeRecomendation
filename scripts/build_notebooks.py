"""Generate deterministic reader-facing notebooks; run from repository root."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def write(name, cells):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    notebook["cells"] = cells
    nbf.write(notebook, NOTEBOOKS / name)


write("01_data_preparation.ipynb", [
    nbf.v4.new_markdown_cell("""# Anime data preparation — Member 1

## Goal

Build and read-back verify the four canonical PySpark tables used by Member 2. The notebook is an orchestration and quality-check artifact; transformation logic lives in tested modules under `src/`."""),
    nbf.v4.new_markdown_cell("""## Setup

Run this notebook from the repository root. It uses explicit feedback only for ALS, keeps `rating=-1` separately, uses seed 42, and prohibits `community_rating` leakage."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import os
import sys

START_DIR = Path.cwd()
ROOT = START_DIR if (START_DIR / "src").exists() else START_DIR.parent
if not (ROOT / "src").exists() or not (ROOT / "database").exists():
    raise RuntimeError("Run this notebook from the repository root")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.config import RAW_RATINGS_PATH, RAW_ANIME_PATH, PROCESSED_DIR, SEED
from src.pipeline import create_spark, run_pipeline

for required in (RAW_RATINGS_PATH, RAW_ANIME_PATH):
    if not required.exists():
        raise FileNotFoundError(f"Required raw input is missing: {required}")
print({"seed": SEED, "ratings": str(RAW_RATINGS_PATH), "anime": str(RAW_ANIME_PATH)})"""),
    nbf.v4.new_markdown_cell("""## Steps

### 1. Run the declared-schema pipeline

The pipeline profiles raw inputs, normalizes catalogue keys/text, classifies ratings, aggregates duplicate pairs, removes orphans, validates reconciliation, writes each Parquet table atomically, and reads every output back."""),
    nbf.v4.new_code_cell("""spark = create_spark()
manifest = run_pipeline(spark)
print(json.dumps({k: v["rows"] for k, v in manifest["outputs"].items()}, indent=2))"""),
    nbf.v4.new_markdown_cell("""### 2. Inspect the long-format audit"""),
    nbf.v4.new_code_cell("""import pandas as pd
audit = pd.read_csv(PROCESSED_DIR / "data_quality_summary.csv")
audit"""),
    nbf.v4.new_markdown_cell("""## Checks

The following cell fails if the output validation recorded anything other than a clean model-input contract."""),
    nbf.v4.new_code_cell("""assert manifest["validation"]["passed"] is True
assert manifest["validation"]["duplicate_pairs"] == 0
assert manifest["validation"]["invalid_rows"] == 0
assert manifest["validation"]["orphan_rows"] == 0
print("All canonical output and read-back quality gates passed.")"""),
    nbf.v4.new_markdown_cell("""## Next steps

Member 2 splits only `clean_ratings` with seed 42, then runs `validate_split`. Split summaries, cold-start exclusions, and train-only popularity are intentionally deferred until that real split exists."""),
    nbf.v4.new_code_cell("spark.stop()"),
])


write("02_eda_clean_data.ipynb", [
    nbf.v4.new_markdown_cell("""# EDA of quality-controlled anime data

## tl;dr

This notebook is a read-only consumer. Executed aggregate outputs quantify the sentinel separation, sparsity/long-tail behavior, catalogue genre/type coverage, and the distinction between interaction popularity and catalogue membership."""),
    nbf.v4.new_markdown_cell("""## Context & Methods

Only canonical Parquet tables and their audit are read. Large interaction data remains in Spark; only small aggregates are converted to pandas. No canonical processed table is modified.

### Key Assumptions

- `-1` is watched-unrated and excluded from explicit labels.
- Genre is a fallback signal, not a replacement for fair ALS evaluation.
- Popularity shown here is descriptive and must not be reused as a test-aware baseline."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json
import os
import sys
import pandas as pd
from pyspark.sql import functions as F

START_DIR = Path.cwd()
ROOT = START_DIR if (START_DIR / "src").exists() else START_DIR.parent
if not (ROOT / "src").exists() or not (ROOT / "database").exists():
    raise RuntimeError("Run this notebook from the repository root")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, FIGURES_DIR
from src.eda import generate_figures
from src.pipeline import create_spark

required = [PROCESSED_DIR / f for f in ["clean_ratings.parquet", "clean_anime.parquet", "watched_unrated.parquet", "genre_features.parquet", "data_quality_summary.csv"]]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(f"Run notebook 01 first; missing outputs: {missing}")
spark = create_spark()"""),
    nbf.v4.new_markdown_cell("## Data"),
    nbf.v4.new_code_cell("""ratings = spark.read.parquet(str(PROCESSED_DIR / "clean_ratings.parquet"))
anime = spark.read.parquet(str(PROCESSED_DIR / "clean_anime.parquet"))
watched = spark.read.parquet(str(PROCESSED_DIR / "watched_unrated.parquet"))
audit = pd.read_csv(PROCESSED_DIR / "data_quality_summary.csv")
print({"explicit_rows": ratings.count(), "watched_unrated_pairs": watched.count(), "catalog_items": anime.count()})"""),
    nbf.v4.new_markdown_cell("## Results\n\n### Generate the six report-ready figures"),
    nbf.v4.new_code_cell("""figures = generate_figures(spark)
figures"""),
    nbf.v4.new_markdown_cell("### Long-tail percentiles and sparsity"),
    nbf.v4.new_code_cell("""user_counts = ratings.groupBy("user_id").count()
item_counts = ratings.groupBy("anime_id").count()
user_q = user_counts.approxQuantile("count", [0.5, 0.9, 0.99], 0.001)
item_q = item_counts.approxQuantile("count", [0.5, 0.9, 0.99], 0.001)
n_users = ratings.select("user_id").distinct().count()
n_items = ratings.select("anime_id").distinct().count()
n_rows = ratings.count()
sparsity = 1 - n_rows / (n_users * n_items)
summary = {"users": n_users, "items": n_items, "explicit_rows": n_rows, "matrix_sparsity": sparsity,
           "user_interaction_p50_p90_p99": user_q, "item_interaction_p50_p90_p99": item_q}
summary"""),
    nbf.v4.new_markdown_cell("""## Takeaways

- The sentinel and explicit-rating chart demonstrates why `-1` cannot be treated as zero or dislike.
- The user/item histograms and percentiles show a sparse, long-tail matrix, motivating latent-factor ALS and explicit cold-start coverage reporting.
- Genre/type coverage supports a separate content-based fallback for cold starts.
- Interaction count and catalogue members are different popularity concepts. Any evaluated baseline must be fitted on train only.
- RMSE/MAE alone cannot measure Top-N usefulness; Member 2 must also report Precision@K, Recall@K, and warm-row/candidate coverage."""),
    nbf.v4.new_code_cell("spark.stop()"),
])

print("Generated:", *(str(p) for p in sorted(NOTEBOOKS.glob("*.ipynb"))), sep="\n- ")
