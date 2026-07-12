# Anime Recommendation System — Member 1 Data Pipeline

This repository turns the CooperUnion Anime Recommendations Database into four quality-controlled PySpark tables for explicit-feedback ALS, serving filters, catalogue lookup, and content-based fallback.

## Environment

- Python 3.11.9
- Apache Spark / PySpark 4.1.2
- Java 17 (Microsoft OpenJDK 17.0.10 tested)
- Windows local Hadoop helper (`winutils.exe`) for Parquet output

Create and activate the environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

On Windows, set Java 17 and the project-local Hadoop helper before Spark commands:

```powershell
$env:JAVA_HOME = [Environment]::GetEnvironmentVariable("JAVA_HOME", "User")
$env:HADOOP_HOME = (Resolve-Path ".tools\hadoop").Path
$env:PATH = "$env:JAVA_HOME\bin;$env:HADOOP_HOME\bin;$env:PATH"
```

If the helper is absent, download it manually:

```powershell
New-Item -ItemType Directory -Force -Path .tools\hadoop\bin | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/winutils.exe" -OutFile ".tools\hadoop\bin\winutils.exe"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/hadoop.dll" -OutFile ".tools\hadoop\bin\hadoop.dll"
```

## Reproduce

Raw inputs must be present at `database/rating.csv` and `database/anime.csv`. Run:

```powershell
.\scripts\run_pipeline.ps1
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\jupyter-nbconvert.exe --execute --to notebook --inplace notebooks\01_data_preparation.ipynb --ExecutePreprocessor.timeout=1800
.\.venv\Scripts\jupyter-nbconvert.exe --execute --to notebook --inplace notebooks\02_eda_clean_data.ipynb --ExecutePreprocessor.timeout=900
```

The first notebook orchestrates and verifies canonical data preparation. The second is a read-only consumer of processed outputs and generates six figures under `outputs/figures/`.

## Contracts and restrictions

- `clean_ratings`: `user_id`, `anime_id`, `rating`; one row per pair; rating in `[1,10]`; all items exist in `clean_anime`.
- `watched_unrated`: unique watched pairs originally marked `-1`; never a label or negative example.
- `clean_anime`: deterministic normalized catalogue with parse/missing flags and `has_explicit_interaction`.
- `genre_features`: stable genre-token arrays; excludes empty/`Unknown` signals.
- `community_rating`, `members`, and full-data popularity are not model inputs for primary ALS evaluation.
- Seed is `42`; relevant items for Top-N have `rating >= 8.0`.
- Popularity must be computed from Member 2's train split only. Split validation helpers are in `src/split_validation.py`.

Large Parquet folders and the local environment are ignored by version control. Regenerate them with `scripts/run_pipeline.ps1`.
