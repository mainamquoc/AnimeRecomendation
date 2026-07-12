import csv
import json

from src.pipeline import run_pipeline


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_end_to_end_sample_exports_all_contracts(spark, tmp_path):
    ratings_path = tmp_path / "rating.csv"
    anime_path = tmp_path / "anime.csv"
    output = tmp_path / "processed"
    _write_csv(ratings_path, ["user_id", "anime_id", "rating"], [
        [1, 10, -1], [1, 10, -1], [1, 10, 8], [1, 10, 10],
        [2, 20, 1], [3, 999, 7], [4, 10, 0],
    ])
    _write_csv(anime_path, ["anime_id", "name", "genre", "type", "episodes", "rating", "members"], [
        [10, "Alpha", "Action, Action, Drama", "TV", "12", "8.5", "100"],
        [20, "Beta", "", "", "Unknown", "11", "-1"],
    ])
    manifest = run_pipeline(spark, ratings_path, anime_path, output)
    assert manifest["validation"]["passed"] is True
    assert manifest["outputs"]["clean_ratings"]["rows"] == 2
    assert manifest["outputs"]["watched_unrated"]["rows"] == 1
    assert {p.name for p in output.iterdir()} >= {
        "clean_ratings.parquet", "clean_anime.parquet", "watched_unrated.parquet",
        "genre_features.parquet", "data_quality_summary.csv", "run_manifest.json",
    }
    saved = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["inputs"][0]["sha256"]
