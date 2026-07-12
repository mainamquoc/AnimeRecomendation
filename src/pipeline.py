import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from .audit import assert_clean_contract, audit_rows, make_run_id, profile_stage, validate_clean_ratings
from .cleaning import build_genre_features, clean_ratings, duplicate_metrics, finalize_anime, standardize_anime
from .config import (APP_NAME, PROCESSED_DIR, RAW_ANIME_PATH, RAW_RATINGS_PATH,
                     SEED, SPARK_TIMEZONE)
from .data_io import load_raw_data, write_parquet_safe


def create_spark() -> SparkSession:
    # The Windows Spark launcher splits absolute executable paths containing spaces.
    # Put the active venv first and let both driver/worker resolve the stable command name.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYSPARK_PYTHON"] = "python"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "python"
    spark = (SparkSession.builder.appName(APP_NAME).master("local[4]")
             .config("spark.driver.memory", "2g")
             .config("spark.sql.session.timeZone", SPARK_TIMEZONE)
             .config("spark.sql.shuffle.partitions", "12")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path.relative_to(path.parents[1])), "bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def _write_single_csv(df, path: Path) -> None:
    temp = path.parent / f".{path.name}.spark-tmp"
    if temp.exists():
        shutil.rmtree(temp)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(temp))
    part = next(temp.glob("part-*.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    shutil.move(str(part), str(path))
    shutil.rmtree(temp)


def run_pipeline(spark: SparkSession, ratings_path=RAW_RATINGS_PATH, anime_path=RAW_ANIME_PATH,
                 processed_dir=PROCESSED_DIR) -> dict:
    ratings_path, anime_path, processed_dir = Path(ratings_path), Path(anime_path), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    raw_ratings, raw_anime = load_raw_data(spark, ratings_path, anime_path)
    raw_ratings.persist(StorageLevel.MEMORY_AND_DISK)
    raw_anime.persist(StorageLevel.MEMORY_AND_DISK)

    raw_audit = profile_stage("raw", raw_ratings, raw_anime, run_id)
    rating_counts = {int(row["rating"]): row["count"] for row in raw_ratings.groupBy("rating").count().collect()
                     if row["rating"] is not None and float(row["rating"]).is_integer()}
    raw_rating_stats = raw_ratings.agg(F.min("rating").alias("min"), F.max("rating").alias("max")).first()
    raw_detail_metrics = {
        "rating_min": (raw_rating_stats["min"], "rating", "raw minimum including sentinel"),
        "rating_max": (raw_rating_stats["max"], "rating", "raw maximum"),
        "invalid_schema_rows": (raw_ratings.filter("invalid_schema").count(), "rows", "unparsable declared-schema fields"),
    }
    for value in [-1, *range(1, 11)]:
        raw_detail_metrics[f"rating_value_{value}_rows"] = (rating_counts.get(value, 0), "rows", f"raw rating={value}")
    raw_audit = raw_audit.unionByName(audit_rows(spark, run_id, "raw", "ratings", raw_detail_metrics))
    raw_anime_details = {
        "duplicate_anime_id_rows": (raw_anime.count() - raw_anime.select("anime_id").distinct().count(), "rows", "catalog rows beyond unique IDs"),
        "missing_episodes": (raw_anime.filter(F.col("episodes").isNull() | (F.trim("episodes") == "")).count(), "rows", "empty raw episodes"),
        "missing_community_rating": (raw_anime.filter(F.col("community_rating_raw").isNull() | (F.trim("community_rating_raw") == "")).count(), "rows", "empty raw community rating"),
        "missing_members": (raw_anime.filter(F.col("members_raw").isNull() | (F.trim("members_raw") == "")).count(), "rows", "empty raw members"),
    }
    raw_audit = raw_audit.unionByName(audit_rows(spark, run_id, "raw", "anime", raw_anime_details))
    standardized_anime, rejected_anime = standardize_anime(raw_anime)
    standardized_anime.persist(StorageLevel.MEMORY_AND_DISK)
    clean, watched, rejected = clean_ratings(raw_ratings, standardized_anime.select("anime_id"))
    for df in (clean, watched, rejected):
        df.persist(StorageLevel.MEMORY_AND_DISK)
    clean_anime = finalize_anime(standardized_anime, clean).persist(StorageLevel.MEMORY_AND_DISK)
    genres = build_genre_features(clean_anime).persist(StorageLevel.MEMORY_AND_DISK)

    assert_clean_contract(clean, clean_anime, genres)
    duplicates = duplicate_metrics(raw_ratings)
    valid_id = (F.col("user_id").isNotNull() & (F.col("user_id") > 0)
                & F.col("anime_id").isNotNull() & (F.col("anime_id") > 0) & ~F.col("invalid_schema"))
    watched_raw_rows = raw_ratings.filter(valid_id & (F.col("rating") == -1)).count()
    watched_raw = raw_ratings.filter(valid_id & (F.col("rating") == -1)).select("user_id", "anime_id")
    watched_dedup_rows = watched_raw.dropDuplicates(["user_id", "anime_id"]).count()
    watched_orphan_rows = rejected.filter("reason = 'orphan_watched_unrated'").count()
    rejected_raw_rows = raw_ratings.filter(~(valid_id & ((F.col("rating") == -1) | F.col("rating").between(1, 10)))).count()
    raw_rows = raw_ratings.count()
    if raw_rows != watched_raw_rows + duplicates["explicit_valid_raw_rows"] + rejected_raw_rows:
        raise AssertionError("Raw rating classification reconciliation failed")
    orphan_explicit_rows = rejected.filter("reason = 'orphan_explicit'").count()
    if duplicates["explicit_after_dedup_rows"] != orphan_explicit_rows + clean.count():
        raise AssertionError("Explicit orphan reconciliation failed")

    metrics = {
        "watched_unrated_raw_rows": (watched_raw_rows, "rows", "rating=-1 before dedup/orphan removal"),
        "watched_unrated_duplicate_rows": (watched_raw_rows - watched_dedup_rows, "rows", "duplicate watched pairs removed"),
        "watched_unrated_orphan_rows": (watched_orphan_rows, "rows", "deduplicated watched pairs absent from catalog"),
        "rejected_invalid_id_or_rating_rows": (rejected_raw_rows, "rows", "invalid raw classification"),
        **{key: (value, "rows" if "pairs" not in key else "pairs", "duplicate policy metric")
           for key, value in duplicates.items()},
        "orphan_explicit_rows": (orphan_explicit_rows, "rows", "deduplicated explicit pairs absent from catalog"),
        "clean_ratings_rows": (clean.count(), "rows", "ALS-ready rows"),
        "clean_distinct_users": (clean.select("user_id").distinct().count(), "users", "ALS-ready users"),
        "clean_distinct_items": (clean.select("anime_id").distinct().count(), "items", "ALS-ready anime"),
        "clean_duplicate_pairs": (0, "pairs", "quality gate passed"),
        "clean_orphan_rows": (0, "rows", "quality gate passed"),
        "watched_unrated_rows": (watched.count(), "pairs", "deduplicated serving filter"),
        "rejected_anime_rows": (rejected_anime.count(), "rows", "invalid catalog IDs"),
    }
    transform_audit = audit_rows(spark, run_id, "clean", "ratings", metrics)
    parsed_audit = audit_rows(spark, run_id, "parsed", "ratings", {
        "classified_raw_rows": (raw_rows, "rows", "mutually exclusive classification reconciled"),
        "invalid_id_or_rating_rows": (rejected_raw_rows, "rows", "raw rows rejected before duplicate/orphan handling"),
    })
    before_dedup_audit = audit_rows(spark, run_id, "explicit_before_dedup", "ratings", {
        "row_count": (duplicates["explicit_valid_raw_rows"], "rows", "valid explicit raw rows"),
        "duplicate_pairs": (duplicates["duplicate_pairs"], "pairs", "pairs with more than one raw explicit row"),
        "conflicting_pairs": (duplicates["conflicting_pairs"], "pairs", "pairs with more than one distinct rating"),
    })
    after_dedup_audit = audit_rows(spark, run_id, "explicit_after_dedup", "ratings", {
        "row_count": (duplicates["explicit_after_dedup_rows"], "pairs", "one row per explicit pair before orphan removal"),
        "rows_removed_as_duplicate_or_aggregated": (duplicates["rows_removed_as_duplicate_or_aggregated"], "rows", "raw rows minus aggregated pairs"),
    })
    anime_clean_audit = audit_rows(spark, run_id, "clean", "anime", {
        "row_count": (clean_anime.count(), "rows", "normalized catalog"),
        "distinct_items": (clean_anime.select("anime_id").distinct().count(), "items", "unique normalized keys"),
        "episodes_missing": (clean_anime.filter("episodes_missing = 1").count(), "rows", "unparsable/missing/non-positive episodes"),
        "members_missing": (clean_anime.filter("members_missing = 1").count(), "rows", "invalid/missing members"),
        "community_rating_missing": (clean_anime.filter(F.col("community_rating").isNull()).count(), "rows", "missing/out-of-range aggregate rating"),
        "without_explicit_interaction": (clean_anime.filter(~F.col("has_explicit_interaction")).count(), "items", "serving-only catalog rows"),
    })

    outputs = {
        "clean_ratings": (clean, ["user_id", "anime_id"]),
        "watched_unrated": (watched, ["user_id", "anime_id"]),
        "clean_anime": (clean_anime, ["anime_id"]),
        "genre_features": (genres, ["anime_id"]),
    }
    readbacks = {}
    for name, (df, keys) in outputs.items():
        readbacks[name] = write_parquet_safe(df, processed_dir / f"{name}.parquet", keys)
    export_metrics = {}
    for name, df in readbacks.items():
        export_metrics[f"{name}_row_count"] = (df.count(), "rows", "read-back verified")
        export_metrics[f"{name}_distinct_keys"] = (
            df.select(*(outputs[name][1])).distinct().count(), "keys", "read-back verified")
    export_audit = audit_rows(spark, run_id, "export_readback", "all", export_metrics)
    audit = (raw_audit.unionByName(parsed_audit).unionByName(before_dedup_audit)
             .unionByName(after_dedup_audit).unionByName(transform_audit)
             .unionByName(anime_clean_audit).unionByName(export_audit)
             .orderBy("stage", "entity", "metric"))
    _write_single_csv(audit, processed_dir / "data_quality_summary.csv")

    manifest = {
        "run_id": run_id, "seed": SEED, "python": platform.python_version(),
        "spark": spark.version, "java": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
        "timezone": SPARK_TIMEZONE,
        "inputs": [_fingerprint(ratings_path), _fingerprint(anime_path)],
        "validation": validate_clean_ratings(clean, clean_anime),
        "rejected": {
            "reason_counts": {row["reason"]: row["count"] for row in rejected.groupBy("reason").count().collect()},
            "orphan_anime_ids": sorted(row["anime_id"] for row in rejected.filter(F.col("reason").startswith("orphan"))
                                       .select("anime_id").distinct().collect()),
        },
        "outputs": {name: {"path": str(processed_dir / f"{name}.parquet"), "rows": df.count(),
                           "schema": df.schema.jsonValue()} for name, df in readbacks.items()},
    }
    (processed_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ALS-ready anime recommendation datasets")
    parser.add_argument("--ratings", default=str(RAW_RATINGS_PATH))
    parser.add_argument("--anime", default=str(RAW_ANIME_PATH))
    parser.add_argument("--output", default=str(PROCESSED_DIR))
    args = parser.parse_args()
    spark = create_spark()
    try:
        manifest = run_pipeline(spark, args.ratings, args.anime, args.output)
        print(json.dumps({"status": "success", "run_id": manifest["run_id"],
                          "outputs": {k: v["rows"] for k, v in manifest["outputs"].items()}}, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
