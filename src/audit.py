from datetime import datetime, timezone

from pyspark.sql import DataFrame, functions as F

from .schemas import AUDIT_SCHEMA


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def audit_rows(spark, run_id: str, stage: str, entity: str, metrics: dict) -> DataFrame:
    rows = []
    for metric, spec in metrics.items():
        if isinstance(spec, tuple):
            value, unit, note = spec
        else:
            value, unit, note = spec, "rows", ""
        rows.append((run_id, stage, entity, metric, float(value), unit, note))
    return spark.createDataFrame(rows, AUDIT_SCHEMA)


def profile_stage(stage_name: str, ratings_df: DataFrame, anime_df: DataFrame | None = None,
                  run_id: str | None = None) -> DataFrame:
    run_id = run_id or make_run_id()
    metrics = {
        "row_count": (ratings_df.count(), "rows", "physical rows at this stage"),
        "distinct_users": (ratings_df.select("user_id").distinct().count(), "users", "distinct user_id"),
        "distinct_items": (ratings_df.select("anime_id").distinct().count(), "items", "distinct anime_id"),
        "distinct_pair_count": (ratings_df.select("user_id", "anime_id").distinct().count(), "pairs", "distinct interaction grain"),
        "null_user_id": (ratings_df.filter(F.col("user_id").isNull()).count(), "rows", "user_id is null"),
        "null_anime_id": (ratings_df.filter(F.col("anime_id").isNull()).count(), "rows", "anime_id is null"),
        "null_rating": (ratings_df.filter(F.col("rating").isNull()).count(), "rows", "rating is null"),
    }
    result = audit_rows(ratings_df.sparkSession, run_id, stage_name, "ratings", metrics)
    if anime_df is not None:
        anime_metrics = {
            "row_count": (anime_df.count(), "rows", "catalogue rows"),
            "distinct_items": (anime_df.select("anime_id").distinct().count(), "items", "distinct anime_id"),
            "null_genre": (anime_df.filter(F.col("genre").isNull() | (F.trim("genre") == "")).count(), "rows", "raw missing genre"),
            "null_type": (anime_df.filter(F.col("type").isNull() | (F.trim("type") == "")).count(), "rows", "raw missing type"),
        }
        result = result.unionByName(audit_rows(ratings_df.sparkSession, run_id, stage_name, "anime", anime_metrics))
    return result


def validate_clean_ratings(ratings_df: DataFrame, anime_df: DataFrame) -> dict[str, bool | int | float]:
    row_count = ratings_df.count()
    duplicate_pairs = row_count - ratings_df.select("user_id", "anime_id").distinct().count()
    invalid_rows = ratings_df.filter(
        F.col("user_id").isNull() | F.col("anime_id").isNull() | F.col("rating").isNull()
        | (F.col("user_id") <= 0) | (F.col("anime_id") <= 0) | ~F.col("rating").between(1.0, 10.0)
    ).count()
    orphan_rows = ratings_df.join(anime_df.select("anime_id").distinct(), "anime_id", "left_anti").count()
    checks = {
        "row_count": row_count,
        "duplicate_pairs": duplicate_pairs,
        "invalid_rows": invalid_rows,
        "orphan_rows": orphan_rows,
        "schema_numeric": all(dict(ratings_df.dtypes).get(c) in {"bigint", "int", "double", "float"}
                              for c in ["user_id", "anime_id", "rating"]),
    }
    checks["passed"] = duplicate_pairs == invalid_rows == orphan_rows == 0 and checks["schema_numeric"]
    return checks


def assert_clean_contract(ratings_df: DataFrame, anime_df: DataFrame, genre_features: DataFrame) -> None:
    checks = validate_clean_ratings(ratings_df, anime_df)
    if not checks["passed"]:
        raise AssertionError(f"clean_ratings quality gate failed: {checks}")
    if anime_df.count() != anime_df.select("anime_id").distinct().count():
        raise AssertionError("clean_anime anime_id is not unique")
    bad_genres = genre_features.filter(F.exists("genres", lambda x: x.isNull() | (F.trim(x) == "") | (F.lower(x) == "unknown"))).count()
    if bad_genres or genre_features.count() != genre_features.select("anime_id").distinct().count():
        raise AssertionError("genre_features contract failed")

