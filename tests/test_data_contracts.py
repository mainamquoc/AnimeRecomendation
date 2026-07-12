import pytest

from src.audit import assert_clean_contract, validate_clean_ratings
from src.data_io import write_parquet_safe


def test_validator_accepts_valid_data(spark):
    ratings = spark.createDataFrame([(1, 10, 8.0), (1, 11, 9.0)], "user_id long, anime_id long, rating double")
    anime = spark.createDataFrame([(10,), (11,)], "anime_id long")
    checks = validate_clean_ratings(ratings, anime)
    assert checks["passed"] is True


@pytest.mark.parametrize("rows", [
    [(1, 10, 11.0)],
    [(None, 10, 8.0)],
    [(1, 10, 8.0), (1, 10, 8.0)],
    [(1, 999, 8.0)],
])
def test_validator_rejects_contract_violations(spark, rows):
    ratings = spark.createDataFrame(rows, "user_id long, anime_id long, rating double")
    anime = spark.createDataFrame([(10,)], "anime_id long")
    assert validate_clean_ratings(ratings, anime)["passed"] is False


def test_roundtrip_preserves_schema_count_and_keys(spark, tmp_path):
    df = spark.createDataFrame([(1, 10, 8.0), (2, 11, 9.0)], "user_id long, anime_id long, rating double")
    readback = write_parquet_safe(df, tmp_path / "ratings.parquet", ["user_id", "anime_id"])
    assert readback.schema == df.schema
    assert readback.count() == 2

