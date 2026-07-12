import csv
from pathlib import Path
from uuid import uuid4

from pyspark.sql import DataFrame, SparkSession, functions as F

from .config import ANIME_HEADER, RATINGS_HEADER
from .schemas import RAW_ANIME_STRING_SCHEMA, RAW_RATINGS_STRING_SCHEMA


def _validate_header(path: Path, expected: list[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required raw input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        actual = next(csv.reader(handle), None)
    if actual != expected:
        raise ValueError(f"Unexpected CSV header for {path}: expected {expected}, got {actual}")


def load_raw_data(spark: SparkSession, ratings_path, anime_path) -> tuple[DataFrame, DataFrame]:
    """Load raw CSV as strings, then expose parsed fields and parse-failure flags."""
    ratings_path, anime_path = Path(ratings_path), Path(anime_path)
    _validate_header(ratings_path, RATINGS_HEADER)
    _validate_header(anime_path, ANIME_HEADER)

    ratings_strings = (spark.read.option("header", True).option("mode", "PERMISSIVE")
                       .schema(RAW_RATINGS_STRING_SCHEMA).csv(str(ratings_path)))
    ratings = ratings_strings.select(
        F.col("user_id").cast("long").alias("user_id"),
        F.col("anime_id").cast("long").alias("anime_id"),
        F.col("rating").cast("double").alias("rating"),
        (F.col("user_id").isNotNull() & F.col("user_id").cast("long").isNull()
         | F.col("anime_id").isNotNull() & F.col("anime_id").cast("long").isNull()
         | F.col("rating").isNotNull() & F.col("rating").cast("double").isNull()).alias("invalid_schema"),
    )

    anime_strings = (spark.read.option("header", True).option("mode", "PERMISSIVE")
                     .schema(RAW_ANIME_STRING_SCHEMA).csv(str(anime_path)))
    anime = anime_strings.select(
        F.col("anime_id").cast("long").alias("anime_id"),
        "name", "genre", "type", "episodes",
        F.col("rating").alias("community_rating_raw"),
        F.col("members").alias("members_raw"),
        (F.col("anime_id").isNotNull() & F.col("anime_id").cast("long").isNull()).alias("invalid_schema"),
    )
    return ratings, anime


def write_parquet_safe(df: DataFrame, path, key_columns: list[str]) -> DataFrame:
    """Write one contract table, read it back, and verify schema/count/key count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    before_count = df.count()
    before_keys = df.select(*key_columns).distinct().count()
    schema_signature = [(field.name, field.dataType.simpleString()) for field in df.schema.fields]
    temp_path = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    df.write.mode("overwrite").parquet(str(temp_path))
    readback = df.sparkSession.read.parquet(str(temp_path))
    if readback.count() != before_count or readback.select(*key_columns).distinct().count() != before_keys:
        raise AssertionError(f"Parquet read-back count/key mismatch for {path}")
    readback_signature = [(field.name, field.dataType.simpleString()) for field in readback.schema.fields]
    if readback_signature != schema_signature:
        raise AssertionError(f"Parquet read-back schema mismatch for {path}")
    # Spark/Hadoop rename is safer than deleting the processed parent. Existing target alone is replaced.
    jvm = df.sparkSession._jvm
    conf = df.sparkSession._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(conf)
    target = jvm.org.apache.hadoop.fs.Path(str(path))
    temp = jvm.org.apache.hadoop.fs.Path(str(temp_path))
    if fs.exists(target):
        fs.delete(target, True)
    if not fs.rename(temp, target):
        raise OSError(f"Could not atomically move {temp_path} to {path}")
    return df.sparkSession.read.parquet(str(path))
