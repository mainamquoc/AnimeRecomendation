from pyspark.sql.types import (
    ArrayType, BooleanType, DoubleType, IntegerType, LongType, StringType,
    StructField, StructType,
)

RAW_RATINGS_STRING_SCHEMA = StructType([
    StructField("user_id", StringType(), True),
    StructField("anime_id", StringType(), True),
    StructField("rating", StringType(), True),
])

RAW_ANIME_STRING_SCHEMA = StructType([
    StructField("anime_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("genre", StringType(), True),
    StructField("type", StringType(), True),
    StructField("episodes", StringType(), True),
    StructField("rating", StringType(), True),
    StructField("members", StringType(), True),
])

CLEAN_RATINGS_SCHEMA = StructType([
    StructField("user_id", LongType(), False),
    StructField("anime_id", LongType(), False),
    StructField("rating", DoubleType(), False),
])

WATCHED_UNRATED_SCHEMA = StructType([
    StructField("user_id", LongType(), False),
    StructField("anime_id", LongType(), False),
])

CLEAN_ANIME_SCHEMA = StructType([
    StructField("anime_id", LongType(), False),
    StructField("name", StringType(), False),
    StructField("genre", StringType(), True),
    StructField("genres", ArrayType(StringType(), False), False),
    StructField("type", StringType(), False),
    StructField("episodes_num", IntegerType(), True),
    StructField("episodes_missing", IntegerType(), False),
    StructField("members", LongType(), True),
    StructField("members_missing", IntegerType(), False),
    StructField("log_members", DoubleType(), True),
    StructField("community_rating", DoubleType(), True),
    StructField("has_explicit_interaction", BooleanType(), False),
])

GENRE_FEATURES_SCHEMA = StructType([
    StructField("anime_id", LongType(), False),
    StructField("genres", ArrayType(StringType(), False), False),
])

AUDIT_SCHEMA = StructType([
    StructField("run_id", StringType(), False),
    StructField("stage", StringType(), False),
    StructField("entity", StringType(), False),
    StructField("metric", StringType(), False),
    StructField("value", DoubleType(), False),
    StructField("unit", StringType(), False),
    StructField("rule_or_note", StringType(), False),
])

