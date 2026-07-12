from pyspark.sql import DataFrame, functions as F


def classify_cold_start(eval_df: DataFrame, train_df: DataFrame) -> DataFrame:
    users = train_df.select("user_id").distinct().withColumn("_user_seen", F.lit(True))
    items = train_df.select("anime_id").distinct().withColumn("_item_seen", F.lit(True))
    return (eval_df.join(users, "user_id", "left").join(items, "anime_id", "left")
            .withColumn("reason", F.when(F.col("_user_seen").isNull() & F.col("_item_seen").isNull(), "both")
                        .when(F.col("_user_seen").isNull(), "user_unseen")
                        .when(F.col("_item_seen").isNull(), "item_unseen")
                        .otherwise(F.lit(None).cast("string")))
            .drop("_user_seen", "_item_seen"))


def _summary_rows(df: DataFrame, split_name: str, train_df: DataFrame):
    classified = classify_cold_start(df, train_df)
    total = df.count()
    warm = classified.filter(F.col("reason").isNull()).count()
    user_counts = df.groupBy("user_id").count()
    item_counts = df.groupBy("anime_id").count()
    stats = df.agg(F.avg("rating").alias("mean"), F.stddev("rating").alias("std")).first()
    return [
        (split_name, "row_count", float(total), "rows"),
        (split_name, "distinct_users", float(df.select("user_id").distinct().count()), "users"),
        (split_name, "distinct_items", float(df.select("anime_id").distinct().count()), "items"),
        (split_name, "rating_mean", float(stats["mean"] or 0), "rating"),
        (split_name, "rating_std", float(stats["std"] or 0), "rating"),
        (split_name, "duplicate_pairs", float(total - df.select("user_id", "anime_id").distinct().count()), "pairs"),
        (split_name, "users_one_interaction", float(user_counts.filter("count = 1").count()), "users"),
        (split_name, "items_one_interaction", float(item_counts.filter("count = 1").count()), "items"),
        (split_name, "warm_rows", float(warm), "rows"),
        (split_name, "cold_rows", float(total - warm), "rows"),
        (split_name, "warm_coverage", float(warm / total if total else 0), "fraction"),
    ]


def validate_split(train_df: DataFrame, validation_df: DataFrame, test_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    spark = train_df.sparkSession
    rows = []
    for name, df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        rows.extend(_summary_rows(df, name, train_df))
    summary = spark.createDataFrame(rows, "split string, metric string, value double, unit string")
    cold = (classify_cold_start(validation_df, train_df).filter(F.col("reason").isNotNull()).withColumn("split", F.lit("validation"))
            .unionByName(classify_cold_start(test_df, train_df).filter(F.col("reason").isNotNull()).withColumn("split", F.lit("test"))))
    return summary, cold.select("split", "user_id", "anime_id", "rating", "reason")


def build_train_only_popularity(train_df: DataFrame, min_count: int = 5, shrinkage: float = 20.0) -> DataFrame:
    global_mean = train_df.agg(F.avg("rating")).first()[0]
    return (train_df.groupBy("anime_id").agg(F.count("*").alias("rating_count"), F.avg("rating").alias("mean_rating"))
            .filter(F.col("rating_count") >= min_count)
            .withColumn("popularity_score", (F.col("rating_count") * F.col("mean_rating") + F.lit(shrinkage * global_mean))
                        / (F.col("rating_count") + F.lit(shrinkage))))


def build_unseen_train_candidates(users_df: DataFrame, train_df: DataFrame,
                                  watched_unrated_df: DataFrame | None = None) -> DataFrame:
    """Candidate pairs from train items only, excluding train and optional watched history."""
    users = users_df.select("user_id").distinct()
    train_items = train_df.select("anime_id").distinct()
    candidates = users.crossJoin(train_items)
    seen = train_df.select("user_id", "anime_id").distinct()
    if watched_unrated_df is not None:
        seen = seen.unionByName(watched_unrated_df.select("user_id", "anime_id")).dropDuplicates()
    return candidates.join(seen, ["user_id", "anime_id"], "left_anti")
