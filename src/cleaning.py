from pyspark.sql import DataFrame, Window, functions as F

from .config import UNKNOWN_TOKEN


def _normalized_genres(column):
    tokens = F.transform(F.split(F.coalesce(column, F.lit("")), ","), lambda x: F.initcap(F.lower(F.trim(x))))
    return F.array_sort(F.array_distinct(F.filter(
        tokens, lambda x: (x != "") & (F.lower(x) != F.lower(F.lit(UNKNOWN_TOKEN)))
    )))


def standardize_anime(raw_anime: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Resolve catalogue keys deterministically and return standardized + rejected rows."""
    invalid = raw_anime.filter(F.col("anime_id").isNull() | (F.col("anime_id") <= 0)).withColumn(
        "reason", F.lit("invalid_anime_id")
    )
    valid = raw_anime.filter(F.col("anime_id").isNotNull() & (F.col("anime_id") > 0))
    name = F.when(F.length(F.trim("name")) > 0, F.trim("name"))
    genre_display = F.when(F.length(F.trim("genre")) > 0, F.trim("genre")).otherwise(F.lit(UNKNOWN_TOKEN))
    anime_type = F.when(F.length(F.trim("type")) > 0, F.trim("type")).otherwise(F.lit(UNKNOWN_TOKEN))
    episode_text = F.trim("episodes")
    episode_valid = episode_text.rlike(r"^[1-9][0-9]*$")
    members_text = F.trim("members_raw")
    members_valid = members_text.rlike(r"^[0-9]+$")
    rating_num = F.col("community_rating_raw").cast("double")

    normalized = valid.select(
        "anime_id", name.alias("name"), genre_display.alias("genre"),
        _normalized_genres(F.col("genre")).alias("genres"), anime_type.alias("type"),
        F.when(episode_valid, episode_text.cast("int")).alias("episodes_num"),
        F.when(episode_valid, 0).otherwise(1).cast("int").alias("episodes_missing"),
        F.when(members_valid, members_text.cast("long")).alias("members"),
        F.when(members_valid, 0).otherwise(1).cast("int").alias("members_missing"),
        F.when(members_valid, F.log1p(members_text.cast("double"))).alias("log_members"),
        F.when(rating_num.between(1.0, 10.0), rating_num).alias("community_rating"),
    )
    # Most complete row wins; lexical serialization is a deterministic tie-break.
    completeness = sum(F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in
                       ["name", "genre", "type", "episodes_num", "members", "community_rating"])
    tie = F.concat_ws("|", *[F.coalesce(F.col(c).cast("string"), F.lit("")) for c in normalized.columns[1:]])
    window = Window.partitionBy("anime_id").orderBy(completeness.desc(), tie.asc())
    standardized = normalized.withColumn("_rank", F.row_number().over(window)).filter("_rank = 1").drop("_rank")
    return standardized, invalid


def clean_ratings(raw_ratings: DataFrame, clean_anime_ids: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Classify, aggregate explicit pairs, and remove orphan interactions."""
    valid_id = (F.col("user_id").isNotNull() & (F.col("user_id") > 0)
                & F.col("anime_id").isNotNull() & (F.col("anime_id") > 0)
                & ~F.col("invalid_schema"))
    watched_raw = raw_ratings.filter(valid_id & (F.col("rating") == -1.0)).select("user_id", "anime_id")
    explicit_raw = raw_ratings.filter(valid_id & F.col("rating").between(1.0, 10.0)).select(
        "user_id", "anime_id", "rating"
    )
    rejected = raw_ratings.filter(~(valid_id & ((F.col("rating") == -1.0) | F.col("rating").between(1.0, 10.0)))).withColumn(
        "reason",
        F.when(~valid_id, F.lit("invalid_id_or_schema")).otherwise(F.lit("invalid_rating")),
    ).select("user_id", "anime_id", "rating", "reason")

    catalog = clean_anime_ids.select("anime_id").distinct()
    watched_dedup = watched_raw.dropDuplicates(["user_id", "anime_id"])
    watched_orphans = watched_dedup.join(catalog, "anime_id", "left_anti").withColumn("rating", F.lit(None).cast("double")).withColumn(
        "reason", F.lit("orphan_watched_unrated")
    ).select("user_id", "anime_id", "rating", "reason")
    watched = watched_dedup.join(catalog, "anime_id", "left_semi").select("user_id", "anime_id")

    deduped = explicit_raw.groupBy("user_id", "anime_id").agg(F.avg("rating").cast("double").alias("rating"))
    orphan_explicit = deduped.join(catalog, "anime_id", "left_anti").withColumn(
        "reason", F.lit("orphan_explicit")
    ).select("user_id", "anime_id", "rating", "reason")
    clean = deduped.join(catalog, "anime_id", "left_semi").select("user_id", "anime_id", "rating")
    return clean, watched, rejected.unionByName(watched_orphans).unionByName(orphan_explicit)


def finalize_anime(standardized_anime: DataFrame, clean_ratings_df: DataFrame) -> DataFrame:
    explicit_ids = clean_ratings_df.select("anime_id").distinct().withColumn("has_explicit_interaction", F.lit(True))
    result = standardized_anime.join(explicit_ids, "anime_id", "left").fillna(False, ["has_explicit_interaction"])
    missing_required = result.filter(F.col("has_explicit_interaction") & F.col("name").isNull()).limit(1).count()
    if missing_required:
        raise AssertionError("An anime with explicit interaction has an empty name")
    return result.withColumn("name", F.coalesce("name", F.lit(UNKNOWN_TOKEN))).select(
        "anime_id", "name", "genre", "genres", "type", "episodes_num", "episodes_missing",
        "members", "members_missing", "log_members", "community_rating", "has_explicit_interaction",
    )


def build_genre_features(clean_anime: DataFrame) -> DataFrame:
    return clean_anime.select("anime_id", F.array_sort(F.array_distinct(F.filter(
        "genres", lambda x: x.isNotNull() & (F.trim(x) != "") & (F.lower(x) != F.lower(F.lit(UNKNOWN_TOKEN)))
    ))).alias("genres"))


def duplicate_metrics(raw_ratings: DataFrame) -> dict[str, int]:
    explicit = raw_ratings.filter(
        (F.col("user_id") > 0) & (F.col("anime_id") > 0) & F.col("rating").between(1.0, 10.0)
    ).select("user_id", "anime_id", "rating")
    exact_unique = explicit.dropDuplicates()
    pair_stats = explicit.groupBy("user_id", "anime_id").agg(
        F.count("*").alias("n"), F.countDistinct("rating").alias("n_ratings")
    )
    raw_count, exact_count = explicit.count(), exact_unique.count()
    return {
        "explicit_valid_raw_rows": raw_count,
        "exact_duplicate_rows": raw_count - exact_count,
        "duplicate_pairs": pair_stats.filter("n > 1").count(),
        "conflicting_pairs": pair_stats.filter("n_ratings > 1").count(),
        "explicit_after_dedup_rows": pair_stats.count(),
        "rows_removed_as_duplicate_or_aggregated": raw_count - pair_stats.count(),
    }
