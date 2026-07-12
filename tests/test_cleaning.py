from pyspark.sql import functions as F

from src.cleaning import build_genre_features, clean_ratings, finalize_anime, standardize_anime


def raw_anime_fixture(spark):
    rows = [
        (1, " Alpha ", "Action, action,  Drama ", " TV ", "12", "8.5", "100", False),
        (1, "Alpha", "Action, Drama", "TV", "12", "8.5", "100", False),
        (2, "Beta", None, None, "Unknown", "11", "-4", False),
        (3, "Gamma", "Unknown", "Movie", "1.5", "7", "20", False),
        (None, "Bad", "Action", "TV", "1", "5", "2", True),
    ]
    return spark.createDataFrame(rows, "anime_id long, name string, genre string, type string, episodes string, community_rating_raw string, members_raw string, invalid_schema boolean")


def raw_ratings_fixture(spark):
    rows = [
        (10, 1, -1.0, False), (10, 1, -1.0, False),
        (10, 1, 1.0, False), (10, 1, 1.0, False), (10, 1, 3.0, False),
        (11, 2, 10.0, False), (12, 999, 8.0, False),
        (13, 1, 0.0, False), (14, 1, None, False), (None, 1, 7.0, True),
    ]
    return spark.createDataFrame(rows, "user_id long, anime_id long, rating double, invalid_schema boolean")


def test_sentinel_duplicate_orphan_and_invalid_classification(spark):
    standardized, _ = standardize_anime(raw_anime_fixture(spark))
    clean, watched, rejected = clean_ratings(raw_ratings_fixture(spark), standardized.select("anime_id"))
    assert watched.collect() == [watched.first()]
    assert watched.first().asDict() == {"anime_id": 1, "user_id": 10}
    values = {(r.user_id, r.anime_id): r.rating for r in clean.collect()}
    assert values[(10, 1)] == 5.0 / 3.0
    assert values[(11, 2)] == 10.0
    assert -1.0 not in values.values() and 0.0 not in values.values()
    reasons = {r.reason for r in rejected.collect()}
    assert {"orphan_explicit", "invalid_rating", "invalid_id_or_schema"} <= reasons


def test_catalog_normalization_and_genres_are_deterministic(spark):
    standardized, rejected = standardize_anime(raw_anime_fixture(spark))
    assert standardized.count() == 3
    assert rejected.count() == 1
    item1 = standardized.filter("anime_id = 1").first()
    assert item1.name == "Alpha"
    assert item1.genres == ["Action", "Drama"]
    item2 = standardized.filter("anime_id = 2").first()
    assert item2.type == "Unknown" and item2.episodes_num is None
    assert item2.members is None and item2.community_rating is None
    clean, _, _ = clean_ratings(raw_ratings_fixture(spark), standardized.select("anime_id"))
    catalog = finalize_anime(standardized, clean)
    features = build_genre_features(catalog)
    all_tokens = [token for row in features.collect() for token in row.genres]
    assert "" not in all_tokens and "Unknown" not in all_tokens
    assert len(all_tokens) == len([token for row in features.collect() for token in set(row.genres)])

