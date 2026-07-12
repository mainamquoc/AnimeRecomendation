from src.split_validation import (build_train_only_popularity, build_unseen_train_candidates,
                                  classify_cold_start, validate_split)


def test_classifies_all_cold_start_reasons(spark):
    train = spark.createDataFrame([(1, 10, 8.0), (2, 20, 9.0)], "user_id long, anime_id long, rating double")
    evaluation = spark.createDataFrame([
        (1, 10, 8.0), (3, 10, 8.0), (1, 30, 8.0), (3, 30, 8.0),
    ], "user_id long, anime_id long, rating double")
    reasons = {(r.user_id, r.anime_id): r.reason for r in classify_cold_start(evaluation, train).collect()}
    assert reasons == {(1, 10): None, (3, 10): "user_unseen", (1, 30): "item_unseen", (3, 30): "both"}
    summary, cold = validate_split(train, evaluation, evaluation)
    assert cold.count() == 6
    assert summary.filter("split = 'test' and metric = 'warm_coverage'").first().value == 0.25


def test_popularity_uses_only_passed_train(spark):
    train = spark.createDataFrame([(1, 10, 8.0), (2, 10, 10.0), (1, 20, 2.0)],
                                  "user_id long, anime_id long, rating double")
    result = build_train_only_popularity(train, min_count=1).orderBy("anime_id").collect()
    assert [r.rating_count for r in result] == [2, 1]


def test_sample_user_candidates_use_train_items_and_filter_all_seen(spark):
    train = spark.createDataFrame([(1, 10, 8.0), (2, 20, 9.0), (2, 30, 7.0)],
                                  "user_id long, anime_id long, rating double")
    users = spark.createDataFrame([(1,)], "user_id long")
    watched = spark.createDataFrame([(1, 20)], "user_id long, anime_id long")
    candidates = build_unseen_train_candidates(users, train, watched).collect()
    assert [(r.user_id, r.anime_id) for r in candidates] == [(1, 30)]
