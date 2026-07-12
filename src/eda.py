from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession, functions as F

from .config import FIGURES_DIR, PROCESSED_DIR


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_figures(spark: SparkSession, processed_dir=PROCESSED_DIR, figures_dir=FIGURES_DIR) -> dict:
    """Read only canonical outputs/audit and generate bounded aggregate plots."""
    processed_dir, figures_dir = Path(processed_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    ratings = spark.read.parquet(str(processed_dir / "clean_ratings.parquet"))
    anime = spark.read.parquet(str(processed_dir / "clean_anime.parquet"))
    audit = pd.read_csv(processed_dir / "data_quality_summary.csv")
    sns.set_theme(style="whitegrid")

    raw_distribution = audit[(audit.stage == "raw") & (audit.entity == "ratings")
                             & audit.metric.str.match(r"rating_value_-?\d+_rows")].copy()
    raw_distribution["rating"] = raw_distribution.metric.str.extract(r"rating_value_(-?\d+)_rows")[0].astype(int)
    distribution = raw_distribution.rename(columns={"value": "count"}).sort_values("rating")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=distribution, x="rating", y="count", ax=ax, color="#5B8FF9")
    ax.set(title="Raw watched-unrated sentinel vs. explicit rating distribution",
           xlabel="Rating (-1 = watched but not rated)", ylabel="Interaction rows")
    ax.ticklabel_format(style="plain", axis="y")
    _save(fig, figures_dir / "rating_distribution.png")

    user_counts = ratings.groupBy("user_id").count().select(F.log10("count").alias("log10_count")).toPandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(user_counts, x="log10_count", bins=45, ax=ax, color="#61DDAA")
    ax.set(title="Explicit interactions per user (log scale)", xlabel="log10(interactions per user)", ylabel="Users")
    _save(fig, figures_dir / "interactions_per_user.png")

    item_counts = ratings.groupBy("anime_id").count().select(F.log10("count").alias("log10_count")).toPandas()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(item_counts, x="log10_count", bins=45, ax=ax, color="#F6BD16")
    ax.set(title="Explicit interactions per anime (log scale)", xlabel="log10(interactions per anime)", ylabel="Anime")
    _save(fig, figures_dir / "interactions_per_anime.png")

    genres = (anime.select(F.explode("genres").alias("genre")).groupBy("genre").count()
              .orderBy(F.desc("count")).limit(15).toPandas())
    types = anime.groupBy("type").count().orderBy(F.desc("count")).limit(10).toPandas()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=genres, y="genre", x="count", ax=axes[0], color="#5D7092")
    axes[0].set(title="Top 15 genres in catalog", xlabel="Anime", ylabel="Genre")
    sns.barplot(data=types, y="type", x="count", ax=axes[1], color="#E8684A")
    axes[1].set(title="Anime type distribution", xlabel="Anime", ylabel="Type")
    _save(fig, figures_dir / "genres_and_types.png")

    raw_value = audit[(audit.stage == "raw") & (audit.entity == "ratings") & (audit.metric == "row_count")].value.iloc[0]
    clean_value = audit[(audit.stage == "clean") & (audit.metric == "clean_ratings_rows")].value.iloc[0]
    raw_users = audit[(audit.stage == "raw") & (audit.entity == "ratings") & (audit.metric == "distinct_users")].value.iloc[0]
    clean_users = audit[(audit.stage == "clean") & (audit.metric == "clean_distinct_users")].value.iloc[0]
    raw_items = audit[(audit.stage == "raw") & (audit.entity == "ratings") & (audit.metric == "distinct_items")].value.iloc[0]
    clean_items = audit[(audit.stage == "clean") & (audit.metric == "clean_distinct_items")].value.iloc[0]
    scale = pd.DataFrame({"metric": ["Interaction rows", "Users", "Items"] * 2,
                          "value": [raw_value, raw_users, raw_items, clean_value, clean_users, clean_items],
                          "stage": ["Raw"] * 3 + ["Clean ALS input"] * 3})
    issues = pd.DataFrame({
        "issue": ["Duplicate pairs", "Orphan rows", "Missing genre", "Missing type"] * 2,
        "value": [
            audit[(audit.stage == "explicit_before_dedup") & (audit.metric == "duplicate_pairs")].value.iloc[0],
            audit[(audit.stage == "clean") & (audit.metric.isin(["orphan_explicit_rows", "watched_unrated_orphan_rows"]))].value.sum(),
            audit[(audit.stage == "raw") & (audit.entity == "anime") & (audit.metric == "null_genre")].value.iloc[0],
            audit[(audit.stage == "raw") & (audit.entity == "anime") & (audit.metric == "null_type")].value.iloc[0],
            0, 0, 0, 0,
        ], "stage": ["Before"] * 4 + ["After"] * 4,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=scale, x="metric", y="value", hue="stage", ax=axes[0])
    axes[0].set(title="Dataset scale before/after cleaning", xlabel="Metric", ylabel="Count (log scale)")
    axes[0].set_yscale("log")
    sns.barplot(data=issues, x="issue", y="value", hue="stage", ax=axes[1])
    axes[1].set(title="Quality issues before/after cleaning", xlabel="Issue", ylabel="Affected rows/pairs")
    axes[1].tick_params(axis="x", rotation=20)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.0f", padding=2, fontsize=9)
    _save(fig, figures_dir / "data_quality_before_after.png")

    top_interactions = (ratings.groupBy("anime_id").count().join(anime.select("anime_id", "name", "members"), "anime_id")
                        .orderBy(F.desc("count")).limit(10).toPandas())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=top_interactions, y="name", x="count", ax=axes[0], color="#6DC8EC")
    axes[0].set(title="Top anime by explicit interaction count", xlabel="Explicit ratings", ylabel="Anime")
    top_members = anime.orderBy(F.desc_nulls_last("members")).select("name", "members").limit(10).toPandas()
    sns.barplot(data=top_members, y="name", x="members", ax=axes[1], color="#FF9D4D")
    axes[1].set(title="Top anime by catalog members", xlabel="Members", ylabel="Anime")
    _save(fig, figures_dir / "top_anime.png")

    return {p.stem: str(p) for p in sorted(figures_dir.glob("*.png"))}
