from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_RATINGS_PATH = PROJECT_ROOT / "database" / "rating.csv"
RAW_ANIME_PATH = PROJECT_ROOT / "database" / "anime.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SEED = 42
POSITIVE_THRESHOLD = 8.0
TOP_K_VALUES = [5, 10]
UNKNOWN_TOKEN = "Unknown"
APP_NAME = "anime-member1-data-preparation"
SPARK_TIMEZONE = "UTC"

RATINGS_HEADER = ["user_id", "anime_id", "rating"]
ANIME_HEADER = ["anime_id", "name", "genre", "type", "episodes", "rating", "members"]

