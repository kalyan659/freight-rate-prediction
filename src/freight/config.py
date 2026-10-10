"""Paths, column names and fixed settings. Everything else imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = ROOT / "artifacts"

TRAIN_FILE = DATA_DIR / "train_test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEMPLATE_FILE = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUT_FILE = DATA_DIR / "december_chart_inputs.csv"

PREDICTIONS_FILE = OUTPUT_DIR / "validation_predictions.csv"
DECEMBER_PREDICTIONS_FILE = OUTPUT_DIR / "december_chart_predictions.csv"
MODEL_FILE = ARTIFACT_DIR / "final_model.joblib"

ID = "load_id"
TARGET = "posted_rate"
DATE = "date"
CITY_COLS = [("pickup", "pickup_lat", "pickup_lon"), ("delivery", "delivery_lat", "delivery_lon")]

TREND_ORIGIN = "2025-01-01"   # day 0 of the linear time trend
WEIGHT_CAP = 47_500.0         # weights are censored at this value
SEED = 42

# Label filter: drop training rows whose rate-per-mile is this many times above/below the
# median of their (equipment x distance band) segment. EDA shows a clean gap: 98% of rows sit
# within 0.86-1.16x and the corrupted ones sit at 0.17-0.47x or 2.1-5.4x, with nothing in between.
OUTLIER_LOW = 0.6
OUTLIER_HIGH = 1.6
DISTANCE_BANDS = [0, 250, 500, 1000, 1500, 2000, 10_000]
