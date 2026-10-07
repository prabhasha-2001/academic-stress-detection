"""Shared constants: paths, column names, ordinal maps and experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "cleaned_data.csv"       # survey export 
DATA_CLEAN = ROOT / "data" / "interim" / "cleaned.csv"        # after clean_data.py
DATA_LABELED = ROOT / "data" / "processed" / "labeled.csv"   # after compute_pss10_label.py
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
MODELS_DIR = RESULTS / "models"

SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 10

PSS_ITEMS = [f"pss{i}" for i in range(1, 11)]
# PSS-10 items 4, 5, 7, 8 are positively worded. 
# still contains the raw (un-reversed) responses. 
REVERSE_ITEMS = []

# PSS-10 total -> class 
BANDS = [(-1, 13, "Low"), (13, 26, "Moderate"), (26, 40, "High")]
CLASS_ORDER = ["Low", "Moderate", "High"]

# Ordered survey answers
ORDINAL_MAPS = {
    "screen_time": ["Less than 2 hours", "2–4 hours", "4–6 hours", "6–8 hours", "More than 8 hours"],
    "study_time": ["Less than 1 hour", "1–2 hours", "3–4 hours", "5–6 hours", "More than 6 hours"],
    "sleep_hours": ["Less than 5 hours", "5–6 hours", "6–7 hours", "7–8 hours", "More than 8 hours"],
    "device_before_sleep": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "phone_check_studying": ["Never", "Rarely", "Sometimes", "Often", "Very Often"],
    "weekly_assignments": ["0–1", "2–3", "4–5", "More than 5"],
    "exercise": ["Never", "Once a week", "2–3 times a week", "More than 3 times a week"],
    "social_activities": ["Never", "Rarely", "Sometimes", "Often", "Very Often"],
    "academic_year": ["First Year", "Second Year", "Third Year", "Fourth Year", "Other"],
}

NUMERIC_FEATURES = [
    "age", "academic_year", "screen_time", "study_time", "sleep_hours",
    "device_before_sleep", "phone_check_studying", "weekly_assignments",
    "academic_workload", "exam_pressure", "academic_satisfaction",
    "exercise", "social_activities", "is_horizon",
]
CATEGORICAL_FEATURES = ["gender", "faculty_group"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "stress_level"
