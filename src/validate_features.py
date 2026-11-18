
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = (
    BASE_DIR / "dataset" / "features" /
    "speech_only_acoustic_features.csv"
)

EXPECTED_ROWS = 840
EXPECTED_COLUMNS = 40

EXPECTED_EMOTIONS = {
    "anger", "disgust", "fear", "happiness",
    "neutral", "sadness", "surprise"
}

EXPECTED_SPEAKERS = {
    "fariba", "wasima", "arfat", "joy"
}

METADATA_COLUMNS = [
    "filename", "emotion", "speaker_id",
    "gender", "sentence_id", "take"
]

print("\n" + "=" * 60)
print("ACOUSTIC FEATURE VALIDATION REPORT")
print("=" * 60)

if not FILE_PATH.exists():
    raise FileNotFoundError(f"Feature file not found: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)

# 1. Dataset shape
print("\n1. DATASET SHAPE")
print("-" * 40)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Expected rows:", EXPECTED_ROWS)
print("Expected columns:", EXPECTED_COLUMNS)

# 2. Required columns
print("\n2. REQUIRED COLUMN CHECK")
print("-" * 40)

missing_columns = [
    col for col in METADATA_COLUMNS
    if col not in df.columns
]

print("Missing metadata columns:", missing_columns)

# 3. Missing values
print("\n3. MISSING VALUE CHECK")
print("-" * 40)

missing_total = df.isnull().sum().sum()
print("Total missing values:", missing_total)

missing_by_column = df.isnull().sum()
print("\nColumns with missing values:")
print(missing_by_column[missing_by_column > 0])

# 4. Numeric feature validation
print("\n4. NUMERICAL FEATURE CHECK")
print("-" * 40)

feature_cols = [
    col for col in df.columns
    if col not in METADATA_COLUMNS
]

non_numeric = [
    col for col in feature_cols
    if not pd.api.types.is_numeric_dtype(df[col])
]

print("Numerical feature count:", len(feature_cols))
print("Non-numeric feature columns:", non_numeric)

# 5. Infinite values
print("\n5. INFINITE VALUE CHECK")
print("-" * 40)

numeric_df = df[feature_cols].select_dtypes(include=[np.number])
infinite_count = np.isinf(numeric_df.to_numpy()).sum()

print("Infinite values:", infinite_count)

# 6. Duplicate checks
print("\n6. DUPLICATE CHECK")
print("-" * 40)

print("Duplicate filenames:", df["filename"].duplicated().sum())
print("Duplicate complete rows:", df.duplicated().sum())

# 7. Emotion distribution
print("\n7. EMOTION DISTRIBUTION")
print("-" * 40)

emotion_counts = df["emotion"].value_counts().sort_index()
print(emotion_counts.to_string())

print("\nUnexpected emotions:",
      sorted(set(df["emotion"].dropna()) - EXPECTED_EMOTIONS))

# 8. Speaker distribution
print("\n8. SPEAKER DISTRIBUTION")
print("-" * 40)

speaker_counts = df["speaker_id"].value_counts().sort_index()
print(speaker_counts.to_string())

print("\nUnexpected speakers:",
      sorted(set(df["speaker_id"].dropna()) - EXPECTED_SPEAKERS))

# 9. Constant features
print("\n9. CONSTANT FEATURE CHECK")
print("-" * 40)

constant_features = [
    col for col in feature_cols
    if df[col].nunique(dropna=True) <= 1
]

print("Constant features:", constant_features)

# 10. Feature statistics
print("\n10. FEATURE STATISTICS")
print("-" * 40)

print(
    numeric_df.describe()
    .round(4)
    .to_string()
)

# 11. Final validation summary
print("\n" + "=" * 60)
print("FINAL VALIDATION SUMMARY")
print("=" * 60)

checks = {
    "Expected row count": len(df) == EXPECTED_ROWS,
    "Expected column count": len(df.columns) == EXPECTED_COLUMNS,
    "All metadata columns present": len(missing_columns) == 0,
    "34 numerical features": len(feature_cols) == 34,
    "All features numeric": len(non_numeric) == 0,
    "No missing values": missing_total == 0,
    "No infinite values": infinite_count == 0,
    "No duplicate filenames": df["filename"].duplicated().sum() == 0,
    "No duplicate rows": df.duplicated().sum() == 0,
    "No constant features": len(constant_features) == 0,
    "Correct emotion counts": (
        set(emotion_counts.index) == EXPECTED_EMOTIONS
        and (emotion_counts == 120).all()
    ),
    "Correct speaker counts": (
        set(speaker_counts.index) == EXPECTED_SPEAKERS
        and (speaker_counts == 210).all()
    )
}

for name, passed in checks.items():
    print(f"{'[PASS]' if passed else '[FAIL]'} {name}")

print("\nOverall result:",
      "PASSED" if all(checks.values()) else "NEEDS ATTENTION")

print("\nValidation completed.")
