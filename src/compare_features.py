
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = BASE_DIR / "dataset" / "features"

FULL_AUDIO_PATH = FEATURE_DIR / "acoustic_features.csv"
SPEECH_ONLY_PATH = FEATURE_DIR / "speech_only_acoustic_features.csv"

OUTPUT_DIR = FEATURE_DIR / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metadata columns: these are not acoustic predictors
METADATA_COLUMNS = [
    "filename",
    "emotion",
    "speaker_id",
    "gender",
    "sentence_id",
    "take"
]


# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

print("\nLoading feature datasets...")

full_df = pd.read_csv(FULL_AUDIO_PATH, encoding="utf-8-sig")
speech_df = pd.read_csv(SPEECH_ONLY_PATH, encoding="utf-8-sig")

print(f"Full-audio dataset: {full_df.shape}")
print(f"Speech-only dataset: {speech_df.shape}")


# --------------------------------------------------
# 3. BASIC VALIDATION
# --------------------------------------------------

if full_df["filename"].duplicated().any():
    raise ValueError("Duplicate filenames found in full-audio dataset!")

if speech_df["filename"].duplicated().any():
    raise ValueError("Duplicate filenames found in speech-only dataset!")

full_files = set(full_df["filename"])
speech_files = set(speech_df["filename"])

if full_files != speech_files:
    missing_in_speech = sorted(full_files - speech_files)
    missing_in_full = sorted(speech_files - full_files)

    print("\nFilename mismatch detected!")
    print("Missing in speech-only:", missing_in_speech)
    print("Missing in full-audio:", missing_in_full)

    raise ValueError("The two datasets do not contain identical recordings.")

print("\nAll 420 recording filenames match!")


# --------------------------------------------------
# 4. CHECK METADATA CONSISTENCY
# --------------------------------------------------

metadata_to_check = [
    "emotion",
    "speaker_id",
    "gender",
    "sentence_id",
    "take"
]

metadata_check = full_df[
    ["filename"] + metadata_to_check
].merge(
    speech_df[["filename"] + metadata_to_check],
    on="filename",
    suffixes=("_full", "_speech"),
    validate="one_to_one"
)

for col in metadata_to_check:
    if not (
        metadata_check[f"{col}_full"]
        .fillna("MISSING")
        .astype(str)
        .equals(
            metadata_check[f"{col}_speech"]
            .fillna("MISSING")
            .astype(str)
        )
    ):
        raise ValueError(f"Metadata mismatch found in: {col}")

print("All metadata values match!")


# --------------------------------------------------
# 5. IDENTIFY ACOUSTIC FEATURES
# --------------------------------------------------

numeric_columns = full_df.select_dtypes(
    include="number"
).columns.tolist()

feature_columns = [
    col for col in numeric_columns
    if col not in METADATA_COLUMNS
]

speech_numeric_columns = speech_df.select_dtypes(
    include="number"
).columns.tolist()

speech_feature_columns = [
    col for col in speech_numeric_columns
    if col not in METADATA_COLUMNS
]

if set(feature_columns) != set(speech_feature_columns):
    raise ValueError("Acoustic feature columns do not match!")

feature_columns = sorted(feature_columns)

print(f"\nNumber of acoustic features: {len(feature_columns)}")


# --------------------------------------------------
# 6. MERGE PAIRED FEATURE VALUES
# --------------------------------------------------

merged = full_df.merge(
    speech_df,
    on="filename",
    suffixes=("_full", "_speech"),
    validate="one_to_one"
)

paired_data = merged[
    ["filename", "emotion_full", "speaker_id_full"]
].copy()

paired_data = paired_data.rename(columns={
    "emotion_full": "emotion",
    "speaker_id_full": "speaker_id"
})

for feature in feature_columns:
    full_values = merged[f"{feature}_full"]
    speech_values = merged[f"{feature}_speech"]

    paired_data[f"{feature}_full"] = full_values
    paired_data[f"{feature}_speech"] = speech_values

    # Positive means the feature increased after silence removal
    paired_data[f"{feature}_difference"] = (
        speech_values - full_values
    )

paired_data.to_csv(
    OUTPUT_DIR / "paired_feature_values.csv",
    index=False
)


# --------------------------------------------------
# 7. OVERALL FEATURE COMPARISON
# --------------------------------------------------

summary_rows = []

for feature in feature_columns:

    full_values = merged[f"{feature}_full"]
    speech_values = merged[f"{feature}_speech"]

    differences = speech_values - full_values

    correlation = full_values.corr(speech_values)

    summary_rows.append({
        "feature": feature,

        "full_audio_mean": full_values.mean(),
        "speech_only_mean": speech_values.mean(),

        "full_audio_std": full_values.std(),
        "speech_only_std": speech_values.std(),

        "mean_difference": differences.mean(),
        "mean_absolute_difference": differences.abs().mean(),
        "median_absolute_difference": differences.abs().median(),

        "paired_correlation": correlation
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "overall_feature_comparison.csv",
    index=False
)


# --------------------------------------------------
# 8. EMOTION-WISE COMPARISON
# --------------------------------------------------

emotion_rows = []

for emotion in sorted(paired_data["emotion"].unique()):

    emotion_data = paired_data[
        paired_data["emotion"] == emotion
    ]

    for feature in feature_columns:

        full_values = emotion_data[f"{feature}_full"]
        speech_values = emotion_data[f"{feature}_speech"]

        differences = speech_values - full_values

        emotion_rows.append({
            "emotion": emotion,
            "feature": feature,

            "full_audio_mean": full_values.mean(),
            "speech_only_mean": speech_values.mean(),

            "mean_difference": differences.mean(),
            "mean_absolute_difference": differences.abs().mean()
        })

emotion_df = pd.DataFrame(emotion_rows)

emotion_df.to_csv(
    OUTPUT_DIR / "emotion_wise_comparison.csv",
    index=False
)


# --------------------------------------------------
# 9. FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 55)
print("PAIRED FEATURE COMPARISON COMPLETED")
print("=" * 55)

print(f"Total recordings compared: {len(merged)}")
print(f"Total acoustic features: {len(feature_columns)}")
print(f"Total emotions: {paired_data['emotion'].nunique()}")

print("\nAverage feature values:")

print(
    summary_df[
        [
            "feature",
            "full_audio_mean",
            "speech_only_mean",
            "mean_difference"
        ]
    ].round(4).to_string(index=False)
)

print("\nOutput files saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("- paired_feature_values.csv")
print("- overall_feature_comparison.csv")
print("- emotion_wise_comparison.csv")

print("\nComparison process finished successfully!")