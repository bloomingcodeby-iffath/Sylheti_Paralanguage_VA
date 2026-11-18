
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "features"
    / "speech_only_acoustic_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "features"
    / "model_results"
    / "LOSO"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

EXPECTED_SPEAKERS = [
    "arfat",
    "fariba",
    "joy",
    "wasima"
]

METADATA_COLUMNS = [
    "filename",
    "emotion",
    "speaker_id",
    "gender",
    "sentence_id",
    "take"
]

EXPECTED_FEATURE_COUNT = 34

# ==================================================
# 2. LOAD DATASET
# ==================================================

print("=" * 65)
print("SYLHETI EMOTION CLASSIFICATION - LOSO BASELINE")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded.")
print("Total recordings:", len(df))

# Check required columns

missing_columns = [
    col for col in METADATA_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing metadata columns: {missing_columns}"
    )

# Check dataset size

if len(df) != 840:
    raise ValueError("Expected exactly 840 recordings.")

# Check duplicate filenames

if df["filename"].duplicated().any():
    raise ValueError("Duplicate filenames detected.")

# Check speakers and emotions

if set(df["speaker_id"].unique()) != set(EXPECTED_SPEAKERS):
    raise ValueError("Unexpected speaker IDs detected.")

if set(df["emotion"].unique()) != set(EMOTIONS):
    raise ValueError("Unexpected emotion labels detected.")

# ==================================================
# 3. SELECT ONLY ACOUSTIC FEATURES
# ==================================================

# Explicitly exclude all metadata columns.
# sentence_id and take must NOT become model inputs.

feature_columns = [
    col for col in df.columns
    if col not in METADATA_COLUMNS
    and pd.api.types.is_numeric_dtype(df[col])
]

if len(feature_columns) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} acoustic features, "
        f"found {len(feature_columns)}: {feature_columns}"
    )

X = df[feature_columns].copy()
y = df["emotion"].copy()
groups = df["speaker_id"].copy()

# Check missing and infinite values

if X.isnull().any().any():
    raise ValueError("Missing feature values detected.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Infinite feature values detected.")

print("\nFeature validation passed.")
print("Acoustic features:", len(feature_columns))
print("Metadata excluded:", METADATA_COLUMNS)

# ==================================================
# 4. DEFINE MODELS
# ==================================================

models = {

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Dummy Uniform": DummyClassifier(
        strategy="uniform",
        random_state=42
    ),

    "Dummy Most Frequent": DummyClassifier(
        strategy="most_frequent"
    )
}

# ==================================================
# 5. LEAVE-ONE-SPEAKER-OUT
# ==================================================

logo = LeaveOneGroupOut()

fold_results = []
per_class_results = []
all_predictions = []

model_confusions = {
    name: np.zeros(
        (len(EMOTIONS), len(EMOTIONS)),
        dtype=int
    )
    for name in models
}

print("\nStarting LOSO evaluation...\n")

for fold_number, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_speakers = sorted(
        groups.iloc[train_idx].unique()
    )

    test_speaker = groups.iloc[test_idx].unique()

    if len(test_speaker) != 1:
        raise ValueError("Each fold must contain exactly one test speaker.")

    test_speaker = test_speaker[0]

    # Verify no speaker leakage

    if test_speaker in train_speakers:
        raise ValueError("Speaker leakage detected!")

    # Verify expected fold sizes

    if len(train_idx) != 630 or len(test_idx) != 210:
        raise ValueError("Unexpected train/test fold size.")

    test_gender_values = df.iloc[test_idx]["gender"].unique()

    if len(test_gender_values) != 1:
        raise ValueError("Inconsistent gender for test speaker.")

    test_gender = test_gender_values[0]

    print("=" * 65)
    print(f"FOLD {fold_number}: TEST SPEAKER = {test_speaker}")
    print("=" * 65)

    print("Training speakers:", train_speakers)
    print("Training recordings:", len(train_idx))
    print("Testing recordings:", len(test_idx))
    print("Test speaker gender:", test_gender)

    # Verify all seven classes in training and test

    if set(y_train.unique()) != set(EMOTIONS):
        raise ValueError("Training fold is missing an emotion.")

    if set(y_test.unique()) != set(EMOTIONS):
        raise ValueError("Test fold is missing an emotion.")

    # ==================================================
    # 6. TRAIN AND EVALUATE EACH MODEL
    # ==================================================

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=EMOTIONS,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            labels=EMOTIONS,
            average="weighted",
            zero_division=0
        )

        print(f"\n{model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(f"Weighted-F1: {weighted_f1:.4f}")

        fold_results.append({
            "fold": fold_number,
            "test_speaker": test_speaker,
            "gender": test_gender,
            "model": model_name,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        })

        # Per-class performance

        report = classification_report(
            y_test,
            predictions,
            labels=EMOTIONS,
            output_dict=True,
            zero_division=0
        )

        for emotion in EMOTIONS:

            per_class_results.append({
                "fold": fold_number,
                "test_speaker": test_speaker,
                "gender": test_gender,
                "model": model_name,
                "emotion": emotion,
                "precision": report[emotion]["precision"],
                "recall": report[emotion]["recall"],
                "f1_score": report[emotion]["f1-score"],
                "support": report[emotion]["support"]
            })

        # Confusion matrix

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=EMOTIONS
        )

        model_confusions[model_name] += cm

        # Save individual predictions

        for filename, actual, predicted in zip(
            df.iloc[test_idx]["filename"],
            y_test,
            predictions
        ):

            all_predictions.append({
                "fold": fold_number,
                "test_speaker": test_speaker,
                "gender": test_gender,
                "model": model_name,
                "filename": filename,
                "actual_emotion": actual,
                "predicted_emotion": predicted,
                "correct": actual == predicted
            })


# ==================================================
# 7. SAVE FOLD RESULTS
# ==================================================

results_df = pd.DataFrame(fold_results)

results_df.to_csv(
    OUTPUT_DIR / "loso_fold_metrics.csv",
    index=False
)

per_class_df = pd.DataFrame(per_class_results)

per_class_df.to_csv(
    OUTPUT_DIR / "loso_per_class_report.csv",
    index=False
)

predictions_df = pd.DataFrame(all_predictions)

predictions_df.to_csv(
    OUTPUT_DIR / "loso_predictions.csv",
    index=False
)

# ==================================================
# 8. MEAN AND STANDARD DEVIATION
# ==================================================

summary = (
    results_df
    .groupby("model")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std")
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "loso_model_summary.csv",
    index=False
)

# ==================================================
# 9. GENDER-WISE ANALYSIS
# ==================================================

gender_summary = (
    results_df[
        results_df["model"].isin(["SVM", "Random Forest"])
    ]
    .groupby(["model", "gender"])
    .agg(
        speaker_count=("test_speaker", "nunique"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean")
    )
    .reset_index()
)

gender_summary.to_csv(
    OUTPUT_DIR / "loso_gender_summary.csv",
    index=False
)

# ==================================================
# 10. SAVE POOLED CONFUSION MATRICES
# ==================================================

for model_name, cm in model_confusions.items():

    cm_df = pd.DataFrame(
        cm,
        index=EMOTIONS,
        columns=EMOTIONS
    )

    safe_name = model_name.lower().replace(" ", "_")

    cm_df.to_csv(
        OUTPUT_DIR / f"confusion_matrix_{safe_name}.csv"
    )

# ==================================================
# 11. FINAL REPORT
# ==================================================

print("\n")
print("=" * 65)
print("LOSO EXPERIMENT COMPLETED")
print("=" * 65)

print("\nFOLD RESULTS:")

print(
    results_df[
        ["fold", "test_speaker", "model", "accuracy", "macro_f1"]
    ].to_string(index=False)
)

print("\nMODEL SUMMARY: MEAN ± STD")

for _, row in summary.iterrows():

    print(f"\n{row['model']}")

    print(
        f"Accuracy: "
        f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
    )

    print(
        f"Weighted-F1: "
        f"{row['weighted_f1_mean']:.4f} ± "
        f"{row['weighted_f1_std']:.4f}"
    )

print("\nGENDER-WISE RESULTS:")
print(gender_summary.to_string(index=False))

print("\nSaved results in:")
print(OUTPUT_DIR)

print("\nOVERALL: LOSO BASELINE FINISHED")