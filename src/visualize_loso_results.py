
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. OUTPUT DIRECTORY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "features"
    / "model_results"
    / "LOSO"
    / "visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("SYLHETI EMOTION RECOGNITION - LOSO RESULT VISUALIZATION")
print("=" * 65)


# ============================================================
# 2. OVERALL MODEL RESULTS
# ============================================================

overall = pd.DataFrame({
    "Model": [
        "SVM",
        "Random Forest",
        "Dummy Uniform",
        "Dummy Most Frequent"
    ],
    "Accuracy": [
        0.1762,
        0.1667,
        0.1333,
        0.1429
    ],
    "Accuracy_SD": [
        0.0447,
        0.0421,
        0.0000,
        0.0000
    ],
    "Macro_F1": [
        0.1180,
        0.0914,
        0.1326,
        0.0357
    ],
    "Macro_F1_SD": [
        0.0459,
        0.0505,
        0.0000,
        0.0000
    ]
})

overall.to_csv(
    OUTPUT_DIR / "overall_model_results.csv",
    index=False
)


# ============================================================
# 3. SPEAKER-WISE RESULTS
# ============================================================

speaker_results = pd.DataFrame({
    "Speaker": [
        "Arfat",
        "Fariba",
        "Joy",
        "Wasima"
    ],
    "Gender": [
        "Male",
        "Female",
        "Male",
        "Female"
    ],
    "SVM_Accuracy": [
        0.195238,
        0.166667,
        0.223810,
        0.119048
    ],
    "SVM_Macro_F1": [
        0.138571,
        0.080853,
        0.173137,
        0.079412
    ],
    "RF_Accuracy": [
        0.223810,
        0.128571,
        0.171429,
        0.142857
    ],
    "RF_Macro_F1": [
        0.155551,
        0.032550,
        0.093993,
        0.083679
    ]
})

speaker_results.to_csv(
    OUTPUT_DIR / "speaker_wise_results.csv",
    index=False
)


# ============================================================
# 4. GENDER-WISE EXPLORATORY RESULTS
# ============================================================

gender_results = pd.DataFrame({
    "Gender": ["Female", "Male"],
    "SVM_Macro_F1": [0.080132, 0.155854],
    "RF_Macro_F1": [0.058114, 0.124772]
})

gender_results.to_csv(
    OUTPUT_DIR / "gender_wise_results.csv",
    index=False
)


# ============================================================
# 5. PER-EMOTION F1 RESULTS
# These are mean per-fold F1 values calculated from
# the reported fold-wise precision and recall.
# They are NOT pooled F1 scores.
# ============================================================

emotion_results = pd.DataFrame({
    "Emotion": [
        "Anger",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise"
    ],
    "SVM_F1": [
        0.247,
        0.030,
        0.043,
        0.133,
        0.065,
        0.143,
        0.000
    ],
    "RF_F1": [
        0.175,
        0.092,
        0.024,
        0.076,
        0.054,
        0.100,
        0.017
    ]
})

emotion_results.to_csv(
    OUTPUT_DIR / "emotion_wise_f1_results.csv",
    index=False
)


# ============================================================
# 6. SVM POOLED CONFUSION MATRIX
# Rows = Actual, Columns = Predicted
# ============================================================

emotions = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise"
]

svm_cm = np.array([
    [54, 8, 25, 11, 8, 9, 5],
    [34, 4, 32, 9, 15, 25, 1],
    [42, 4, 19, 3, 2, 50, 0],
    [28, 2, 23, 20, 27, 13, 7],
    [33, 2, 23, 12, 30, 20, 0],
    [34, 4, 25, 8, 28, 21, 0],
    [42, 9, 23, 13, 12, 21, 0]
])

cm_df = pd.DataFrame(
    svm_cm,
    index=emotions,
    columns=emotions
)

cm_df.to_csv(
    OUTPUT_DIR / "svm_pooled_confusion_matrix.csv"
)


# ============================================================
# 7. HELPER FUNCTION FOR SAVING GRAPHS
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# GRAPH 1: OVERALL ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    overall["Model"],
    overall["Accuracy"],
    yerr=overall["Accuracy_SD"],
    capsize=5
)

plt.title("Overall Accuracy Comparison - 4-Fold LOSO")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 0.30)
plt.xticks(rotation=15)

save_plot("01_overall_accuracy.png")


# ============================================================
# GRAPH 2: OVERALL MACRO-F1 COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    overall["Model"],
    overall["Macro_F1"],
    yerr=overall["Macro_F1_SD"],
    capsize=5
)

plt.title("Overall Macro-F1 Comparison - 4-Fold LOSO")
plt.ylabel("Macro-F1")
plt.xlabel("Model")
plt.ylim(0, 0.25)
plt.xticks(rotation=15)

save_plot("02_overall_macro_f1.png")


# ============================================================
# GRAPH 3: SPEAKER-WISE MACRO-F1
# ============================================================

x = np.arange(len(speaker_results))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    speaker_results["SVM_Macro_F1"],
    width,
    label="SVM"
)

plt.bar(
    x + width / 2,
    speaker_results["RF_Macro_F1"],
    width,
    label="Random Forest"
)

plt.xticks(x, speaker_results["Speaker"])
plt.ylabel("Macro-F1")
plt.xlabel("Held-out Test Speaker")
plt.title("Speaker-wise Macro-F1 - LOSO")
plt.ylim(0, 0.25)
plt.legend()

save_plot("03_speaker_wise_macro_f1.png")


# ============================================================
# GRAPH 4: ACCURACY VS MACRO-F1
# ============================================================

x = np.arange(len(overall))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    overall["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x + width / 2,
    overall["Macro_F1"],
    width,
    label="Macro-F1"
)

plt.xticks(x, overall["Model"], rotation=15)
plt.ylabel("Score")
plt.title("Accuracy vs Macro-F1")
plt.ylim(0, 0.25)
plt.legend()

save_plot("04_accuracy_vs_macro_f1.png")


# ============================================================
# GRAPH 5: SVM CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(svm_cm, interpolation="nearest")
plt.title("SVM Pooled Confusion Matrix - LOSO")
plt.colorbar(label="Number of Recordings")

plt.xticks(
    np.arange(len(emotions)),
    emotions,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(len(emotions)),
    emotions
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

threshold = svm_cm.max() / 2

for i in range(svm_cm.shape[0]):
    for j in range(svm_cm.shape[1]):
        plt.text(
            j,
            i,
            str(svm_cm[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if svm_cm[i, j] > threshold
                else "black"
            )
        )

save_plot("05_svm_confusion_matrix.png")


# ============================================================
# GRAPH 6: NORMALIZED SVM CONFUSION MATRIX
# Each row represents 100% of one actual emotion.
# ============================================================

row_sums = svm_cm.sum(axis=1, keepdims=True)

normalized_cm = svm_cm / row_sums

plt.figure(figsize=(10, 8))

plt.imshow(
    normalized_cm,
    interpolation="nearest",
    vmin=0,
    vmax=1
)

plt.title("SVM Normalized Confusion Matrix - LOSO")
plt.colorbar(label="Proportion")

plt.xticks(
    np.arange(len(emotions)),
    emotions,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(len(emotions)),
    emotions
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

for i in range(normalized_cm.shape[0]):
    for j in range(normalized_cm.shape[1]):
        plt.text(
            j,
            i,
            f"{normalized_cm[i, j]:.2f}",
            ha="center",
            va="center"
        )

save_plot("06_svm_normalized_confusion_matrix.png")


# ============================================================
# GRAPH 7: PER-EMOTION F1 COMPARISON
# ============================================================

x = np.arange(len(emotion_results))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    emotion_results["SVM_F1"],
    width,
    label="SVM"
)

plt.bar(
    x + width / 2,
    emotion_results["RF_F1"],
    width,
    label="Random Forest"
)

plt.xticks(x, emotion_results["Emotion"])
plt.ylabel("Mean Per-Fold F1")
plt.xlabel("Emotion")
plt.title("Per-Emotion F1 Comparison")
plt.ylim(0, 0.35)
plt.legend()

save_plot("07_emotion_wise_f1.png")


# ============================================================
# GRAPH 8: GENDER-WISE EXPLORATORY COMPARISON
# ============================================================

x = np.arange(len(gender_results))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    gender_results["SVM_Macro_F1"],
    width,
    label="SVM"
)

plt.bar(
    x + width / 2,
    gender_results["RF_Macro_F1"],
    width,
    label="Random Forest"
)

plt.xticks(x, gender_results["Gender"])
plt.ylabel("Mean Macro-F1")
plt.title("Gender-wise Exploratory Comparison")
plt.ylim(0, 0.22)
plt.legend()

save_plot("08_gender_wise_comparison.png")


# ============================================================
# 9. TABLE IMAGE GENERATOR
# ============================================================

def save_table(dataframe, filename, title):
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    ax.set_title(
        title,
        fontsize=14,
        pad=20
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# Overall table
save_table(
    overall.round(4),
    "09_overall_results_table.png",
    "Overall LOSO Model Results"
)

# Speaker-wise table
save_table(
    speaker_results.round(4),
    "10_speaker_wise_results_table.png",
    "Speaker-wise LOSO Results"
)

# Gender-wise table
save_table(
    gender_results.round(4),
    "11_gender_wise_results_table.png",
    "Gender-wise Exploratory Results"
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("VISUALIZATION COMPLETED")
print("=" * 65)

print("\nOverall Results:")
print(overall.round(4).to_string(index=False))

print("\nSpeaker-wise Results:")
print(speaker_results.round(4).to_string(index=False))

print("\nAll outputs saved to:")
print(OUTPUT_DIR)

print("\nImportant:")
print("- SVM confusion matrix uses pooled LOSO predictions.")
print("- Emotion-wise F1 contains mean per-fold F1 values.")
print("- Gender-wise results are exploratory only.")
print("- Dummy baseline SD is displayed as zero.")
print("=" * 65)
