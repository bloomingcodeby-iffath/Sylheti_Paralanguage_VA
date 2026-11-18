
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

FULL_AUDIO_PATH = (
    BASE_DIR / "dataset" / "features" / "acoustic_features.csv"
)

SPEECH_ONLY_PATH = (
    BASE_DIR / "dataset" / "features" /
    "speech_only_acoustic_features.csv"
)

PLOT_DIR = (
    BASE_DIR / "dataset" / "features" / "comparison_plots"
)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

full_df = pd.read_csv(FULL_AUDIO_PATH)
speech_df = pd.read_csv(SPEECH_ONLY_PATH)

# Check matching recordings
if len(full_df) != len(speech_df):
    raise ValueError("Dataset sizes do not match!")

if set(full_df["filename"]) != set(speech_df["filename"]):
    raise ValueError("Recording filenames do not match!")

# Align rows by filename
full_df = full_df.sort_values("filename").reset_index(drop=True)
speech_df = speech_df.sort_values("filename").reset_index(drop=True)

if not (full_df["filename"] == speech_df["filename"]).all():
    raise ValueError("Dataset alignment failed!")

if not (full_df["emotion"] == speech_df["emotion"]).all():
    raise ValueError("Emotion labels do not match!")

selected_features = [
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
    "pitch_mean",
    "rms_mean",
    "zcr_mean"
]

emotion_colors = {
    "anger": "#E63946",
    "disgust": "#2A9D8F",
    "fear": "#9B5DE5",
    "happiness": "#F4A261",
    "neutral": "#457B9D",
    "sadness": "#4361EE",
    "surprise": "#F15BB5"
}

emotion_order = list(emotion_colors.keys())

dataset_colors = {
    "Full Audio": "#457B9D",
    "Speech-only": "#F4A261"
}

print("Generating comparative feature plots...\n")

for feature in selected_features:

    if feature not in full_df.columns or feature not in speech_df.columns:
        raise ValueError(f"Feature not found: {feature}")

    fig, ax = plt.subplots(figsize=(14, 7))

    positions = []
    plot_data = []
    colors = []

    for i, emotion in enumerate(emotion_order):

        full_values = full_df.loc[
            full_df["emotion"] == emotion, feature
        ].dropna()

        speech_values = speech_df.loc[
            speech_df["emotion"] == emotion, feature
        ].dropna()

        base_position = i * 3

        positions.extend([
            base_position + 1,
            base_position + 2
        ])

        plot_data.extend([
            full_values,
            speech_values
        ])

        colors.extend([
            dataset_colors["Full Audio"],
            dataset_colors["Speech-only"]
        ])

    box = ax.boxplot(
        plot_data,
        positions=positions,
        patch_artist=True,
        widths=0.7,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
        flierprops=dict(
            marker="o",
            markersize=3,
            alpha=0.35
        )
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    centers = [i * 3 + 1.5 for i in range(len(emotion_order))]

    ax.set_xticks(centers)
    ax.set_xticklabels(emotion_order)

    ax.set_title(
        f"{feature.replace('_', ' ').title()} "
        "Comparison Across Emotions",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel("Emotion")
    ax.set_ylabel(feature)

    from matplotlib.patches import Patch

    legend_items = [
        Patch(
            facecolor=dataset_colors["Full Audio"],
            label="Full Audio"
        ),
        Patch(
            facecolor=dataset_colors["Speech-only"],
            label="Speech-only"
        )
    ]

    ax.legend(handles=legend_items, loc="best")

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_file = PLOT_DIR / f"{feature}_comparison.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_file.name)

print("\nAll comparative plots generated successfully!")
