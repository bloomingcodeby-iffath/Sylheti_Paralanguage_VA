
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "features" / "speech_only_acoustic_features.csv"
PLOT_DIR = BASE_DIR / "dataset" / "features" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# EMOTION COLORS
# Same color for each emotion in every graph
# --------------------------------------------------

emotion_order = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
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

# --------------------------------------------------
# GENERAL STYLE
# --------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "DejaVu Sans"
})


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Rows:", len(df))

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

if df[numeric_features].isnull().any().any():
    raise ValueError("Missing numerical values found.")

if not np.isfinite(df[numeric_features].to_numpy()).all():
    raise ValueError("Infinite values found.")

if "emotion" not in df.columns:
    raise ValueError("Emotion column is missing.")

df = df[df["emotion"].isin(emotion_order)].copy()

print("Numerical features:", len(numeric_features))
print("Emotion classes:", df["emotion"].nunique())


# --------------------------------------------------
# 1. EMOTION DISTRIBUTION
# --------------------------------------------------

emotion_counts = (
    df["emotion"]
    .value_counts()
    .reindex(emotion_order, fill_value=0)
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    emotion_counts.index,
    emotion_counts.values,
    color=[emotion_colors[e] for e in emotion_order],
    edgecolor="black",
    linewidth=0.7,
    width=0.65
)

plt.title("Distribution of Sylheti Emotion Dataset", fontweight="bold")
plt.xlabel("Emotion")
plt.ylabel("Number of Recordings")
plt.grid(axis="y", alpha=0.25)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        str(int(bar.get_height())),
        ha="center",
        fontweight="bold"
    )

save_plot("emotion_distribution.png")


# --------------------------------------------------
# 2. ACOUSTIC FEATURE BOXPLOTS BY EMOTION
# --------------------------------------------------

selected_features = [
    "pitch_mean",
    "rms_mean",
    "zcr_mean",
    "centroid_mean"
]

selected_features = [
    f for f in selected_features if f in numeric_features
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10)
)

axes = axes.flatten()

for i, feature in enumerate(selected_features):

    data = [
        df.loc[df["emotion"] == emotion, feature].dropna()
        for emotion in emotion_order
    ]

    bp = axes[i].boxplot(
        data,
        patch_artist=True,
        tick_labels=emotion_order,
        showfliers=False
    )

    for patch, emotion in zip(bp["boxes"], emotion_order):
        patch.set_facecolor(emotion_colors[emotion])
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    axes[i].set_title(feature.replace("_", " ").title(), fontweight="bold")
    axes[i].set_xlabel("Emotion")
    axes[i].set_ylabel(feature)
    axes[i].grid(axis="y", alpha=0.25)
    axes[i].tick_params(axis="x", rotation=25)

plt.suptitle(
    "Acoustic Feature Distribution Across Emotions",
    fontsize=16,
    fontweight="bold"
)

save_plot("acoustic_feature_boxplots.png")


# --------------------------------------------------
# 3. MFCC EMOTION HEATMAP
# --------------------------------------------------

mfcc_features = [
    f"mfcc_{i}_mean" for i in range(1, 14)
]

mfcc_features = [
    f for f in mfcc_features if f in numeric_features
]

mfcc_means = df.groupby("emotion")[mfcc_features].mean()
mfcc_means = mfcc_means.reindex(emotion_order)

plt.figure(figsize=(13, 6))

plt.imshow(
    mfcc_means.values,
    aspect="auto",
    cmap="viridis"
)

plt.colorbar(label="Mean MFCC Value")

plt.xticks(
    range(len(mfcc_features)),
    mfcc_features,
    rotation=45
)

plt.yticks(
    range(len(emotion_order)),
    emotion_order
)

plt.title(
    "MFCC Feature Patterns Across Emotions",
    fontweight="bold"
)

plt.xlabel("MFCC Features")
plt.ylabel("Emotion")

save_plot("mfcc_emotion_heatmap.png")


# --------------------------------------------------
# 4. ALL FEATURE PROFILES BY EMOTION
# --------------------------------------------------

emotion_feature_means = (
    df.groupby("emotion")[numeric_features]
    .mean()
    .reindex(emotion_order)
)

# Standardize each feature to make different scales comparable
scaler = StandardScaler()

standardized_means = scaler.fit_transform(
    emotion_feature_means
)

plt.figure(figsize=(18, 7))

plt.imshow(
    standardized_means,
    aspect="auto",
    cmap="coolwarm",
    vmin=-2,
    vmax=2
)

plt.colorbar(label="Standardized Mean Feature Value")

plt.yticks(
    range(len(emotion_order)),
    emotion_order
)

plt.xticks(
    range(len(numeric_features)),
    numeric_features,
    rotation=90
)

plt.title(
    "Standardized Acoustic Feature Profiles by Emotion",
    fontweight="bold"
)

plt.xlabel("Acoustic Features")
plt.ylabel("Emotion")

save_plot("emotion_feature_profile_heatmap.png")


# --------------------------------------------------
# 5. PCA VISUALIZATION BY EMOTION
# --------------------------------------------------

X = df[numeric_features].values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

plt.figure(figsize=(11, 8))

for emotion in emotion_order:

    subset = df[df["emotion"] == emotion]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        color=emotion_colors[emotion],
        label=emotion,
        alpha=0.75,
        s=38,
        edgecolors="white",
        linewidths=0.3
    )

variance = pca.explained_variance_ratio_ * 100

plt.title(
    "PCA Visualization of Sylheti Emotion Features",
    fontweight="bold"
)

plt.xlabel(f"PC1 ({variance[0]:.2f}% variance)")
plt.ylabel(f"PC2 ({variance[1]:.2f}% variance)")

plt.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(alpha=0.2)

print("\nPCA explained variance:")
print(f"PC1: {variance[0]:.2f}%")
print(f"PC2: {variance[1]:.2f}%")
print(f"Combined: {variance.sum():.2f}%")

save_plot("pca_emotion_visualization.png")


# --------------------------------------------------
# 6. EMOTION-WISE ACOUSTIC FEATURE COMPARISON
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feature in enumerate(selected_features):

    means = (
        df.groupby("emotion")[feature]
        .mean()
        .reindex(emotion_order)
    )

    bars = axes[i].bar(
        emotion_order,
        means.values,
        color=[emotion_colors[e] for e in emotion_order],
        edgecolor="black",
        linewidth=0.5
    )

    axes[i].set_title(
        f"Mean {feature.replace('_', ' ').title()} by Emotion",
        fontweight="bold"
    )

    axes[i].set_xlabel("Emotion")
    axes[i].set_ylabel("Mean Value")
    axes[i].tick_params(axis="x", rotation=25)
    axes[i].grid(axis="y", alpha=0.25)

plt.suptitle(
    "Emotion-Wise Acoustic Feature Comparison",
    fontsize=16,
    fontweight="bold"
)

save_plot("emotion_acoustic_feature_comparison.png")


# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\n" + "=" * 55)
print("EMOTION-BASED VISUALIZATION COMPLETED")
print("=" * 55)

print("Total recordings:", len(df))
print("Total numerical features:", len(numeric_features))
print("Total emotion classes:", len(emotion_order))

print("\nGenerated plots:")
for file in sorted(PLOT_DIR.glob("*.png")):
    print(file.name)

print("\nSaved in:", PLOT_DIR)
