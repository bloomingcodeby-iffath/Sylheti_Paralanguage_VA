
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR / "dataset" / "speech_only"

DIAGNOSTIC_PATH = (
    BASE_DIR
    / "dataset"
    / "features"
    / "pitch_validation"
    / "frame_level_pitch_diagnostics.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "features"
    / "pitch_validation"
    / "pitch_contour_plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# SETTINGS
# ==========================================

SR = 16000
FMIN = 65
FMAX = 400
FRAME_LENGTH = 2048
HOP_LENGTH = 256

OUTLIER_THRESHOLD = 20


# ==========================================
# LOAD DIAGNOSTIC REPORT
# ==========================================

diagnostics = pd.read_csv(DIAGNOSTIC_PATH)

flagged_files = (
    diagnostics[
        diagnostics["outlier"] == True
    ]["filename"]
    .unique()
)

print("=" * 55)
print("PITCH CONTOUR VISUALIZATION")
print("=" * 55)

print(f"Flagged recordings: {len(flagged_files)}")
print("\nGenerating plots...\n")


# ==========================================
# PROCESS EACH FLAGGED RECORDING
# ==========================================

for index, filename in enumerate(flagged_files):

    audio_filename = Path(filename).stem + ".wav"
    audio_path = AUDIO_DIR / audio_filename

    if not audio_path.exists():
        print(f"Missing audio: {audio_filename}")
        continue

    # Load audio
    y, sr = librosa.load(
        audio_path,
        sr=SR,
        mono=True
    )

    # YIN
    yin_f0 = librosa.yin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    # pYIN
    pyin_f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    # Align
    common_length = min(
        len(yin_f0),
        len(pyin_f0),
        len(voiced_flag)
    )

    yin_f0 = yin_f0[:common_length]
    pyin_f0 = pyin_f0[:common_length]
    voiced_flag = voiced_flag[:common_length]

    frame_times = librosa.frames_to_time(
        np.arange(common_length),
        sr=sr,
        hop_length=HOP_LENGTH
    )

    valid_mask = (
        voiced_flag
        & np.isfinite(yin_f0)
        & np.isfinite(pyin_f0)
    )

    differences = np.full(
        common_length,
        np.nan
    )

    differences[valid_mask] = np.abs(
        yin_f0[valid_mask] - pyin_f0[valid_mask]
    )

    # --------------------------------------
    # CREATE FIGURE
    # --------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True
    )

    fig.suptitle(
        f"Pitch Diagnostic: {Path(filename).stem}",
        fontsize=14
    )

    # 1. Waveform
    audio_times = np.arange(len(y)) / sr

    axes[0].plot(
        audio_times,
        y,
        linewidth=0.6
    )

    axes[0].set_title("Original Speech Waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # 2. Pitch contour
    axes[1].plot(
        frame_times,
        yin_f0,
        label="YIN",
        linewidth=1.2
    )

    axes[1].plot(
        frame_times,
        pyin_f0,
        label="pYIN",
        linewidth=1.2
    )

    axes[1].set_title("YIN vs pYIN Pitch Contour")
    axes[1].set_ylabel("Pitch (Hz)")
    axes[1].set_ylim(FMIN - 10, FMAX + 20)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Absolute difference
    axes[2].plot(
        frame_times,
        differences,
        linewidth=1
    )

    axes[2].axhline(
        OUTLIER_THRESHOLD,
        linestyle="--",
        label="20 Hz inspection threshold"
    )

    axes[2].set_title("Absolute Pitch Difference")
    axes[2].set_ylabel("Difference (Hz)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{Path(filename).stem}_pitch_diagnostic.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"{index + 1}/{len(flagged_files)} Saved: "
        f"{output_path.name}"
    )


print("\n" + "=" * 55)
print("VISUALIZATION COMPLETED")
print("=" * 55)

print(f"Plots saved in:\n{OUTPUT_DIR}")
