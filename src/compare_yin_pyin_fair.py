
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR / "dataset" / "speech_only"
METADATA_PATH = BASE_DIR / "dataset" / "metadata.csv"

OUTPUT_DIR = BASE_DIR / "dataset" / "features" / "pitch_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "fair_yin_pyin_comparison.csv"
DIAGNOSTIC_PATH = OUTPUT_DIR / "frame_level_pitch_diagnostics.csv"


# ==========================================
# PITCH SETTINGS
# ==========================================

SR = 16000
FMIN = 65
FMAX = 400
FRAME_LENGTH = 2048
HOP_LENGTH = 256

# Frames with absolute difference above this
# threshold will be marked for inspection.
OUTLIER_THRESHOLD_HZ = 20


# ==========================================
# SELECT 14 PILOT RECORDINGS
# ==========================================

metadata = pd.read_csv(METADATA_PATH)

selected = (
    metadata
    .sort_values("filename")
    .groupby(["speaker_id", "emotion"], sort=True)
    .head(1)
    .reset_index(drop=True)
)

print("=" * 55)
print("FAIR YIN vs pYIN - FRAME LEVEL DIAGNOSTICS")
print("=" * 55)

print(f"Selected recordings: {len(selected)}")
print("\nProcessing...\n")


# ==========================================
# PROCESS RECORDINGS
# ==========================================

summary_results = []
diagnostic_results = []

for index, row in selected.iterrows():

    filename = row["filename"]
    speaker = row["speaker_id"]
    emotion = row["emotion"]

    audio_filename = Path(filename).stem + ".wav"
    audio_path = AUDIO_DIR / audio_filename

    try:

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio not found: {audio_path}"
            )

        # Load audio
        y, sr = librosa.load(
            audio_path,
            sr=SR,
            mono=True
        )

        # --------------------------------------
        # YIN
        # --------------------------------------

        yin_f0 = librosa.yin(
            y,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        # --------------------------------------
        # pYIN
        # --------------------------------------

        pyin_f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        # --------------------------------------
        # ALIGN OUTPUTS
        # --------------------------------------

        common_length = min(
            len(yin_f0),
            len(pyin_f0),
            len(voiced_flag),
            len(voiced_prob)
        )

        yin_f0 = yin_f0[:common_length]
        pyin_f0 = pyin_f0[:common_length]
        voiced_flag = voiced_flag[:common_length]
        voiced_prob = voiced_prob[:common_length]

        # Same pYIN-voiced frames for both methods
        valid_mask = (
            voiced_flag
            & np.isfinite(yin_f0)
            & np.isfinite(pyin_f0)
        )

        valid_indices = np.where(valid_mask)[0]

        yin_voiced = yin_f0[valid_mask]
        pyin_voiced = pyin_f0[valid_mask]

        if len(yin_voiced) < 2:
            raise ValueError(
                "Not enough common voiced frames"
            )

        # --------------------------------------
        # FRAME LEVEL DIAGNOSTICS
        # --------------------------------------

        differences = yin_voiced - pyin_voiced
        absolute_differences = np.abs(differences)

        for frame_position, frame_index in enumerate(valid_indices):

            yin_value = float(yin_voiced[frame_position])
            pyin_value = float(pyin_voiced[frame_position])

            signed_difference = (
                yin_value - pyin_value
            )

            absolute_difference = abs(
                signed_difference
            )

            frame_time = (
                frame_index * HOP_LENGTH / sr
            )

            diagnostic_results.append({

                "filename": filename,
                "speaker_id": speaker,
                "emotion": emotion,

                "frame_index": int(frame_index),

                "time_seconds": round(
                    float(frame_time), 4
                ),

                "yin_pitch_hz": round(
                    yin_value, 2
                ),

                "pyin_pitch_hz": round(
                    pyin_value, 2
                ),

                "signed_difference_hz": round(
                    float(signed_difference), 2
                ),

                "absolute_difference_hz": round(
                    float(absolute_difference), 2
                ),

                "voiced_probability": round(
                    float(voiced_prob[frame_index]), 4
                ) if np.isfinite(voiced_prob[frame_index]) else np.nan,

                "within_10hz": (
                    absolute_difference <= 10
                ),

                "within_20hz": (
                    absolute_difference <= 20
                ),

                "outlier": (
                    absolute_difference > OUTLIER_THRESHOLD_HZ
                )
            })

        # --------------------------------------
        # SUMMARY METRICS
        # --------------------------------------

        correlation = np.nan

        if (
            np.std(yin_voiced) > 0
            and np.std(pyin_voiced) > 0
        ):
            correlation = np.corrcoef(
                yin_voiced,
                pyin_voiced
            )[0, 1]

        rmse = float(
            np.sqrt(np.mean(differences ** 2))
        )

        summary_results.append({

            "filename": filename,
            "speaker_id": speaker,
            "emotion": emotion,

            "total_frames": common_length,

            "common_voiced_frames": len(yin_voiced),

            "voiced_percentage": round(
                len(yin_voiced) / common_length * 100, 2
            ),

            "yin_mean_same_frames": round(
                float(np.mean(yin_voiced)), 2
            ),

            "pyin_mean_same_frames": round(
                float(np.mean(pyin_voiced)), 2
            ),

            "mean_absolute_difference_hz": round(
                float(np.mean(absolute_differences)), 2
            ),

            "median_absolute_difference_hz": round(
                float(np.median(absolute_differences)), 2
            ),

            "rmse_hz": round(rmse, 2),

            "max_absolute_difference_hz": round(
                float(np.max(absolute_differences)), 2
            ),

            "frames_above_20hz": int(
                np.sum(absolute_differences > 20)
            ),

            "correlation": round(
                float(correlation), 3
            ) if np.isfinite(correlation) else np.nan,

            "within_10hz_percentage": round(
                float(np.mean(absolute_differences <= 10) * 100),
                2
            ),

            "within_20hz_percentage": round(
                float(np.mean(absolute_differences <= 20) * 100),
                2
            )
        })

        print(
            f"{index + 1}/{len(selected)} Processed: {filename} "
            f"| Common voiced frames: {len(yin_voiced)} "
            f"| MAE: {np.mean(absolute_differences):.2f} Hz "
            f"| Outliers: {np.sum(absolute_differences > 20)}"
        )

    except Exception as error:

        print(
            f"{index + 1}/{len(selected)} FAILED: {filename}"
        )

        print(f"Reason: {error}")


# ==========================================
# SAVE REPORTS
# ==========================================

summary_report = pd.DataFrame(summary_results)
diagnostic_report = pd.DataFrame(diagnostic_results)

summary_report.to_csv(
    SUMMARY_PATH,
    index=False
)

diagnostic_report.to_csv(
    DIAGNOSTIC_PATH,
    index=False
)


# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 55)
print("FRAME LEVEL DIAGNOSTICS COMPLETED")
print("=" * 55)

print(
    f"Successful recordings: {len(summary_report)}"
)

print(
    f"Expected recordings: {len(selected)}"
)

print(
    f"Total common voiced frames: {len(diagnostic_report)}"
)

if not summary_report.empty:

    print(
        "Average MAE:",
        round(
            summary_report[
                "mean_absolute_difference_hz"
            ].mean(),
            2
        ),
        "Hz"
    )

    print(
        "Average correlation:",
        round(
            summary_report["correlation"].mean(),
            3
        )
    )

    print(
        "Total frames above 20 Hz:",
        int(
            diagnostic_report["outlier"].sum()
        )
    )

    print("\nRecordings requiring inspection:")

    flagged = summary_report[
        (summary_report["mean_absolute_difference_hz"] > 10)
        | (summary_report["correlation"] < 0.8)
        | (summary_report["frames_above_20hz"] > 0)
    ]

    if flagged.empty:
        print("No recordings flagged by these criteria.")
    else:
        print(
            flagged[
                [
                    "filename",
                    "mean_absolute_difference_hz",
                    "correlation",
                    "frames_above_20hz"
                ]
            ].to_string(index=False)
        )

print("\nSummary report saved at:")
print(SUMMARY_PATH)

print("\nFrame-level diagnostic report saved at:")
print(DIAGNOSTIC_PATH)
