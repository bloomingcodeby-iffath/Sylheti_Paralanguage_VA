
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_DIR / "dataset" / "processed"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "features" / "audio_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "clipping_report.csv"

# Samples at or above this amplitude are considered near full scale
CLIP_THRESHOLD = 0.999


def longest_consecutive_run(mask):
    """Find the longest consecutive run of True values."""
    if len(mask) == 0 or not np.any(mask):
        return 0

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return int(np.max(ends - starts))


def analyze_audio(filepath):
    audio, sample_rate = sf.read(
        filepath,
        dtype="float32",
        always_2d=True
    )

    total_samples = audio.size

    if total_samples == 0:
        raise ValueError("Empty audio file")

    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))

    # Samples very close to the digital full-scale limit
    near_clip_mask = abs_audio >= CLIP_THRESHOLD

    near_clip_count = int(np.sum(near_clip_mask))
    near_clip_ratio = near_clip_count / total_samples

    positive_peak_count = int(
        np.sum(audio >= CLIP_THRESHOLD)
    )

    negative_peak_count = int(
        np.sum(audio <= -CLIP_THRESHOLD)
    )

    # Longest consecutive run in each channel
    longest_run = max(
        longest_consecutive_run(near_clip_mask[:, ch])
        for ch in range(audio.shape[1])
    )

    peak_dbfs = (
        20 * np.log10(peak)
        if peak > 0 else float("-inf")
    )

    # Conservative screening rule, not a definitive diagnosis
    if longest_run >= 3 or near_clip_ratio >= 0.001:
        status = "Review - possible clipping"
    elif near_clip_count > 0:
        status = "Check isolated peaks"
    else:
        status = "No clipping detected"

    return {
        "filename": filepath.name,
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "duration_seconds": len(audio) / sample_rate,
        "peak_amplitude": round(peak, 6),
        "peak_dbfs": round(peak_dbfs, 2),
        "near_clip_samples": near_clip_count,
        "near_clip_percentage": round(
            near_clip_ratio * 100, 5
        ),
        "positive_peak_samples": positive_peak_count,
        "negative_peak_samples": negative_peak_count,
        "longest_consecutive_near_clip_samples": longest_run,
        "status": status
    }


def main():
    audio_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not audio_files:
        print("ERROR: No WAV files found in", AUDIO_DIR)
        return

    results = []
    failed = []

    print("AUDIO CLIPPING DETECTION")
    print("=" * 50)
    print("Total files found:", len(audio_files))

    for index, filepath in enumerate(audio_files, start=1):
        try:
            result = analyze_audio(filepath)
            results.append(result)

            print(
                f"{index}/{len(audio_files)} "
                f"{filepath.name} | "
                f"Peak: {result['peak_dbfs']} dBFS | "
                f"{result['status']}"
            )

        except Exception as error:
            failed.append({
                "filename": filepath.name,
                "error": str(error)
            })
            print("FAILED:", filepath.name, error)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nCLIPPING DETECTION SUMMARY")
    print("=" * 50)

    print("Successfully analyzed:", len(results))
    print("Failed:", len(failed))

    if not df.empty:
        print("\nStatus distribution:")
        print(df["status"].value_counts().to_string())

        print(
            "\nFiles with possible clipping:",
            (df["status"] == "Review - possible clipping").sum()
        )

        print(
            "Files with isolated peaks to check:",
            (df["status"] == "Check isolated peaks").sum()
        )

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(
            OUTPUT_DIR / "clipping_failed_files.csv",
            index=False
        )

    print("\nReport saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()