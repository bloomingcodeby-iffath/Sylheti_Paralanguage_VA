
import librosa
import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR / "dataset" / "speech_only"
METADATA_PATH = BASE_DIR / "dataset" / "metadata.csv"
FEATURE_PATH = BASE_DIR / "dataset" / "features" / "speech_only_acoustic_features.csv"

OUTPUT_DIR = BASE_DIR / "dataset" / "features" / "pitch_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "pilot_pitch_quality_report.csv"

# Audio settings
SAMPLE_RATE = 16000
FMIN = 65
FMAX = 400
FRAME_LENGTH = 2048
HOP_LENGTH = 256


def validate_pitch(row, existing_features):

    filename = row["filename"]

    # Metadata has .m4a, speech_only contains .wav
    audio_filename = Path(filename).stem + ".wav"
    audio_path = AUDIO_DIR / audio_filename

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_filename}")

    # Load speech-only audio
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Estimate pitch and voiced frames using pYIN
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )

    # Keep only voiced-frame pitch values
    voiced_pitch = f0[voiced_flag & np.isfinite(f0)]

    total_frames = len(f0)
    voiced_frames = len(voiced_pitch)

    voiced_percentage = (
        (voiced_frames / total_frames) * 100
        if total_frames > 0 else 0
    )

    # Match existing YIN features by filename stem
    target_stem = Path(filename).stem

    feature_stems = existing_features["filename"].apply(
        lambda x: Path(str(x)).stem
    )

    old_pitch = existing_features.loc[
        feature_stems == target_stem,
        "pitch_mean"
    ]

    old_pitch_mean = (
        old_pitch.iloc[0]
        if not old_pitch.empty else np.nan
    )

    return {
        "filename": filename,
        "speaker_id": row["speaker_id"],
        "emotion": row["emotion"],

        "total_frames": total_frames,
        "voiced_frames": voiced_frames,
        "unvoiced_frames": total_frames - voiced_frames,
        "voiced_percentage": round(voiced_percentage, 2),

        "yin_pitch_mean": old_pitch_mean,

        "pyin_pitch_mean": (
            round(float(np.mean(voiced_pitch)), 2)
            if voiced_frames > 0 else np.nan
        ),

        "pyin_pitch_std": (
            round(float(np.std(voiced_pitch)), 2)
            if voiced_frames > 0 else np.nan
        ),

        "pyin_pitch_median": (
            round(float(np.median(voiced_pitch)), 2)
            if voiced_frames > 0 else np.nan
        ),

        "pyin_pitch_min": (
            round(float(np.min(voiced_pitch)), 2)
            if voiced_frames > 0 else np.nan
        ),

        "pyin_pitch_max": (
            round(float(np.max(voiced_pitch)), 2)
            if voiced_frames > 0 else np.nan
        ),

        "mean_all_frame_probability": (
            round(float(np.nanmean(voiced_prob)), 3)
            if len(voiced_prob) > 0 else np.nan
        ),

        "mean_voiced_frame_probability": (
            round(float(np.nanmean(voiced_prob[voiced_flag])), 3)
            if np.any(voiced_flag) else np.nan
        )
    }


def main():

    print("PITCH QUALITY VALIDATION - PILOT")
    print("=" * 45)

    metadata = pd.read_csv(
        METADATA_PATH,
        encoding="utf-8-sig"
    )

    features = pd.read_csv(
        FEATURE_PATH,
        encoding="utf-8-sig"
    )

    # Select one recording per speaker and emotion
    pilot_data = (
        metadata.sort_values("filename")
        .groupby(["speaker_id", "emotion"], as_index=False)
        .first()
    )

    print(f"Selected recordings: {len(pilot_data)}")
    print("\nProcessing...\n")

    results = []

    for index, row in pilot_data.iterrows():

        try:
            result = validate_pitch(row, features)
            results.append(result)

            print(
                f"{len(results)}/{len(pilot_data)} "
                f"Processed: {row['filename']} | "
                f"Voiced: {result['voiced_percentage']}%"
            )

        except Exception as error:
            print(f"FAILED: {row['filename']} | {error}")

    if not results:
        print("\nNo recordings were processed successfully.")
        return

    report = pd.DataFrame(results)

    report.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 45)
    print("PILOT VALIDATION COMPLETED")
    print("=" * 45)

    print(f"Successful recordings: {len(report)}")
    print(f"Expected recordings: {len(pilot_data)}")

    print(
        f"Average voiced percentage: "
        f"{report['voiced_percentage'].mean():.2f}%"
    )

    print(
        f"Missing YIN pitch values: "
        f"{report['yin_pitch_mean'].isna().sum()}"
    )

    print(f"\nReport saved at:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
