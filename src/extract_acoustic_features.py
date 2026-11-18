
import numpy as np
import pandas as pd
import librosa

from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = BASE_DIR / "dataset" / "metadata.csv"

# Segmented speech-only audio
AUDIO_DIR = BASE_DIR / "dataset" / "speech_only"

FEATURE_DIR = BASE_DIR / "dataset" / "features"

OUTPUT_PATH = FEATURE_DIR / "speech_only_acoustic_features.csv"

REPORT_PATH = FEATURE_DIR / "feature_extraction_report.csv"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_RECORDINGS = 840

SAMPLE_RATE = 16000

N_MFCC = 13

N_FFT = 1024

HOP_LENGTH = 256

PITCH_MIN = 65

PITCH_MAX = 400


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(audio_path, metadata_row):

    # Load segmented audio
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Audio is empty")

    if len(y) < N_FFT:
        raise ValueError("Audio is too short for feature extraction")

    if not np.all(np.isfinite(y)):
        raise ValueError("Audio contains non-finite values")

    features = {

        "filename": Path(
            str(metadata_row["filename"])
        ).stem + ".wav",

        "emotion": metadata_row["emotion"],

        "speaker_id": metadata_row["speaker_id"],

        "gender": metadata_row["gender"],

        "sentence_id": metadata_row["sentence_id"],

        "take": metadata_row["take"]

    }

    # --------------------------------------------------------
    # 1. MFCC: 13 coefficients
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    for i in range(N_MFCC):

        features[f"mfcc_{i+1}_mean"] = float(
            np.mean(mfcc[i])
        )

        features[f"mfcc_{i+1}_std"] = float(
            np.std(mfcc[i])
        )

    # --------------------------------------------------------
    # 2. Pitch: Fundamental Frequency using YIN
    # --------------------------------------------------------

    pitch = librosa.yin(
        y,
        fmin=PITCH_MIN,
        fmax=PITCH_MAX,
        sr=sr,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    valid_pitch = pitch[
        np.isfinite(pitch) &
        (pitch >= PITCH_MIN) &
        (pitch <= PITCH_MAX)
    ]

    # No valid pitch: preserve as NaN, not artificial zero
    if len(valid_pitch) > 0:

        features["pitch_mean"] = float(
            np.mean(valid_pitch)
        )

        features["pitch_std"] = float(
            np.std(valid_pitch)
        )

    else:

        features["pitch_mean"] = np.nan

        features["pitch_std"] = np.nan

    # --------------------------------------------------------
    # 3. RMS Energy
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    features["rms_mean"] = float(np.mean(rms))

    features["rms_std"] = float(np.std(rms))

    # --------------------------------------------------------
    # 4. Zero Crossing Rate
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    features["zcr_mean"] = float(np.mean(zcr))

    features["zcr_std"] = float(np.std(zcr))

    # --------------------------------------------------------
    # 5. Spectral Centroid
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )[0]

    features["centroid_mean"] = float(np.mean(centroid))

    features["centroid_std"] = float(np.std(centroid))

    # --------------------------------------------------------
    # Final numerical feature validation
    # --------------------------------------------------------

    numerical_values = [
        value
        for key, value in features.items()
        if key not in [
            "filename",
            "emotion",
            "speaker_id",
            "gender",
            "sentence_id",
            "take"
        ]
    ]

    finite_values = [
        value
        for value in numerical_values
        if np.isfinite(value)
    ]

    if not finite_values:
        raise ValueError("No valid numerical features extracted")

    return features, len(valid_pitch)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SPEECH-ONLY ACOUSTIC FEATURE EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {AUDIO_DIR}"
        )

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Never overwrite existing outputs
    if OUTPUT_PATH.exists() or REPORT_PATH.exists():

        raise FileExistsError(
            "\nFeature output already exists!\n"
            "Please inspect the existing files before proceeding.\n"
            "This script will not overwrite them."
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA_PATH,
        encoding="utf-8-sig"
    )

    if len(df) != EXPECTED_RECORDINGS:

        raise ValueError(
            f"Expected {EXPECTED_RECORDINGS} recordings, "
            f"found {len(df)}"
        )

    if df["filename"].duplicated().any():

        raise ValueError(
            "Duplicate filenames found in metadata"
        )

    print(f"\nTotal recordings: {len(df)}")

    print(f"Audio directory: {AUDIO_DIR}")

    print(f"Output directory: {FEATURE_DIR}")

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    results = []

    report = []

    successful = 0

    failed = 0

    pitch_warning = 0

    print("\nStarting feature extraction...\n")

    for index, row in df.iterrows():

        filename = Path(
            str(row["filename"])
        ).stem + ".wav"

        audio_path = AUDIO_DIR / filename

        status = "FAILED"

        error_message = ""

        valid_pitch_frames = 0

        try:

            if not audio_path.exists():

                raise FileNotFoundError(
                    f"Audio file not found: {filename}"
                )

            features, valid_pitch_frames = extract_features(
                audio_path,
                row
            )

            results.append(features)

            successful += 1

            status = "SUCCESS"

            if valid_pitch_frames == 0:

                pitch_warning += 1

                error_message = (
                    "No valid pitch frames; pitch features are NaN"
                )

                status = "SUCCESS_WITH_PITCH_WARNING"

            print(
                f"[{index + 1}/{len(df)}] "
                f"SUCCESS: {filename}"
            )

        except Exception as error:

            failed += 1

            error_message = str(error)

            print(
                f"[{index + 1}/{len(df)}] "
                f"FAILED: {filename} | {error}"
            )

        report.append({

            "filename": filename,

            "status": status,

            "valid_pitch_frames": valid_pitch_frames,

            "error": error_message

        })

        if (index + 1) % 50 == 0:

            print(
                f"\nProgress: {index + 1}/{len(df)}"
            )

    # --------------------------------------------------------
    # Create feature DataFrame
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Save feature CSV
    # --------------------------------------------------------

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Save extraction report
    # --------------------------------------------------------

    report_df = pd.DataFrame(report)

    report_df.to_csv(
        REPORT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("FEATURE EXTRACTION SUMMARY")

    print("=" * 60)

    print("Total recordings:", len(df))

    print("Successfully extracted:", successful)

    print("Failed:", failed)

    print("Pitch warnings:", pitch_warning)

    print("Total feature rows:", len(result_df))

    print("Total columns:", len(result_df.columns))

    print("\nFeature columns:")

    print(result_df.columns.tolist())

    print("\nFeature dataset saved to:")

    print(OUTPUT_PATH)

    print("\nExtraction report saved to:")

    print(REPORT_PATH)

    print("\nFeature extraction completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
