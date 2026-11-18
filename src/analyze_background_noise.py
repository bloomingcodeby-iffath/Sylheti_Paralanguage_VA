
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"
SPEECH_DIR = PROJECT_DIR / "dataset" / "speech_only"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "features" / "audio_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "background_noise_report.csv"

SR_EXPECTED = 16000


def rms_dbfs(audio):
    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0:
        return np.nan

    rms = np.sqrt(np.mean(audio ** 2))

    if rms <= 0:
        return -120.0

    return float(20 * np.log10(rms))


def read_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)

    # Convert to mono for analysis
    audio = np.mean(audio, axis=1)

    return audio, sr


def analyze_recording(filename):
    original_path = PROCESSED_DIR / filename
    speech_path = SPEECH_DIR / filename

    if not original_path.exists():
        raise FileNotFoundError(f"Original WAV missing: {filename}")

    if not speech_path.exists():
        raise FileNotFoundError(f"Speech-only WAV missing: {filename}")

    original, sr = read_audio(original_path)
    speech, speech_sr = read_audio(speech_path)

    if sr != SR_EXPECTED or speech_sr != SR_EXPECTED:
        raise ValueError(
            f"Unexpected sample rate: original={sr}, speech={speech_sr}"
        )

    if len(original) == 0 or len(speech) == 0:
        raise ValueError("Empty audio")

    # Use the original recording's first and last 1 second
    # as approximate background-noise samples.
    edge_samples = min(sr, len(original) // 3)

    if edge_samples == 0:
        raise ValueError("Recording too short for edge analysis")

    leading = original[:edge_samples]
    trailing = original[-edge_samples:]

    leading_noise_dbfs = rms_dbfs(leading)
    trailing_noise_dbfs = rms_dbfs(trailing)

    # Conservative estimate: use the noisier edge
    # so the estimated noise floor is not understated.
    noise_floor_dbfs = max(
        leading_noise_dbfs,
        trailing_noise_dbfs
    )

    speech_rms_dbfs = rms_dbfs(speech)

    estimated_snr_db = speech_rms_dbfs - noise_floor_dbfs

    return {
        "filename": filename,
        "duration_original_sec": round(len(original) / sr, 3),
        "duration_speech_sec": round(len(speech) / sr, 3),
        "leading_noise_dbfs": round(leading_noise_dbfs, 2),
        "trailing_noise_dbfs": round(trailing_noise_dbfs, 2),
        "estimated_noise_floor_dbfs": round(noise_floor_dbfs, 2),
        "speech_rms_dbfs": round(speech_rms_dbfs, 2),
        "estimated_snr_db": round(estimated_snr_db, 2),
    }


def main():
    files = sorted(SPEECH_DIR.glob("*.wav"))

    print("BACKGROUND NOISE ASSESSMENT")
    print("=" * 50)
    print("Speech-only files:", len(files))

    results = []
    failures = []

    for i, path in enumerate(files, start=1):
        try:
            result = analyze_recording(path.name)
            results.append(result)

            print(
                f"{i}/{len(files)} {path.name} | "
                f"Noise: {result['estimated_noise_floor_dbfs']} dBFS | "
                f"Estimated SNR: {result['estimated_snr_db']} dB"
            )

        except Exception as error:
            failures.append({
                "filename": path.name,
                "error": str(error)
            })
            print("FAILED:", path.name, error)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)

        print("\nSUMMARY")
        print("=" * 50)
        print("Successfully analyzed:", len(results))
        print("Failed:", len(failures))

        print("\nEstimated noise floor (dBFS):")
        print(df["estimated_noise_floor_dbfs"].describe().round(2))

        print("\nEstimated SNR (dB):")
        print(df["estimated_snr_db"].describe().round(2))

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "noise_analysis_failed_files.csv",
            index=False
        )

    print("\nReport saved to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()