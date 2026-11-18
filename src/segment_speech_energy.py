
import os
import wave
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_DIR / "dataset" / "metadata.csv"
INPUT_DIR = PROJECT_DIR / "dataset" / "processed"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "speech_only"
REPORT_PATH = OUTPUT_DIR / "segmentation_report.csv"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_RECORDINGS = 840

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM

FRAME_DURATION = 0.020       # 20 ms
HOP_DURATION = 0.010         # 10 ms

MIN_ACTIVE_DURATION = 0.050  # 50 ms
SMOOTHING_FRAMES = 5

PADDING_DURATION = 0.250     # 250 ms before and after speech

NOISE_MULTIPLIER = 3.0
PEAK_MULTIPLIER = 0.04
MIN_THRESHOLD = 0.0015


# ============================================================
# AUDIO READING
# ============================================================

def read_wav(filepath):

    with wave.open(str(filepath), "rb") as wav:

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        compression = wav.getcomptype()
        frame_count = wav.getnframes()

        if channels != CHANNELS:
            raise ValueError(
                f"Expected mono audio, found {channels} channels"
            )

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz, found {sample_rate} Hz"
            )

        if sample_width != SAMPLE_WIDTH:
            raise ValueError(
                f"Expected 16-bit audio, found {sample_width * 8}-bit"
            )

        if compression != "NONE":
            raise ValueError("Unsupported WAV compression")

        audio_bytes = wav.readframes(frame_count)

    audio = np.frombuffer(audio_bytes, dtype=np.int16)

    if len(audio) == 0:
        raise ValueError("Audio file is empty")

    audio_float = audio.astype(np.float32) / 32768.0

    return audio, audio_float, sample_rate


# ============================================================
# RMS CALCULATION
# ============================================================

def calculate_rms(audio, frame_length, hop_length):

    if len(audio) < frame_length:
        raise ValueError("Audio is shorter than one analysis frame")

    rms_values = []

    for start in range(0, len(audio) - frame_length + 1, hop_length):

        frame = audio[start:start + frame_length]

        rms = np.sqrt(np.mean(frame ** 2))

        rms_values.append(rms)

    return np.array(rms_values, dtype=np.float32)


# ============================================================
# SPEECH BOUNDARY DETECTION
# ============================================================

def detect_speech_boundaries(audio, sample_rate):

    frame_length = int(FRAME_DURATION * sample_rate)
    hop_length = int(HOP_DURATION * sample_rate)

    rms_values = calculate_rms(
        audio,
        frame_length,
        hop_length
    )

    # Smooth RMS values
    kernel = np.ones(SMOOTHING_FRAMES) / SMOOTHING_FRAMES

    smoothed_rms = np.convolve(
        rms_values,
        kernel,
        mode="same"
    )

    # Estimate noise floor
    noise_floor = float(np.percentile(smoothed_rms, 20))

    peak_energy = float(np.max(smoothed_rms))

    threshold = max(
        noise_floor * NOISE_MULTIPLIER,
        peak_energy * PEAK_MULTIPLIER,
        MIN_THRESHOLD
    )

    # Detect active frames
    active_mask = smoothed_rms > threshold

    # Find contiguous active regions
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        raise ValueError("No active speech region detected")

    split_positions = np.where(
        np.diff(active_indices) > 1
    )[0] + 1

    regions = np.split(active_indices, split_positions)

    min_active_frames = max(
        1,
        int(np.ceil(MIN_ACTIVE_DURATION / HOP_DURATION))
    )

    valid_regions = [
        region
        for region in regions
        if len(region) >= min_active_frames
    ]

    if len(valid_regions) == 0:
        raise ValueError("No valid speech region detected")

    # Keep everything between first and last valid regions.
    # Internal pauses remain untouched.

    first_region = valid_regions[0]
    last_region = valid_regions[-1]

    detected_start = (
        first_region[0] * hop_length
    ) / sample_rate

    detected_end = (
        last_region[-1] * hop_length + frame_length
    ) / sample_rate

    # Add padding
    padding_samples = int(
        PADDING_DURATION * sample_rate
    )

    crop_start_sample = max(
        0,
        int(first_region[0] * hop_length) - padding_samples
    )

    crop_end_sample = min(
        len(audio),
        int(last_region[-1] * hop_length)
        + frame_length
        + padding_samples
    )

    return {
        "detected_start": detected_start,
        "detected_end": detected_end,
        "crop_start_sample": crop_start_sample,
        "crop_end_sample": crop_end_sample,
        "noise_floor": noise_floor,
        "threshold": threshold
    }


# ============================================================
# SAVE SEGMENTED WAV
# ============================================================

def save_wav(filepath, audio):

    audio = np.asarray(audio, dtype=np.int16)

    with wave.open(str(filepath), "wb") as wav:

        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(audio.tobytes())


# ============================================================
# MAIN PROCESS
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("FULL DATASET SPEECH SEGMENTATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check required paths
    # --------------------------------------------------------

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    df = pd.read_csv(METADATA_PATH)

    if len(df) != EXPECTED_RECORDINGS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDINGS} metadata rows, "
            f"found {len(df)}"
        )

    if "filename" not in df.columns:
        raise ValueError("Metadata is missing filename column")

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found in metadata")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Never overwrite previous outputs
    existing_files = list(OUTPUT_DIR.iterdir())

    if existing_files:
        raise FileExistsError(
            "\nOutput directory is not empty!\n"
            f"Directory: {OUTPUT_DIR}\n"
            "Please inspect existing files before proceeding. "
            "This script will not overwrite them."
        )

    print(f"\nMetadata recordings: {len(df)}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Process recordings
    # --------------------------------------------------------

    results = []

    successful = 0
    failed = 0

    for index, row in df.iterrows():

        filename = str(row["filename"])

        stem = Path(filename).stem

        input_path = INPUT_DIR / f"{stem}.wav"
        output_path = OUTPUT_DIR / f"{stem}.wav"

        print(
            f"\n[{index + 1}/{len(df)}] Processing: {filename}"
        )

        result = {
            "filename": filename,
            "total_duration": None,
            "detected_start": None,
            "detected_end": None,
            "crop_start": None,
            "crop_end": None,
            "leading_removed": None,
            "trailing_removed": None,
            "output_duration": None,
            "noise_floor": None,
            "threshold": None,
            "status": "FAILED",
            "error": None
        }

        try:

            # Check input
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Processed WAV not found: {input_path.name}"
                )

            # Read audio
            audio_int16, audio_float, sr = read_wav(input_path)

            total_duration = len(audio_int16) / sr

            result["total_duration"] = round(
                total_duration, 4
            )

            # Detect boundaries
            boundaries = detect_speech_boundaries(
                audio_float,
                sr
            )

            start_sample = boundaries["crop_start_sample"]
            end_sample = boundaries["crop_end_sample"]

            if end_sample <= start_sample:
                raise ValueError("Invalid crop boundaries")

            # Crop audio
            segmented_audio = audio_int16[
                start_sample:end_sample
            ]

            if len(segmented_audio) == 0:
                raise ValueError("Segmented audio is empty")

            # Save output
            save_wav(output_path, segmented_audio)

            output_duration = len(segmented_audio) / sr

            crop_start = start_sample / sr
            crop_end = end_sample / sr

            # Report information
            result.update({

                "detected_start": round(
                    boundaries["detected_start"], 4
                ),

                "detected_end": round(
                    boundaries["detected_end"], 4
                ),

                "crop_start": round(crop_start, 4),

                "crop_end": round(crop_end, 4),

                "leading_removed": round(
                    crop_start, 4
                ),

                "trailing_removed": round(
                    total_duration - crop_end, 4
                ),

                "output_duration": round(
                    output_duration, 4
                ),

                "noise_floor": round(
                    boundaries["noise_floor"], 6
                ),

                "threshold": round(
                    boundaries["threshold"], 6
                ),

                "status": "SUCCESS"

            })

            successful += 1

            print(
                f"SUCCESS | "
                f"Original: {total_duration:.2f}s | "
                f"Output: {output_duration:.2f}s"
            )

        except Exception as error:

            failed += 1

            result["error"] = str(error)

            # Remove incomplete output if one was created
            if output_path.exists():
                output_path.unlink()

            print(f"FAILED: {error}")

        results.append(result)

    # --------------------------------------------------------
    # Save segmentation report
    # --------------------------------------------------------

    report_df = pd.DataFrame(results)

    report_df.to_csv(
        REPORT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FULL SEGMENTATION SUMMARY")
    print("=" * 60)

    print(f"Total recordings: {len(df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    print(f"\nReport saved at:\n{REPORT_PATH}")

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nSegmentation completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
