import csv
import wave
import webrtcvad

from pathlib import Path


# ==============================
# PROJECT PATHS
# ==============================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_DIR / "dataset" / "metadata.csv"
INPUT_DIR = PROJECT_DIR / "dataset" / "processed"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "speech_only"

REPORT_PATH = OUTPUT_DIR / "segmentation_report.csv"


# ==============================
# VAD SETTINGS
# ==============================

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30

# Less aggressive to reduce the risk of cutting quiet speech
VAD_MODE = 1

# Preserve a little audio before and after detected speech
PADDING_MS = 300

# Detect speech using a 300 ms window
WINDOW_DURATION_MS = 300
SPEECH_RATIO_THRESHOLD = 0.5


# ==============================
# DETECT SPEECH BOUNDARIES
# ==============================

def detect_boundaries(audio_data, vad, sample_width):

    bytes_per_frame = int(
        SAMPLE_RATE * (FRAME_DURATION_MS / 1000)
    ) * sample_width

    frames = []

    for start in range(0, len(audio_data), bytes_per_frame):

        frame = audio_data[start:start + bytes_per_frame]

        # WebRTC VAD requires complete frames
        if len(frame) != bytes_per_frame:
            break

        frames.append(frame)

    if not frames:
        return None

    speech_flags = []

    for frame in frames:
        try:
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_speech = False

        speech_flags.append(is_speech)

    window_frames = int(
        WINDOW_DURATION_MS / FRAME_DURATION_MS
    )

    active_windows = []

    for i in range(len(speech_flags) - window_frames + 1):

        window = speech_flags[i:i + window_frames]

        speech_ratio = sum(window) / window_frames

        if speech_ratio >= SPEECH_RATIO_THRESHOLD:
            active_windows.append(i)

    if not active_windows:
        return None

    # First and last detected speech windows
    first_frame = active_windows[0]
    last_frame = active_windows[-1] + window_frames

    detected_start = first_frame * FRAME_DURATION_MS / 1000
    detected_end = last_frame * FRAME_DURATION_MS / 1000

    return detected_start, detected_end


# ==============================
# PROCESS ONE AUDIO FILE
# ==============================

def process_audio(row, vad):

    filename = Path(row["filename"]).stem + ".wav"

    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(f"Audio not found: {input_path}")

    with wave.open(str(input_path), "rb") as audio:

        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()

        if channels != 1:
            raise ValueError("Audio must be mono.")

        if sample_rate != SAMPLE_RATE:
            raise ValueError("Audio must be 16000 Hz.")

        if sample_width != 2:
            raise ValueError("Audio must be 16-bit PCM.")

        audio_data = audio.readframes(audio.getnframes())

        total_duration = len(audio_data) / (
            sample_rate * sample_width
        )

        boundaries = detect_boundaries(
            audio_data,
            vad,
            sample_width
        )

        if boundaries is None:
            raise ValueError("No speech detected.")

        detected_start, detected_end = boundaries

        # Add padding around the detected speech boundaries
        crop_start = max(
            0,
            detected_start - PADDING_MS / 1000
        )

        crop_end = min(
            total_duration,
            detected_end + PADDING_MS / 1000
        )

        # Convert time boundaries to byte positions
        start_sample = int(crop_start * sample_rate)
        end_sample = int(crop_end * sample_rate)

        start_byte = start_sample * sample_width
        end_byte = end_sample * sample_width

        # Continuous crop preserves internal pauses
        cropped_audio = audio_data[start_byte:end_byte]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_path), "wb") as output:

            output.setnchannels(channels)
            output.setsampwidth(sample_width)
            output.setframerate(sample_rate)
            output.writeframes(cropped_audio)

        output_duration = len(cropped_audio) / (
            sample_rate * sample_width
        )

        return {
            "filename": filename,
            "total_duration": round(total_duration, 3),
            "detected_start": round(detected_start, 3),
            "detected_end": round(detected_end, 3),
            "crop_start": round(crop_start, 3),
            "crop_end": round(crop_end, 3),
            "leading_removed": round(crop_start, 3),
            "trailing_removed": round(total_duration - crop_end, 3),
            "output_duration": round(output_duration, 3),
            "status": "success"
        }


# ==============================
# MAIN
# ==============================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vad = webrtcvad.Vad(VAD_MODE)

    with open(
        METADATA_PATH,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    results = []

    successful = 0
    failed = 0

    print("Starting speech boundary detection...")
    print("=" * 45)

    for index, row in enumerate(rows, start=1):

        try:
            result = process_audio(row, vad)

            results.append(result)
            successful += 1

        except Exception as error:

            failed += 1

            results.append({
                "filename": row.get("filename", "unknown"),
                "total_duration": "",
                "detected_start": "",
                "detected_end": "",
                "crop_start": "",
                "crop_end": "",
                "leading_removed": "",
                "trailing_removed": "",
                "output_duration": "",
                "status": f"failed: {error}"
            })

        if index % 50 == 0 or index == len(rows):
            print(f"Processed: {index}/{len(rows)}")

    # Save segmentation report
    fieldnames = [
        "filename",
        "total_duration",
        "detected_start",
        "detected_end",
        "crop_start",
        "crop_end",
        "leading_removed",
        "trailing_removed",
        "output_duration",
        "status"
    ]

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSEGMENTATION COMPLETED!")
    print("=" * 45)
    print(f"Total recordings: {len(rows)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
