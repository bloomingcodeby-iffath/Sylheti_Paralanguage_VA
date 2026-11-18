
import sys
import csv
from pathlib import Path

# Import the existing segmentation functions
from segment_speech_energy import process_audio

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_DIR / "dataset" / "metadata.csv"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "speech_only"

TARGET_FILENAME = "f_8_fariba_surprise_3.wav"


def main():

    with open(
        METADATA_PATH,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    target_row = None

    for row in rows:
        filename = Path(row["filename"]).stem + ".wav"

        if filename == TARGET_FILENAME:
            target_row = row
            break

    if target_row is None:
        print("Recording not found in metadata.")
        sys.exit(1)

    print("Recovering:", TARGET_FILENAME)

    try:
        result = process_audio(target_row)

        output_path = OUTPUT_DIR / TARGET_FILENAME

        # Verify that the output really exists
        if not output_path.exists():
            raise FileNotFoundError(
                "Output file was not created."
            )

        if output_path.stat().st_size == 0:
            raise ValueError("Output file is empty.")

        print("\nRECOVERY SUCCESSFUL")
        print("Filename:", TARGET_FILENAME)
        print("Duration:", result["output_duration"], "seconds")
        print("File size:", output_path.stat().st_size, "bytes")
        print("Saved at:", output_path)

    except Exception as error:
        print("\nRECOVERY FAILED:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
