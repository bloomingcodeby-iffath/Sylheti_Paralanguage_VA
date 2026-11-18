
import subprocess
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_DIR / "dataset" / "metadata.csv"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "processed"
REPORT_DIR = PROJECT_DIR / "dataset" / "features" / "audio_quality"
REPORT_FILE = REPORT_DIR / "conversion_report.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with METADATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
    metadata = list(csv.DictReader(file))

results = []

converted = 0
skipped = 0
failed = 0

print("Starting audio conversion...\n")

for index, row in enumerate(metadata, start=1):

    filename = row["filename"]
    input_path = PROJECT_DIR / row["filepath"]
    output_path = OUTPUT_DIR / (Path(filename).stem + ".wav")

    status = ""
    message = ""

    if not input_path.exists():
        status = "FAILED"
        message = "Input file not found"
        failed += 1

    elif output_path.exists():
        status = "SKIPPED"
        message = "Output already exists; not overwritten"
        skipped += 1

    else:
        # Check whether the source contains an audio stream
        probe_command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path)
        ]

        probe = subprocess.run(
            probe_command,
            capture_output=True,
            text=True
        )

        if probe.returncode != 0 or not probe.stdout.strip():
            status = "FAILED"
            message = "No valid audio stream: " + probe.stderr.strip()
            failed += 1

        else:
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel", "error",
                "-i", str(input_path),
                "-map", "0:a:0",
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                "-n",
                str(output_path)
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and output_path.exists():
                status = "SUCCESS"
                message = "Converted successfully"
                converted += 1
            else:
                status = "FAILED"
                message = result.stderr.strip()

                if output_path.exists():
                    output_path.unlink()

                failed += 1

    results.append({
        "filename": filename,
        "status": status,
        "message": message
    })

    print(f"[{index}/{len(metadata)}] {filename}: {status}")

# Save conversion report
with REPORT_FILE.open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "status", "message"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nCONVERSION SUMMARY")
print("=" * 35)
print("Total recordings:", len(metadata))
print("Successfully converted:", converted)
print("Skipped:", skipped)
print("Failed:", failed)
print("\nReport saved to:", REPORT_FILE)
print("Output folder:", OUTPUT_DIR)
