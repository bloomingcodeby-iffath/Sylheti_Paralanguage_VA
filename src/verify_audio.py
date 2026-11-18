
import subprocess
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_DIR / "dataset" / "metadata.csv"
PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"

df = pd.read_csv(METADATA_PATH)

successful = 0
failed = []

print("Starting WAV verification...\n")

expected_files = set()

for index, row in df.iterrows():

    filename = Path(row["filename"]).stem + ".wav"
    expected_files.add(filename)

    wav_path = PROCESSED_DIR / filename

    if not wav_path.exists():
        failed.append((filename, "File not found"))
        print(f"Verified: {index + 1}/{len(df)}")
        continue

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample",
        "-of", "default=noprint_wrappers=1",
        str(wav_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        failed.append((filename, "Unreadable audio"))
        print(f"Verified: {index + 1}/{len(df)}")
        continue

    info = {}

    for line in result.stdout.strip().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = value

    bit_depth = info.get("bits_per_sample")

    if (
        info.get("codec_name") == "pcm_s16le"
        and info.get("sample_rate") == "16000"
        and info.get("channels") == "1"
        and bit_depth == "16"
    ):
        successful += 1
    else:
        failed.append(
            (filename, f"Incorrect properties: {info}")
        )

    print(f"Verified: {index + 1}/{len(df)}")


# Check for unexpected WAV files
actual_files = {
    path.name for path in PROCESSED_DIR.glob("*.wav")
}

extra_files = actual_files - expected_files

print("\nVERIFICATION SUMMARY")
print("=" * 40)
print("Expected recordings:", len(df))
print("Verified successfully:", successful)
print("Failed:", len(failed))
print("Extra WAV files:", len(extra_files))

if extra_files:
    print("\nUnexpected files:")
    for filename in sorted(extra_files):
        print(filename)

if failed:
    print("\nProblematic files:")
    for filename, reason in failed:
        print(filename, "->", reason)

if not failed and not extra_files and successful == len(df):
    print("\nAll WAV files passed verification!")
