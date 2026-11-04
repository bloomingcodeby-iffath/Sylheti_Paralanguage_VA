import os
import subprocess
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METADATA_PATH = os.path.join(PROJECT_DIR, "dataset", "metadata.csv")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "dataset", "processed")

df = pd.read_csv(METADATA_PATH)

successful = 0
failed = []

print("Starting WAV verification...\n")

for index, row in df.iterrows():

    filename = os.path.splitext(row["filename"])[0] + ".wav"
    wav_path = os.path.join(PROCESSED_DIR, filename)

    if not os.path.exists(wav_path):
        failed.append((filename, "File not found"))
        continue

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,bits_per_sample",
        "-of", "default=noprint_wrappers=1",
        wav_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        failed.append((filename, "Unreadable audio"))
        continue

    info = {}

    for line in result.stdout.strip().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = value

    if (
        info.get("sample_rate") == "16000"
        and info.get("channels") == "1"
        and info.get("bits_per_sample") == "16"
    ):
        successful += 1
    else:
        failed.append((filename, f"Incorrect properties: {info}"))

    print(f"Verified: {index + 1}/{len(df)}")

print("\nVERIFICATION SUMMARY")
print("--------------------")
print("Total recordings:", len(df))
print("Verified successfully:", successful)
print("Failed:", len(failed))

if failed:
    print("\nProblematic files:")
    for filename, reason in failed:
        print(filename, "->", reason)
else:
    print("\nAll WAV files passed verification!")
