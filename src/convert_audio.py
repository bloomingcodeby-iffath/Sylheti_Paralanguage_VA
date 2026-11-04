import os
import subprocess
import pandas as pd

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METADATA_PATH = os.path.join(PROJECT_DIR, "dataset", "metadata.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "dataset", "processed")

# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read metadata
df = pd.read_csv(METADATA_PATH)

successful = 0
failed = []

print("Starting audio conversion...\n")

for index, row in df.iterrows():

    input_path = os.path.join(PROJECT_DIR, row["filepath"])

    filename = os.path.splitext(row["filename"])[0] + ".wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode == 0:
        successful += 1
    else:
        failed.append(row["filename"])

    print(f"Processed: {index + 1}/{len(df)}")

print("\nCONVERSION SUMMARY")
print("------------------")
print("Total recordings:", len(df))
print("Successfully converted:", successful)
print("Failed:", len(failed))

if failed:
    print("\nFailed files:")
    for filename in failed:
        print(filename)
else:
    print("\nAll audio files converted successfully!")
