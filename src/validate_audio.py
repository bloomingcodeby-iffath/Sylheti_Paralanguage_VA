
from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_DIR / "dataset" / "metadata.csv"

total = 0
missing_files = []
existing_files = []

with METADATA_FILE.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    reader = csv.DictReader(file)

    for row in reader:
        total += 1
        audio_path = PROJECT_DIR / row["filepath"]

        if audio_path.is_file():
            existing_files.append(audio_path)
        else:
            missing_files.append(row["filepath"])

print("Total metadata records:", total)
print("Files found:", len(existing_files))
print("Files missing:", len(missing_files))

if missing_files:
    print("\nMissing files:")
    for filename in missing_files:
        print(filename)
else:
    print("\nAll audio paths are valid!")