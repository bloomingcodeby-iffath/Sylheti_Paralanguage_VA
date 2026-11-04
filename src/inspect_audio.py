from pathlib import Path
import csv
from mutagen import File

PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_DIR / "dataset" / "metadata.csv"

with METADATA_FILE.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    rows = list(csv.DictReader(file))

print("Total recordings:", len(rows))
print("=" * 60)

success = 0
failed = 0
durations = []

for row in rows:
    audio_path = PROJECT_DIR / row["filepath"]

    try:
        audio = File(audio_path)

        if audio is None or audio.info is None:
            print("Could not read:", audio_path.name)
            failed += 1
            continue

        info = audio.info
        duration = info.length
        sample_rate = getattr(info, "sample_rate", "Unknown")
        channels = getattr(info, "channels", "Unknown")

        durations.append(duration)
        success += 1

        print(
            f"{audio_path.name} | "
            f"Duration: {duration:.2f}s | "
            f"Sample rate: {sample_rate} Hz | "
            f"Channels: {channels}"
        )

    except Exception as error:
        print(f"Error: {audio_path.name} - {error}")
        failed += 1

print("\n" + "=" * 60)
print("SUMMARY")
print("Total recordings:", len(rows))
print("Successfully inspected:", success)
print("Failed:", failed)

if durations:
    print(f"Shortest duration: {min(durations):.2f} seconds")
    print(f"Longest duration: {max(durations):.2f} seconds")
    print(f"Average duration: {sum(durations) / len(durations):.2f} seconds")
