
from pathlib import Path
import csv
from collections import Counter, defaultdict

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "dataset" / "raw"
OUTPUT_FILE = PROJECT_DIR / "dataset" / "metadata.csv"

# Supported audio formats
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".mp4", ".wav", ".flac"}

# Dataset configuration
VALID_SPEAKERS = {"fariba", "wasima", "arfat", "joy"}

VALID_EMOTIONS = {
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
}

VALID_GENDERS = {"f", "m"}

EXPECTED_TOTAL = 840
EXPECTED_PER_SPEAKER = 210
EXPECTED_PER_EMOTION = 120

columns = [
    "filename",
    "filepath",
    "gender",
    "speaker_id",
    "sentence_id",
    "emotion",
    "take",
]

rows = []
errors = []
speaker_genders = defaultdict(set)

if not RAW_DIR.exists():
    raise FileNotFoundError(f"Audio folder not found: {RAW_DIR}")

audio_files = sorted(
    path for path in RAW_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
)

# Check duplicate filenames and filename stems
filename_counts = Counter(path.name.lower() for path in audio_files)
stem_counts = Counter(path.stem.lower() for path in audio_files)

for filename, count in filename_counts.items():
    if count > 1:
        errors.append(f"Duplicate filename: {filename}")

for stem, count in stem_counts.items():
    if count > 1:
        errors.append(f"Duplicate filename stem: {stem}")

# Parse filenames
for audio_path in audio_files:

    parts = audio_path.stem.split("_")

    if len(parts) != 5:
        errors.append(f"Invalid filename format: {audio_path.name}")
        continue

    gender, sentence, speaker, emotion, take = parts

    gender = gender.lower()
    speaker = speaker.lower()
    emotion = emotion.lower()

    if gender not in VALID_GENDERS:
        errors.append(f"Invalid gender: {audio_path.name}")
        continue

    if speaker not in VALID_SPEAKERS:
        errors.append(f"Unknown speaker: {audio_path.name}")
        continue

    if emotion not in VALID_EMOTIONS:
        errors.append(f"Invalid emotion: {audio_path.name}")
        continue

    if not sentence.isdigit() or not take.isdigit():
        errors.append(f"Sentence/take must be numeric: {audio_path.name}")
        continue

    sentence_number = int(sentence)
    take_number = int(take)

    if not 1 <= sentence_number <= 10:
        errors.append(f"Invalid sentence number: {audio_path.name}")
        continue

    if take_number not in {1, 2, 3}:
        errors.append(f"Invalid take number: {audio_path.name}")
        continue

    speaker_genders[speaker].add(gender)

    rows.append({
        "filename": audio_path.name,
        "filepath": audio_path.relative_to(PROJECT_DIR).as_posix(),
        "gender": gender,
        "speaker_id": speaker,
        "sentence_id": f"S{sentence_number:02d}",
        "emotion": emotion,
        "take": take_number,
    })

# Check gender consistency for each speaker
for speaker, genders in speaker_genders.items():
    if len(genders) > 1:
        errors.append(
            f"Mixed gender prefixes for speaker '{speaker}': {sorted(genders)}"
        )

# Check total count
if len(audio_files) != EXPECTED_TOTAL:
    errors.append(
        f"Expected {EXPECTED_TOTAL} audio files, found {len(audio_files)}"
    )

# Check speaker distribution
speaker_counts = Counter(row["speaker_id"] for row in rows)

for speaker in sorted(VALID_SPEAKERS):
    count = speaker_counts[speaker]
    if count != EXPECTED_PER_SPEAKER:
        errors.append(
            f"Speaker {speaker}: expected {EXPECTED_PER_SPEAKER}, found {count}"
        )

# Check emotion distribution
emotion_counts = Counter(row["emotion"] for row in rows)

for emotion in sorted(VALID_EMOTIONS):
    count = emotion_counts[emotion]
    if count != EXPECTED_PER_EMOTION:
        errors.append(
            f"Emotion {emotion}: expected {EXPECTED_PER_EMOTION}, found {count}"
        )

# Print summary
print("\nDATASET METADATA VALIDATION")
print("=" * 40)
print("Total audio files:", len(audio_files))
print("Valid metadata rows:", len(rows))

print("\nSpeaker distribution:")
for speaker in sorted(VALID_SPEAKERS):
    print(f"{speaker}: {speaker_counts[speaker]}")

print("\nEmotion distribution:")
for emotion in sorted(VALID_EMOTIONS):
    print(f"{emotion}: {emotion_counts[emotion]}")

print("\nValidation errors:", len(errors))

if errors:
    print("\nERROR DETAILS")
    for error in errors:
        print("ERROR:", error)

    print("\nMetadata CSV was NOT created.")
else:
    rows.sort(key=lambda row: row["filename"].lower())

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nAll checks passed!")
    print("Metadata saved to:", OUTPUT_FILE)