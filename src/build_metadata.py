from pathlib import Path
import csv

# Project folder automatically locate
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "dataset" / "raw"
OUTPUT_FILE = PROJECT_DIR / "dataset" / "metadata.csv"

# Supported audio formats
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

# Seven emotion labels used in our research
VALID_EMOTIONS = {
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
}

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

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Audio folder not found: {RAW_DIR}"
    )

for audio_path in sorted(RAW_DIR.iterdir()):

    if not audio_path.is_file():
        continue

    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        continue

    # Example: f_1_iffath_anger_1.wav
    parts = audio_path.stem.split("_")

    if len(parts) != 5:
        errors.append(
            f"Invalid filename format: {audio_path.name}"
        )
        continue

    gender, sentence, speaker, emotion, take = parts

    gender = gender.lower()
    speaker = speaker.lower()
    emotion = emotion.lower()

    if gender not in {"f", "m"}:
        errors.append(
            f"Invalid gender: {audio_path.name}"
        )
        continue

    if emotion not in VALID_EMOTIONS:
        errors.append(
            f"Invalid emotion: {audio_path.name}"
        )
        continue

    if not sentence.isdigit() or not take.isdigit():
        errors.append(
            f"Sentence/take must be numeric: {audio_path.name}"
        )
        continue

    rows.append({
        "filename": audio_path.name,
        "filepath": audio_path.relative_to(
            PROJECT_DIR
        ).as_posix(),
        "gender": gender,
        "speaker_id": speaker,
        "sentence_id": f"S{int(sentence):02d}",
        "emotion": emotion,
        "take": int(take),
    })

# Save metadata
with OUTPUT_FILE.open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"Audio files indexed: {len(rows)}")
print(f"Metadata saved to: {OUTPUT_FILE}")
print(f"Filename/label errors: {len(errors)}")

for error in errors:
    print("ERROR:", error)