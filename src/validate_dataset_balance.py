
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "dataset" / "metadata.csv"

EXPECTED_SPEAKERS = ["arfat", "fariba", "joy", "wasima"]

EXPECTED_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise"
]

EXPECTED_TOTAL = 840
EXPECTED_PER_SPEAKER = 210
EXPECTED_PER_EMOTION = 120
EXPECTED_PER_SPEAKER_EMOTION = 30

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(METADATA_PATH)

print("=" * 60)
print("DATASET BALANCE VERIFICATION")
print("=" * 60)

print(f"\nTotal recordings: {len(df)}")

required_columns = [
    "filename",
    "speaker_id",
    "emotion",
    "gender",
    "sentence_id",
    "take"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# --------------------------------------------------
# 1. TOTAL RECORDING CHECK
# --------------------------------------------------

print("\n[1] TOTAL RECORDING CHECK")

print("Expected:", EXPECTED_TOTAL)
print("Actual:", len(df))

total_pass = len(df) == EXPECTED_TOTAL
print("Result:", "PASS" if total_pass else "FAIL")

# --------------------------------------------------
# 2. SPEAKER DISTRIBUTION
# --------------------------------------------------

print("\n[2] SPEAKER DISTRIBUTION")

speaker_counts = df["speaker_id"].value_counts().reindex(
    EXPECTED_SPEAKERS, fill_value=0
)

print(speaker_counts)

speaker_pass = (
    len(speaker_counts) == len(EXPECTED_SPEAKERS)
    and (speaker_counts == EXPECTED_PER_SPEAKER).all()
)

print("Result:", "PASS" if speaker_pass else "FAIL")

# --------------------------------------------------
# 3. EMOTION DISTRIBUTION
# --------------------------------------------------

print("\n[3] EMOTION DISTRIBUTION")

emotion_counts = df["emotion"].value_counts().reindex(
    EXPECTED_EMOTIONS, fill_value=0
)

print(emotion_counts)

emotion_pass = (
    (emotion_counts == EXPECTED_PER_EMOTION).all()
)

print("Result:", "PASS" if emotion_pass else "FAIL")

# --------------------------------------------------
# 4. SPEAKER × EMOTION DISTRIBUTION
# --------------------------------------------------

print("\n[4] SPEAKER × EMOTION DISTRIBUTION")

cross_table = pd.crosstab(
    df["speaker_id"],
    df["emotion"]
).reindex(
    index=EXPECTED_SPEAKERS,
    columns=EXPECTED_EMOTIONS,
    fill_value=0
)

print(cross_table)

speaker_emotion_pass = (
    (cross_table == EXPECTED_PER_SPEAKER_EMOTION).all().all()
)

print(
    "Expected recordings per speaker-emotion pair:",
    EXPECTED_PER_SPEAKER_EMOTION
)

print("Result:", "PASS" if speaker_emotion_pass else "FAIL")

# --------------------------------------------------
# 5. GENDER DISTRIBUTION
# --------------------------------------------------

print("\n[5] GENDER DISTRIBUTION")

gender_counts = (
    df.groupby("speaker_id")["gender"]
    .agg(lambda x: x.dropna().unique().tolist())
)

gender_pass = True

for speaker, genders in gender_counts.items():
    print(f"{speaker}: {genders}")

    if len(genders) != 1:
        gender_pass = False

print("Result:", "PASS" if gender_pass else "FAIL")

# --------------------------------------------------
# 6. DUPLICATE FILENAME CHECK
# --------------------------------------------------

print("\n[6] DUPLICATE FILENAME CHECK")

duplicate_count = df["filename"].duplicated().sum()

print("Duplicate filenames:", duplicate_count)

duplicate_pass = duplicate_count == 0

print("Result:", "PASS" if duplicate_pass else "FAIL")

# --------------------------------------------------
# 7. SENTENCE AND TAKE DISTRIBUTION
# --------------------------------------------------

print("\n[7] SENTENCE AND TAKE DISTRIBUTION")

sentence_counts = (
    df.groupby(["speaker_id", "emotion"])["sentence_id"]
    .nunique()
)

take_counts = (
    df.groupby(["speaker_id", "emotion", "sentence_id"])["take"]
    .nunique()
)

sentence_pass = (sentence_counts == 10).all()
take_pass = (take_counts == 3).all()

print("Unique sentences per speaker-emotion pair:")
print(sentence_counts)

print("\nUnique takes per sentence:")
print(take_counts.value_counts().sort_index())

print("Sentence check:", "PASS" if sentence_pass else "FAIL")
print("Take check:", "PASS" if take_pass else "FAIL")

# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

checks = {
    "Total recordings": total_pass,
    "Speaker distribution": speaker_pass,
    "Emotion distribution": emotion_pass,
    "Speaker-emotion balance": speaker_emotion_pass,
    "Gender consistency": gender_pass,
    "Duplicate filenames": duplicate_pass,
    "Sentence distribution": sentence_pass,
    "Take distribution": take_pass
}

print("\n" + "=" * 60)
print("FINAL VALIDATION SUMMARY")
print("=" * 60)

for check, result in checks.items():
    print(f"{check}: {'PASS' if result else 'FAIL'}")

print(f"\nPassed: {sum(checks.values())}/{len(checks)}")

if all(checks.values()):
    print("\nOVERALL RESULT: PASSED")
else:
    print("\nOVERALL RESULT: FAILED")
    print("Please investigate the failed checks before model training.")