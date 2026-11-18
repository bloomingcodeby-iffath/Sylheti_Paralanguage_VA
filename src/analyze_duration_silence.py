import librosa
import pandas as pd
import numpy as np
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "dataset" / "metadata.csv"
AUDIO_DIR = BASE_DIR / "dataset" / "processed"
OUTPUT_PATH = AUDIO_DIR / "audio_analysis.csv"

# Load metadata
df = pd.read_csv(METADATA_PATH)

results = []
failed = []

print("Starting audio duration and silence analysis...\n")

for index, row in df.iterrows():

    filename = Path(str(row["filename"])).stem + ".wav"
    audio_path = AUDIO_DIR / filename

    try:
        # Load audio at 16 kHz
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Total audio duration
        duration = len(y) / sr

        # Detect non-silent regions
        intervals = librosa.effects.split(y, top_db=30)

        # Total non-silent duration
        active_samples = sum(
            end - start for start, end in intervals
        )
        active_duration = active_samples / sr

        # Estimated silence duration
        silence_duration = max(0, duration - active_duration)

        # Count internal pauses of at least 0.2 seconds
        pause_count = 0

        for i in range(len(intervals) - 1):
            gap = intervals[i + 1][0] - intervals[i][1]

            if gap / sr >= 0.2:
                pause_count += 1

        # Silence percentage
        silence_percentage = (
            (silence_duration / duration) * 100
            if duration > 0 else 0
        )

        results.append({
            "filename": filename,
            "emotion": row["emotion"],
            "speaker_id": row["speaker_id"],
            "duration_seconds": round(duration, 2),
            "active_seconds": round(active_duration, 2),
            "estimated_silence_seconds": round(silence_duration, 2),
            "silence_percentage": round(silence_percentage, 2),
            "pause_count": pause_count
        })

    except Exception as e:
        failed.append((filename, str(e)))

# Save analysis results
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_PATH, index=False)

# Summary
print("\nANALYSIS SUMMARY")
print("-------------------------")
print("Total recordings:", len(df))
print("Successfully analyzed:", len(results))
print("Failed:", len(failed))

if results:
    print("\nDURATION STATISTICS")
    print("-------------------------")
    print("Shortest:", result_df["duration_seconds"].min(), "seconds")
    print("Longest:", result_df["duration_seconds"].max(), "seconds")
    print("Average:", round(result_df["duration_seconds"].mean(), 2), "seconds")

    print("\nAVERAGE SILENCE")
    print("-------------------------")
    print(
        round(result_df["estimated_silence_seconds"].mean(), 2),
        "seconds"
    )

    print("\nResults saved to:", OUTPUT_PATH)

if failed:
    print("\nFailed files:")
    for filename, error in failed:
        print(filename, ":", error)
