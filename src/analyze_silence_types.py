
import librosa
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "dataset" / "metadata.csv"
AUDIO_DIR = BASE_DIR / "dataset" / "processed"
OUTPUT_PATH = AUDIO_DIR / "silence_analysis.csv"

df = pd.read_csv(METADATA_PATH)

results = []
failed = []

print("Starting silence classification...\n")

for _, row in df.iterrows():

    filename = Path(str(row["filename"])).stem + ".wav"
    audio_path = AUDIO_DIR / filename

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        duration = len(y) / sr

        # Detect active audio regions
        intervals = librosa.effects.split(y, top_db=30)

        if len(intervals) == 0:
            raise ValueError("No active audio detected")

        # Beginning silence
        leading_silence = intervals[0][0] / sr

        # Ending silence
        trailing_silence = (
            len(y) - intervals[-1][1]
        ) / sr

        # Active duration
        active_samples = sum(
            end - start for start, end in intervals
        )
        active_duration = active_samples / sr

        # Internal pauses
        internal_pause = 0
        pause_count = 0

        for i in range(len(intervals) - 1):

            gap = (
                intervals[i + 1][0] - intervals[i][1]
            ) / sr

            if gap > 0:
                internal_pause += gap

            if gap >= 0.2:
                pause_count += 1

        results.append({
            "filename": filename,
            "emotion": row["emotion"],
            "speaker_id": row["speaker_id"],
            "total_duration": round(duration, 2),
            "leading_silence": round(leading_silence, 2),
            "active_duration": round(active_duration, 2),
            "internal_pause": round(internal_pause, 2),
            "trailing_silence": round(trailing_silence, 2),
            "pause_count": pause_count
        })

    except Exception as e:
        failed.append((filename, str(e)))

# Save results
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_PATH, index=False)

print("\nSILENCE ANALYSIS SUMMARY")
print("----------------------------")
print("Total recordings:", len(df))
print("Successfully analyzed:", len(results))
print("Failed:", len(failed))

if results:

    print("\nAVERAGE MEASUREMENTS")
    print("----------------------------")

    columns = [
        "total_duration",
        "leading_silence",
        "active_duration",
        "internal_pause",
        "trailing_silence",
        "pause_count"
    ]

    print(result_df[columns].mean().round(2))

    print("\nResults saved to:", OUTPUT_PATH)

if failed:
    print("\nFailed files:")
    for filename, error in failed:
        print(filename, error)