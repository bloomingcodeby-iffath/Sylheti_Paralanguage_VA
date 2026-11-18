
from pathlib import Path

import pandas as pd
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = PROJECT_DIR / "dataset" / "processed"
REPORT = PROJECT_DIR / "dataset" / "features" / "audio_quality" / "background_noise_report.csv"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "features" / "audio_quality" / "edge_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(REPORT)
selected = df.nsmallest(10, "estimated_snr_db")

for _, row in selected.iterrows():
    filename = row["filename"]
    audio, sr = sf.read(AUDIO_DIR / filename, dtype="float32")

    edge_samples = sr * 2

    leading = audio[:edge_samples]
    trailing = audio[-edge_samples:]

    stem = Path(filename).stem

    sf.write(OUTPUT_DIR / f"{stem}_leading.wav", leading, sr)
    sf.write(OUTPUT_DIR / f"{stem}_trailing.wav", trailing, sr)

print("Edge review clips created:", len(selected) * 2)
print("Output folder:", OUTPUT_DIR)
