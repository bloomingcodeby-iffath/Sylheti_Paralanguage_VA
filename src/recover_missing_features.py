
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = BASE_DIR / "dataset" / "metadata.csv"
AUDIO_DIR = BASE_DIR / "dataset" / "speech_only"

FEATURE_PATH = (
    BASE_DIR / "dataset" / "features" /
    "speech_only_acoustic_features.csv"
)

TARGET = "f_8_fariba_surprise_3.wav"

metadata = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
df = pd.read_csv(FEATURE_PATH)

row = metadata[
    metadata["filename"].astype(str).apply(
        lambda x: Path(x).stem + ".wav"
    ) == TARGET
]

if row.empty:
    raise ValueError("Recording not found in metadata.")

if TARGET in df["filename"].values:
    print("Feature already exists. No changes made.")

else:
    row = row.iloc[0]
    audio_path = AUDIO_DIR / TARGET

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    features = {
        "filename": TARGET,
        "emotion": row["emotion"],
        "speaker_id": row["speaker_id"],
        "gender": row["gender"],
        "sentence_id": row["sentence_id"],
        "take": row["take"]
    }

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=1024, hop_length=256
    )

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    pitch = librosa.yin(
        y, fmin=65, fmax=400,
        sr=sr, frame_length=1024, hop_length=256
    )

    valid_pitch = pitch[
        np.isfinite(pitch) &
        (pitch >= 65) & (pitch <= 400)
    ]

    features["pitch_mean"] = (
        np.mean(valid_pitch) if len(valid_pitch) else 0
    )
    features["pitch_std"] = (
        np.std(valid_pitch) if len(valid_pitch) else 0
    )

    rms = librosa.feature.rms(
        y=y, frame_length=1024, hop_length=256
    )[0]

    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=1024, hop_length=256
    )[0]

    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=1024, hop_length=256
    )[0]

    features["centroid_mean"] = np.mean(centroid)
    features["centroid_std"] = np.std(centroid)

    new_row = pd.DataFrame([features])
    updated_df = pd.concat([df, new_row], ignore_index=True)

    updated_df.to_csv(FEATURE_PATH, index=False)

    print("Missing feature extracted successfully!")
    print("Total recordings:", len(updated_df))
    print("Total columns:", len(updated_df.columns))
    print("Updated file:", FEATURE_PATH)
