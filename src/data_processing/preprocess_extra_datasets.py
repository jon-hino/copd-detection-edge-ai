import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

SAMPLE_RATE = 16000
ABNORMAL_SOUNDS = {"rhonchi", "wheeze", "stridor", "coarse crackle", "fine crackle", "wheeze+crackle"}

def normalize_sound(sound):
    """
    Standardizes annotation labels for comparison.
    """
    return " ".join(str(sound).lower().strip().split())

def read_spr_annotations(json_path):
    """
    Loads SPRSound JSON annotations and converts event timestamps to seconds.
    Returns labels for normal and abnormal respiratory sounds.
    """

    #Load JSON with labels
    with json_path.open(encoding="utf-8") as file:
        annotations = json.load(file)

    #Skip poor quality recordings
    quality = normalize_sound(annotations.get("record_annotation", annotations.get("recording_annotation", "")))
    if quality == "poor quality":
        return None

    #Read event annotations
    events = []
    for event in annotations.get("event_annotation", []):
        sound_type = normalize_sound(event.get("type", ""))
        if sound_type not in ABNORMAL_SOUNDS and sound_type != "normal":
            raise ValueError(f"Unknown SPRSound annotation {sound_type!r} in {json_path}")

        start = float(event["start"]) / 1000
        end = float(event["end"]) / 1000
        if end > start:
            events.append((start, end, int(sound_type in ABNORMAL_SOUNDS)))

    return events

def read_kauh_annotations(wav_path):
    """
    Reads KAUH recording-level annotations from the WAV filename.
    Sound labels are inherited by each chunk rather than timestamped.
    """

    #Read patient and diagnosis information
    parts = wav_path.stem.split(",")
    if len(parts) < 2:
        return None

    match = re.match(r"^[BDE]?P(\d+)_([^,]+)$", parts[0].strip(), re.IGNORECASE)
    if not match:
        return None

    patient_id, diagnosis = match.groups()

    #Read recording sound labels
    sound_codes = set(re.findall(r"\b[IEWCN]\b", parts[1].upper()))
    if not sound_codes or not sound_codes.issubset({"I", "E", "W", "C", "N"}):
        return None
    if not ({"W", "C", "N"} & sound_codes):
        return None

    return {
        "patient_id": "P" + patient_id,
        "diagnosis": diagnosis,
        "label": int(bool({"W", "C"} & sound_codes))
    }

def apply_bandpass_filter(audio_data, sample_rate, lowcut=50.0, highcut=2000.0, order=5):
    """
    Applies the bandpass filter to the data.
    """

    #Filter audio
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio_data) if len(audio_data) > 40 else audio_data

def process_and_chunk_audio(wav_path, output_dir, source_id, patient_id, annotations,
                            diagnosis, dataset, chunk_length_sec=5.0, overlap_sec=2.5):
    """
    Preprocessing pipeline for KAUH and SPRSound audio.
    Filters, chunks, and converts data to spectrogram for CNNs.
    Maintains chunk_length_sec .wav files for LEAN model.
    """

    #Load and filter the audio
    audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    filtered_audio = apply_bandpass_filter(audio, sr).astype(np.float32)

    #Establish chunk constants
    chunk_samples = int(round(chunk_length_sec * sr))
    step_samples = int(round((chunk_length_sec - overlap_sec) * sr))

    #Making directories
    wav_out_dir = output_dir / "1d_wavs"
    mel_out_dir = output_dir / "2d_mels"
    wav_out_dir.mkdir(parents=True, exist_ok=True)
    mel_out_dir.mkdir(parents=True, exist_ok=True)

    #Chunking
    chunk_metadata = []
    for i, start_idx in enumerate(range(0, len(filtered_audio) - chunk_samples + 1, step_samples)):
        chunk_start_time = start_idx / sr
        chunk_end_time = (start_idx + chunk_samples) / sr

        #SPRSound uses timestamps; KAUH uses recording labels
        if isinstance(annotations, list):
            overlapping_events = [label for start, end, label in annotations
                                  if start < chunk_end_time and end > chunk_start_time]
            if not overlapping_events:
                continue
            label = int(any(overlapping_events))
            label_scope = "event_overlap"
        else:
            label = int(annotations)
            label_scope = "recording_weak"

        chunk = filtered_audio[start_idx : start_idx + chunk_samples]

        mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=2000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        chunk_name = f"{dataset}_{source_id}_chunk{i}"
        sf.write(str(wav_out_dir / f"{chunk_name}.wav"), chunk, sr)
        np.save(str(mel_out_dir / f"{chunk_name}.npy"), mel_spec_db)

        chunk_metadata.append({
            "chunk_name": chunk_name,
            "patient_id": patient_id,
            "label": label,
            "diagnosis": diagnosis or "",
            "dataset": dataset,
            "label_scope": label_scope,
            "start_sec": chunk_start_time,
            "end_sec": chunk_end_time
        })
    return chunk_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess KAUH or SPRSound Respiratory Audio into .wav and mel spectrogram files."
    )
    parser.add_argument(
        "--dataset",
        choices=["kauh", "sprsound"],
        required=True,
        help="Dataset to preprocess (kauh or sprsound)",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder containing raw .wav files and annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Folder for output processed chunks",
    )
    parser.add_argument(
        "--chunk_sec",
        type=float,
        default=5.0,
        help="Length of each audio window (default: 5.0)",
    )
    parser.add_argument(
        "--overlap_sec",
        type=float,
        default=2.5,
        help="Overlap between consecutive windows (default: 2.5)",
    )
    args = parser.parse_args()

    if args.chunk_sec <= 0 or not (0 <= args.overlap_sec < args.chunk_sec):
        parser.error("Require chunk_sec > 0 and 0 <= overlap_sec < chunk_sec")

    #Find WAV files
    wav_files = sorted(args.input_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in '{args.input_dir}'. Verify --input_dir.")

    #Preprocess files
    all_metadata = []
    skipped_files = 0
    print(f"Found {len(wav_files)} files in '{args.input_dir}'. Saving output to '{args.output_dir}'...")
    for wav_path in tqdm(wav_files, desc="Preprocessing"):
        if args.dataset == "sprsound":
            json_path = wav_path.with_suffix(".json")
            if not json_path.exists():
                matches = list(args.input_dir.rglob(wav_path.stem + ".json"))
                if len(matches) != 1:
                    skipped_files += 1
                    continue
                json_path = matches[0]

            annotations = read_spr_annotations(json_path)
            if not annotations:
                skipped_files += 1
                continue
            patient_id = wav_path.stem.split("_")[0]
            diagnosis = ""
        else:
            recording_info = read_kauh_annotations(wav_path)
            if recording_info is None:
                skipped_files += 1
                continue
            patient_id = recording_info["patient_id"]
            diagnosis = recording_info["diagnosis"]
            annotations = recording_info["label"]

        #Generate unique recording names across folders
        source_id = hashlib.sha1(str(wav_path.relative_to(args.input_dir)).encode()).hexdigest()[:12]
        file_metadata = process_and_chunk_audio(
            wav_path, args.output_dir, source_id, patient_id, annotations,
            diagnosis, args.dataset, chunk_length_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
        )
        all_metadata.extend(file_metadata)

    #Save metadata
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["chunk_name", "patient_id", "label", "diagnosis", "dataset", "label_scope", "start_sec", "end_sec"]
    metadata_df = pd.DataFrame(all_metadata, columns=columns)
    csv_path = args.output_dir / "metadata.csv"
    metadata_df.to_csv(csv_path, index=False)

    print(f"Preprocessing complete. Metadata saved to {csv_path} with {len(metadata_df)} total chunks.")
    print(f"Skipped {skipped_files} files with missing or invalid annotations.")
    if metadata_df.empty:
        raise RuntimeError("No chunks produced. Inspect filenames and annotation layout.")
    print("Label counts:\n", metadata_df["label"].value_counts().to_string())
    if args.dataset == "kauh":
        print("NOTE: KAUH chunk labels are inherited from the whole recording, not timestamped events.")
