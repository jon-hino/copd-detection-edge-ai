import argparse
import glob
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from tqdm import tqdm

def apply_bandpass_filter(audio_data, sample_rate, lowcut=50.0, highcut=2000.0, order=5):
    """
    Applies the titular bandpass filter to the data.
    """

    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='bandpass')
    filtered_audio = filtfilt(b, a, audio_data)

    return filtered_audio

def process_and_chunk_audio(input_wav_path, output_dir, chunk_length_sec=5.0, overlap_sec=2.5):
    """
    Preprocessing pipeline for the data. 
    Uses apply_bandpass_filter to filter, chunks, and converts data to spectrogram for CNNs.
    Maintains chunk_length_sec .wav files for LEAN model.
    """

    #Load txt with labels
    txt_path = input_wav_path.replace(".wav", ".txt")
    if not os.path.exists(txt_path):
        return []

    annotations = pd.read_csv(txt_path, sep='\t', header=None, names=['start', 'end', 'crackle', 'wheeze'])

    #Load and filter the audio
    target_sr = 16000
    audio, sr = librosa.load(input_wav_path, sr=target_sr)

    filtered_audio = apply_bandpass_filter(audio, sr)

    #Establish chunk constants
    chunk_samples = int(chunk_length_sec * sr)
    step_samples = int((chunk_length_sec - overlap_sec) * sr)

    #Making directories
    wav_out_dir = os.path.join(output_dir, "1d_wavs")
    mel_out_dir = os.path.join(output_dir, "2d_mels")
    os.makedirs(wav_out_dir, exist_ok=True)
    os.makedirs(mel_out_dir, exist_ok=True)

    #Chunking
    base_name = os.path.splitext(os.path.basename(input_wav_path))[0]
    chunk_metadata = []

    for i, start_idx in enumerate(range(0, len(filtered_audio) - chunk_samples + 1, step_samples)):
        chunk_start_time = start_idx / sr
        chunk_end_time = (start_idx + chunk_samples) / sr

        overlapping_cycles = annotations[
            (annotations['start'] < chunk_end_time) & (annotations['end'] > chunk_start_time)
        ]

        if overlapping_cycles.empty:
            continue

        has_anomaly = (overlapping_cycles['crackle'].sum() > 0) or (overlapping_cycles['wheeze'].sum() > 0)
        label = 1 if has_anomaly else 0

        chunk = filtered_audio[start_idx : start_idx + chunk_samples]

        mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=2000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        chunk_name = f"{base_name}_chunk{i}"
        sf.write(os.path.join(wav_out_dir, f"{chunk_name}.wav"), chunk, sr)
        np.save(os.path.join(mel_out_dir, f"{chunk_name}.npy"), mel_spec_db)

        chunk_metadata.append({
            "chunk_name": chunk_name,
            "patient_id": base_name.split("_")[0],
            "label": label
        })
    return chunk_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ICBHI Respiratory Audio into .wav (waveforms) and .mel (spectrogram) files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing raw .wav files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/ICBHIprocessed",
        help="Folder for output processed chunks (default: data/ICBHIprocessed)",
    )
    parser.add_argument(
        "--chunk_sec",
        type=float,
        default=5.0,
        help="Length of each audio window (chunk) in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--overlap_sec",
        type=float,
        default=2.5,
        help="Overlap between consecutive windows in seconds (default: 2.5)",
    )
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir.strip())
    wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in '{args.input_dir}'. Verify --input_dir."
        )

    all_metadata = []
    print(f"Found {len(wav_files)} files in '{args.input_dir}'. Saving output to '{args.output_dir}...'")
    for wav_path in tqdm(wav_files, desc="Preprocessing"):
        file_metadata = process_and_chunk_audio(
            wav_path,
            args.output_dir,
            chunk_length_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
        )
        all_metadata.extend(file_metadata)

    metadata_df = pd.DataFrame(all_metadata)
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    metadata_df.to_csv(csv_path, index=False)
    print("Preprocessing complete. Metadata saved to {csv_path} with {len(metadata_df)} total chunks.")


