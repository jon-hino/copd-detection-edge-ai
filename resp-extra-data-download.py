import argparse
import shutil
import subprocess
from pathlib import Path

import kagglehub


def download_kauh():
    """
    Downloads the KAUH lung sound dataset from Kaggle.
    Keeps raw recordings in data/KAUH for preprocessing.
    """

    #Download dataset
    cache_path = Path(kagglehub.dataset_download("arashnic/lung-dataset"))

    #Copy files to data folder
    target_dir = Path("data/KAUH")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)

    #Check downloaded audio
    wav_files = list(target_dir.rglob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No WAV recordings found in {target_dir}. Check the dataset contents.")

    print(f"KAUH dataset ready at: {target_dir.resolve()} ({len(wav_files)} WAV files)")
    print("Check the mirror's filenames and labels before running KAUH preprocessing.")


def download_sprsound():
    """
    Downloads the official SPRSound 2022 dataset from GitHub.
    Maintains the original WAV and JSON annotation folders.
    """

    #Establish download directory
    target_dir = Path("data/SPRSound")
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    #Download only the BioCAS2022 directory
    if not (target_dir / ".git").exists():
        if target_dir.exists() and any(target_dir.iterdir()):
            raise FileExistsError(f"{target_dir} already contains files. Move or rename it before cloning.")
        subprocess.run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git",
            str(target_dir)
        ], check=True)

    subprocess.run([
        "git", "-C", str(target_dir), "sparse-checkout", "set", "BioCAS2022"
    ], check=True)

    #Check downloaded audio and annotations
    data_dir = target_dir / "BioCAS2022"
    wav_files = list(data_dir.rglob("*.wav"))
    json_files = list(data_dir.rglob("*.json"))
    if not wav_files or not json_files:
        raise RuntimeError(
            f"Missing SPRSound WAV/JSON files in {data_dir}. "
            "If the repository uses Git LFS, install Git LFS and run 'git lfs pull' inside data/SPRSound."
        )

    print(f"SPRSound dataset ready at: {data_dir.resolve()}")
    print(f"Found {len(wav_files)} WAV and {len(json_files)} JSON files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download KAUH and SPRSound respiratory sound datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["kauh", "sprsound", "all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    args = parser.parse_args()

    if args.dataset in ("kauh", "all"):
        download_kauh()
    if args.dataset in ("sprsound", "all"):
        download_sprsound()
