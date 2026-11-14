import kagglehub
import shutil
from pathlib import Path

cache_path = Path(kagglehub.dataset_download("vbookshelf/respiratory-sound-database"))

target_dir = Path("data")
target_dir.mkdir(exist_ok=True)
shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)
print(f"Dataset ready at: {target_dir.resolve()}")
