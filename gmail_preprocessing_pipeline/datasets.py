from pathlib import Path
import re


TIMESTAMP_PATTERN = re.compile(r"(\d{8}_\d{6})")


def extract_dataset_version(mbox_path: Path) -> str:
    match = TIMESTAMP_PATTERN.search(mbox_path.stem)
    if not match:
        raise ValueError(f"Could not extract dataset timestamp from {mbox_path.name}")
    return match.group(1)


def resolve_dataset_dir(data_dir: Path, version: str) -> Path:
    return data_dir / "gmail" / version


def resolve_dataset_dir_from_mbox(data_dir: Path, mbox_path: Path) -> Path:
    return resolve_dataset_dir(data_dir, extract_dataset_version(mbox_path))


def find_latest_export(exports_dir: Path) -> Path:
    mbox_files = sorted(exports_dir.glob("new_threads_*.mbox"))
    if not mbox_files:
        raise FileNotFoundError(f"No mbox exports found in {exports_dir}")
    return mbox_files[-1]


def find_latest_dataset_dir(data_dir: Path) -> Path:
    dataset_root = data_dir / "gmail"
    if not dataset_root.exists():
        raise FileNotFoundError(f"No dataset versions found in {dataset_root}")
    versions = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if not versions:
        raise FileNotFoundError(f"No dataset versions found in {dataset_root}")
    return versions[-1]