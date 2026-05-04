"""Write data/manifest.json with content hashes, line counts, and timestamps."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"


def _file_entry(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "sha256": hashlib.sha256(content).hexdigest()[:12],
        "lines": content.count(b"\n"),
        "bytes": len(content),
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(data_dir: Path, filenames: list[str]) -> Path:
    manifest = {}
    for name in filenames:
        path = data_dir / name
        if path.exists():
            manifest[name] = _file_entry(path)

    manifest_path = data_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
