from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "zarqankhn/quickdraw-345-tflite"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEST_DIR = PROJECT_ROOT / "draw_game" / "models" / "quickdraw-345-tflite"
FILES_TO_COPY = [
    "quickdraw_model.tflite",
    "quickdraw_model_int8.tflite",
    "labels.txt",
    "categories.txt",
    "model_metadata.json",
]


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=FILES_TO_COPY,
            local_dir=None,
        )
    )

    copied: list[Path] = []
    for filename in FILES_TO_COPY:
        source = snapshot_path / filename
        if not source.exists():
            print(f"Optional file not found in snapshot: {filename}")
            continue
        destination = DEST_DIR / filename
        shutil.copy2(source, destination)
        copied.append(destination)
        print(f"Copied: {destination}")

    required = [DEST_DIR / "quickdraw_model.tflite", DEST_DIR / "labels.txt"]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing required downloaded files:")
        for path in missing:
            print(f"  {path}")
        return 1

    print()
    print("Recommended env vars:")
    print("MODEL_BACKEND=tflite")
    print("MODEL_PATH=draw_game/models/quickdraw-345-tflite/quickdraw_model.tflite")
    print("LABELS_PATH=draw_game/models/quickdraw-345-tflite/labels.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
