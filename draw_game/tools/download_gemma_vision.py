from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError


DEFAULT_REPO_ID = "google/paligemma-3b-mix-224"
REQUIRED_PATTERNS = [
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
    "tokenizer*",
    "preprocessor_config.json",
    "processor_config.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the configured local Gemma/PaliGemma vision model."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repo to download. Default: {DEFAULT_REPO_ID}",
    )
    args = parser.parse_args()

    print(f"Downloading Gemma vision model: {args.repo_id}")
    print("This can take several minutes and multiple GB on first download.")
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=args.repo_id,
                allow_patterns=REQUIRED_PATTERNS,
            )
        )
    except GatedRepoError:
        print()
        print(f"Cannot download {args.repo_id}: this Hugging Face repo is gated.")
        print("Open the model page, accept the terms, then authenticate locally:")
        print("  .venv/bin/hf auth login")
        print()
        print("Then rerun:")
        print("  .venv/bin/python draw_game/tools/download_gemma_vision.py")
        return 1
    except RepositoryNotFoundError:
        print()
        print(f"Cannot find model repo: {args.repo_id}")
        return 1

    print()
    print(f"Downloaded snapshot: {snapshot_path}")
    print("Recommended env vars:")
    print("GEMMA_ENABLED=true")
    print(f"GEMMA_MODEL={args.repo_id}")
    print("GEMMA_INTERVAL_SEC=2.0")
    print("GEMMA_CONFIDENCE=0.9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
