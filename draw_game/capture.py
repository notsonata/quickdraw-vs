from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - dependency checked at runtime
    cv2 = None

try:
    import mss
except Exception:  # pragma: no cover - dependency checked at runtime
    mss = None

try:
    from .config import settings
except ImportError:  # pragma: no cover
    from config import settings


def _validate_crop() -> None:
    if settings.CANVAS_W <= 0 or settings.CANVAS_H <= 0:
        raise ValueError(
            "Invalid crop dimensions. Set CANVAS_W and CANVAS_H to positive values."
        )


def capture_canvas_crop() -> np.ndarray:
    """Capture the configured canvas rectangle and return a BGR OpenCV image."""
    _validate_crop()
    if mss is None:
        raise RuntimeError("mss is not installed. Run pip install -r requirements.txt.")

    monitor = {
        "left": settings.CANVAS_X,
        "top": settings.CANVAS_Y,
        "width": settings.CANVAS_W,
        "height": settings.CANVAS_H,
    }
    with mss.mss() as screenshotter:
        raw = screenshotter.grab(monitor)
    frame = np.asarray(raw)
    return frame[:, :, :3][:, :, ::-1].copy()


def save_debug_crop(frame: np.ndarray | None = None, directory: Path | None = None) -> Path:
    """Save a crop for debugging and return the written path."""
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Run pip install -r requirements.txt.")
    if frame is None:
        frame = capture_canvas_crop()
    target_dir = directory or Path(__file__).resolve().parent / "logs"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = target_dir / f"debug-crop-{stamp}.png"
    cv2.imwrite(str(path), frame)
    return path


def save_debug_artifacts(
    raw_frame: np.ndarray | None,
    preview_image: np.ndarray | None,
    directory: Path | None = None,
) -> list[Path]:
    """Save raw crop, 28x28 preview, and enlarged model preview for inspection."""
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Run pip install -r requirements.txt.")
    if raw_frame is None:
        raw_frame = capture_canvas_crop()

    target_dir = directory or Path(__file__).resolve().parent / "logs"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    written: list[Path] = []

    raw_path = target_dir / f"debug-raw-{stamp}.png"
    cv2.imwrite(str(raw_path), raw_frame)
    written.append(raw_path)

    if preview_image is not None:
        preview_path = target_dir / f"debug-preview-28x28-{stamp}.png"
        cv2.imwrite(str(preview_path), preview_image)
        written.append(preview_path)

        enlarged = cv2.resize(preview_image, (280, 280), interpolation=cv2.INTER_NEAREST)
        enlarged_path = target_dir / f"debug-preview-280x280-{stamp}.png"
        cv2.imwrite(str(enlarged_path), enlarged)
        written.append(enlarged_path)

    return written
