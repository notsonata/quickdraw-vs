from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - dependency checked at runtime
    cv2 = None

try:
    from .config import settings
except ImportError:  # pragma: no cover
    from config import settings


MIN_STROKE_PIXELS = 8
STROKE_PADDING_RATIO = 0.35


def _center_strokes_on_square(binary: np.ndarray) -> np.ndarray:
    """Crop visible strokes and place them on a square white canvas."""
    stroke_y, stroke_x = np.where(binary < 128)
    if stroke_x.size < MIN_STROKE_PIXELS:
        return binary

    x1 = int(stroke_x.min())
    x2 = int(stroke_x.max()) + 1
    y1 = int(stroke_y.min())
    y2 = int(stroke_y.max()) + 1

    stroke_w = x2 - x1
    stroke_h = y2 - y1
    side = max(stroke_w, stroke_h)
    padding = max(4, int(side * STROKE_PADDING_RATIO))
    side += padding * 2

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    crop_x1 = max(0, center_x - side // 2)
    crop_y1 = max(0, center_y - side // 2)
    crop_x2 = min(binary.shape[1], crop_x1 + side)
    crop_y2 = min(binary.shape[0], crop_y1 + side)

    if crop_x2 - crop_x1 < side:
        crop_x1 = max(0, crop_x2 - side)
    if crop_y2 - crop_y1 < side:
        crop_y1 = max(0, crop_y2 - side)

    cropped = binary[crop_y1:crop_y2, crop_x1:crop_x2]
    square_side = max(cropped.shape[:2])
    square = np.full((square_side, square_side), 255, dtype=np.uint8)
    offset_y = (square_side - cropped.shape[0]) // 2
    offset_x = (square_side - cropped.shape[1]) // 2
    square[offset_y : offset_y + cropped.shape[0], offset_x : offset_x + cropped.shape[1]] = cropped
    return square


def preprocess_for_classifier(
    frame: np.ndarray, input_size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a screen crop for a QuickDraw-style image classifier.

    The returned tensor is NHWC float32 in the range 0..1 where white
    background is 1.0 and dark strokes are 0.0.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Run pip install -r requirements.txt.")
    if frame is None or frame.size == 0:
        raise ValueError("Cannot preprocess an empty frame.")

    width = int(input_size or settings.MODEL_INPUT_W or settings.CLASSIFIER_INPUT_SIZE)
    height = int(input_size or settings.MODEL_INPUT_H or settings.CLASSIFIER_INPUT_SIZE)
    if width <= 0 or height <= 0:
        raise ValueError("Classifier input size must be positive.")

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # THRESH_BINARY keeps the TFLite polarity: background white, strokes black.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    centered = _center_strokes_on_square(binary)
    preview = cv2.resize(centered, (width, height), interpolation=cv2.INTER_AREA)
    preview = np.where(preview < 240, 0, 255).astype(np.uint8)
    tensor = (preview.astype(np.float32) / 255.0)[None, :, :, None]
    return tensor, preview
