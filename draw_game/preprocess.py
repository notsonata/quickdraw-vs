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
LEGACY_STROKE_PADDING_RATIO = 0.35
PREPROCESS_PROFILES = (
    "current",
    "dilate_before_resize",
    "dilate_after_resize",
    "antialias_grayscale",
    "more_margin",
)


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Run pip install -r requirements.txt.")


def _resolve_model_size(input_size: int | None = None) -> tuple[int, int]:
    width = int(input_size or settings.MODEL_INPUT_W or settings.CLASSIFIER_INPUT_SIZE)
    height = int(input_size or settings.MODEL_INPUT_H or settings.CLASSIFIER_INPUT_SIZE)
    if width <= 0 or height <= 0:
        raise ValueError("Classifier input size must be positive.")
    return width, height


def _prepare_gray(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Cannot preprocess an empty frame.")
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 2:
        return frame
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _extract_binary_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return blurred, binary


def _stroke_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    stroke_y, stroke_x = np.where(binary < 128)
    if stroke_x.size < MIN_STROKE_PIXELS:
        return None
    return (
        int(stroke_x.min()),
        int(stroke_y.min()),
        int(stroke_x.max()) + 1,
        int(stroke_y.max()) + 1,
    )


def _center_strokes_on_square(binary: np.ndarray) -> np.ndarray:
    """Crop visible strokes and place them on a square white canvas."""
    bbox = _stroke_bbox(binary)
    if bbox is None:
        return binary

    x1, y1, x2, y2 = bbox
    stroke_w = x2 - x1
    stroke_h = y2 - y1
    side = max(stroke_w, stroke_h)
    padding = max(4, int(side * LEGACY_STROKE_PADDING_RATIO))
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


def _crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return image
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def _square_pad(image: np.ndarray, padding_ratio: float) -> np.ndarray:
    side = max(image.shape[:2])
    pad = max(2, int(round(side * padding_ratio)))
    square = np.full((side + (pad * 2), side + (pad * 2)), 255, dtype=np.uint8)
    offset_y = pad + (side - image.shape[0]) // 2
    offset_x = pad + (side - image.shape[1]) // 2
    square[offset_y : offset_y + image.shape[0], offset_x : offset_x + image.shape[1]] = image
    return square


def _dilate_black_strokes(image: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    kernel_side = max(1, int(kernel_size))
    kernel = np.ones((kernel_side, kernel_side), dtype=np.uint8)
    iterations = max(1, int(iterations))
    inverted = 255 - image
    dilated = cv2.dilate(inverted, kernel, iterations=iterations)
    return 255 - dilated


def _resize_preview(
    image: np.ndarray,
    width: int,
    height: int,
    *,
    binary_output: bool,
    allow_antialias: bool,
    interpolation: int = cv2.INTER_AREA,
    binary_threshold: int = 240,
) -> np.ndarray:
    if allow_antialias and settings.PREPROCESS_INTERMEDIATE_SIZE > max(width, height):
        intermediate_size = int(settings.PREPROCESS_INTERMEDIATE_SIZE)
        image = cv2.resize(image, (intermediate_size, intermediate_size), interpolation=cv2.INTER_AREA)
    preview = cv2.resize(image, (width, height), interpolation=interpolation)
    if binary_output:
        preview = np.where(preview < binary_threshold, 0, 255).astype(np.uint8)
    else:
        preview = np.clip(np.rint(preview), 0, 255).astype(np.uint8)
    return preview


def _tensor_from_preview(preview: np.ndarray) -> np.ndarray:
    return (preview.astype(np.float32) / 255.0)[None, :, :, None]


def preprocess_for_classifier_with_profile(
    frame: np.ndarray,
    profile_name: str,
    input_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a screen crop for a QuickDraw-style classifier using a named profile."""
    _ensure_cv2()

    width, height = _resolve_model_size(input_size)
    profile = (profile_name or "current").strip().lower()
    if profile not in PREPROCESS_PROFILES:
        raise ValueError(f"Unknown preprocess profile: {profile_name!r}")

    gray = _prepare_gray(frame)
    blurred, binary = _extract_binary_mask(gray)
    bbox = _stroke_bbox(binary)

    if profile == "current":
        centered = _center_strokes_on_square(binary)
        preview = _resize_preview(
            centered,
            width,
            height,
            binary_output=True,
            allow_antialias=False,
        )
        return _tensor_from_preview(preview), preview

    padding_ratio = float(settings.PREPROCESS_PADDING_RATIO)
    use_intermediate = profile == "antialias_grayscale"
    kernel = int(settings.PREPROCESS_DILATE_KERNEL)
    iterations = int(settings.PREPROCESS_DILATE_ITERATIONS)

    if profile == "more_margin":
        padding_ratio = 0.30
    elif profile == "dilate_before_resize":
        # Keep thin tall sketches from shrinking back down after pre-resize dilation.
        padding_ratio = min(padding_ratio, 0.12)

    if profile == "antialias_grayscale":
        grayscale_strokes = np.where(binary < 128, blurred, 255).astype(np.uint8)
        cropped = _crop_to_bbox(grayscale_strokes, bbox)
        centered = _square_pad(cropped, padding_ratio)
        preview = _resize_preview(
            centered,
            width,
            height,
            binary_output=not settings.PREPROCESS_USE_GRAYSCALE_ANTIALIAS,
            allow_antialias=use_intermediate and settings.PREPROCESS_USE_GRAYSCALE_ANTIALIAS,
            interpolation=cv2.INTER_AREA,
        )
        return _tensor_from_preview(preview), preview

    cropped_binary = _crop_to_bbox(binary, bbox)
    if profile == "dilate_before_resize" or settings.PREPROCESS_DILATE_BEFORE_RESIZE:
        before_kernel = kernel if settings.PREPROCESS_DILATE_BEFORE_RESIZE else max(3, kernel)
        cropped_binary = _dilate_black_strokes(cropped_binary, before_kernel, iterations)

    centered = _square_pad(cropped_binary, padding_ratio)
    preview = _resize_preview(
        centered,
        width,
        height,
        binary_output=True,
        allow_antialias=use_intermediate,
        interpolation=cv2.INTER_AREA,
        binary_threshold=252 if profile == "dilate_before_resize" else 240,
    )

    if profile == "dilate_after_resize" or settings.PREPROCESS_DILATE_AFTER_RESIZE:
        preview = _dilate_black_strokes(preview, 2, iterations)
        preview = np.where(preview < 240, 0, 255).astype(np.uint8)

    return _tensor_from_preview(preview), preview


def preprocess_for_classifier(
    frame: np.ndarray,
    input_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a screen crop for a QuickDraw-style image classifier.

    The returned tensor is NHWC float32 in the range 0..1 where white
    background is 1.0 and dark strokes are 0.0.
    """
    return preprocess_for_classifier_with_profile(frame, settings.PREPROCESS_PROFILE, input_size)
