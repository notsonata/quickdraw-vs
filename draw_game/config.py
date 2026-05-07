from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional until dependencies are installed
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(Path.cwd() / ".env", override=True)


def _get(name: str, default: Any, caster: Callable[[str], Any]) -> Any:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return caster(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using default {default!r}")
        return default


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "draw_game":
        return REPO_ROOT / path
    return BASE_DIR / path


def _backend(value: str) -> str:
    backend = value.strip().lower()
    if backend in {"tflite", "onnx", "stub"}:
        return backend
    print(f"Invalid MODEL_BACKEND={value!r}; using stub.")
    return "stub"


def _canvas_source(value: str) -> str:
    source = value.strip().lower()
    if source in {"screen", "web"}:
        return source
    print(f"Invalid CANVAS_SOURCE={value!r}; using 'screen'.")
    return "screen"


def _preprocess_profile(value: str) -> str:
    profile = value.strip().lower()
    allowed = {
        "current",
        "dilate_before_resize",
        "dilate_after_resize",
        "antialias_grayscale",
        "more_margin",
    }
    if profile in allowed:
        return profile
    print(f"Invalid PREPROCESS_PROFILE={value!r}; using 'current'.")
    return "current"


@dataclass(frozen=True)
class Settings:
    CANVAS_SOURCE: str = _get("CANVAS_SOURCE", "screen", _canvas_source)
    CANVAS_X: int = _get("CANVAS_X", 0, int)
    CANVAS_Y: int = _get("CANVAS_Y", 0, int)
    CANVAS_W: int = _get("CANVAS_W", 800, int)
    CANVAS_H: int = _get("CANVAS_H", 600, int)
    WEB_CANVAS_HOST: str = _get("WEB_CANVAS_HOST", "0.0.0.0", str)
    WEB_CANVAS_PORT: int = _get("WEB_CANVAS_PORT", 8765, int)
    CLASSIFY_INTERVAL_SEC: float = _get("CLASSIFY_INTERVAL_SEC", 0.25, float)
    ROUND_DURATION_SEC: float = _get("ROUND_DURATION_SEC", 60.0, float)
    AI_FIRST_GUESS_DELAY_SEC: float = _get("AI_FIRST_GUESS_DELAY_SEC", 1.0, float)
    AI_MIN_CONFIDENCE: float = _get("AI_MIN_CONFIDENCE", 0.65, float)
    AI_STABLE_FOR_SEC: float = _get("AI_STABLE_FOR_SEC", 0.5, float)
    AI_SPEECH_COOLDOWN_SEC: float = _get("AI_SPEECH_COOLDOWN_SEC", 2.5, float)
    MAX_AI_GUESSES_PER_ROUND: int = _get("MAX_AI_GUESSES_PER_ROUND", 3, int)
    AI_SPEAK_EVERY_SCAN: bool = _get("AI_SPEAK_EVERY_SCAN", False, _bool)
    AI_LOW_CONFIDENCE_TAUNT_SEC: float = _get("AI_LOW_CONFIDENCE_TAUNT_SEC", 2.5, float)
    AI_TAUNT_COOLDOWN_SEC: float = _get("AI_TAUNT_COOLDOWN_SEC", 5.0, float)
    GEMMA_ENABLED: bool = _get("GEMMA_ENABLED", False, _bool)
    GEMMA_MODEL: str = _get("GEMMA_MODEL", "google/paligemma-3b-mix-224", str)
    GEMMA_INTERVAL_SEC: float = _get("GEMMA_INTERVAL_SEC", 2.0, float)
    GEMMA_CONFIDENCE: float = _get("GEMMA_CONFIDENCE", 0.9, float)
    MODEL_BACKEND: str = _get("MODEL_BACKEND", "stub", _backend)
    MODEL_PATH: Path = _get(
        "MODEL_PATH",
        BASE_DIR / "models" / "quickdraw_stroke_tflite" / "quickdraw_stroke_model_float32.tflite",
        _path,
    )
    LABELS_PATH: Path = _get(
        "LABELS_PATH",
        BASE_DIR / "models" / "quickdraw_stroke_tflite" / "labels.json",
        _path,
    )
    MODEL_INPUT_W: int = _get("MODEL_INPUT_W", 28, int)
    MODEL_INPUT_H: int = _get("MODEL_INPUT_H", 28, int)
    MODEL_INPUT_CHANNELS: int = _get("MODEL_INPUT_CHANNELS", 1, int)
    MODEL_BACKGROUND_VALUE: float = _get("MODEL_BACKGROUND_VALUE", 1.0, float)
    MODEL_STROKE_VALUE: float = _get("MODEL_STROKE_VALUE", 0.0, float)
    # Stroke-sequence model settings
    MODEL_SEQ_LEN: int = _get("MODEL_SEQ_LEN", 128, int)
    MODEL_FEATURES: int = _get("MODEL_FEATURES", 5, int)
    PREPROCESS_PROFILE: str = _get("PREPROCESS_PROFILE", "current", _preprocess_profile)
    PREPROCESS_COMPARE_PROFILES: bool = _get("PREPROCESS_COMPARE_PROFILES", False, _bool)
    PREPROCESS_INTERMEDIATE_SIZE: int = _get("PREPROCESS_INTERMEDIATE_SIZE", 64, int)
    PREPROCESS_PADDING_RATIO: float = _get("PREPROCESS_PADDING_RATIO", 0.22, float)
    PREPROCESS_DILATE_BEFORE_RESIZE: bool = _get(
        "PREPROCESS_DILATE_BEFORE_RESIZE", False, _bool
    )
    PREPROCESS_DILATE_AFTER_RESIZE: bool = _get(
        "PREPROCESS_DILATE_AFTER_RESIZE", False, _bool
    )
    PREPROCESS_DILATE_KERNEL: int = _get("PREPROCESS_DILATE_KERNEL", 2, int)
    PREPROCESS_DILATE_ITERATIONS: int = _get("PREPROCESS_DILATE_ITERATIONS", 1, int)
    PREPROCESS_USE_GRAYSCALE_ANTIALIAS: bool = _get(
        "PREPROCESS_USE_GRAYSCALE_ANTIALIAS", True, _bool
    )
    TTS_ENABLED: bool = _get("TTS_ENABLED", True, _bool)
    DEBUG_SAVE_FRAMES: bool = _get("DEBUG_SAVE_FRAMES", False, _bool)
    DEBUG_PRINT_JSON: bool = _get("DEBUG_PRINT_JSON", True, _bool)
    CLASSIFIER_INPUT_SIZE: int = _get("CLASSIFIER_INPUT_SIZE", 28, int)
    KOKORO_VOICE: str = _get("KOKORO_VOICE", "am_adam", str)
    KOKORO_SPEED: float = _get("KOKORO_SPEED", 1.0, float)
    KOKORO_AUDIO_DEVICE: str = _get("KOKORO_AUDIO_DEVICE", "", str)


settings = Settings()
