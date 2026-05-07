from __future__ import annotations

import logging
import re
import threading
import time
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

try:
    from .config import settings
except ImportError:  # pragma: no cover
    from config import settings


UNKNOWN_LABELS = {"", "unknown", "none", "no_detection", "no detection"}
SEMANTIC_LABEL_ALIASES = {
    "person": "face",
    "human": "face",
    "man": "face",
    "woman": "face",
    "boy": "face",
    "girl": "face",
    "portrait": "face",
    "head": "face",
}
GEMMA_FALLBACK_LABELS = [
    "face",
    "circle",
    "line",
    "eye",
    "mouth",
    "cat",
    "dog",
]


def frame_to_pil_image(frame: np.ndarray) -> Image.Image:
    image = np.asarray(frame)
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    if image.ndim == 3 and image.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if image.ndim == 3 and image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
    raise ValueError(f"Unsupported frame shape for Gemma vision: {image.shape}")


def _normalize_label(label: object) -> str:
    return str(label or "").strip().lower().replace("_", " ")


def _canonical_label(label: object, labels: Iterable[str]) -> str | None:
    normalized = _normalize_label(label)
    if normalized in UNKNOWN_LABELS:
        return None

    lookup = {_normalize_label(item): item for item in labels}
    alias = SEMANTIC_LABEL_ALIASES.get(normalized)
    if alias is not None and alias in lookup:
        return lookup[alias]
    return lookup.get(normalized)


def extract_label_from_text(text: str, labels: list[str]) -> str | None:
    cleaned = _normalize_label(text)
    if cleaned in UNKNOWN_LABELS:
        return None

    direct = _canonical_label(cleaned, labels)
    if direct is not None:
        return direct

    lookup = {_normalize_label(item): item for item in labels}
    for alias, target in SEMANTIC_LABEL_ALIASES.items():
        if target not in lookup:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", cleaned):
            return lookup[target]

    for label in sorted(labels, key=len, reverse=True):
        normalized = re.escape(_normalize_label(label))
        if re.search(rf"(?<![a-z0-9]){normalized}(?![a-z0-9])", cleaned):
            return label
    return None


def format_gemma_detection(
    text: str,
    labels: list[str],
    confidence: float | None = None,
) -> dict | None:
    label = extract_label_from_text(text, labels)
    if label is None:
        return None

    if confidence is None:
        confidence = 1.0
    confidence = max(0.0, min(1.0, round(float(confidence), 4)))

    top_labels = [label]
    preferred_fallbacks = GEMMA_FALLBACK_LABELS + list(labels)
    for fallback in preferred_fallbacks:
        if fallback not in top_labels:
            canonical = _canonical_label(fallback, labels)
            if canonical is not None and canonical not in top_labels:
                top_labels.append(canonical)
        if len(top_labels) >= 5:
            break

    top5 = [[top_labels[0], confidence]]
    top5.extend([[candidate, 0.0] for candidate in top_labels[1:5]])

    return {
        "source": "gemma",
        "top1": top5[0][0],
        "confidence": top5[0][1],
        "top3": top5[:3],
        "top5": top5,
    }


def build_gemma_prompt(labels: list[str] | None = None) -> str:
    return "<image>answer en what object is this drawing?"


class GemmaVisionDetector:
    def __init__(
        self,
        labels: list[str],
        model: str | None = None,
        interval_sec: float | None = None,
        confidence: float | None = None,
    ) -> None:
        self.labels = labels
        self.model_id = model or settings.GEMMA_MODEL
        self.interval_sec = float(interval_sec if interval_sec is not None else settings.GEMMA_INTERVAL_SEC)
        self.confidence = float(confidence if confidence is not None else settings.GEMMA_CONFIDENCE)
        self._last_scan_time = -1_000_000.0
        self._disabled = False
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._lock = threading.Lock()
        self._inflight = False
        self._pending_result: dict | None = None

    def should_scan(self, now: float | None = None) -> bool:
        if self._disabled:
            return False
        current = time.monotonic() if now is None else now
        return current - self._last_scan_time >= self.interval_sec

    def predict(self, frame: np.ndarray, now: float | None = None) -> dict | None:
        current = time.monotonic() if now is None else now
        result = self._consume_pending_result()
        if self.should_scan(current):
            self._start_background_scan(frame, current)
        return result

    def preload(self) -> None:
        self._ensure_model()

    def clear_pending_result(self) -> None:
        with self._lock:
            self._pending_result = None

    def _consume_pending_result(self) -> dict | None:
        with self._lock:
            result = self._pending_result
            self._pending_result = None
            return result

    def _start_background_scan(self, frame: np.ndarray, now: float) -> None:
        with self._lock:
            if self._disabled or self._inflight:
                return
            self._inflight = True
            self._last_scan_time = now

        frame_copy = np.array(frame, copy=True)
        thread = threading.Thread(
            target=self._scan_worker,
            args=(frame_copy,),
            daemon=True,
        )
        thread.start()

    def _scan_worker(self, frame: np.ndarray) -> None:
        result = None
        try:
            text = self._request_detection(frame)
            logging.info("gemma raw output: %r", text)
            result = format_gemma_detection(text, self.labels, confidence=self.confidence)
            if result is None:
                logging.info("gemma output did not match a QuickDraw label")
        except Exception:
            if self._model is None or self._processor is None:
                self._disabled = True
            logging.exception("Gemma vision detection failed")
        finally:
            with self._lock:
                if result is not None:
                    self._pending_result = result
                self._inflight = False

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        except Exception as exc:
            self._disabled = True
            raise RuntimeError("Gemma vision requires torch and transformers.") from exc

        if torch.backends.mps.is_available():
            self._device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            self._device = "cuda"
            dtype = torch.float16
        else:
            self._device = "cpu"
            dtype = torch.float32

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()

    def _request_detection(self, frame: np.ndarray) -> str:
        self._ensure_model()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None
        assert self._device is not None

        prompt = build_gemma_prompt(self.labels)
        image = frame_to_pil_image(frame)
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._device)
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=12, do_sample=False)
        input_token_count = int(inputs["input_ids"].shape[-1])
        generated_tokens = output[0][input_token_count:]
        return self._processor.decode(generated_tokens, skip_special_tokens=True).strip()
