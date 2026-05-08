from __future__ import annotations

import json
import random
import sys
from itertools import cycle
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .config import settings
except ImportError:  # pragma: no cover
    from config import settings


FALLBACK_LABELS = ["cat", "dog", "bottlecap", "spreadsheet", "stitches"]
QUICKDRAW_LABEL_COUNT = 345
PYTHON_314_TFLITE_WARNING = (
    "Python 3.14 is not currently supported for the TensorFlow/TFLite runtime "
    "used by this project. Use Python 3.11 or 3.12."
)


def load_labels(path: Path | str) -> list[str]:
    labels_path = Path(path)
    if not labels_path.exists():
        return FALLBACK_LABELS.copy()
    text = labels_path.read_text(encoding="utf-8")
    if labels_path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                labels = [str(item).strip() for item in data if str(item).strip()]
                return labels or FALLBACK_LABELS.copy()
        except Exception:
            pass
    labels = [line.strip() for line in text.splitlines() if line.strip()]
    return labels or FALLBACK_LABELS.copy()


def load_required_labels(path: Path | str, expected_count: int | None = None) -> list[str]:
    """Load labels from a .txt or .json file.

    Pass expected_count to enforce a specific count; omit (or pass None) to
    accept any non-empty label list.
    """
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = load_labels(labels_path)
    if not labels:
        raise ValueError(f"No labels found in {labels_path}.")
    if expected_count is not None and len(labels) != expected_count:
        raise ValueError(
            f"Expected {expected_count} labels in {labels_path}, found {len(labels)}."
        )
    return labels


def _format_prediction(labels: list[str], probabilities: Iterable[float]) -> dict:
    probs = np.asarray(list(probabilities), dtype=np.float32)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("Classifier probabilities must be a non-empty vector.")

    count = min(5, probs.size, len(labels))
    top_indices = np.argsort(probs)[::-1][:count]
    top5 = [[labels[i], round(float(probs[i]), 4)] for i in top_indices]
    top3 = top5[:3]
    while len(top3) < 3:
        fallback_label = FALLBACK_LABELS[len(top3) % len(FALLBACK_LABELS)]
        top3.append([fallback_label, 0.0])
    while len(top5) < 5:
        fallback_label = FALLBACK_LABELS[len(top5) % len(FALLBACK_LABELS)]
        top5.append([fallback_label, 0.0])

    return {
        "top1": top3[0][0],
        "confidence": top3[0][1],
        "top3": top3,
        "top5": top5,
    }


class StubClassifier:
    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = labels or FALLBACK_LABELS.copy()
        self._labels = cycle([label for label in FALLBACK_LABELS if label in self.labels] or self.labels)

    def predict(self, model_input: np.ndarray | None) -> dict:
        top_label = next(self._labels)
        other_labels = [label for label in self.labels if label != top_label]
        random.shuffle(other_labels)
        chosen = [top_label] + other_labels[:2]
        while len(chosen) < 3:
            chosen.append(FALLBACK_LABELS[len(chosen) % len(FALLBACK_LABELS)])

        confidence = round(random.uniform(0.66, 0.88), 4)
        remaining = max(0.0, 1.0 - confidence)
        second = round(random.uniform(0.05, min(0.22, remaining)), 4)
        third = round(max(0.0, remaining - second), 4)
        return {
            "top1": chosen[0],
            "confidence": confidence,
            "top3": [[chosen[0], confidence], [chosen[1], second], [chosen[2], third]],
            "top5": [
                [chosen[0], confidence],
                [chosen[1], second],
                [chosen[2], third],
                [FALLBACK_LABELS[3], 0.0],
                [FALLBACK_LABELS[4], 0.0],
            ],
        }


class OnnxClassifier:
    def __init__(self, model_path: Path, labels: list[str]) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("onnxruntime is not installed.") from exc

        self.labels = labels
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, model_input: np.ndarray) -> dict:
        outputs = self.session.run(None, {self.input_name: model_input.astype(np.float32)})
        logits = np.asarray(outputs[0]).reshape(-1)
        logits = logits[: len(self.labels)]
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probabilities = exp / np.sum(exp)
        return _format_prediction(self.labels, probabilities)


class TFLiteClassifier:
    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        interpreter_factory=None,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model file not found: {self.model_path}")
        self.labels = load_required_labels(self.labels_path)
        self.interpreter = interpreter_factory(str(self.model_path)) if interpreter_factory else self._load_interpreter()
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # For single-input models keep a shortcut; multi-input models use _set_inputs().
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        self._print_diagnostics()

    def _load_interpreter(self):
        tf_error = None
        try:
            import tensorflow as tf

            return tf.lite.Interpreter(model_path=str(self.model_path))
        except Exception as exc:
            tf_error = exc

        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter(model_path=str(self.model_path))
        except Exception as exc:
            raise RuntimeError(
                "Failed to load TFLite interpreter. Tried tensorflow.lite.Interpreter "
                f"({tf_error}) and tflite_runtime.Interpreter ({exc})."
            ) from exc

    def _print_diagnostics(self) -> None:
        print("Classifier backend: TFLite")
        print(f"Model path: {self.model_path}")
        print(f"Labels path: {self.labels_path}")
        print(f"Loaded labels: {len(self.labels)}")
        print(f"First 10 labels: {self.labels[:10]}")
        print(f"Input details: {self.input_details}")
        print(f"Output details: {self.output_details}")

    def _prepare_input(self, model_input: np.ndarray) -> np.ndarray:
        x = np.asarray(model_input, dtype=np.float32)

        # Stroke-sequence input: [1, seq_len, features] — pass through as-is.
        # dx/dy values are signed so range checks do not apply.
        if x.ndim == 3:
            return x

        # Image input: infer expected shape from the model's first input tensor.
        # This avoids depending on MODEL_INPUT_W/H settings for single-input models
        # of any size (e.g. legacy 28x28, fused image branch at 64x64, etc.).
        detail_shape = tuple(self.input_details[0]["shape"].tolist())  # e.g. (1, H, W, C)
        if len(detail_shape) == 4:
            _, exp_h, exp_w, exp_c = detail_shape
        else:
            # Unusual shape: fall back to settings
            exp_h = settings.MODEL_INPUT_H
            exp_w = settings.MODEL_INPUT_W
            exp_c = settings.MODEL_INPUT_CHANNELS

        if x.shape == (exp_h, exp_w):
            x = x[None, :, :, None]
        elif x.shape == (exp_h, exp_w, 1):
            x = x[None, :, :, :]
        elif x.shape == (1, exp_h, exp_w, exp_c):
            pass
        else:
            try:
                x = x.reshape(1, exp_h, exp_w, exp_c)
            except ValueError as exc:
                raise ValueError(
                    f"TFLite input shape {x.shape} is not a valid "
                    "stroke sequence [1, seq_len, features] or image "
                    f"[{exp_h}, {exp_w}]."
                ) from exc

        expected_shape = (1, exp_h, exp_w, exp_c)
        if x.shape != expected_shape:
            raise ValueError(f"TFLite input shape must be {expected_shape}; got {x.shape}.")

        min_value = float(np.min(x))
        max_value = float(np.max(x))
        if min_value < 0.0 or max_value > 1.0:
            print(
                "Warning: TFLite input values were outside [0.0, 1.0]; "
                f"clipping min={min_value:.4f} max={max_value:.4f}."
            )
            x = np.clip(x, 0.0, 1.0)
        return x.astype(np.float32)


    def _set_inputs(self, model_input: np.ndarray | dict) -> None:
        """Route model_input to one or more interpreter tensor slots.

        Accepts:
          - np.ndarray  — single-input path; only valid when model has exactly one input.
          - dict        — multi-input path; keys "image" and "stroke";
                          routed by tensor name substring, then by shape ndim.
        """
        if isinstance(model_input, np.ndarray):
            if len(self.input_details) >= 2:
                raise RuntimeError(
                    f"This TFLite model has {len(self.input_details)} inputs and requires a dict "
                    'input: {"image": image_tensor, "stroke": stroke_tensor}. '
                    "Passing a single np.ndarray is only valid for single-input models."
                )
            x = self._prepare_input(model_input)
            self.interpreter.set_tensor(self.input_index, x)
            return

        if not isinstance(model_input, dict):
            raise TypeError(
                f"model_input must be np.ndarray or dict, got {type(model_input).__name__}."
            )

        image_tensor = model_input.get("image")
        stroke_tensor = model_input.get("stroke")

        for detail in self.input_details:
            idx = detail["index"]
            name = detail.get("name", "").lower()
            expected_shape = tuple(detail["shape"].tolist())

            # --- name-based routing (primary) ---
            if "image" in name:
                if image_tensor is None:
                    raise RuntimeError(
                        f"Tensor '{detail['name']}' (index {idx}) expects an image input, "
                        "but 'image' key is missing from the input dict."
                    )
                tensor = np.asarray(image_tensor, dtype=np.float32)

            elif "stroke" in name:
                if stroke_tensor is None:
                    raise RuntimeError(
                        f"Tensor '{detail['name']}' (index {idx}) expects a stroke input, "
                        "but 'stroke' key is missing from the input dict."
                    )
                tensor = np.asarray(stroke_tensor, dtype=np.float32)

            else:
                # --- shape-based fallback ---
                # Use the model's actual shape ndim as the discriminator:
                #   4-D (1, H, W, C) → image tensor
                #   3-D (1, seq, feat) → stroke sequence tensor
                # This avoids hardcoding config values that may not match the model.
                if len(expected_shape) == 4 and image_tensor is not None:
                    tensor = np.asarray(image_tensor, dtype=np.float32)
                elif len(expected_shape) == 3 and stroke_tensor is not None:
                    tensor = np.asarray(stroke_tensor, dtype=np.float32)
                else:
                    raise RuntimeError(
                        f"Cannot route input to tensor '{detail['name']}' (index {idx}): "
                        f"name does not contain 'image' or 'stroke', "
                        f"and shape {expected_shape} (ndim={len(expected_shape)}) "
                        "does not unambiguously identify an image (4-D) or stroke (3-D) tensor. "
                        'Provide a tensor name containing \'image\' or \'stroke\'.'
                    )

            # --- shape validation ---
            actual_shape = tensor.shape
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"Shape mismatch for tensor '{detail['name']}' (index {idx}): "
                    f"expected {expected_shape}, got {actual_shape}."
                )

            self.interpreter.set_tensor(idx, tensor)

    def predict(self, model_input: np.ndarray | dict) -> dict:
        self._set_inputs(model_input)
        self.interpreter.invoke()
        output = np.asarray(self.interpreter.get_tensor(self.output_index), dtype=np.float32)
        probabilities = output.reshape(-1)
        if probabilities.size != len(self.labels):
            raise ValueError(
                f"Expected output size {len(self.labels)}, got {probabilities.size}."
            )
        return _format_prediction(self.labels, probabilities)


def create_classifier(
    backend: str | None = None,
    model_path: Path | str | None = None,
    labels_path: Path | str | None = None,
    python_version: tuple[int, int] | None = None,
):
    selected_backend = (backend or settings.MODEL_BACKEND).strip().lower()
    if selected_backend not in {"tflite", "onnx", "stub"}:
        print(f"Warning: invalid MODEL_BACKEND={selected_backend!r}; using StubClassifier.")
        selected_backend = "stub"

    resolved_model_path = Path(model_path or settings.MODEL_PATH)
    resolved_labels_path = Path(labels_path or settings.LABELS_PATH)
    labels = load_labels(resolved_labels_path)

    if selected_backend == "tflite":
        runtime_version = python_version or (sys.version_info.major, sys.version_info.minor)
        if runtime_version >= (3, 14):
            print(f"Warning: {PYTHON_314_TFLITE_WARNING}")
            print("Warning: using StubClassifier.")
            return StubClassifier(labels)
        try:
            return TFLiteClassifier(resolved_model_path, resolved_labels_path)
        except Exception as exc:
            print(f"Warning: failed to load TFLite classifier ({exc}); using StubClassifier.")
            return StubClassifier(labels)

    if selected_backend == "onnx":
        if not resolved_model_path.exists():
            print(f"Warning: MODEL_PATH not found ({resolved_model_path}); using StubClassifier.")
            return StubClassifier(labels)
        if resolved_model_path.suffix.lower() != ".onnx":
            print(
                f"Warning: MODEL_BACKEND=onnx but model is not ONNX ({resolved_model_path}); "
                "using StubClassifier."
            )
            return StubClassifier(labels)
        try:
            print(f"Loading ONNX classifier: {resolved_model_path}")
            return OnnxClassifier(resolved_model_path, labels)
        except Exception as exc:
            print(f"Warning: failed to load ONNX classifier ({exc}); using StubClassifier.")
            return StubClassifier(labels)

    print("Classifier backend: StubClassifier")
    return StubClassifier(labels)


_CLASSIFIER = None


def predict(model_input: np.ndarray) -> dict:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = create_classifier()
    return _CLASSIFIER.predict(model_input)
