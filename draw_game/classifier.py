from __future__ import annotations

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
    labels = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return labels or FALLBACK_LABELS.copy()


def load_required_labels(path: Path | str, expected_count: int = QUICKDRAW_LABEL_COUNT) -> list[str]:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != expected_count:
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
        if x.shape == (settings.MODEL_INPUT_H, settings.MODEL_INPUT_W):
            x = x[None, :, :, None]
        elif x.shape == (settings.MODEL_INPUT_H, settings.MODEL_INPUT_W, 1):
            x = x[None, :, :, :]
        elif x.shape == (
            1,
            settings.MODEL_INPUT_H,
            settings.MODEL_INPUT_W,
            settings.MODEL_INPUT_CHANNELS,
        ):
            pass
        else:
            try:
                x = x.reshape(
                    1,
                    settings.MODEL_INPUT_H,
                    settings.MODEL_INPUT_W,
                    settings.MODEL_INPUT_CHANNELS,
                )
            except ValueError as exc:
                raise ValueError(
                    "TFLite model input must be [28, 28], [28, 28, 1], "
                    f"or [1, 28, 28, 1]; got {x.shape}."
                ) from exc

        expected_shape = (
            1,
            settings.MODEL_INPUT_H,
            settings.MODEL_INPUT_W,
            settings.MODEL_INPUT_CHANNELS,
        )
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

    def predict(self, model_input: np.ndarray) -> dict:
        x = self._prepare_input(model_input)
        self.interpreter.set_tensor(self.input_index, x)
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
