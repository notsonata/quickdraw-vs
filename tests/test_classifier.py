import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from draw_game.classifier import (
    PYTHON_314_TFLITE_WARNING,
    StubClassifier,
    TFLiteClassifier,
    create_classifier,
)


class FakeInterpreter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.input = None

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, 28, 28, 1]), "dtype": np.float32}]

    def get_output_details(self):
        return [{"index": 0, "shape": np.array([1, 345]), "dtype": np.float32}]

    def set_tensor(self, index, value):
        self.input = value

    def invoke(self):
        return None

    def get_tensor(self, index):
        output = np.zeros((1, 345), dtype=np.float32)
        output[0, 10] = 0.72
        output[0, 11] = 0.18
        output[0, 12] = 0.06
        return output


class ClassifierTests(unittest.TestCase):
    def test_stub_classifier_returns_required_json_shape(self):
        classifier = StubClassifier(labels=["cat", "dog", "bottlecap", "spreadsheet", "stitches"])

        prediction = classifier.predict(None)

        self.assertEqual(set(prediction), {"top1", "confidence", "top3"})
        self.assertIsInstance(prediction["top1"], str)
        self.assertLessEqual(0.0, prediction["confidence"])
        self.assertLessEqual(prediction["confidence"], 1.0)
        self.assertEqual(len(prediction["top3"]), 3)
        self.assertEqual(prediction["top3"][0][0], prediction["top1"])
        self.assertEqual(prediction["top3"][0][1], prediction["confidence"])

    def test_tflite_classifier_loads_345_labels(self):
        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.txt"
            model_path = Path(tmp) / "quickdraw_model.tflite"
            labels_path.write_text("\n".join(f"label_{i:03d}" for i in range(345)), encoding="utf-8")
            model_path.write_bytes(b"not a real model")

            classifier = TFLiteClassifier(
                str(model_path),
                str(labels_path),
                interpreter_factory=FakeInterpreter,
            )

            self.assertEqual(len(classifier.labels), 345)
            self.assertEqual(classifier.labels[:10], [f"label_{i:03d}" for i in range(10)])

    def test_tflite_inference_smoke_when_model_exists(self):
        model_path = Path("draw_game/models/quickdraw-345-tflite/quickdraw_model.tflite")
        labels_path = Path("draw_game/models/quickdraw-345-tflite/labels.txt")
        if not model_path.exists() or not labels_path.exists():
            self.skipTest("QuickDraw TFLite model files are not downloaded")

        try:
            classifier = TFLiteClassifier(str(model_path), str(labels_path))
        except RuntimeError as exc:
            self.skipTest(f"TFLite interpreter unavailable: {exc}")
        x = np.ones((1, 28, 28, 1), dtype=np.float32)
        x[:, 12:16, 12:16, :] = 0.0

        prediction = classifier.predict(x)

        self.assertIn("top1", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("top3", prediction)
        self.assertEqual(len(prediction["top3"]), 3)

    def test_tflite_backend_missing_model_falls_back_to_stub(self):
        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.txt"
            labels_path.write_text("\n".join(f"label_{i:03d}" for i in range(345)), encoding="utf-8")

            classifier = create_classifier(
                backend="tflite",
                model_path=Path(tmp) / "missing.tflite",
                labels_path=labels_path,
            )

            self.assertIsInstance(classifier, StubClassifier)

    def test_tflite_backend_python_314_warns_and_falls_back_to_stub(self):
        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.txt"
            model_path = Path(tmp) / "quickdraw_model.tflite"
            labels_path.write_text("\n".join(f"label_{i:03d}" for i in range(345)), encoding="utf-8")
            model_path.write_bytes(b"not a real model")

            from contextlib import redirect_stdout
            from io import StringIO

            output = StringIO()
            with redirect_stdout(output):
                classifier = create_classifier(
                    backend="tflite",
                    model_path=model_path,
                    labels_path=labels_path,
                    python_version=(3, 14),
                )

            self.assertIsInstance(classifier, StubClassifier)
            self.assertIn(PYTHON_314_TFLITE_WARNING, output.getvalue())
