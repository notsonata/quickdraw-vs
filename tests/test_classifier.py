import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from time import monotonic, sleep
from unittest.mock import Mock

import numpy as np

from draw_game.classifier import (
    PYTHON_314_TFLITE_WARNING,
    StubClassifier,
    TFLiteClassifier,
    create_classifier,
)
from draw_game.tts_kokoro import TTSWorker, prepare_playback_audio, prepare_stream_audio, resample_audio


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


class FakeInterpreterStroke:
    """Fake TFLite interpreter simulating the 29-class stroke sequence model."""

    NUM_CLASSES = 29
    SEQ_LEN = 128
    FEATURES = 5

    def __init__(self, model_path):
        self.model_path = model_path
        self.input = None

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, self.SEQ_LEN, self.FEATURES]), "dtype": np.float32}]

    def get_output_details(self):
        return [{"index": 0, "shape": np.array([1, self.NUM_CLASSES]), "dtype": np.float32}]

    def set_tensor(self, index, value):
        self.input = value

    def invoke(self):
        return None

    def get_tensor(self, index):
        output = np.zeros((1, self.NUM_CLASSES), dtype=np.float32)
        output[0, 2] = 0.80   # "cat" at index 2
        output[0, 3] = 0.12   # "dog" at index 3
        output[0, 0] = 0.05   # "apple" at index 0
        return output


class ClassifierTests(unittest.TestCase):
    def test_stub_classifier_returns_required_json_shape(self):
        classifier = StubClassifier(labels=["cat", "dog", "bottlecap", "spreadsheet", "stitches"])

        prediction = classifier.predict(None)

        self.assertEqual(set(prediction), {"top1", "confidence", "top3", "top5"})
        self.assertIsInstance(prediction["top1"], str)
        self.assertLessEqual(0.0, prediction["confidence"])
        self.assertLessEqual(prediction["confidence"], 1.0)
        self.assertEqual(len(prediction["top3"]), 3)
        self.assertEqual(len(prediction["top5"]), 5)
        self.assertEqual(prediction["top3"][0][0], prediction["top1"])
        self.assertEqual(prediction["top3"][0][1], prediction["confidence"])
        self.assertEqual(prediction["top5"][0][0], prediction["top1"])

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
        self.assertIn("top5", prediction)
        self.assertEqual(len(prediction["top3"]), 3)
        self.assertEqual(len(prediction["top5"]), 5)

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

    def test_resample_audio_expands_to_target_rate(self):
        source = np.linspace(-1.0, 1.0, num=240, dtype=np.float32)
        resampled = resample_audio(source, source_rate=24000, target_rate=48000)

        self.assertEqual(resampled.dtype, np.float32)
        self.assertGreater(len(resampled), len(source))

    def test_prepare_playback_audio_uses_device_default_samplerate(self):
        class FakeSoundDevice:
            default = type("Default", (), {"device": (-1, 0)})()

            @staticmethod
            def query_devices(device=None):
                if device is None:
                    return []
                return {"default_samplerate": 48000}

        source = np.linspace(-1.0, 1.0, num=240, dtype=np.float32)
        playback_audio, samplerate = prepare_playback_audio(
            FakeSoundDevice,
            source,
            device=0,
            source_rate=24000,
        )

        self.assertEqual(samplerate, 48000)
        self.assertGreater(len(playback_audio), len(source))

    def test_prepare_stream_audio_adds_padding_and_stereo(self):
        class FakeSoundDevice:
            @staticmethod
            def query_devices(device=None):
                return {"max_output_channels": 2}

        source = np.ones(100, dtype=np.float32)
        stream_audio = prepare_stream_audio(
            FakeSoundDevice,
            source,
            device=0,
            samplerate=48000,
            padding_ms=100,
        )

        self.assertEqual(stream_audio.dtype, np.float32)
        self.assertEqual(stream_audio.shape[1], 2)
        self.assertGreater(stream_audio.shape[0], len(source))
        self.assertTrue(np.allclose(stream_audio[0], 0.0))

    def test_tts_worker_speak_dispatches_immediately(self):
        worker = TTSWorker()
        worker._speak_now = Mock()

        worker.speak("cat")

        deadline = monotonic() + 1.0
        while monotonic() < deadline:
            if worker._speak_now.call_count == 1:
                break
            sleep(0.01)

        worker._speak_now.assert_called_once_with("cat")

    def test_tts_worker_keeps_only_latest_pending_guess_while_speaking(self):
        worker = TTSWorker()
        started = Event()
        release_first = Event()
        call_lock = Lock()
        calls: list[tuple[str, str]] = []

        def fake_speak_now(text: str) -> None:
            with call_lock:
                calls.append(("start", text))
            if text == "cat":
                started.set()
                release_first.wait(timeout=1.0)
            with call_lock:
                calls.append(("end", text))

        worker._speak_now = fake_speak_now

        first = Thread(target=worker.speak, args=("cat",))
        first.start()
        self.assertTrue(started.wait(timeout=1.0))

        worker.speak("dog")
        worker.speak("bird")
        release_first.set()
        first.join(timeout=1.0)

        deadline = monotonic() + 1.0
        while monotonic() < deadline:
            with call_lock:
                snapshot = list(calls)
            if ("end", "bird") in snapshot:
                break
            sleep(0.01)

        with call_lock:
            self.assertEqual(
                calls,
                [
                    ("start", "cat"),
                    ("end", "cat"),
                    ("start", "bird"),
                    ("end", "bird"),
                ],
            )

    def test_tts_worker_can_interrupt_taunt_for_guess(self):
        worker = TTSWorker()
        started = Event()
        calls: list[tuple[str, str]] = []
        call_lock = Lock()

        def fake_speak_now(text: str) -> None:
            with call_lock:
                calls.append(("start", text))
            started.set()
            deadline = monotonic() + 1.0
            while monotonic() < deadline and not worker.should_interrupt_current():
                sleep(0.01)
            with call_lock:
                calls.append(("end", text))

        worker._speak_now = fake_speak_now

        first = Thread(target=worker.speak, kwargs={"text": "Draw better.", "speech_kind": "taunt"})
        first.start()
        self.assertTrue(started.wait(timeout=1.0))

        worker.speak("door.", interrupt=True, speech_kind="guess")

        deadline = monotonic() + 1.5
        while monotonic() < deadline:
            with call_lock:
                snapshot = list(calls)
            if ("end", "door.") in snapshot:
                break
            sleep(0.01)

        with call_lock:
            self.assertEqual(
                calls,
                [
                    ("start", "Draw better."),
                    ("end", "Draw better."),
                    ("start", "door."),
                    ("end", "door."),
                ],
            )


import json as _json  # noqa: E402 — appended test utilities


class StrokeModelClassifierTests(unittest.TestCase):
    """Tests for 29-class stroke-sequence TFLite model integration."""

    # --- label loading ---

    def test_load_labels_reads_json_array(self):
        from draw_game.classifier import load_labels

        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.json"
            labels_path.write_text(
                _json.dumps(["apple", "banana", "cat"]), encoding="utf-8"
            )
            labels = load_labels(labels_path)
        self.assertEqual(labels, ["apple", "banana", "cat"])

    def test_load_required_labels_accepts_any_count_when_expected_count_is_none(self):
        from draw_game.classifier import load_required_labels

        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.json"
            labels_path.write_text(
                _json.dumps([f"class_{i}" for i in range(29)]), encoding="utf-8"
            )
            labels = load_required_labels(labels_path)  # no expected_count
        self.assertEqual(len(labels), 29)

    def test_load_required_labels_still_enforces_count_when_given(self):
        from draw_game.classifier import load_required_labels

        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.json"
            labels_path.write_text(_json.dumps(["a", "b"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_required_labels(labels_path, expected_count=10)

    # --- [1, 128, 5] input handling ---

    def test_tflite_classifier_loads_29_json_labels(self):
        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.json"
            model_path = Path(tmp) / "stroke_model.tflite"
            labels_path.write_text(
                _json.dumps([f"class_{i}" for i in range(29)]), encoding="utf-8"
            )
            model_path.write_bytes(b"not a real model")

            clf = TFLiteClassifier(
                str(model_path),
                str(labels_path),
                interpreter_factory=FakeInterpreterStroke,
            )

        self.assertEqual(len(clf.labels), 29)

    def test_tflite_classifier_accepts_sequence_input_shape(self):
        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.json"
            model_path = Path(tmp) / "stroke_model.tflite"
            classes = ["apple", "banana", "cat", "dog", "fish", "bird", "airplane",
                       "car", "bicycle", "bus", "tree", "flower", "house", "chair",
                       "table", "cup", "fork", "umbrella", "star", "moon", "sun",
                       "cloud", "crown", "pizza", "ice cream", "book", "clock", "eye", "face"]
            labels_path.write_text(_json.dumps(classes), encoding="utf-8")
            model_path.write_bytes(b"not a real model")

            clf = TFLiteClassifier(
                str(model_path),
                str(labels_path),
                interpreter_factory=FakeInterpreterStroke,
            )

        # [1, 128, 5] should pass through _prepare_input without reshaping
        x = np.zeros((1, 128, 5), dtype=np.float32)
        x[0, 0, :] = [0.01, 0.02, 1.0, 0.0, 0.0]  # typical first timestep

        prediction = clf.predict(x)

        self.assertIn("top1", prediction)
        self.assertIn("confidence", prediction)
        self.assertEqual(len(prediction["top3"]), 3)
        self.assertEqual(prediction["top1"], "cat")  # FakeInterpreterStroke sets index 2 highest
        self.assertAlmostEqual(prediction["confidence"], 0.80, places=2)

    def test_tflite_inference_smoke_stroke_model(self):
        """Smoke test against the real float32 stroke model if present."""
        model_path = Path("draw_game/models/quickdraw_stroke_tflite_export_15k_256/quickdraw_stroke_model_float32.tflite")
        labels_path = Path("draw_game/models/quickdraw_stroke_tflite_export_15k_256/labels.json")
        if not model_path.exists() or not labels_path.exists():
            self.skipTest("Stroke TFLite model files not present")

        try:
            clf = TFLiteClassifier(str(model_path), str(labels_path))
        except RuntimeError as exc:
            self.skipTest(f"TFLite interpreter unavailable: {exc}")

        # Minimal sequence: one horizontal stroke
        x = np.zeros((1, 256, 5), dtype=np.float32)
        for t in range(10):
            x[0, t] = [0.01, 0.0, 1.0, 0.0, 0.0]  # pen down, small dx steps
        x[0, 10] = [0.0, 0.0, 0.0, 1.0, 0.0]       # pen up
        x[0, 11] = [0.0, 0.0, 0.0, 0.0, 1.0]        # end token

        prediction = clf.predict(x)

        self.assertIn("top1", prediction)
        self.assertIn("confidence", prediction)
        self.assertEqual(len(prediction["top3"]), 3)
        self.assertEqual(len(prediction["top5"]), 5)
        # top1 must be one of the 29 classes
        from draw_game.classifier import load_labels
        valid_labels = load_labels(labels_path)
        self.assertIn(prediction["top1"], valid_labels)

        # Report interpreter details
        input_details = clf.input_details
        output_details = clf.output_details
        print(f"\n[smoke] input shape : {input_details[0]['shape'].tolist()}")
        print(f"[smoke] input dtype : {input_details[0]['dtype']}")
        print(f"[smoke] output shape: {output_details[0]['shape'].tolist()}")
        print(f"[smoke] output dtype: {output_details[0]['dtype']}")
        print(f"[smoke] top1={prediction['top1']}  confidence={prediction['confidence']:.4f}")
        print(f"[smoke] top3={prediction['top3']}")


class FakeFusedInterpreter:
    """Fake TFLite interpreter simulating the 30-class fused image+stroke model.

    Two inputs with real-world tensor names matching the exported model.
    """

    NUM_CLASSES = 30
    IMAGE_SIZE = 64
    SEQ_LEN = 256
    FEATURES = 5

    def __init__(self, model_path):
        self.model_path = model_path
        self._tensors: dict = {}

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [
            {
                "index": 0,
                "name": "serving_default_image_input:0",
                "shape": np.array([1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1]),
                "dtype": np.float32,
            },
            {
                "index": 1,
                "name": "serving_default_stroke_input:0",
                "shape": np.array([1, self.SEQ_LEN, self.FEATURES]),
                "dtype": np.float32,
            },
        ]

    def get_output_details(self):
        return [{"index": 2, "shape": np.array([1, self.NUM_CLASSES]), "dtype": np.float32}]

    def set_tensor(self, index, value):
        self._tensors[index] = value

    def invoke(self):
        return None

    def get_tensor(self, index):
        output = np.zeros((1, self.NUM_CLASSES), dtype=np.float32)
        output[0, 3] = 0.75   # "cat" at index 3
        output[0, 4] = 0.15   # "dog" at index 4
        output[0, 0] = 0.06   # "The Mona Lisa" at index 0
        return output


class FakeFusedInterpreterShapeOnly:
    """Same model but with generic tensor names — forces shape-based fallback routing."""

    NUM_CLASSES = 30
    IMAGE_SIZE = 64
    SEQ_LEN = 256
    FEATURES = 5

    def __init__(self, model_path):
        self.model_path = model_path
        self._tensors: dict = {}

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [
            {
                "index": 0,
                "name": "input_0",  # generic — no "image" / "stroke" substring
                "shape": np.array([1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1]),
                "dtype": np.float32,
            },
            {
                "index": 1,
                "name": "input_1",  # generic
                "shape": np.array([1, self.SEQ_LEN, self.FEATURES]),
                "dtype": np.float32,
            },
        ]

    def get_output_details(self):
        return [{"index": 2, "shape": np.array([1, self.NUM_CLASSES]), "dtype": np.float32}]

    def set_tensor(self, index, value):
        self._tensors[index] = value

    def invoke(self):
        return None

    def get_tensor(self, index):
        output = np.zeros((1, self.NUM_CLASSES), dtype=np.float32)
        output[0, 2] = 0.88  # "banana" at index 2
        return output


class FakeFusedInterpreterAmbiguous:
    """Inputs with 2-D shapes — neither image (4-D) nor stroke (3-D) by ndim.

    The shape-based fallback cannot route these, so predict() must raise.
    """

    NUM_CLASSES = 30

    def __init__(self, model_path):
        self.model_path = model_path

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        return [
            {
                "index": 0,
                "name": "weird_input_a",
                "shape": np.array([64, 64]),    # 2-D: not image (4-D) nor stroke (3-D)
                "dtype": np.float32,
            },
            {
                "index": 1,
                "name": "weird_input_b",
                "shape": np.array([256, 5]),    # 2-D: same — unroutable
                "dtype": np.float32,
            },
        ]

    def get_output_details(self):
        return [{"index": 2, "shape": np.array([1, self.NUM_CLASSES]), "dtype": np.float32}]

    def set_tensor(self, index, value):
        pass

    def invoke(self):
        return None

    def get_tensor(self, index):
        return np.zeros((1, self.NUM_CLASSES), dtype=np.float32)


FUSED_LABELS = [
    "The Mona Lisa", "apple", "banana", "cat", "dog", "fish", "bird",
    "airplane", "car", "bicycle", "bus", "tree", "flower", "house", "chair",
    "table", "cup", "fork", "umbrella", "star", "moon", "sun", "cloud",
    "crown", "pizza", "ice cream", "book", "clock", "eye", "face",
]


class FusedModelClassifierTests(unittest.TestCase):
    """Tests for the fused image+stroke multi-input TFLite integration."""

    def _make_clf(self, factory=FakeFusedInterpreter):
        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.json"
            model_path = Path(tmp) / "fused_model.tflite"
            labels_path.write_text(_json.dumps(FUSED_LABELS), encoding="utf-8")
            model_path.write_bytes(b"not a real model")
            return TFLiteClassifier(str(model_path), str(labels_path), interpreter_factory=factory)

    def _valid_inputs(self):
        image_tensor = np.ones((1, 64, 64, 1), dtype=np.float32) * 0.95
        stroke_tensor = np.zeros((1, 256, 5), dtype=np.float32)
        stroke_tensor[0, 0] = [0.01, 0.02, 1.0, 0.0, 0.0]
        stroke_tensor[0, 1] = [0.0, 0.0, 0.0, 1.0, 0.0]
        stroke_tensor[0, 2] = [0.0, 0.0, 0.0, 0.0, 1.0]
        return image_tensor, stroke_tensor

    # --- routing by tensor name ---

    def test_fused_routing_by_tensor_name_returns_prediction(self):
        clf = self._make_clf(FakeFusedInterpreter)
        image_tensor, stroke_tensor = self._valid_inputs()
        result = clf.predict({"image": image_tensor, "stroke": stroke_tensor})
        self.assertIn("top1", result)
        self.assertIn("confidence", result)
        self.assertEqual(len(result["top3"]), 3)
        self.assertEqual(len(result["top5"]), 5)
        self.assertEqual(result["top1"], "cat")    # FakeFusedInterpreter sets index 3 highest; FUSED_LABELS[3]="cat"

    def test_fused_routing_by_tensor_name_feeds_both_tensors(self):
        clf = self._make_clf(FakeFusedInterpreter)
        interp = clf.interpreter
        image_tensor, stroke_tensor = self._valid_inputs()
        clf.predict({"image": image_tensor, "stroke": stroke_tensor})
        # Both tensor slots must have been written
        self.assertIn(0, interp._tensors)
        self.assertIn(1, interp._tensors)

    # --- routing by shape fallback ---

    def test_fused_routing_by_shape_fallback_returns_prediction(self):
        clf = self._make_clf(FakeFusedInterpreterShapeOnly)
        image_tensor, stroke_tensor = self._valid_inputs()
        result = clf.predict({"image": image_tensor, "stroke": stroke_tensor})
        self.assertIn("top1", result)
        self.assertEqual(result["top1"], "banana")   # index 2 in FUSED_LABELS

    # --- ambiguous routing raises ---

    def test_fused_ambiguous_routing_raises_runtime_error(self):
        """Inputs with 2-D shapes cannot be routed by ndim and must raise."""
        clf = self._make_clf(FakeFusedInterpreterAmbiguous)
        image_tensor, stroke_tensor = self._valid_inputs()
        with self.assertRaises(RuntimeError) as ctx:
            clf.predict({"image": image_tensor, "stroke": stroke_tensor})
        self.assertIn("Cannot route", str(ctx.exception))

    # --- shape mismatch raises ---

    def test_fused_shape_mismatch_raises_with_clear_message(self):
        clf = self._make_clf(FakeFusedInterpreter)
        wrong_image = np.ones((1, 28, 28, 1), dtype=np.float32)   # wrong size
        stroke_tensor = self._valid_inputs()[1]
        with self.assertRaises(RuntimeError) as ctx:
            clf.predict({"image": wrong_image, "stroke": stroke_tensor})
        self.assertIn("Shape mismatch", str(ctx.exception))

    # --- single ndarray on multi-input model raises ---

    def test_single_ndarray_raises_for_multi_input_model(self):
        """np.ndarray passed to a multi-input (fused) model must raise RuntimeError."""
        clf = self._make_clf(FakeFusedInterpreter)
        x = np.ones((1, 64, 64, 1), dtype=np.float32)
        with self.assertRaises(RuntimeError) as ctx:
            clf.predict(x)
        self.assertIn("requires a dict", str(ctx.exception))

    # --- single ndarray still works for single-input models ---

    def test_single_ndarray_input_still_works_on_single_input_model(self):
        """Single np.ndarray must still work for single-input models."""
        with TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.json"
            model_path = Path(tmp) / "stroke_model.tflite"
            classes = [f"class_{i}" for i in range(29)]
            labels_path.write_text(_json.dumps(classes), encoding="utf-8")
            model_path.write_bytes(b"not a real model")
            clf = TFLiteClassifier(
                str(model_path), str(labels_path),
                interpreter_factory=FakeInterpreterStroke,
            )
        x = np.zeros((1, 128, 5), dtype=np.float32)
        result = clf.predict(x)
        self.assertIn("top1", result)

    # --- output label count validation ---

    def test_fused_output_label_count_validated_against_labels_json(self):
        """If output size ≠ len(labels), predict must raise."""
        with TemporaryDirectory() as tmp:
            # Give it 10 labels but the fake interpreter outputs 30
            labels_path = Path(tmp) / "labels.json"
            model_path = Path(tmp) / "fused_model.tflite"
            labels_path.write_text(_json.dumps([f"c{i}" for i in range(10)]), encoding="utf-8")
            model_path.write_bytes(b"not a real model")
            clf = TFLiteClassifier(
                str(model_path), str(labels_path),
                interpreter_factory=FakeFusedInterpreter,
            )
        image_tensor, stroke_tensor = self._valid_inputs()
        with self.assertRaises(ValueError) as ctx:
            clf.predict({"image": image_tensor, "stroke": stroke_tensor})
        self.assertIn("Expected output size", str(ctx.exception))

    # --- real fused TFLite smoke test ---

    def test_fused_tflite_smoke_real_model(self):
        """Load the real fused model and run a dummy prediction if the file exists."""
        model_path = Path("draw_game/models/quickdraw_fused_tflite/quickdraw_fused_model_float32.tflite")
        labels_path = Path("draw_game/models/quickdraw_fused_tflite/labels.json")
        if not model_path.exists() or not labels_path.exists():
            self.skipTest("Fused TFLite model files not present")

        try:
            clf = TFLiteClassifier(str(model_path), str(labels_path))
        except RuntimeError as exc:
            self.skipTest(f"TFLite interpreter unavailable: {exc}")

        # Dummy white image with a simulated stroke pixel in the centre
        image_tensor = np.ones((1, 64, 64, 1), dtype=np.float32)
        image_tensor[0, 28:36, 28:36, 0] = 0.0   # black patch

        # Minimal stroke: horizontal pen strokes
        stroke_tensor = np.zeros((1, 256, 5), dtype=np.float32)
        for t in range(10):
            stroke_tensor[0, t] = [0.01, 0.0, 1.0, 0.0, 0.0]
        stroke_tensor[0, 10] = [0.0, 0.0, 0.0, 1.0, 0.0]   # pen up
        stroke_tensor[0, 11] = [0.0, 0.0, 0.0, 0.0, 1.0]   # end token

        result = clf.predict({"image": image_tensor, "stroke": stroke_tensor})

        self.assertIn("top1", result)
        self.assertIn("confidence", result)
        self.assertIn("top3", result)
        self.assertEqual(len(result["top3"]), 3)
        self.assertEqual(len(result["top5"]), 5)

        from draw_game.classifier import load_labels
        valid_labels = load_labels(labels_path)
        self.assertIn(result["top1"], valid_labels)

        # Print observed details for the report
        print(f"\n[fused-smoke] input details:")
        for d in clf.input_details:
            print(f"  name={d['name']}  index={d['index']}  shape={d['shape'].tolist()}  dtype={d['dtype']}")
        print(f"[fused-smoke] output shape: {clf.output_details[0]['shape'].tolist()}")
        print(f"[fused-smoke] label count: {len(valid_labels)}")
        print(f"[fused-smoke] top1={result['top1']}  confidence={result['confidence']:.4f}")
        print(f"[fused-smoke] top3={result['top3']}")

