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

