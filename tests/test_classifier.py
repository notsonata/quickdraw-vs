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
