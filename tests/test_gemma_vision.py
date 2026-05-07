import unittest
from threading import Event
from time import monotonic

import numpy as np

from draw_game import gemma_vision


class GemmaVisionTests(unittest.TestCase):
    def test_formats_valid_gemma_label_as_classifier_result(self):
        result = gemma_vision.format_gemma_detection(
            "cat",
            labels=["cat", "dog", "rabbit"],
            confidence=0.83,
        )

        self.assertEqual(result["source"], "gemma")
        self.assertEqual(result["top1"], "cat")
        self.assertEqual(result["confidence"], 0.83)
        self.assertEqual(result["top3"], [["cat", 0.83], ["dog", 0.0], ["rabbit", 0.0]])
        self.assertEqual(result["top5"][0], ["cat", 0.83])

    def test_rejects_gemma_label_outside_quickdraw_label_set(self):
        result = gemma_vision.format_gemma_detection(
            "dragon",
            labels=["cat", "dog"],
            confidence=0.9,
        )

        self.assertIsNone(result)

    def test_parses_label_from_chatty_model_text(self):
        label = gemma_vision.extract_label_from_text(
            "The closest QuickDraw label is: fire hydrant",
            labels=["cat", "fire hydrant", "dog"],
        )

        self.assertEqual(label, "fire hydrant")

    def test_prompt_is_short_and_does_not_dump_full_label_set(self):
        prompt = gemma_vision.build_gemma_prompt(["cat", "dog", "fire hydrant"])

        self.assertEqual(prompt, "<image>answer en what object is this drawing?")
        self.assertNotIn("fire hydrant", prompt)

    def test_parses_simple_caption_against_label_set(self):
        label = gemma_vision.extract_label_from_text(
            "a drawing of an owl",
            labels=["cat", "owl", "dog"],
        )

        self.assertEqual(label, "owl")

    def test_maps_person_to_face_when_face_label_exists(self):
        label = gemma_vision.extract_label_from_text(
            "person",
            labels=["face", "owl", "dog"],
        )

        self.assertEqual(label, "face")

    def test_maps_person_caption_to_face_when_face_label_exists(self):
        label = gemma_vision.extract_label_from_text(
            "a simple drawing of a person",
            labels=["face", "owl", "dog"],
        )

        self.assertEqual(label, "face")

    def test_gemma_fallback_candidates_prefer_drawable_labels(self):
        result = gemma_vision.format_gemma_detection(
            "person",
            labels=[
                "The Eiffel Tower",
                "The Mona Lisa",
                "face",
                "circle",
                "line",
                "eye",
                "mouth",
                "cat",
            ],
            confidence=0.9,
        )

        self.assertEqual(result["top1"], "face")
        self.assertEqual(result["top3"], [["face", 0.9], ["circle", 0.0], ["line", 0.0]])
        self.assertNotIn(["The Eiffel Tower", 0.0], result["top5"])

    def test_encodes_canvas_frame_as_pil_image(self):
        frame = np.full((24, 32, 3), 255, dtype=np.uint8)
        frame[8:16, 10:20] = 0

        image = gemma_vision.frame_to_pil_image(frame)

        self.assertEqual(image.size, (32, 24))
        self.assertEqual(image.mode, "RGB")

    def test_predict_returns_immediately_while_gemma_runs_in_background(self):
        started = Event()
        release = Event()

        class SlowDetector(gemma_vision.GemmaVisionDetector):
            def _request_detection(self, frame):
                started.set()
                release.wait(timeout=1.0)
                return "cat"

        detector = SlowDetector(labels=["cat", "dog"], interval_sec=2.0)
        frame = np.full((8, 8, 3), 255, dtype=np.uint8)

        before = monotonic()
        first = detector.predict(frame, now=0.0)
        elapsed = monotonic() - before

        self.assertIsNone(first)
        self.assertLess(elapsed, 0.1)
        self.assertTrue(started.wait(timeout=1.0))
        release.set()

        result = None
        deadline = monotonic() + 1.0
        while monotonic() < deadline:
            result = detector.predict(frame, now=0.1)
            if result is not None:
                break

        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "gemma")
        self.assertEqual(result["top1"], "cat")

    def test_predict_does_not_start_overlapping_gemma_jobs(self):
        release = Event()
        calls = []

        class SlowDetector(gemma_vision.GemmaVisionDetector):
            def _request_detection(self, frame):
                calls.append("call")
                release.wait(timeout=1.0)
                return "cat"

        detector = SlowDetector(labels=["cat", "dog"], interval_sec=0.0)
        frame = np.full((8, 8, 3), 255, dtype=np.uint8)

        detector.predict(frame, now=0.0)
        detector.predict(frame, now=0.1)
        release.set()

        self.assertEqual(calls, ["call"])

    def test_clear_pending_result_discards_stale_gemma_guess(self):
        class IdleDetector(gemma_vision.GemmaVisionDetector):
            def should_scan(self, now=None):
                return False

        detector = IdleDetector(labels=["cat", "dog"])
        detector._pending_result = {"source": "gemma", "top1": "cat"}

        detector.clear_pending_result()

        self.assertIsNone(detector.predict(np.full((8, 8, 3), 255, dtype=np.uint8), now=0.0))

    def test_preload_model_calls_ensure_model(self):
        calls = []

        class FakeDetector(gemma_vision.GemmaVisionDetector):
            def _ensure_model(self):
                calls.append("loaded")

        detector = FakeDetector(labels=["cat"])

        detector.preload()

        self.assertEqual(calls, ["loaded"])
