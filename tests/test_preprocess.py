import unittest

import cv2
import numpy as np

from draw_game.preprocess import (
    preprocess_for_classifier,
    preprocess_for_classifier_with_profile,
    preprocess_for_web_canvas,
)


PROFILES = [
    "current",
    "dilate_before_resize",
    "dilate_after_resize",
    "antialias_grayscale",
    "more_margin",
]


class PreprocessTests(unittest.TestCase):
    def test_preprocess_returns_normalized_tensor_and_white_background_preview(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        frame[50:53, 20:80] = 0

        tensor, preview = preprocess_for_classifier(frame, input_size=28)

        self.assertEqual(tensor.shape, (1, 28, 28, 1))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertLessEqual(0.0, float(tensor.min()))
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertEqual(preview.shape, (28, 28))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertGreater(float(tensor.max()), 0.0)
        self.assertGreater(float(tensor[:, :5, :5, :].mean()), 0.9)
        self.assertLess(float(tensor.min()), 0.2)

    def test_preprocess_scales_small_strokes_into_model_frame(self):
        frame = np.full((300, 300, 3), 255, dtype=np.uint8)
        frame[145:155, 145:155] = 0

        tensor, _preview = preprocess_for_classifier(frame, input_size=28)

        dark_pixels = int((tensor[0, :, :, 0] < 0.5).sum())
        self.assertGreater(dark_pixels, 20)

    def test_all_profiles_return_float32_nhwc_in_range(self):
        frame = np.full((160, 160, 3), 255, dtype=np.uint8)
        frame[20:140, 74:86] = 0

        for profile in PROFILES:
            with self.subTest(profile=profile):
                tensor, preview = preprocess_for_classifier_with_profile(frame, profile)
                self.assertEqual(tensor.shape, (1, 28, 28, 1))
                self.assertEqual(tensor.dtype, np.float32)
                self.assertEqual(preview.shape, (28, 28))
                self.assertLessEqual(0.0, float(tensor.min()))
                self.assertLessEqual(float(tensor.max()), 1.0)

    def test_all_profiles_keep_white_background_and_black_strokes_polarity(self):
        frame = np.full((160, 160, 3), 255, dtype=np.uint8)
        frame[20:140, 74:86] = 0

        for profile in PROFILES:
            with self.subTest(profile=profile):
                tensor, _preview = preprocess_for_classifier_with_profile(frame, profile)
                image = tensor[0, :, :, 0]
                self.assertGreater(float(image[:4, :4].mean()), 0.9)
                self.assertLess(float(image.min()), 0.2)

    def test_dilate_profiles_increase_stroke_pixel_count_compared_to_current(self):
        frame = np.full((160, 160, 3), 255, dtype=np.uint8)
        frame[20:140, 78:82] = 0

        current, _ = preprocess_for_classifier_with_profile(frame, "current")
        before, _ = preprocess_for_classifier_with_profile(frame, "dilate_before_resize")
        after, _ = preprocess_for_classifier_with_profile(frame, "dilate_after_resize")

        current_dark = int((current[0, :, :, 0] < 0.5).sum())
        before_dark = int((before[0, :, :, 0] < 0.5).sum())
        after_dark = int((after[0, :, :, 0] < 0.5).sum())

        self.assertGreater(before_dark, current_dark)
        self.assertGreater(after_dark, current_dark)

    def test_more_margin_keeps_tall_drawing_smaller_than_current(self):
        frame = np.full((220, 220, 3), 255, dtype=np.uint8)
        frame[20:200, 96:124] = 0

        current, _ = preprocess_for_classifier_with_profile(frame, "current")
        more_margin, _ = preprocess_for_classifier_with_profile(frame, "more_margin")

        current_rows = np.where((current[0, :, :, 0] < 0.5).any(axis=1))[0]
        margin_rows = np.where((more_margin[0, :, :, 0] < 0.5).any(axis=1))[0]

        self.assertGreater(current_rows.size, 0)
        self.assertGreater(margin_rows.size, 0)
        self.assertLess(margin_rows.size, current_rows.size)

    def test_web_canvas_preprocess_keeps_drawing_position_instead_of_recentering(self):
        frame = np.full((224, 224, 3), 255, dtype=np.uint8)
        frame[20:180, 18:34] = 0

        web_tensor, web_preview = preprocess_for_web_canvas(frame, input_size=28)
        expected = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (28, 28), interpolation=cv2.INTER_AREA)

        self.assertEqual(web_tensor.shape, (1, 28, 28, 1))
        self.assertEqual(web_preview.shape, (28, 28))
        self.assertLess(float(web_tensor.min()), 0.2)
        self.assertGreater(float(web_tensor[:, :3, :3, :].mean()), 0.9)
        self.assertLess(int(web_preview[:, :8].min()), 10)
        self.assertTrue(np.array_equal(web_preview, expected))


from draw_game.preprocess import MIN_STROKE_PEN_POINTS, preprocess_strokes


def _make_stroke(points, tool="pen"):
    """Helper: build a stroke event dict like SharedCanvasState produces."""
    return {"type": "stroke", "tool": tool, "points": points}


def _line_stroke(n=20, x_start=0.1, y_start=0.1, dx=0.01, dy=0.0):
    """Return a single horizontal pen stroke with n points."""
    pts = [[round(x_start + i * dx, 4), round(y_start + i * dy, 4)] for i in range(n)]
    return _make_stroke(pts)


class StrokePreprocessTests(unittest.TestCase):
    def test_returns_none_when_no_events(self):
        result = preprocess_strokes([])
        self.assertIsNone(result)

    def test_returns_none_when_fewer_than_min_points(self):
        stroke = _make_stroke([[0.1, 0.1], [0.2, 0.2]])  # only 2 points
        result = preprocess_strokes([stroke])
        self.assertIsNone(result)

    def test_eraser_strokes_are_ignored(self):
        # Eraser-only → not enough pen points
        eraser = _make_stroke([[0.0, 0.0]] * 20, tool="eraser")
        result = preprocess_strokes([eraser])
        self.assertIsNone(result)

    def test_eraser_mixed_with_pen_only_counts_pen_points(self):
        eraser = _make_stroke([[0.0, 0.0]] * 50, tool="eraser")
        pen = _line_stroke(n=MIN_STROKE_PEN_POINTS)
        result = preprocess_strokes([eraser, pen])
        self.assertIsNotNone(result)

    def test_output_shape_is_1_128_5(self):
        stroke = _line_stroke(n=20)
        result = preprocess_strokes([stroke], seq_len=128)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 128, 5))

    def test_output_dtype_is_float32(self):
        stroke = _line_stroke(n=20)
        result = preprocess_strokes([stroke])
        self.assertEqual(result.dtype, np.float32)

    def test_end_token_is_present(self):
        stroke = _line_stroke(n=20)
        result = preprocess_strokes([stroke], seq_len=128)
        tensor = result[0]  # (128, 5)
        # Find end token: feature index 4 == 1.0
        end_positions = np.where(tensor[:, 4] == 1.0)[0]
        self.assertEqual(len(end_positions), 1, "Exactly one end token expected")
        end_pos = end_positions[0]
        # After end token, remaining slots must be all zeros
        if end_pos < 127:
            self.assertTrue(np.all(tensor[end_pos + 1:] == 0.0))

    def test_pen_down_and_pen_up_are_mutually_exclusive(self):
        stroke = _line_stroke(n=30)
        result = preprocess_strokes([stroke], seq_len=128)
        tensor = result[0]  # (128, 5)
        pen_down = tensor[:, 2]
        pen_up = tensor[:, 3]
        end_flag = tensor[:, 4]
        # Where end=1, both pen_down and pen_up should be 0
        end_mask = end_flag == 1.0
        self.assertTrue(np.all(pen_down[end_mask] == 0.0))
        self.assertTrue(np.all(pen_up[end_mask] == 0.0))
        # No timestep should have both pen_down and pen_up set
        both_set = (pen_down == 1.0) & (pen_up == 1.0)
        self.assertFalse(np.any(both_set))

    def test_dx_dy_values_are_normalized_not_255_scale(self):
        """dx/dy should be small normalized deltas (~[-1, 1]), not pixel values."""
        stroke = _line_stroke(n=20, dx=0.01)
        result = preprocess_strokes([stroke], seq_len=128)
        tensor = result[0]
        # All dx values in content region should be around 0.01, not ~2.55
        content_dx = tensor[:20, 0]
        self.assertTrue(
            np.all(np.abs(content_dx) <= 1.0),
            f"dx values exceed 1.0: {content_dx}",
        )

    def test_last_point_of_stroke_has_pen_up(self):
        stroke = _line_stroke(n=10)
        result = preprocess_strokes([stroke], seq_len=128)
        tensor = result[0]
        # Content uses 10 slots (indices 0-9); end token at 10.
        # Index 9 is the last content point of the stroke → pen_up=1.
        self.assertEqual(tensor[9, 3], 1.0, "Last stroke point should have pen_up=1")
        self.assertEqual(tensor[9, 2], 0.0, "Last stroke point should have pen_down=0")

    def test_subsampling_preserves_shape_when_too_many_points(self):
        # 300 points exceeds seq_len=128 content slots
        stroke = _line_stroke(n=300, dx=0.003)
        result = preprocess_strokes([stroke], seq_len=128)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 128, 5))

    def test_subsampled_output_still_has_end_token(self):
        stroke = _line_stroke(n=300, dx=0.003)
        result = preprocess_strokes([stroke], seq_len=128)
        tensor = result[0]
        end_positions = np.where(tensor[:, 4] == 1.0)[0]
        self.assertEqual(len(end_positions), 1)

    def test_two_strokes_produce_two_pen_up_events_before_end(self):
        s1 = _line_stroke(n=10)
        s2 = _line_stroke(n=10, x_start=0.5)
        result = preprocess_strokes([s1, s2], seq_len=128)
        tensor = result[0]
        # pen_up column: two 1.0 values before the end token, then 0.0
        end_pos = int(np.where(tensor[:, 4] == 1.0)[0][0])
        pen_up_count = int(tensor[:end_pos, 3].sum())
        self.assertEqual(pen_up_count, 2)

    def test_clear_event_in_list_does_not_crash(self):
        """preprocess_strokes should silently ignore non-stroke dict entries."""
        # SharedCanvasState.get_pen_strokes() already filters clears,
        # but the function should be robust.
        clear = {"type": "clear"}
        pen = _line_stroke(n=20)
        result = preprocess_strokes([clear, pen])
        # clear entry has no "points", so it is skipped; pen should produce output
        self.assertIsNotNone(result)

