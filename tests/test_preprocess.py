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
