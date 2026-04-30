import unittest

import numpy as np

from draw_game.preprocess import preprocess_for_classifier


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
