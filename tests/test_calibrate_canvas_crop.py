import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from draw_game.tools.calibrate_canvas_crop import compute_bounding_box, update_env_crop


class CalibrateCanvasCropTests(unittest.TestCase):
    def test_compute_bounding_box_uses_enclosing_rectangle(self):
        bbox = compute_bounding_box([(705, 226), (270, 790), (270, 226), (705, 790)])

        self.assertEqual(bbox, (270, 226, 435, 564))

    def test_compute_bounding_box_requires_four_points(self):
        with self.assertRaises(ValueError):
            compute_bounding_box([(1, 2), (3, 4), (5, 6)])

    def test_update_env_crop_replaces_existing_values(self):
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "CANVAS_X=1\nCANVAS_Y=2\nCANVAS_W=3\nCANVAS_H=4\nDEBUG_PRINT_JSON=true\n",
                encoding="utf-8",
            )

            update_env_crop(env_path, 270, 226, 435, 564)

            content = env_path.read_text(encoding="utf-8")
            self.assertIn("CANVAS_X=270\n", content)
            self.assertIn("CANVAS_Y=226\n", content)
            self.assertIn("CANVAS_W=435\n", content)
            self.assertIn("CANVAS_H=564\n", content)
            self.assertIn("DEBUG_PRINT_JSON=true\n", content)

    def test_update_env_crop_appends_missing_values(self):
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("DEBUG_PRINT_JSON=true\n", encoding="utf-8")

            update_env_crop(env_path, 10, 20, 30, 40)

            content = env_path.read_text(encoding="utf-8")
            self.assertIn("CANVAS_X=10\n", content)
            self.assertIn("CANVAS_Y=20\n", content)
            self.assertIn("CANVAS_W=30\n", content)
            self.assertIn("CANVAS_H=40\n", content)
