import unittest

from draw_game.responses import make_spoken_line


class ResponseTests(unittest.TestCase):
    def test_make_spoken_line_replaces_underscores_and_ends_cleanly(self):
        line = make_spoken_line("wine_bottle", 0.9)

        self.assertIn("wine bottle", line)
        self.assertNotIn("_", line)
        self.assertTrue(line.endswith("."))

    def test_make_spoken_line_uses_medium_templates_below_high_confidence(self):
        line = make_spoken_line("cat", 0.7)

        self.assertIn("cat", line)
        self.assertIn(
            line,
            {
                "Maybe cat.",
                "I am guessing cat.",
                "Could be cat.",
                "My current guess is cat.",
            },
        )
