import unittest

from draw_game.responses import make_low_confidence_taunt, make_spoken_line


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
                "cat.",
                "I think it's cat.",
            },
        )

    def test_make_spoken_line_uses_short_high_confidence_templates(self):
        line = make_spoken_line("soccer_ball", 0.9)

        self.assertIn(
            line,
            {
                "soccer ball.",
                "It's soccer ball.",
            },
        )

    def test_make_low_confidence_taunt_is_short_and_mild(self):
        line = make_low_confidence_taunt()

        self.assertIn(
            line,
            {
                "Draw better.",
                "This is rough.",
                "Help me out here.",
                "That drawing is brutal.",
                "Give me something clearer.",
            },
        )
