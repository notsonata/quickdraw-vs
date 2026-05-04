import unittest

from draw_game.responses import LOW_CONFIDENCE_TAUNTS, make_low_confidence_taunt, make_spoken_line


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
                "Is it cat?",
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

    def test_make_spoken_line_can_hedge_between_two_labels(self):
        line = make_spoken_line("watermelon", 0.28, alternate_label="camouflage")

        self.assertIn(
            line,
            {
                "watermelon or camouflage.",
                "Maybe watermelon or camouflage.",
                "watermelon... or camouflage.",
                "Is it watermelon or camouflage?",
            },
        )

    def test_make_low_confidence_taunt_is_short_and_mild(self):
        line = make_low_confidence_taunt()

        self.assertIn(line, set(LOW_CONFIDENCE_TAUNTS))
