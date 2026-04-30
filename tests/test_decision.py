import unittest
from types import SimpleNamespace

from draw_game.decision import SpeechGate


def result(label="cat", confidence=0.8):
    return {
        "top1": label,
        "confidence": confidence,
        "top3": [[label, confidence], ["dog", 0.1], ["rabbit", 0.05]],
    }


class SpeechGateTests(unittest.TestCase):
    def test_gate_rejects_when_round_is_inactive(self):
        gate = SpeechGate(now_func=lambda: 100.0)
        from unittest.mock import patch
        base_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=1.0,
            AI_MIN_CONFIDENCE=0.65,
            AI_STABLE_FOR_SEC=0.5,
            AI_SPEECH_COOLDOWN_SEC=2.5,
            MAX_AI_GUESSES_PER_ROUND=3,
        )

        with patch("draw_game.decision.settings", base_settings):
            decision = gate.update(result())

        self.assertFalse(decision["should_speak"])
        self.assertEqual(decision["reason"], "round_not_active")

    def test_gate_accepts_after_delay_confidence_and_stability(self):
        times = iter([0.0, 1.1, 1.7])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        base_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=1.0,
            AI_MIN_CONFIDENCE=0.65,
            AI_STABLE_FOR_SEC=0.5,
            AI_SPEECH_COOLDOWN_SEC=2.5,
            MAX_AI_GUESSES_PER_ROUND=3,
        )

        with patch("draw_game.decision.settings", base_settings):
            first = gate.update(result("bottlecap", 0.68))
            second = gate.update(result("bottlecap", 0.68))

        self.assertEqual(first["reason"], "not_stable")
        self.assertTrue(second["should_speak"])
        self.assertEqual(second["reason"], "stable_confident_guess")
        self.assertEqual(second["stable_ms"], 600)
        self.assertEqual(second["ai_guesses_this_round"], 1)

    def test_gate_blocks_duplicate_label_after_spoken_guess(self):
        times = iter([0.0, 1.1, 1.7, 4.5])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        base_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=1.0,
            AI_MIN_CONFIDENCE=0.65,
            AI_STABLE_FOR_SEC=0.5,
            AI_SPEECH_COOLDOWN_SEC=2.5,
            MAX_AI_GUESSES_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", base_settings):
            gate.update(result("cat", 0.8))
            gate.update(result("cat", 0.8))

            duplicate = gate.update(result("cat", 0.9))
        self.assertFalse(duplicate["should_speak"])
        self.assertEqual(duplicate["reason"], "duplicate_label")

    def test_gate_allows_fast_speaking_with_shorter_runtime_thresholds(self):
        times = iter([0.0, 0.4, 0.65])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

        from unittest.mock import patch

        fast_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=0.35,
            AI_MIN_CONFIDENCE=0.10,
            AI_STABLE_FOR_SEC=0.2,
            AI_SPEECH_COOLDOWN_SEC=2.5,
            MAX_AI_GUESSES_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", fast_settings):
            first = gate.update(result("balloon", 0.25))
            second = gate.update(result("balloon", 0.25))

        self.assertEqual(first["reason"], "not_stable")
        self.assertTrue(second["should_speak"])

    def test_gate_can_speak_every_scan_even_for_duplicate_labels(self):
        times = iter([0.0, 0.1, 0.2])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

        from unittest.mock import patch

        scan_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=True,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.10,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            MAX_AI_GUESSES_PER_ROUND=999,
        )
        with patch("draw_game.decision.settings", scan_settings):
            first = gate.update(result("camouflage", 0.2))
            second = gate.update(result("camouflage", 0.2))

        self.assertTrue(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertEqual(first["reason"], "speak_every_scan")
        self.assertEqual(second["reason"], "speak_every_scan")

    def test_gate_emits_taunt_after_long_low_confidence_period(self):
        times = iter([0.0, 0.1, 3.0])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

        from unittest.mock import patch

        taunt_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=True,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.10,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            MAX_AI_GUESSES_PER_ROUND=999,
            AI_LOW_CONFIDENCE_TAUNT_SEC=2.5,
            AI_TAUNT_COOLDOWN_SEC=5.0,
        )
        with patch("draw_game.decision.settings", taunt_settings):
            first = gate.update(result("camouflage", 0.02))
            second = gate.update(result("camouflage", 0.02))

        self.assertFalse(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertEqual(second["reason"], "low_confidence_taunt")
        self.assertEqual(second["speech_kind"], "taunt")
