import unittest

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

        decision = gate.update(result())

        self.assertFalse(decision["should_speak"])
        self.assertEqual(decision["reason"], "round_not_active")

    def test_gate_accepts_after_delay_confidence_and_stability(self):
        times = iter([0.0, 1.1, 1.7])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

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
        gate.update(result("cat", 0.8))
        gate.update(result("cat", 0.8))

        duplicate = gate.update(result("cat", 0.9))
        self.assertFalse(duplicate["should_speak"])
        self.assertEqual(duplicate["reason"], "duplicate_label")
