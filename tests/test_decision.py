import unittest
from types import SimpleNamespace

from draw_game.decision import SpeechGate


def result(label="cat", confidence=0.8, top3=None, top5=None):
    top3_value = top3 or [[label, confidence], ["dog", 0.1], ["rabbit", 0.05]]
    return {
        "top1": label,
        "confidence": confidence,
        "top3": top3_value,
        "top5": top5 or top3_value + [["horse", 0.03], ["frog", 0.02]],
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
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
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
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=3,
        )

        with patch("draw_game.decision.settings", base_settings):
            first = gate.update(result("bottlecap", 0.68))
            second = gate.update(result("bottlecap", 0.68))

        self.assertEqual(first["reason"], "not_stable")
        self.assertTrue(second["should_speak"])
        self.assertEqual(second["reason"], "normal_guess")
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
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", base_settings):
            gate.update(result("cat", 0.8))
            gate.update(result("cat", 0.8))

            duplicate = gate.update(result("cat", 0.9))
        self.assertFalse(duplicate["should_speak"])
        self.assertEqual(duplicate["reason"], "duplicate_label_cooldown")

    def test_gate_speaks_each_gemma_detection_even_for_duplicate_label(self):
        times = iter([0.0, 0.1, 0.2])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        base_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=1.0,
            AI_MIN_CONFIDENCE=0.65,
            AI_STABLE_FOR_SEC=0.5,
            AI_SPEECH_COOLDOWN_SEC=2.5,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=3,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
        )

        with patch("draw_game.decision.settings", base_settings):
            first = gate.update({**result("cat", 0.8), "source": "gemma"})
            second = gate.update({**result("cat", 0.8), "source": "gemma"})

        self.assertTrue(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertEqual(first["reason"], "gemma_detection")
        self.assertEqual(second["reason"], "gemma_detection")

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
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=3,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
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
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
        )
        with patch("draw_game.decision.settings", scan_settings):
            r = result("camouflage", 0.2)
            r["top3"] = [["camouflage", 0.2], ["tent", 0.18], ["leaf", 0.16]]
            r["top5"] = r["top3"]
            first = gate.update(r)
            second = gate.update(r)

        self.assertTrue(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertEqual(first["reason"], "speak_every_scan")
        self.assertEqual(second["reason"], "stalled_confidence_top5")
        self.assertNotEqual(second["spoken_label"], "camouflage")
        self.assertEqual(second["alternate_label"], "camouflage")

    def test_gate_prefers_recent_consensus_label_for_spoken_guess(self):
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
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
        )
        with patch("draw_game.decision.settings", scan_settings):
            gate.update(
                result(
                    "camouflage",
                    0.26,
                    top3=[["camouflage", 0.26], ["watermelon", 0.24], ["apple", 0.10]],
                )
            )
            second = gate.update(
                result(
                    "watermelon",
                    0.27,
                    top3=[["watermelon", 0.27], ["camouflage", 0.22], ["apple", 0.08]],
                )
            )

        self.assertEqual(second["spoken_label"], "watermelon")
        self.assertEqual(second["speech_kind"], "guess")

    def test_gate_can_request_hedged_speech_when_two_labels_are_close(self):
        times = iter([0.0, 0.1])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

        from unittest.mock import patch

        scan_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=True,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.10,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
        )
        with patch("draw_game.decision.settings", scan_settings):
            decision = gate.update(
                result(
                    "camouflage",
                    0.21,
                    top3=[["camouflage", 0.21], ["watermelon", 0.20], ["apple", 0.05]],
                )
            )

        self.assertEqual(decision["spoken_label"], "camouflage")
        self.assertEqual(decision["alternate_label"], "watermelon")

    def test_gate_taunts_immediately_when_below_min_confidence(self):
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
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
            AI_LOW_CONFIDENCE_TAUNT_SEC=2.5,
            AI_TAUNT_COOLDOWN_SEC=5.0,
        )
        with patch("draw_game.decision.settings", taunt_settings):
            very_low = result(
                "camouflage",
                0.02,
                top3=[["camouflage", 0.02], ["dog", 0.01], ["rabbit", 0.01]],
                top5=[["camouflage", 0.02], ["dog", 0.01], ["rabbit", 0.01], ["horse", 0.01], ["frog", 0.01]],
            )
            first = gate.update(very_low)
            second = gate.update(very_low)

        self.assertTrue(first["should_speak"])
        self.assertEqual(first["reason"], "low_confidence_taunt")
        self.assertEqual(first["speech_kind"], "taunt")
        self.assertFalse(second["should_speak"])
        self.assertEqual(second["reason"], "low_confidence")

    def test_gate_uses_top5_alternative_when_stuck_on_same_label(self):
        times = iter([0.0, 0.1, 0.2, 0.3])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

        from unittest.mock import patch

        scan_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=True,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.10,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
            AI_LOW_CONFIDENCE_TAUNT_SEC=2.5,
            AI_TAUNT_COOLDOWN_SEC=5.0,
        )
        with patch("draw_game.decision.settings", scan_settings):
            first = gate.update(
                result(
                    "dishwasher",
                    0.24,
                    top3=[["dishwasher", 0.24], ["oven", 0.22], ["stove", 0.20]],
                    top5=[
                        ["dishwasher", 0.24],
                        ["oven", 0.22],
                        ["stove", 0.20],
                        ["microwave", 0.18],
                        ["door", 0.17],
                    ],
                )
            )
            second = gate.update(
                result(
                    "dishwasher",
                    0.23,
                    top3=[["dishwasher", 0.23], ["oven", 0.22], ["stove", 0.19]],
                    top5=[
                        ["dishwasher", 0.23],
                        ["oven", 0.22],
                        ["stove", 0.19],
                        ["microwave", 0.18],
                        ["door", 0.17],
                    ],
                )
            )
            third = gate.update(
                result(
                    "dishwasher",
                    0.22,
                    top3=[["dishwasher", 0.22], ["oven", 0.21], ["stove", 0.20]],
                    top5=[
                        ["dishwasher", 0.22],
                        ["oven", 0.21],
                        ["stove", 0.20],
                        ["microwave", 0.18],
                        ["door", 0.17],
                    ],
                )
            )

        self.assertEqual(first["spoken_label"], "dishwasher")
        self.assertEqual(second["spoken_label"], "oven")
        self.assertEqual(second["alternate_label"], "dishwasher")
        self.assertEqual(third["spoken_label"], "microwave")  # Shifted due to anti-repeat excluding oven
        self.assertEqual(third["alternate_label"], "dishwasher")

    def test_gate_cycles_top5_when_confidence_stalls(self):
        times = iter([0.0, 0.1, 0.2, 0.3])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

        from unittest.mock import patch

        scan_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=True,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.10,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
            AI_LOW_CONFIDENCE_TAUNT_SEC=2.5,
            AI_TAUNT_COOLDOWN_SEC=5.0,
        )
        with patch("draw_game.decision.settings", scan_settings):
            first = gate.update(
                result(
                    "dishwasher",
                    0.91,
                    top3=[["dishwasher", 0.91], ["oven", 0.88], ["stove", 0.87]],
                    top5=[
                        ["dishwasher", 0.91],
                        ["oven", 0.88],
                        ["stove", 0.87],
                        ["microwave", 0.86],
                        ["door", 0.85],
                    ],
                )
            )
            second = gate.update(
                result(
                    "dishwasher",
                    0.91,
                    top3=[["dishwasher", 0.91], ["oven", 0.88], ["stove", 0.87]],
                    top5=[
                        ["dishwasher", 0.91],
                        ["oven", 0.88],
                        ["stove", 0.87],
                        ["microwave", 0.86],
                        ["door", 0.85],
                    ],
                )
            )
            third = gate.update(
                result(
                    "dishwasher",
                    0.91,
                    top3=[["dishwasher", 0.91], ["oven", 0.88], ["stove", 0.87]],
                    top5=[
                        ["dishwasher", 0.91],
                        ["oven", 0.88],
                        ["stove", 0.87],
                        ["microwave", 0.86],
                        ["door", 0.85],
                    ],
                )
            )

        self.assertEqual(first["spoken_label"], "dishwasher")
        self.assertEqual(second["spoken_label"], "oven")
        self.assertEqual(second["alternate_label"], "dishwasher")
        self.assertEqual(second["reason"], "stalled_confidence_top5")
        self.assertEqual(third["spoken_label"], "microwave")  # Shifted due to anti-repeat excluding oven
        self.assertEqual(third["alternate_label"], "dishwasher")

    def test_gate_taunt_cooldown_blocks_repeat(self):
        times = iter([0.0, 0.1, 3.0])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

        from unittest.mock import patch

        scan_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=True,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.10,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
            AI_LOW_CONFIDENCE_TAUNT_SEC=2.5,
            AI_TAUNT_COOLDOWN_SEC=5.0,
        )
        near_confident = result(
            "camouflage",
            0.05,
            top3=[["camouflage", 0.05], ["door", 0.09], ["oven", 0.07]],
            top5=[["camouflage", 0.05], ["door", 0.09], ["oven", 0.07], ["stove", 0.06], ["map", 0.05]],
        )
        with patch("draw_game.decision.settings", scan_settings):
            first = gate.update(near_confident)
            second = gate.update(near_confident)

        self.assertTrue(first["should_speak"])
        self.assertEqual(first["reason"], "low_confidence_taunt")
        self.assertEqual(first["speech_kind"], "taunt")
        self.assertFalse(second["should_speak"])
        self.assertEqual(second["reason"], "low_confidence")

    def test_gate_excludes_labels_below_min_spoken_confidence(self):
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
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
        )
        with patch("draw_game.decision.settings", scan_settings):
            r = result("apple", 0.9)
            # Make sure top5 has labels below 0.15 threshold
            r["top3"] = [["apple", 0.9], ["banana", 0.05], ["pear", 0.01]]
            r["top5"] = [["apple", 0.9], ["banana", 0.05], ["pear", 0.01], ["zero", 0.0], ["none", 0.0]]
            first = gate.update(r)
            second = gate.update(r)

        self.assertEqual(first["spoken_label"], "apple")
        self.assertEqual(second["reason"], "duplicate_label_cooldown")
        self.assertFalse(second["should_speak"])

    def test_gate_enforces_max_guesses_even_with_speak_every_scan(self):
        times = iter([0.0, 1.0, 2.0, 3.0, 4.0])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()

        from unittest.mock import patch
        scan_settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=True,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.10,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=2,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", scan_settings):
            r1 = result("apple", 0.9)
            r1["top5"] = [["apple", 0.9], ["banana", 0.8], ["pear", 0.7]]
            r2 = result("banana", 0.9)
            r2["top5"] = [["banana", 0.9], ["pear", 0.8], ["apple", 0.7]]
            r3 = result("pear", 0.9)
            r3["top5"] = [["pear", 0.9], ["apple", 0.8], ["banana", 0.7]]
            
            first = gate.update(r1)
            second = gate.update(r2)
            third = gate.update(r3)

        self.assertTrue(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertFalse(third["should_speak"])
        self.assertEqual(third["reason"], "max_guesses_per_round")

    def test_duplicate_label_cooldown_prevents_immediate_repeat(self):
        times = iter([0.0, 1.0, 5.0])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        from types import SimpleNamespace
        settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.5,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=1.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=10,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", settings):
            first = gate.update(result("cat", 0.9))
            second = gate.update(result("cat", 0.9))
            
        self.assertTrue(first["should_speak"])
        self.assertFalse(second["should_speak"])
        self.assertEqual(second["reason"], "duplicate_label_cooldown")

    def test_same_label_repeated_after_cooldown_with_high_confidence(self):
        times = iter([0.0, 1.0, 12.0])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        from types import SimpleNamespace
        settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.5,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=10,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", settings):
            first = gate.update(result("cat", 0.9))
            second = gate.update(result("cat", 0.9))
            
        self.assertTrue(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertEqual(second["reason"], "same_label_repeat_allowed")

    def test_same_label_repeated_after_cooldown_with_low_confidence(self):
        times = iter([0.0, 1.0, 12.0])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        from types import SimpleNamespace
        settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.5,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=10,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", settings):
            first = gate.update(result("cat", 0.9))
            second = gate.update(result("cat", 0.6))
            
        self.assertTrue(first["should_speak"])
        self.assertFalse(second["should_speak"])
        self.assertEqual(second["reason"], "low_confidence")

    def test_same_label_reaches_repeat_cap(self):
        times = iter([0.0, 1.0, 12.0, 23.0, 34.0])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        from types import SimpleNamespace
        settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.5,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=10,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", settings):
            first = gate.update(result("cat", 0.9))
            second = gate.update(result("cat", 0.9))
            third = gate.update(result("cat", 0.9))
            fourth = gate.update(result("cat", 0.9))
            
        self.assertTrue(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertTrue(third["should_speak"])
        self.assertFalse(fourth["should_speak"])
        self.assertEqual(fourth["reason"], "same_label_repeat_cap")

    def test_gate_never_speaks_zero_confidence_labels(self):
        times = iter([0.0, 0.1, 0.2])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        from types import SimpleNamespace
        settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=True,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.01,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=999,
            AI_REPEAT_LABEL_COOLDOWN_SEC=0.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.0,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=5,
        )
        with patch("draw_game.decision.settings", settings):
            r = result("cloud", 0.2)
            r["top5"] = [["cloud", 0.2], ["zero", 0.0], ["none", 0.0], ["door", 0.16], ["map", 0.15]]
            first = gate.update(r)
            second = gate.update(r)

        self.assertEqual(first["spoken_label"], "cloud")
        self.assertNotIn(second["spoken_label"], {"zero", "none"})

    def test_top1_change_resets_duplicate_suppression(self):
        times = iter([0.0, 1.0, 2.0, 3.0])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        from types import SimpleNamespace
        settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.5,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=10,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", settings):
            first = gate.update(result("cat", 0.9))
            second = gate.update(result("dog", 0.9))
            third = gate.update(result("cat", 0.9))
            
        self.assertTrue(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertTrue(third["should_speak"])

    def test_max_guesses_per_round_is_enforced(self):
        times = iter([float(i) for i in range(10)])
        gate = SpeechGate(now_func=lambda: next(times))
        gate.start_round()
        from unittest.mock import patch
        from types import SimpleNamespace
        settings = SimpleNamespace(
            AI_SPEAK_EVERY_SCAN=False,
            AI_FIRST_GUESS_DELAY_SEC=0.0,
            AI_MIN_CONFIDENCE=0.5,
            AI_STABLE_FOR_SEC=0.0,
            AI_SPEECH_COOLDOWN_SEC=0.0,
            AI_MIN_SPOKEN_LABEL_CONFIDENCE=0.15,
            MAX_AI_GUESSES_PER_ROUND=2,
            AI_REPEAT_LABEL_COOLDOWN_SEC=10.0,
            AI_HIGH_CONFIDENCE_REPEAT=0.85,
            AI_MAX_REPEAT_SAME_LABEL_PER_ROUND=3,
        )
        with patch("draw_game.decision.settings", settings):
            first = gate.update(result("cat", 0.9))
            second = gate.update(result("dog", 0.9))
            third = gate.update(result("bird", 0.9))
            
        self.assertTrue(first["should_speak"])
        self.assertTrue(second["should_speak"])
        self.assertFalse(third["should_speak"])
        self.assertEqual(third["reason"], "max_guesses_per_round")
