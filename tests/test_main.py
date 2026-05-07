import unittest

from draw_game import main


class FakeGate:
    def __init__(self, round_active=True, round_start_time=10.0):
        self.round_active = round_active
        self.round_start_time = round_start_time
        self.ended = False

    def end_round(self):
        self.round_active = False
        self.ended = True


class FakeCanvasState:
    def __init__(self):
        self.ended = False

    def end_round(self):
        self.ended = True


class MainLoopTests(unittest.TestCase):
    def test_auto_end_round_expires_gate_and_canvas_state(self):
        gate = FakeGate(round_active=True, round_start_time=10.0)
        canvas_state = FakeCanvasState()

        ended = main._auto_end_round_if_needed(
            gate,
            canvas_state,
            None,       # gemma_detector — not relevant to this test
            now=40.1,
            duration_sec=30.0,
        )

        self.assertTrue(ended)
        self.assertTrue(gate.ended)
        self.assertTrue(canvas_state.ended)

    def test_auto_end_round_ignores_disabled_timer(self):
        gate = FakeGate(round_active=True, round_start_time=10.0)
        canvas_state = FakeCanvasState()

        ended = main._auto_end_round_if_needed(
            gate,
            canvas_state,
            None,       # gemma_detector — not relevant to this test
            now=100.0,
            duration_sec=0.0,
        )

        self.assertFalse(ended)
        self.assertFalse(gate.ended)
        self.assertFalse(canvas_state.ended)
