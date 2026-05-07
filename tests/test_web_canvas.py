import http.client
import json
import unittest
from threading import Thread

import cv2
import numpy as np

from draw_game.web_canvas import SharedCanvasState, create_server


class SharedCanvasStateTests(unittest.TestCase):
    def test_round_status_counts_down_and_expires(self):
        now = 100.0
        store = SharedCanvasState(now_func=lambda: now)

        store.start_round(30.0)
        active = store.get_round_status()
        now = 131.0
        expired = store.get_round_status()

        self.assertTrue(active["round_active"])
        self.assertEqual(active["remaining_sec"], 30)
        self.assertFalse(expired["round_active"])
        self.assertEqual(expired["remaining_sec"], 0)

    def test_end_round_clears_round_status(self):
        store = SharedCanvasState(now_func=lambda: 200.0)

        store.start_round(60.0)
        store.end_round()

        status = store.get_round_status()
        self.assertFalse(status["round_active"])
        self.assertEqual(status["remaining_sec"], 0)

    def test_get_latest_frame_requires_any_events(self):
        store = SharedCanvasState()

        with self.assertRaises(RuntimeError):
            store.get_latest_frame()

    def test_add_stroke_event_renders_dark_pixels(self):
        store = SharedCanvasState(canvas_size=128)
        store.add_event(
            {
                "type": "stroke",
                "tool": "pen",
                "width": 0.02,
                "points": [[0.2, 0.2], [0.8, 0.8]],
            }
        )

        frame = store.get_latest_frame()

        self.assertEqual(frame.shape, (128, 128, 3))
        self.assertGreater(int((frame[:, :, 0] < 250).sum()), 0)

    def test_clear_event_resets_canvas(self):
        store = SharedCanvasState(canvas_size=64)
        store.add_event(
            {
                "type": "stroke",
                "tool": "pen",
                "width": 0.03,
                "points": [[0.1, 0.1], [0.9, 0.9]],
            }
        )
        store.add_event({"type": "clear"})

        frame = store.get_latest_frame()

        self.assertEqual(int((frame[:, :, 0] < 250).sum()), 0)

    def test_events_since_returns_incremental_updates(self):
        store = SharedCanvasState()
        first = store.add_event({"type": "clear"})
        second = store.add_event(
            {
                "type": "stroke",
                "tool": "pen",
                "width": 0.01,
                "points": [[0.0, 0.0], [1.0, 1.0]],
            }
        )

        events = store.get_events_since(first)

        self.assertEqual(second, 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["seq"], 2)
        self.assertEqual(events[0]["type"], "stroke")


class WebCanvasServerTests(unittest.TestCase):
    def test_server_accepts_event_upload_and_serves_status(self):
        store = SharedCanvasState(canvas_size=64)
        try:
            server = create_server("127.0.0.1", 0, store)
        except PermissionError:
            self.skipTest("Sandbox does not allow binding a local test server")
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            conn.request(
                "POST",
                "/event",
                body=json.dumps(
                    {
                        "type": "stroke",
                        "tool": "pen",
                        "width": 0.02,
                        "points": [[0.2, 0.2], [0.8, 0.8]],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            conn.close()

            self.assertEqual(response.status, 200)
            payload = json.loads(body)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["seq"], 1)

            saved = store.get_latest_frame()
            self.assertGreater(int((saved[:, :, 0] < 250).sum()), 0)

            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            conn.request("GET", "/status")
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            conn.close()

            self.assertEqual(response.status, 200)
            payload = json.loads(body)
            self.assertTrue(payload["has_image"])
            self.assertEqual(payload["seq"], 1)
        finally:
            server.shutdown()
            server.server_close()
