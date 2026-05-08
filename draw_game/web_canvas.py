from __future__ import annotations

import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - dependency checked at runtime
    cv2 = None


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Draw</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f2f2f2;
      --panel: rgba(255,255,255,0.94);
      --line: #111111;
      --muted: #6b7280;
      --border: rgba(17,17,17,0.12);
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      height: 100%;
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    body {
      overscroll-behavior: none;
      touch-action: none;
    }
    .app {
      position: fixed;
      inset: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      padding: env(safe-area-inset-top) 0 env(safe-area-inset-bottom) 0;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 12px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
    }
    .tool-group {
      display: flex;
      gap: 8px;
    }
    button {
      appearance: none;
      border: 1px solid var(--border);
      background: white;
      color: #111111;
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 16px;
      font-weight: 600;
      min-height: 48px;
    }
    button.active {
      background: #111111;
      color: white;
      border-color: #111111;
    }
    .status {
      margin-left: auto;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    .canvas-wrap {
      min-height: 0;
      padding: 10px;
      display: grid;
      place-items: center;
      position: relative;
    }
    .timer-badge {
      position: absolute;
      top: 20px;
      right: 20px;
      min-width: 72px;
      padding: 8px 10px;
      border-radius: 8px;
      background: rgba(17,17,17,0.88);
      color: white;
      font-size: 20px;
      font-weight: 700;
      text-align: center;
      font-variant-numeric: tabular-nums;
      pointer-events: none;
    }
    canvas {
      width: min(calc(100vw - 20px), calc(100vh - 100px));
      height: min(calc(100vw - 20px), calc(100vh - 100px));
      max-width: 100%;
      max-height: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      background: white;
      border-radius: 14px;
      box-shadow: inset 0 0 0 1px rgba(17,17,17,0.06);
      touch-action: none;
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <div class="tool-group">
        <button id="penButton" class="active" type="button">Pen</button>
        <button id="eraserButton" type="button">Eraser</button>
        <button id="clearButton" type="button">Clear</button>
      </div>
      <div id="status" class="status">Connecting</div>
    </div>
    <div class="canvas-wrap">
      <canvas id="board"></canvas>
      <div id="timerBadge" class="timer-badge">0:00</div>
    </div>
  </div>
  <script>
    const canvas = document.getElementById('board');
    const context = canvas.getContext('2d');
    const statusNode = document.getElementById('status');
    const penButton = document.getElementById('penButton');
    const eraserButton = document.getElementById('eraserButton');
    const clearButton = document.getElementById('clearButton');
    const timerBadge = document.getElementById('timerBadge');

    const clientId = (crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now() + Math.random());
    const pollIntervalMs = 250;
    let tool = 'pen';
    let drawing = false;
    let lastPoint = null;
    let activePoints = [];
    let baseEvents = [];
    let pendingEvents = [];
    let lastSeenSeq = 0;
    let pollTimer = null;
    let roundActive = false;
    let remainingSec = 0;

    function setStatus(text) {
      statusNode.textContent = text;
    }

    function formatTimer(seconds) {
      const value = Math.max(0, Math.ceil(Number(seconds) || 0));
      const minutes = Math.floor(value / 60);
      const remainder = String(value % 60).padStart(2, '0');
      return `${minutes}:${remainder}`;
    }

    function setRoundStatus(round) {
      roundActive = Boolean(round && round.round_active);
      remainingSec = round ? Number(round.remaining_sec || 0) : 0;
      if (roundActive) {
        const text = formatTimer(remainingSec);
        setStatus(text);
        timerBadge.textContent = text;
      } else {
        setStatus('Round ended');
        timerBadge.textContent = '0:00';
      }
    }

    function setTool(nextTool) {
      tool = nextTool;
      penButton.classList.toggle('active', tool === 'pen');
      eraserButton.classList.toggle('active', tool === 'eraser');
    }

    function resizeCanvas() {
      const ratio = window.devicePixelRatio || 1;
      const bounds = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(bounds.width * ratio));
      canvas.height = Math.max(1, Math.round(bounds.height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.lineCap = 'round';
      context.lineJoin = 'round';
      redrawCanvas();
    }

    function clearCanvasPixels() {
      context.save();
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.restore();
    }

    function currentCanvasBounds() {
      return canvas.getBoundingClientRect();
    }

    function getPoint(event) {
      const bounds = currentCanvasBounds();
      return {
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      };
    }

    function pointToNormalized(point) {
      const bounds = currentCanvasBounds();
      return [
        Math.min(1, Math.max(0, point.x / Math.max(1, bounds.width))),
        Math.min(1, Math.max(0, point.y / Math.max(1, bounds.height))),
      ];
    }

    function normalizedToPoint(point) {
      const bounds = currentCanvasBounds();
      return {
        x: point[0] * bounds.width,
        y: point[1] * bounds.height,
      };
    }

    function widthForTool(activeTool) {
      const bounds = currentCanvasBounds();
      const minSide = Math.max(1, Math.min(bounds.width, bounds.height));
      const pxWidth = activeTool === 'eraser' ? 28 : 8;
      return pxWidth / minSide;
    }

    function drawStrokeEvent(event) {
      const points = event.points || [];
      if (!points.length) {
        return;
      }
      const activeTool = event.tool || 'pen';
      const bounds = currentCanvasBounds();
      const minSide = Math.max(1, Math.min(bounds.width, bounds.height));
      context.strokeStyle = activeTool === 'eraser' ? '#ffffff' : '#000000';
      context.lineWidth = Math.max(1, (event.width || 0.01) * minSide);
      context.beginPath();
      const first = normalizedToPoint(points[0]);
      context.moveTo(first.x, first.y);
      for (let index = 1; index < points.length; index += 1) {
        const point = normalizedToPoint(points[index]);
        context.lineTo(point.x, point.y);
      }
      if (points.length === 1) {
        context.lineTo(first.x + 0.001, first.y + 0.001);
      }
      context.stroke();
    }

    function redrawCanvas() {
      clearCanvasPixels();
      for (const event of baseEvents) {
        if (event.type === 'clear') {
          clearCanvasPixels();
        } else if (event.type === 'stroke') {
          drawStrokeEvent(event);
        }
      }
      for (const event of pendingEvents) {
        if (event.type === 'clear') {
          clearCanvasPixels();
        } else if (event.type === 'stroke') {
          drawStrokeEvent(event);
        }
      }
      if (drawing && activePoints.length) {
        drawStrokeEvent({
          type: 'stroke',
          tool,
          width: widthForTool(tool),
          points: activePoints.map(pointToNormalized),
        });
      }
    }

    async function sendEvent(event) {
      const response = await fetch('/event', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...event, client_id: clientId }),
      });
      if (!response.ok) {
        throw new Error('event upload failed');
      }
      return response.json();
    }

    async function flushPendingEvents() {
      if (!pendingEvents.length) {
        return;
      }
      const nextEvent = pendingEvents[0];
      try {
        const payload = await sendEvent(nextEvent);
        setRoundStatus(payload.round);
        nextEvent.seq = payload.seq;
        lastSeenSeq = Math.max(lastSeenSeq, payload.seq);
        baseEvents.push(nextEvent);
        pendingEvents.shift();
        redrawCanvas();
        if (pendingEvents.length) {
          queueMicrotask(flushPendingEvents);
        }
      } catch (_error) {
        setStatus('Offline');
      }
    }

    function enqueueEvent(event) {
      pendingEvents.push(event);
      redrawCanvas();
      if (pendingEvents.length === 1) {
        queueMicrotask(flushPendingEvents);
      }
    }

    function startStroke(event) {
      if (!roundActive) {
        setStatus('Round ended');
        return;
      }
      event.preventDefault();
      drawing = true;
      lastPoint = getPoint(event);
      activePoints = [lastPoint];
      redrawCanvas();
      canvas.setPointerCapture(event.pointerId);
    }

    function moveStroke(event) {
      if (!drawing) return;
      event.preventDefault();
      const nextPoint = getPoint(event);
      lastPoint = nextPoint;
      activePoints.push(nextPoint);
      redrawCanvas();
    }

    function endStroke(event) {
      if (!drawing) return;
      event.preventDefault();
      drawing = false;
      if (activePoints.length) {
        enqueueEvent({
          type: 'stroke',
          tool,
          width: widthForTool(tool),
          points: activePoints.map(pointToNormalized),
        });
      }
      activePoints = [];
      lastPoint = null;
      redrawCanvas();
    }

    async function pollEvents() {
      try {
        const response = await fetch(`/events?since=${lastSeenSeq}`, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error('poll failed');
        }
        const payload = await response.json();
        setRoundStatus(payload.round);
        const events = payload.events || [];
        if (events.length) {
          const pendingKeys = new Set(pendingEvents.map((event) => JSON.stringify(event)));
          for (const event of events) {
            lastSeenSeq = Math.max(lastSeenSeq, event.seq || 0);
            const comparable = JSON.stringify({
              type: event.type,
              tool: event.tool,
              width: event.width,
              points: event.points,
              client_id: event.client_id,
            });
            if (event.client_id === clientId && pendingKeys.has(comparable)) {
              continue;
            }
            baseEvents.push(event);
          }
          redrawCanvas();
        }
        if (!pendingEvents.length && !roundActive) {
          setStatus('Round ended');
        } else if (!pendingEvents.length && roundActive) {
          setStatus(formatTimer(remainingSec));
        }
      } catch (_error) {
        if (!pendingEvents.length) {
          setStatus('Offline');
        }
      } finally {
        pollTimer = window.setTimeout(pollEvents, pollIntervalMs);
      }
    }

    penButton.addEventListener('click', () => setTool('pen'));
    eraserButton.addEventListener('click', () => setTool('eraser'));
    clearButton.addEventListener('click', () => {
      enqueueEvent({ type: 'clear' });
    });
    canvas.addEventListener('pointerdown', startStroke);
    canvas.addEventListener('pointermove', moveStroke);
    canvas.addEventListener('pointerup', endStroke);
    canvas.addEventListener('pointercancel', endStroke);
    canvas.addEventListener('pointerleave', endStroke);
    window.addEventListener('resize', resizeCanvas);

    resizeCanvas();
    pollEvents();
  </script>
</body>
</html>
"""


class NoCanvasFrameError(RuntimeError):
    pass


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed. Run pip install -r requirements.txt.")


class SharedCanvasState:
    def __init__(self, canvas_size: int = 1024, now_func=None) -> None:
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self._next_seq = 1
        self._canvas_size = max(64, int(canvas_size))
        self._now = now_func or time.monotonic
        self._round_active = False
        self._round_ends_at = 0.0

    def start_round(self, duration_sec: float) -> None:
        duration = max(0.0, float(duration_sec))
        with self._lock:
            self._round_active = True
            self._round_ends_at = self._now() + duration if duration > 0.0 else 0.0

    def end_round(self) -> None:
        with self._lock:
            self._round_active = False
            self._round_ends_at = 0.0

    def get_round_status(self) -> dict:
        with self._lock:
            active = self._round_active
            remaining = 0.0
            if active and self._round_ends_at > 0.0:
                remaining = max(0.0, self._round_ends_at - self._now())
                if remaining <= 0.0:
                    active = False
                    self._round_active = False
                    self._round_ends_at = 0.0
            return {
                "round_active": active,
                "remaining_sec": int(round(remaining)),
            }

    def _validate_event(self, event: dict) -> dict:
        event_type = str(event.get("type", "")).strip().lower()
        if event_type not in {"stroke", "clear"}:
            raise ValueError("Canvas event type must be 'stroke' or 'clear'.")
        validated = {"type": event_type, "client_id": str(event.get("client_id", ""))}
        if event_type == "clear":
            return validated

        tool = str(event.get("tool", "pen")).strip().lower()
        if tool not in {"pen", "eraser"}:
            raise ValueError("Canvas stroke tool must be 'pen' or 'eraser'.")
        width = float(event.get("width", 0.01))
        if width <= 0.0:
            raise ValueError("Canvas stroke width must be positive.")
        points = event.get("points", [])
        if not isinstance(points, list) or not points:
            raise ValueError("Canvas stroke points are required.")

        normalized_points: list[list[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Canvas stroke points must be [x, y] pairs.")
            x = min(1.0, max(0.0, float(point[0])))
            y = min(1.0, max(0.0, float(point[1])))
            normalized_points.append([x, y])

        validated["tool"] = tool
        validated["width"] = min(0.25, width)
        validated["points"] = normalized_points
        return validated

    def add_event(self, event: dict) -> int:
        validated = self._validate_event(event)
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            stored = {"seq": seq, **validated}
            self._events.append(stored)
        return seq

    def has_image(self) -> bool:
        with self._lock:
            return bool(self._events)

    def current_seq(self) -> int:
        with self._lock:
            return self._next_seq - 1

    def get_events_since(self, seq: int) -> list[dict]:
        seq = int(seq)
        with self._lock:
            return [event.copy() for event in self._events if int(event["seq"]) > seq]

    def _render_locked(self) -> np.ndarray:
        _ensure_cv2()
        image = np.full((self._canvas_size, self._canvas_size, 3), 255, dtype=np.uint8)
        for event in self._events:
            if event["type"] == "clear":
                image.fill(255)
                continue
            color = (255, 255, 255) if event["tool"] == "eraser" else (0, 0, 0)
            thickness = max(1, int(round(float(event["width"]) * self._canvas_size)))
            points = [
                (
                    int(round(point[0] * (self._canvas_size - 1))),
                    int(round(point[1] * (self._canvas_size - 1))),
                )
                for point in event["points"]
            ]
            if len(points) == 1:
                cv2.circle(image, points[0], max(1, thickness // 2), color, thickness=-1)
                continue
            for index in range(1, len(points)):
                cv2.line(image, points[index - 1], points[index], color, thickness, lineType=cv2.LINE_AA)
        return image

    def get_latest_frame(self) -> np.ndarray:
        with self._lock:
            if not self._events:
                raise NoCanvasFrameError("No canvas image has been uploaded yet.")
            return self._render_locked()

    def get_pen_strokes(self) -> list[dict]:
        """Return pen stroke events that occurred after the most recent clear.

        Eraser strokes are excluded. The returned list is safe to read outside
        the lock (each item is a shallow copy made while the lock is held).
        """
        with self._lock:
            result: list[dict] = []
            for event in self._events:
                if event["type"] == "clear":
                    result = []
                    continue
                if event["type"] == "stroke" and event.get("tool") == "pen":
                    result.append(event.copy())
            return result

    def clear_strokes(self) -> None:
        """Inject a system-generated clear event to reset the drawing canvas.

        Call this at round start to discard strokes from the previous round
        without requiring the user to press the Clear button.
        """
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            self._events.append({"seq": seq, "type": "clear", "client_id": "__system__"})


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def create_server(host: str, port: int, canvas_state: SharedCanvasState) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
        ) -> None:
            try:
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _parse_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            if not payload:
                raise ValueError("Missing request body.")
            return json.loads(payload.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                self._send_bytes(HTTPStatus.OK, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/status":
                self._send_bytes(
                    HTTPStatus.OK,
                    _json_bytes(
                        {
                            "status": "ok",
                            "has_image": canvas_state.has_image(),
                            "seq": canvas_state.current_seq(),
                            "round": canvas_state.get_round_status(),
                        }
                    ),
                    "application/json",
                )
                return
            if path == "/events":
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                since = 0
                for pair in query.split("&"):
                    if pair.startswith("since="):
                        try:
                            since = int(pair.split("=", 1)[1])
                        except ValueError:
                            since = 0
                self._send_bytes(
                    HTTPStatus.OK,
                    _json_bytes(
                        {
                            "status": "ok",
                            "events": canvas_state.get_events_since(since),
                            "round": canvas_state.get_round_status(),
                        }
                    ),
                    "application/json",
                )
                return
            self._send_bytes(
                HTTPStatus.NOT_FOUND,
                _json_bytes({"status": "error", "message": "not found"}),
                "application/json",
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/event":
                self._send_bytes(
                    HTTPStatus.NOT_FOUND,
                    _json_bytes({"status": "error", "message": "not found"}),
                    "application/json",
                )
                return
            try:
                event = self._parse_json_body()
                seq = canvas_state.add_event(event)
            except Exception as exc:
                self._send_bytes(
                    HTTPStatus.BAD_REQUEST,
                    _json_bytes({"status": "error", "message": str(exc)}),
                    "application/json",
                )
                return
            self._send_bytes(
                HTTPStatus.OK,
                _json_bytes(
                    {
                        "status": "ok",
                        "seq": seq,
                        "id": str(uuid.uuid4()),
                        "round": canvas_state.get_round_status(),
                    }
                ),
                "application/json",
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return ThreadingHTTPServer((host, int(port)), Handler)
