from __future__ import annotations

import json
import logging
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

try:
    from . import capture, classifier, gemma_vision, preprocess, responses, tts_kokoro, web_canvas
    from .config import settings
    from .decision import SpeechGate
except ImportError:  # pragma: no cover - supports python main.py from draw_game/
    import capture
    import classifier
    import gemma_vision
    import preprocess
    import responses
    import tts_kokoro
    import web_canvas
    from config import settings
    from decision import SpeechGate


LOG_DIR = Path(__file__).resolve().parent / "logs"


def _run_profile_comparison(frame, clf) -> tuple[dict[str, list[list[str | float]]], dict[str, object]]:
    comparison: dict[str, list[list[str | float]]] = {}
    previews: dict[str, object] = {}
    if settings.CANVAS_SOURCE == "web":
        model_input, preview = preprocess.preprocess_for_web_canvas(frame)
        result = clf.predict(model_input)
        comparison["web_canvas"] = result["top3"]
        previews["web_canvas"] = preview
        return comparison, previews
    for profile_name in preprocess.PREPROCESS_PROFILES:
        model_input, preview = preprocess.preprocess_for_classifier_with_profile(frame, profile_name)
        result = clf.predict(model_input)
        comparison[profile_name] = result["top3"]
        previews[profile_name] = preview
    return comparison, previews


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


class RawTerminal:
    def __enter__(self):
        if not sys.stdin.isatty():
            self._old = None
            return self
        self._old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if readable:
        return sys.stdin.read(1).lower()
    return None


def print_controls() -> None:
    print("Controls:")
    print("  r = start round")
    print("  e = end round")
    print("  q = quit")
    print("  s = save debug frame")
    print()
    print(
        "Set the crop rectangle with CANVAS_X, CANVAS_Y, CANVAS_W, and CANVAS_H "
        "in draw_game/.env or environment variables."
    )
    if settings.CANVAS_W <= 0 or settings.CANVAS_H <= 0:
        print("Warning: crop dimensions are invalid. CANVAS_W and CANVAS_H must be positive.")
    if settings.CANVAS_SOURCE == "web":
        print(
            f"Web canvas mode: open http://localhost:{settings.WEB_CANVAS_PORT} "
            "or your Cloudflare Tunnel URL on the phone."
        )


def log_startup_config() -> None:
    logging.info(
        "startup config: source=%s crop=(%s,%s,%s,%s) interval=%s backend=%s model=%s labels=%s tts=%s",
        settings.CANVAS_SOURCE,
        settings.CANVAS_X,
        settings.CANVAS_Y,
        settings.CANVAS_W,
        settings.CANVAS_H,
        settings.CLASSIFY_INTERVAL_SEC,
        settings.MODEL_BACKEND,
        settings.MODEL_PATH,
        settings.LABELS_PATH,
        settings.TTS_ENABLED,
    )
    logging.info(
        "gemma config: enabled=%s model=%s interval=%s confidence=%s",
        settings.GEMMA_ENABLED,
        settings.GEMMA_MODEL,
        settings.GEMMA_INTERVAL_SEC,
        settings.GEMMA_CONFIDENCE,
    )


def _start_web_canvas_server() -> tuple[web_canvas.SharedCanvasState, object, threading.Thread]:
    image_store = web_canvas.SharedCanvasState()
    server = web_canvas.create_server(
        settings.WEB_CANVAS_HOST,
        settings.WEB_CANVAS_PORT,
        image_store,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.info(
        "web canvas server listening on %s:%s",
        settings.WEB_CANVAS_HOST,
        server.server_port,
    )
    return image_store, server, thread


def _auto_end_round_if_needed(gate, image_store, gemma_detector, now: float, duration_sec: float) -> bool:
    duration = max(0.0, float(duration_sec))
    if not gate.round_active or duration <= 0.0:
        return False
    if now - gate.round_start_time < duration:
        return False

    gate.end_round()
    tts_kokoro.clear_pending()
    if gemma_detector is not None:
        gemma_detector.clear_pending_result()
    if image_store is not None and hasattr(image_store, "end_round"):
        image_store.end_round()
    return True


def main() -> int:
    setup_logging()
    log_startup_config()
    print_controls()
    print("Loading classifier. Wait for 'Ready' before starting a round.")

    running = True
    image_store = None
    web_server = None
    with RawTerminal():
        if settings.CANVAS_SOURCE == "web":
            image_store, web_server, _web_thread = _start_web_canvas_server()
        clf = classifier.create_classifier()
        logging.info("classifier type: %s", clf.__class__.__name__)
        gemma_detector = None
        if settings.GEMMA_ENABLED:
            labels = getattr(clf, "labels", classifier.load_labels(settings.LABELS_PATH))
            gemma_detector = gemma_vision.GemmaVisionDetector(labels)
            logging.info("Gemma vision detector enabled with %s labels", len(labels))
            print("Loading Gemma vision model. This can take a bit.")
            gemma_detector.preload()
            logging.info("Gemma vision model loaded")
        tts_kokoro.prime()
        gate = SpeechGate()
        last_classify_time = 0.0
        last_frame = None
        last_preview = None
        print("Ready. Press r to start a round.")
        while running:
            key = read_key()
            if key == "q":
                running = False
                continue
            if key == "r":
                gate.start_round()
                if gemma_detector is not None:
                    gemma_detector.clear_pending_result()
                if image_store is not None:
                    image_store.start_round(settings.ROUND_DURATION_SEC)
                    image_store.clear_strokes()  # discard strokes from previous round
                logging.info("round start")
                print("Round started.")
            elif key == "e":
                gate.end_round()
                tts_kokoro.clear_pending()
                if gemma_detector is not None:
                    gemma_detector.clear_pending_result()
                if image_store is not None:
                    image_store.end_round()
                logging.info("round end")
                print("Round ended.")
            elif key == "s":
                try:
                    if settings.CANVAS_SOURCE == "web":
                        last_frame = image_store.get_latest_frame()
                        last_preview = None  # no image preview in stroke mode
                    else:
                        last_frame = capture.capture_canvas_crop()
                        _model_input, last_preview = preprocess.preprocess_for_classifier(last_frame)
                    profile_previews = None
                    if settings.PREPROCESS_COMPARE_PROFILES and settings.CANVAS_SOURCE != "web":
                        _comparison, profile_previews = _run_profile_comparison(last_frame, clf)
                    paths = capture.save_debug_artifacts(last_frame, last_preview, profile_previews)
                    logging.info("saved debug artifacts: %s", paths)
                    print("Saved debug artifacts:")
                    for path in paths:
                        print(f"  {path}")
                except Exception as exc:
                    logging.exception("failed to save debug artifacts")
                    print(f"Could not save debug artifacts: {exc}")

            if not gate.round_active:
                time.sleep(0.03)
                continue

            now = time.monotonic()
            if _auto_end_round_if_needed(gate, image_store, gemma_detector, now, settings.ROUND_DURATION_SEC):
                logging.info("round auto-ended after %.2f seconds", settings.ROUND_DURATION_SEC)
                print("Round ended.")
                continue
            if now - last_classify_time < settings.CLASSIFY_INTERVAL_SEC:
                time.sleep(0.01)
                continue
            last_classify_time = now

            try:
                if settings.CANVAS_SOURCE == "web":
                    # --- stroke-sequence path ---
                    stroke_events = image_store.get_pen_strokes()
                    model_input = preprocess.preprocess_strokes(stroke_events)
                    if model_input is None:
                        # Too few pen points — wait for more drawing.
                        time.sleep(0.05)
                        continue
                    last_preview = None
                    # Rendered frame is only needed for Gemma or debug saves.
                    last_frame = None
                    if gemma_detector is not None or settings.DEBUG_SAVE_FRAMES:
                        try:
                            last_frame = image_store.get_latest_frame()
                        except web_canvas.NoCanvasFrameError:
                            last_frame = None
                else:
                    last_frame = capture.capture_canvas_crop()
                    model_input, last_preview = preprocess.preprocess_for_classifier(last_frame)
                result = clf.predict(model_input)
                gemma_result = None
                if gemma_detector is not None:
                    gemma_result = gemma_detector.predict(last_frame, now=now)
                    if gemma_result is not None:
                        logging.info(
                            "gemma detection: %s confidence=%s",
                            gemma_result["top1"],
                            gemma_result["confidence"],
                        )
                decision_result = gemma_result or result
                profile_previews = None
                if settings.PREPROCESS_COMPARE_PROFILES:
                    comparison, profile_previews = _run_profile_comparison(last_frame, clf)
                    print(json.dumps(comparison, indent=2))
                if settings.DEBUG_SAVE_FRAMES or settings.PREPROCESS_COMPARE_PROFILES:
                    paths = capture.save_debug_artifacts(last_frame, last_preview, profile_previews)
                    logging.info("saved debug artifacts: %s", paths)
                decision = gate.update(decision_result)
                if decision["should_speak"]:
                    if decision.get("speech_kind") == "taunt":
                        line = responses.make_low_confidence_taunt()
                        log_label = "spoken taunt"
                    else:
                        line = responses.make_spoken_line(
                            decision["spoken_label"],
                            decision["confidence"],
                            alternate_label=decision.get("alternate_label"),
                        )
                        log_label = "spoken guess"
                    if settings.DEBUG_PRINT_JSON:
                        log_message = f"{log_label}: {line}\n{line}\n{json.dumps(decision, indent=2)}"
                    else:
                        log_message = f"{log_label}: {line}\n{line}"
                    logging.info(log_message)
                    tts_kokoro.speak(
                        line,
                        interrupt=bool(decision.get("interrupt_current")),
                        speech_kind=str(decision.get("speech_kind", "guess")),
                    )
                elif settings.DEBUG_PRINT_JSON:
                    print(json.dumps(decision, indent=2))
            except web_canvas.NoCanvasFrameError:
                time.sleep(0.05)
            except KeyboardInterrupt:
                running = False
            except Exception as exc:
                logging.exception("loop error")
                print(f"Loop error: {exc}")
                time.sleep(0.25)

    if web_server is not None:
        web_server.shutdown()
        web_server.server_close()
    logging.info("shutdown")
    print("Exited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
