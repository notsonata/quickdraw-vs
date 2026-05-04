from __future__ import annotations

import json
import logging
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

try:
    from . import capture, classifier, preprocess, responses, tts_kokoro
    from .config import settings
    from .decision import SpeechGate
except ImportError:  # pragma: no cover - supports python main.py from draw_game/
    import capture
    import classifier
    import preprocess
    import responses
    import tts_kokoro
    from config import settings
    from decision import SpeechGate


LOG_DIR = Path(__file__).resolve().parent / "logs"


def _run_profile_comparison(frame, clf) -> tuple[dict[str, list[list[str | float]]], dict[str, object]]:
    comparison: dict[str, list[list[str | float]]] = {}
    previews: dict[str, object] = {}
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


def log_startup_config() -> None:
    logging.info(
        "startup config: crop=(%s,%s,%s,%s) interval=%s backend=%s model=%s labels=%s tts=%s",
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


def main() -> int:
    setup_logging()
    log_startup_config()
    print_controls()
    print("Loading classifier. Wait for 'Ready' before starting a round.")

    running = True
    with RawTerminal():
        clf = classifier.create_classifier()
        logging.info("classifier type: %s", clf.__class__.__name__)
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
                logging.info("round start")
                print("Round started.")
            elif key == "e":
                gate.end_round()
                logging.info("round end")
                print("Round ended.")
            elif key == "s":
                try:
                    last_frame = capture.capture_canvas_crop()
                    _model_input, last_preview = preprocess.preprocess_for_classifier(last_frame)
                    profile_previews = None
                    if settings.PREPROCESS_COMPARE_PROFILES:
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
            if now - last_classify_time < settings.CLASSIFY_INTERVAL_SEC:
                time.sleep(0.01)
                continue
            last_classify_time = now

            try:
                last_frame = capture.capture_canvas_crop()
                model_input, last_preview = preprocess.preprocess_for_classifier(last_frame)
                result = clf.predict(model_input)
                profile_previews = None
                if settings.PREPROCESS_COMPARE_PROFILES:
                    comparison, profile_previews = _run_profile_comparison(last_frame, clf)
                    print(json.dumps(comparison, indent=2))
                if settings.DEBUG_SAVE_FRAMES or settings.PREPROCESS_COMPARE_PROFILES:
                    paths = capture.save_debug_artifacts(last_frame, last_preview, profile_previews)
                    logging.info("saved debug artifacts: %s", paths)
                decision = gate.update(result)
                if settings.DEBUG_PRINT_JSON:
                    print(json.dumps(decision, indent=2))
                if decision["should_speak"]:
                    if decision.get("speech_kind") == "taunt":
                        line = responses.make_low_confidence_taunt()
                        logging.info("spoken taunt: %s", line)
                    else:
                        line = responses.make_spoken_line(decision["top1"], decision["confidence"])
                        logging.info("spoken guess: %s", line)
                    print(line)
                    tts_kokoro.speak(line)
            except KeyboardInterrupt:
                running = False
            except Exception as exc:
                logging.exception("loop error")
                print(f"Loop error: {exc}")
                time.sleep(0.25)

    logging.info("shutdown")
    print("Exited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
