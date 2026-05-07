# Project Plan

## Product Goal

Build a local MVP for a "Humans vs AI Draw Guessing" game. A human artist draws on a Canva canvas shared through Google Meet, while this app watches a configured screen crop, classifies the sketch, and speaks AI guesses locally.

## MVP Scope

- Capture a fixed screen rectangle where the Canva canvas is visible.
- Optionally serve a first-party single-user web canvas and classify uploaded snapshots instead of screen capture.
- Preprocess the crop for a QuickDraw-style classifier.
- Classify the image and emit JSON with `top1`, `confidence`, and `top3`.
- Gate guesses by delay, confidence, label stability, cooldown, duplicate label, and per-round guess count, with an optional fast-talk mode.
- Convert the accepted guess into a preset phrase using a spoken primary label and optional alternate label.
- Speak the phrase with Kokoro TTS when available, with console and optional `pyttsx3` fallback.
- Stay modular and debuggable.

## Non-Goals

- Google Meet API integration.
- Speech recognition for human guesses.
- Automatic winner detection.
- Ollama, YOLO, or free-form LLM commentary.
- Automatic Canva canvas detection.
- Web UI or database.

## Model Target

The intended classifier model is `zarqankhn/quickdraw-345-tflite`. Runtime supports the downloaded `quickdraw_model.tflite` file through `MODEL_BACKEND=tflite`, and must fall back to a stub classifier when model loading fails.

## Acceptance Criteria

- `python main.py` starts the app from the `draw_game` directory.
- `r` starts a round, `e` ends a round, `q` exits, and `s` saves a debug frame.
- The app captures the configured screen crop or reads the latest uploaded web-canvas image.
- Preprocessing handles frames without crashing.
- Missing classifier model uses `StubClassifier`.
- The decision gate enforces first-guess delay, confidence, stability, cooldown, duplicate label, and max guesses.
- Debug JSON contains `round_active`, `top1`, `confidence`, `top3`, `spoken_label`, `alternate_label`, `stable_ms`, `should_speak`, `reason`, and `ai_guesses_this_round`.
- Preset phrase generation can use a spoken primary label plus optional alternate label.
- Optional Gemma/PaliGemma vision can classify raw canvas frames, but must choose only from the QuickDraw label set.
- TTS failures do not crash the app.
