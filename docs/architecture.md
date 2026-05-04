# Architecture

## Pipeline

```text
Canva screen crop
-> preprocess sketch
-> classify sketch
-> emit JSON
-> confidence/stability gate
-> preset phrase
-> Kokoro TTS
-> local audio output
```

## Modules

- `draw_game/config.py`: Loads tunable settings from defaults, `.env`, and environment variables.
- `draw_game/capture.py`: Captures a fixed rectangle using `mss` and returns an OpenCV/numpy image.
- `draw_game/preprocess.py`: Converts a crop into a normalized QuickDraw tensor plus a debug preview, with selectable rendering profiles and optional side-by-side profile comparison support. TFLite polarity remains white background `1.0`, black strokes `0.0`.
- `draw_game/classifier.py`: Loads labels and wraps TFLite or ONNX inference when configured; otherwise uses `StubClassifier`.
- `draw_game/tools/download_quickdraw_tflite.py`: Downloads the QuickDraw 345-class TFLite model files from Hugging Face into `draw_game/models/quickdraw-345-tflite/`.
- `draw_game/decision.py`: Maintains per-round speech gate state and summarizes recent `top3` predictions into a spoken primary label plus optional alternate label for chatty gameplay modes.
- `draw_game/responses.py`: Maps spoken labels, confidence tiers, and optional alternates to preset spoken phrases.
- `draw_game/tts_kokoro.py`: Non-fatal TTS wrapper with Kokoro-first behavior that starts speaking immediately when idle, never interrupts a current line, and replaces any pending line with the latest guess.
- `draw_game/web_canvas.py`: Serves a minimal mobile drawing page, keeps a shared in-memory stroke log for all connected sessions, and renders the canonical canvas frame for web-canvas mode.
- `draw_game/main.py`: Keyboard-controlled capture/classify/speak loop.

## Runtime Notes

The MVP is intentionally local-only. Google Meet remains the social layer, and audio routing into Meet is handled outside the app with a virtual audio cable.
