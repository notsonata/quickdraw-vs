# Tasks

### [High] Build Local Humans vs AI Draw Guessing MVP

Implement the requested Python MVP pipeline with fixed-region capture, preprocessing, classifier/stub, JSON output, speech gate, preset responses, Kokoro-compatible TTS fallback, logging, and usage docs.

- **Files**: `draw_game/`, `README.md`, `requirements.txt`, `docs/`
- **Context**: The app lets AI guess early from a Canva canvas shared through Google Meet, without integrating with Meet APIs.
- **Status**: Done

### [High] Wire QuickDraw TFLite Model

Add Hugging Face download support and a TensorFlow Lite classifier backend for `zarqankhn/quickdraw-345-tflite`, while preserving ONNX and stub fallback behavior.

- **Files**: `draw_game/config.py`, `draw_game/classifier.py`, `draw_game/preprocess.py`, `draw_game/tools/`, `README.md`, `requirements.txt`, `tests/`
- **Context**: Normal runtime should use the real 345-class QuickDraw TFLite model when configured, but missing model files must still fall back cleanly to `StubClassifier`.
- **Status**: Done

Implemented and verified under Python 3.11 with TensorFlow. The local `.env` points at `draw_game/models/quickdraw-345-tflite/quickdraw_model.tflite`.

### [High] Improve QuickDraw Preprocessing Scale

Adjust preprocessing so small sketches inside the configured Canva crop are centered and scaled into the 28x28 QuickDraw model input instead of shrinking the full screen crop.

- **Files**: `draw_game/preprocess.py`, `tests/test_preprocess.py`
- **Context**: Live TFLite predictions stayed low-confidence and unrelated because the full Canva crop was being resized directly to 28x28, making drawings too small for the classifier.
- **Status**: Done

### [High] Add Click-Based Crop Calibration

Add a calibration tool that lets the user click the four visible Canva corners on a live screenshot and writes the enclosing rectangle into `draw_game/.env`.

- **Files**: `draw_game/tools/`, `tests/`, `README.md`
- **Context**: Manual crop coordinates are error-prone and directly affect recognition quality. The existing runtime still consumes an axis-aligned rectangle, so calibration should write `CANVAS_X`, `CANVAS_Y`, `CANVAS_W`, and `CANVAS_H`.
- **Status**: Done

### [Low] Expand Snarky AI Responses

Add more variety to the low-confidence taunts to keep the game engaging and frustrating for the artist.

- **Files**: `draw_game/responses.py`
- **Context**: The AI should have a wider range of snarky insults when it can't figure out what the human is drawing.
- **Status**: Done

### [High] Use Latest-After-Finish TTS Handoff

Speak accepted guesses immediately when idle, never interrupt a line already in progress, and keep only the latest pending guess to speak next after the current line finishes.

- **Files**: `draw_game/tts_kokoro.py`, `tests/test_classifier.py`, `docs/architecture.md`, `docs/devlog.md`
- **Context**: The game should react immediately when possible without talking over itself, and stale pending guesses should be dropped in favor of the most recent detection.
- **Status**: Done
