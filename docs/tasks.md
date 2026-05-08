# Tasks

### [High] Build Local Humans vs AI Draw Guessing MVP

Implement the requested Python MVP pipeline with fixed-region capture, preprocessing, classifier/stub, JSON output, speech gate, preset responses, Kokoro-compatible TTS fallback, logging, and usage docs.

- **Files**: `draw_game/`, `README.md`, `requirements.txt`, `docs/`
- **Context**: The app lets AI guess early from a Canva canvas shared through Google Meet, without integrating with Meet APIs.
- **Status**: Done

### [High] Loosen SpeechGate Repeat Suppression

Allow the AI to repeat a stable, high-confidence label occasionally while still preventing spam and enforcing per-round caps.

- **Files**: `draw_game/decision.py`, `tests/test_decision.py`, `docs/devlog.md`
- **Context**: Duplicate suppression was blocking the same label indefinitely, even after long high-confidence stability.
- **Status**: Done

### [High] Fix Web Canvas Fused Loop NameError

Resolve the missing numpy import that crashes the fused web canvas path during debug logging.

- **Files**: `draw_game/main.py`, `docs/devlog.md`
- **Context**: The runtime loop hit `NameError: np is not defined` during fused image tensor logging.
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

### [High] Add QuickDraw Preprocessing Profiles

Add multiple preprocessing profiles plus a comparison mode so the 28x28 QuickDraw rendering can be evaluated on the same captured frame without changing model loading, classifier JSON shape, gate behavior, or TTS.

- **Files**: `draw_game/config.py`, `draw_game/preprocess.py`, `draw_game/main.py`, `draw_game/capture.py`, `tests/test_preprocess.py`, `docs/`
- **Context**: Tall thin sketches such as `door` can become too jagged or too thin after downscaling, which pushes predictions toward unrelated labels. The app needs profile-level experimentation focused on the final 28x28 model input.
- **Status**: Done

### [High] Keep Fast-Talk Gameplay While Hedging Noisy Guesses

Make `AI_SPEAK_EVERY_SCAN` mode stay chatty without blindly repeating single-frame `top1` labels by summarizing recent `top3` candidates into a spoken label and optional alternate label.

- **Files**: `draw_game/decision.py`, `draw_game/main.py`, `draw_game/responses.py`, `tests/test_decision.py`, `tests/test_responses.py`, `docs/`
- **Context**: Fast-paced rounds work better when the AI talks often, but raw 28x28 QuickDraw predictions can flicker between related labels like `watermelon` and `camouflage`. Speech should stay lively while sounding less arbitrarily wrong.
- **Status**: Done

### [High] Add Single-User Mobile Web Canvas Mode

Serve a minimal phone-friendly drawing canvas with only pen, eraser, and clear tools, and let the classifier loop read uploaded PNG snapshots instead of a screen crop when configured.

- **Files**: `draw_game/config.py`, `draw_game/main.py`, `draw_game/web_canvas.py`, `tests/test_web_canvas.py`, `README.md`, `docs/`
- **Context**: The existing Canva workflow is convenient, but a first-party drawing surface gives cleaner input without adding multi-user complexity. The app only needs to support one active drawer.
- **Status**: Done

### [High] Sync All Web Canvas Sessions

Make every open web canvas session share the same drawing state so multiple viewers or drawers stay aligned and the classifier always reads the server-side canonical canvas.

- **Files**: `draw_game/web_canvas.py`, `draw_game/main.py`, `tests/test_web_canvas.py`, `README.md`, `docs/`
- **Context**: A single-user snapshot uploader is not enough once multiple phones or browsers open the canvas. The app needs one shared in-memory drawing that every client can replay and the classifier can render consistently.
- **Status**: Done

### [High] Rotate Weak Repeated Guesses Through Top-5 Candidates

When the model gets stuck repeating the same weak `top1` prediction in fast-talk mode, rotate through the current top-5 candidates instead of parroting one label forever, and lower the live confidence floor for gameplay.

- **Files**: `draw_game/decision.py`, `draw_game/.env`, `tests/test_decision.py`
- **Context**: The current QuickDraw model often latches onto one wrong label. The speech layer needs to stay chatty while moving through plausible alternates instead of sounding broken.
- **Status**: Done

### [High] Add Gemma Vision Label Arbiter

Add an optional local Gemma/PaliGemma vision path that reads the raw canvas frame, constrains detections to the same QuickDraw label set, and prioritizes each valid Gemma detection for speech.

- **Files**: `draw_game/gemma_vision.py`, `draw_game/config.py`, `draw_game/main.py`, `draw_game/decision.py`, `tests/`, `docs/`
- **Context**: QuickDraw often gets stuck on weak repeated guesses. Gemma/PaliGemma should provide a higher-level read of the canvas without inventing labels outside the game prompt list.
- **Status**: Done

### [High] Add Web Canvas Round Timer

Show a countdown on the drawing canvas and automatically end active rounds when the configured duration elapses.

- **Files**: `draw_game/config.py`, `draw_game/main.py`, `draw_game/web_canvas.py`, `tests/`, `docs/`
- **Context**: Rounds should end without manual keyboard input, and drawers should see the remaining time directly on the drawing surface.
- **Status**: Done

### [High] Cycle Through Top-5 On Confidence Stall

After speaking the current `top1`, detect when subsequent scans keep the same `top1` confidence and rotate speech through the current top-5 labels instead of repeating one guess.

- **Files**: `draw_game/decision.py`, `tests/test_decision.py`, `docs/`
- **Context**: Fast-talk gameplay sounds stuck when confidence does not change frame-to-frame; speech should walk alternatives automatically.
- **Status**: Done

### [Medium] Taunt Whenever Confidence Falls Below Minimum

When `confidence < AI_MIN_CONFIDENCE`, speak a taunt immediately (respecting the taunt cooldown) instead of staying silent.

- **Files**: `draw_game/decision.py`, `tests/test_decision.py`, `docs/`
- **Context**: Low-confidence moments should always produce an audible taunt rather than a quiet skip.
- **Status**: Done

### [Medium] Log Spoken Lines With Decision JSON

When a spoken line is emitted, log the spoken line followed by the full decision JSON so the log file mirrors the console output sequence.

- **Files**: `draw_game/main.py`, `docs/`
- **Context**: Troubleshooting spoken guesses is easier when the log includes the exact line and the full decision payload.
- **Status**: Done

### [High] Add Fused Image+Stroke TFLite Model Support

Integrate a dual-input TFLite model (`quickdraw_fused_model_float32.tflite`) that accepts both a 64×64 grayscale image tensor and a [256, 5] stroke-sequence tensor simultaneously, while preserving full backward compatibility with single-input stroke and image models.

- **Files**: `draw_game/config.py`, `draw_game/preprocess.py`, `draw_game/classifier.py`, `draw_game/main.py`, `draw_game/.env`, `tests/test_classifier.py`, `tests/test_preprocess.py`
- **Context**: The new fused model combines image and stroke features for better 30-class recognition. It must plug into the existing web canvas runtime without breaking stroke-only or image-only paths.
- **Status**: Done
