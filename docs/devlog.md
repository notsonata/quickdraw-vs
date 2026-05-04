# Devlog

## 2026-04-30

- Started MVP scaffolding for the local Humans vs AI Draw Guessing game.
- Implemented the local Python MVP with fixed crop capture, preprocessing, classifier/stub contract, speech gating, preset responses, Kokoro-first TTS fallback, logging, README, and focused tests.
- Verified with `.venv/bin/python -m unittest discover -s tests -v` and `.venv/bin/python -m py_compile draw_game/*.py`.
- Started wiring the Hugging Face `zarqankhn/quickdraw-345-tflite` model through a TFLite backend with downloader, model polarity fixes, and fallback tests.
- Downloaded the Hugging Face TFLite model files into `draw_game/models/quickdraw-345-tflite/`.
- Verified tests and syntax after TFLite wiring. Live TFLite inference is blocked in the current Python 3.14 `.venv` because `tensorflow` and `tflite-runtime` are not available for this interpreter.
- Added Python 3.14 runtime fallback warning for `MODEL_BACKEND=tflite` and documented Python 3.11 as the recommended real-model environment.
- Recreated `.venv` with Python 3.11, installed TensorFlow, loaded the real TFLite model, and verified a prediction smoke test plus the full unit suite.
- Installed Kokoro TTS and sounddevice, verified `KPipeline`, and updated the TTS wrapper for Kokoro `KPipeline.Result` audio extraction.
- Fixed QuickDraw preprocessing so detected strokes are cropped, padded, centered, and scaled into the 28x28 model input instead of shrinking the full Canva crop; verified with focused preprocessing coverage, full unit tests, and syntax compilation.
- Added a click-based calibration tool that captures a full-screen screenshot, lets the user click four Canva corners, and writes the enclosing crop rectangle into `draw_game/.env`.
- Expanded the snarky low-confidence AI response list in `draw_game/responses.py` with more insults to improve game personality.
- Updated TTS dispatch so accepted guesses start immediately when idle, active speech is never interrupted, and only the latest pending guess is kept for playback after the current line finishes; added regression coverage for both prompt dispatch and latest-after-finish handoff.
- Added configurable QuickDraw preprocessing profiles (`current`, `dilate_before_resize`, `dilate_after_resize`, `antialias_grayscale`, `more_margin`), plus comparison-mode top-3 debug output and per-profile 280x280 previews for side-by-side rendering checks.
