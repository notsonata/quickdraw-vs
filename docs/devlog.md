# Devlog

## 2026-05-08 - Fix Fused Loop Numpy Import

Resolved a runtime `NameError` by importing numpy in the main loop where fused image tensor stats are logged.

- **Changes**: Added `import numpy as np` in `draw_game/main.py`.
- **Validation**: Not run (runtime-only change).

## 2026-05-08 - SpeechGate Repeat Cooldown Tuning

Adjusted SpeechGate repeat suppression to allow high-confidence repeats after a per-label cooldown while enforcing a per-round cap and clearer reason tags.

- **Changes**: Added per-label repeat counts, enforced `AI_REPEAT_LABEL_COOLDOWN_SEC` + `AI_HIGH_CONFIDENCE_REPEAT` for same-label repeats, applied per-round label cap, and standardized `normal_guess`/`duplicate_label_cooldown`/`same_label_repeat_allowed`/`same_label_repeat_cap`/`low_confidence` reasons.
- **Tests**: Updated repeat suppression coverage and added a zero-confidence top5 guard test.
- **Validation**: `.venv/bin/python -m unittest discover -s tests -v` (112 tests, 0 failures).

## 2026-05-08 - Fused Image+Stroke TFLite Model Integration

Added multi-input fused TFLite support for `quickdraw_fused_model_float32.tflite` (30-class, image+stroke dual input).

- **Config**: Added `MODEL_INPUT_MODE` (`image`|`stroke`|`fused`) and `MODEL_IMAGE_SIZE` to `config.py`; `.env` updated to point at the fused model with `MODEL_INPUT_MODE=fused`, `MODEL_IMAGE_SIZE=64`, `MODEL_SEQ_LEN=256`.
- **Preprocess**: Added `preprocess_image_for_fused(frame, size)` in `preprocess.py` as a thin wrapper over `preprocess_for_web_canvas`. Returns `[1, 64, 64, 1]` float32 tensor, white=1.0/black=0.0 convention, no inversion.
- **Classifier**: Updated `TFLiteClassifier._prepare_input` to read the model's actual input shape from `input_details[0]` instead of global settings (allows 28×28 and 64×64 single-input models to coexist). Added `_set_inputs` to support both `np.ndarray` (single-input) and `dict {"image": ..., "stroke": ...}` (multi-input). Routing: tensor name substring first (`"image"` or `"stroke"`), then shape fallback. Raises clear `RuntimeError` on shape mismatch or ambiguous routing. `predict()` now accepts both forms.
- **Main**: Web canvas branch gains a fused path gated on `settings.MODEL_INPUT_MODE == "fused"`. Strokes are preprocessed first; image rendering only occurs after a valid stroke tensor is confirmed.
- **Tests**: Added 8 classifier tests (`FakeFusedInterpreter`, `FakeFusedInterpreterShapeOnly`, `FakeFusedInterpreterAmbiguous`, routing/shape/label-count/real-model tests) and 5 preprocess tests (`FusedPreprocessTests`). All 100 tests pass (1 skipped: sandbox network binding).
- **Smoke test result**: Real fused model loaded; `top1=apple confidence=0.5646 top3=[['apple', 0.5646], ['moon', 0.1585], ['banana', 0.1456]]` on dummy all-ones image + minimal horizontal stroke.
- **Validation**: `.venv/bin/python -m unittest discover -s tests -v` → 100 pass, 1 skip, 0 fail.


## 2026-05-08 - Log Spoken Lines With Decision JSON

Summary of spoken guess logging format update.

- **Changes**: When a line is spoken, the log now records the spoken line followed by the full decision JSON (in the same order as console output).
- **Impact**: Log files provide complete context for each spoken line without needing to cross-reference console output.
- **Validation**: Manual inspection of a spoken guess log entry.

## 2026-05-07 - Always Taunt Below Minimum Confidence

Summary of low-confidence taunt behavior update.

- **Changes**: When `confidence` falls below `AI_MIN_CONFIDENCE`, the speech gate now emits a taunt immediately (subject to taunt cooldown) instead of waiting for a prolonged low-confidence window.
- **Impact**: The AI always responds audibly during weak guesses, keeping the banter continuous.
- **Validation**: Updated unit coverage to confirm immediate taunts and cooldown suppression of repeats.

## 2026-05-07 - Confidence-Stall Top-5 Speech Cycling

Summary of fast-talk confidence-stall speech behavior update.

- **Changes**: Added confidence-stall tracking in the speech gate and, when `top1` confidence stops changing across repeated scans, rotates spoken guesses through the current top-5 candidates instead of repeating only `top1`.
- **Impact**: Chatty gameplay no longer sounds frozen on one label when the classifier confidence is flat.
- **Validation**: Added unit coverage that verifies unchanged-confidence scans walk through top-5 labels in order.

## 2026-05-07 - Gemma Alias Handling

Summary of Gemma label validation fix.

- **Changes**: Extended Gemma semantic alias matching so captions containing words like `person` map to the QuickDraw `face` label, and made Gemma fallback candidates prefer drawable labels.
- **Impact**: Valid Gemma reads are less likely to be discarded, and Gemma speech alternates avoid odd landmark fallbacks when possible.
- **Validation**: Added focused unit coverage for caption aliases and Gemma fallback candidates.

## 2026-05-07 - Web Canvas Round Timer

Summary of countdown and automatic round-ending behavior.

- **Changes**: Added `ROUND_DURATION_SEC`, shared web-canvas round state, a visible countdown over the drawing field, and main-loop auto-end handling.
- **Impact**: Starting a round now starts the web timer; the app ends the round automatically when the timer expires.
- **Validation**: Added unit coverage for timer countdown/expiry and main-loop auto-end behavior.

## 2026-05-07 - Nonblocking Gemma Detection

Summary of Gemma runtime latency fix.

- **Changes**: Moved PaliGemma detection into a single background worker, added raw Gemma output logging, decoded only newly generated tokens, and replaced the giant label-list prompt with a short VQA prompt while keeping QuickDraw label validation.
- **Impact**: QuickDraw can continue speaking on its normal scan cadence while Gemma works; when a valid Gemma label is ready, it is prioritized on the next loop tick.
- **Validation**: Added unit coverage that Gemma prediction returns immediately, does not start overlapping model jobs, and uses a short prompt.

## 2026-05-07 - Preload Gemma Weights

Summary of PaliGemma load timing fix.

- **Changes**: Added explicit Gemma model preloading during startup and cleared stale pending Gemma results when a new round starts.
- **Impact**: The heavy Torch weight load happens before `Ready` instead of during a live round, so the first Gemma guess can arrive during play instead of after the round ends.
- **Validation**: Added unit coverage for Gemma preload and stale-result clearing.

## 2026-05-07 - Gemma Vision Label Arbiter

Summary of optional local Gemma/PaliGemma vision integration for raw canvas detection.

- **Changes**: Added a local PaliGemma vision adapter that reads canvas frames through Transformers, constrains output to QuickDraw labels, returns the existing classifier JSON shape, and lets Gemma detections bypass duplicate/cooldown speech gates.
- **Impact**: QuickDraw remains the fast local scanner, while valid Gemma detections are spoken whenever they arrive.
- **Validation**: Added focused unit coverage for Gemma label validation, frame conversion, and Gemma speech priority.

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
- Changed the fast-talk speech layer to summarize recent `top3` predictions into a spoken primary label plus optional alternate label, so the AI can keep talking frequently while sounding less random on unstable sketches.
- Added a single-user mobile web canvas mode served directly from the Python app, with pen, eraser, and clear tools plus PNG snapshot uploads that the existing classifier loop can read instead of a screen crop.
- Replaced the snapshot-based web canvas with a shared event-synced canvas so every open session stays in sync and the classifier renders from one canonical server-side stroke log.
- Fixed fast-talk fallback rotation so repeated weak `top1` guesses now walk the current top-5 candidates in order instead of getting stuck, and lowered the live confidence floor in `draw_game/.env` to `0.03` for more aggressive guessing.
