# Humans vs AI Draw Guessing MVP

Local Python MVP for a "Humans vs AI Draw Guessing" game. A human can either draw on a Canva canvas shared in Google Meet or use a built-in phone-friendly web canvas, while this app classifies the sketch, gates early AI guesses, and speaks preset guesses through local TTS.

Google Meet is only the social layer. This app does not integrate with Google Meet APIs.

## Current Limitations

- Screen capture uses a fixed rectangle only.
- The target classifier model, `zarqankhn/quickdraw-345-tflite`, is not installed automatically.
- TFLite is the default real-model backend when configured; ONNX support remains available.
- If no supported model exists, the app uses `StubClassifier`.
- No speech recognition, winner detection, Canva detection, or database.
- Kokoro support is best-effort because local package and playback APIs vary; console and optional `pyttsx3` fallback keep the loop running.

## Installation

```bash
cd quickdraw-vs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Real Model Environment Setup

Use Python 3.11 for the real TFLite QuickDraw model. Python 3.14 is not currently suitable for this project because TensorFlow/TFLite wheels are unavailable or unreliable there. With Python 3.14, the app will warn and fall back to `StubClassifier`.

Recommended macOS setup:

```bash
cd quickdraw-vs
deactivate 2>/dev/null || true
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If `python3.11` is not installed, install it with Homebrew:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
```

On an Intel Mac, use:

```bash
/usr/local/bin/python3.11 -m venv .venv
```

Verify TensorFlow and the TFLite interpreter:

```bash
python - <<'PY'
import sys
print("Python:", sys.version)
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("TFLite interpreter:", tf.lite.Interpreter)
PY
```

Run the real model:

```bash
MODEL_BACKEND=tflite \
MODEL_PATH=draw_game/models/quickdraw-345-tflite/quickdraw_model.tflite \
LABELS_PATH=draw_game/models/quickdraw-345-tflite/labels.txt \
python -m draw_game.main
```

## Configure the Input Source

Two input modes are available:

- `CANVAS_SOURCE=screen`: capture a fixed rectangle from Canva or another shared app
- `CANVAS_SOURCE=web`: serve the built-in single-user mobile canvas

### Screen Crop Mode

Copy the example config and edit it:

```bash
cp draw_game/.env.example draw_game/.env
```

Example values:

```env
CANVAS_X=100
CANVAS_Y=200
CANVAS_W=800
CANVAS_H=600
CLASSIFIER_INPUT_SIZE=28
DEBUG_PRINT_JSON=true
DEBUG_SAVE_FRAMES=false
TTS_ENABLED=true
```

The crop rectangle should cover only the visible Canva drawing area in the screen share. Use `s` while the app is running to save raw and preprocessed debug images into `draw_game/logs/`.

You can also calibrate the crop by clicking the four visible Canva corners:

```bash
python draw_game/tools/calibrate_canvas_crop.py
```

Calibration controls:

- Click the four corners in any order.
- Press `r` to reset the points.
- Press `Enter` to save the crop into `draw_game/.env`.
- Press `q` or `Esc` to cancel.

### Web Canvas Mode

Set these values in `draw_game/.env`:

```env
CANVAS_SOURCE=web
WEB_CANVAS_HOST=0.0.0.0
WEB_CANVAS_PORT=8765
```

Then run the app and open `http://localhost:8765` locally or expose port `8765` with Cloudflare Tunnel for the phone.

The web canvas is intentionally minimal:

- pen
- eraser
- clear
- black ink on a white canvas

All open web canvas sessions now stay in sync by replaying shared stroke events from the Python server. The classifier reads the server-rendered shared canvas, not the last client snapshot.

## Run

```bash
cd quickdraw-vs
python3 -m draw_game.main
```

Controls:

- `r`: start round
- `e`: end round
- `q`: quit
- `s`: save debug frame

During an active round, the runtime is:

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

## Test Without a Real Model

No model is required for local loop testing. If `MODEL_PATH` is missing or unsupported, the app prints a warning and uses `StubClassifier`, which emits valid prediction JSON with labels such as `cat`, `dog`, `bottlecap`, `spreadsheet`, and `stitches`.

Expected prediction shape:

```json
{
  "top1": "bottlecap",
  "confidence": 0.68,
  "top3": [
    ["bottlecap", 0.68],
    ["circle", 0.14],
    ["gear", 0.08]
  ]
}
```

Expected speech gate JSON:

```json
{
  "round_active": true,
  "top1": "bottlecap",
  "confidence": 0.68,
  "top3": [
    ["bottlecap", 0.68],
    ["circle", 0.14],
    ["gear", 0.08]
  ],
  "spoken_label": "bottlecap",
  "alternate_label": null,
  "stable_ms": 620,
  "should_speak": true,
  "reason": "stable_confident_guess",
  "ai_guesses_this_round": 1
}
```

The spoken line will be a preset phrase such as:

```text
bottlecap.
```

In fast-talk mode, the speech layer can also hedge between two recent candidates, for example `watermelon or camouflage.`

## Using the Real QuickDraw TFLite Model

Install dependencies and download the Hugging Face model files:

```bash
cd quickdraw-vs
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python draw_game/tools/download_quickdraw_tflite.py
```

TensorFlow currently needs a Python version with published wheels for your platform. Use the Python 3.11 setup above if `pip install -r requirements.txt` reports that no `tensorflow` distribution exists. The app still falls back to `StubClassifier` when no TFLite interpreter is available.

Run with the real model:

```bash
MODEL_BACKEND=tflite \
MODEL_PATH=draw_game/models/quickdraw-345-tflite/quickdraw_model.tflite \
LABELS_PATH=draw_game/models/quickdraw-345-tflite/labels.txt \
.venv/bin/python -m draw_game.main
```

Model input spec:

- Shape: `[1, 28, 28, 1]`
- Dtype: `float32`
- Range: `[0.0, 1.0]`
- Polarity: white background is `1.0`, black strokes are `0.0`
- Output: `[1, 345]` softmax probabilities
- Labels: 345 labels, one per line, alphabetically sorted

The preprocessing polarity matters. If it is inverted, predictions will be poor even when the model loads correctly. Bad crop coordinates or UI noise in the crop will also cause bad guesses.

Startup diagnostics for a working TFLite run should include:

```text
Classifier backend: TFLite
Loaded labels: 345
Input details: ... [1, 28, 28, 1] ...
Output details: ... [1, 345] ...
```

To inspect preprocessing, press `s` during runtime or set `DEBUG_SAVE_FRAMES=true`. The app writes:

- `debug-raw-*.png`: raw screen crop
- `debug-preview-28x28-*.png`: exact model preview
- `debug-preview-280x280-*.png`: enlarged preview for visual inspection

Prompts should use exact QuickDraw labels from `labels.txt`. Mismatched prompt labels make the game harder to score and can make correct model guesses look wrong.

ONNX support is still present for experiments. Set `MODEL_BACKEND=onnx` and point `MODEL_PATH` at an `.onnx` file.

## Kokoro and Google Meet Audio

Kokoro is an 82M parameter open-weight local TTS model. If Kokoro and audio playback dependencies are installed, `tts_kokoro.py` will try to use it. If Kokoro fails, the app falls back to `pyttsx3` when available, then console output.

Install Kokoro and local playback support:

```bash
python -m pip install kokoro sounddevice
```

If you see `PortAudioError Error querying device -1`, Python cannot see a default audio output device. Set your macOS output device or virtual audio cable first, then rerun the app from the activated `.venv`.

`pyttsx3` is intentionally not listed in `requirements.txt` because on macOS it can pull in a large PyObjC dependency set. Install it separately only if you want that fallback:

```bash
pip install pyttsx3
```

To route AI speech into Google Meet:

1. Install a virtual audio cable such as BlackHole, VB-Cable, or Loopback.
2. Set system or app output so this Python app plays into the virtual cable.
3. In Google Meet, choose the virtual cable as the microphone input.
4. Test with `TTS_ENABLED=true` and a stub round before playing.

## Game Rules

Rules:

1. One artist draws per round.
2. The artist receives a secret prompt from the approved QuickDraw-compatible label list.
3. Humans guess through Google Meet voice or chat.
4. AI guesses through Kokoro TTS.
5. First correct guess wins the round.
6. If AI says the correct answer first, AI wins.
7. If a human says the correct answer first, humans win.
8. If no one guesses within the time limit, no point.
9. Illegal drawing means redraw or no point.

Drawing restrictions:

- Draw only the target object itself.
- No written words.
- No letters.
- No labels.
- No arrows.
- No pointing marks.
- No speech bubbles.
- No clue diagrams.
- No second object used only to explain the target.

Example:

Prompt: `bottlecap`

Allowed: draw a top-down ridged circular cap by itself.

Not allowed: draw a bottle, put a cap on it, then draw an arrow to the cap.

## Logs

The app creates `draw_game/logs/` automatically and writes daily log files with:

- startup config
- classifier type
- spoken guesses
- errors
- round start/end

## Development Checks

```bash
python3 -m unittest discover -s tests
python3 -m py_compile draw_game/*.py
```
