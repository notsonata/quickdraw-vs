# Setup

## Install

```bash
cd draw_game
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow/TFLite should use Python 3.11 for this project. Python 3.14 is not currently suitable because TensorFlow/TFLite wheels are unavailable or unreliable.

If `python3.11` is missing:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
```

On Intel Macs:

```bash
/usr/local/bin/python3.11 -m venv .venv
```

## Download QuickDraw TFLite Model

```bash
.venv/bin/python draw_game/tools/download_quickdraw_tflite.py
```

## Run

```bash
python3 main.py
```

## Run With Web Canvas

Set `CANVAS_SOURCE=web` in `draw_game/.env` or the shell, then start the app normally.

Important settings:

```env
CANVAS_SOURCE=web
WEB_CANVAS_HOST=0.0.0.0
WEB_CANVAS_PORT=8765
ROUND_DURATION_SEC=60.0
```

Open `http://localhost:8765` on the local machine or expose that port through Cloudflare Tunnel for a phone browser.

When a round starts, the web canvas shows the countdown over the drawing field. The main loop automatically ends the round when `ROUND_DURATION_SEC` elapses. Set `ROUND_DURATION_SEC=0` to disable automatic round ending.

## Optional Gemma Vision

Gemma vision is disabled by default. To let a local PaliGemma model inspect the raw canvas frame and speak each valid detection, set:

```env
GEMMA_ENABLED=true
GEMMA_MODEL=google/paligemma-3b-mix-224
GEMMA_INTERVAL_SEC=2.0
GEMMA_CONFIDENCE=0.9
```

Gemma output is accepted only when the label matches the configured QuickDraw `labels.txt` file.

Download the model ahead of time:

```bash
.venv/bin/python draw_game/tools/download_gemma_vision.py
```

## Test

```bash
python3 -m unittest discover -s tests
```
