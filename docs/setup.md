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

## Test

```bash
python3 -m unittest discover -s tests
```
