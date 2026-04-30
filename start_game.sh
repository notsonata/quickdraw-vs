#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
MODEL_PATH="$PROJECT_DIR/draw_game/models/quickdraw-345-tflite/quickdraw_model.tflite"
LABELS_PATH="$PROJECT_DIR/draw_game/models/quickdraw-345-tflite/labels.txt"

cd "$PROJECT_DIR"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing virtualenv Python: $VENV_PYTHON"
  echo "Create it with Python 3.11, then install requirements."
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model file: $MODEL_PATH"
  exit 1
fi

if [[ ! -f "$LABELS_PATH" ]]; then
  echo "Missing labels file: $LABELS_PATH"
  exit 1
fi

echo "Starting Humans vs AI Draw Guessing..."
echo "Controls: r=start round, e=end round, s=save debug frame, q=quit"
echo

exec "$VENV_PYTHON" -m draw_game.main
