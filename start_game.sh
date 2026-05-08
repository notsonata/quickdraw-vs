#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
ENV_FILE="$PROJECT_DIR/draw_game/.env"

# Extract paths from .env if present
if [[ -f "$ENV_FILE" ]]; then
  ENV_MODEL=$(grep "^MODEL_PATH=" "$ENV_FILE" | cut -d'=' -f2 | sed "s/^'//;s/'$//;s/^\"//;s/\"$//")
  ENV_LABELS=$(grep "^LABELS_PATH=" "$ENV_FILE" | cut -d'=' -f2 | sed "s/^'//;s/'$//;s/^\"//;s/\"$//")
fi

MODEL_PATH="${ENV_MODEL:-draw_game/models/quickdraw_stroke_tflite_export_15k_256/quickdraw_stroke_model_float32.tflite}"
LABELS_PATH="${ENV_LABELS:-draw_game/models/quickdraw_stroke_tflite_export_15k_256/labels.json}"

# Resolve relative paths against PROJECT_DIR
if [[ ! "$MODEL_PATH" = /* ]]; then MODEL_PATH="$PROJECT_DIR/$MODEL_PATH"; fi
if [[ ! "$LABELS_PATH" = /* ]]; then LABELS_PATH="$PROJECT_DIR/$LABELS_PATH"; fi

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
