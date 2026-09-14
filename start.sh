#!/bin/sh

set -e

MODEL_DIR="/app/models/layoutlmv3_receipt_500"
MODEL_FILE="$MODEL_DIR/model.safetensors"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading LayoutLMv3 model..."

    mkdir -p "$MODEL_DIR"

    curl -L "$MODEL_URL" -o "$MODEL_FILE"
fi

echo "Starting Django..."

exec gunicorn config.wsgi:application --bind 0.0.0.0:8000