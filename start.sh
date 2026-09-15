#!/bin/sh

set -e

MODEL_DIR="/tmp/layoutlmv3_receipt_500"
MODEL_FILE="$MODEL_DIR/model.safetensors"
CONFIG_FILE="$MODEL_DIR/config.json"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading LayoutLMv3 model..."
    curl -fL "$MODEL_URL" -o "$MODEL_FILE"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Downloading LayoutLMv3 config..."
    curl -fL "$CONFIG_URL" -o "$CONFIG_FILE"
fi

echo "Starting Django..."

exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --timeout 180