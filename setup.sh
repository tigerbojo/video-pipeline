#!/bin/bash
# AI Video Pipeline - Quick Setup
# Usage: bash setup.sh

set -e

echo "=== AI Video Pipeline Setup ==="

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "[ERROR] Python not found. Install Python 3.10+ first."
    exit 1
fi

PY=$(command -v python3 || command -v python)
echo "Python: $($PY --version)"

# Check FFmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "[WARNING] ffmpeg not found."
    if command -v brew &>/dev/null; then
        echo "Installing via Homebrew..."
        brew install ffmpeg
    else
        echo "Please install ffmpeg: https://ffmpeg.org/download.html"
    fi
fi

# Install Python dependencies
echo ""
echo "Installing Python packages..."
$PY -m pip install --upgrade pip
$PY -m pip install \
    gradio>=5.0 \
    edge-tts>=6.1 \
    faster-whisper>=1.0 \
    opencc-python-reimplemented \
    speechbrain \
    scikit-learn \
    scipy \
    Pillow>=10.0 \
    google-generativeai>=0.8 \
    pyyaml>=6.0

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Start the app:"
echo "  $PY app.py"
echo ""
echo "Then open: http://localhost:7860"
echo ""
echo "Optional tools:"
echo "  pip install auto-editor    # Smart silence removal"
echo "  pip install demucs         # AI vocal separation"
echo ""
echo "If using Ollama on another machine, change the URL in the UI."
