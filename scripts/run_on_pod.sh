#!/bin/bash
# One-command setup for Vast.ai / RunPod — installs deps, runs full cloud pipeline.
# Usage: bash scripts/run_on_pod.sh [--cycles 1000] [--model qwen14b|llama8b] [--epochs 1]
#
# Python 3.12 is required (Unsloth / training wheels). If venv was created with another
# Python, delete ./venv and re-run this script.

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "=== BoggersTheMind-1 — Pod Setup ==="
echo "Root: $ROOT"

# 1. Python 3.12 venv (if not already in one)
if [ -z "$VIRTUAL_ENV" ]; then
    if ! command -v python3.12 &>/dev/null; then
        echo "ERROR: python3.12 not found. Install Python 3.12 for Unsloth compatibility."
        exit 1
    fi
    if [ ! -d "venv" ]; then
        echo "Creating venv with python3.12..."
        python3.12 -m venv venv
    fi
    echo "Activating venv..."
    . venv/bin/activate
fi

# 2. Base deps (mind + trace gen + processor)
echo "Installing base dependencies..."
pip install -q -r requirements.txt

# 3. Unsloth + training deps (separate to avoid conflicts)
echo "Installing Unsloth + training stack..."
pip install -q unsloth
pip install -q "trl>=0.13" "datasets" "transformers" "accelerate" "bitsandbytes" "peft"

# 4. Run full pipeline
echo ""
echo "Starting full cloud pipeline..."
python scripts/full_cloud_train.py "$@"

echo ""
echo "=== Done. Model at outputs/boggersmind-1 ==="
