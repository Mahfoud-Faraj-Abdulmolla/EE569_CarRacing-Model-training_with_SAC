#!/bin/bash

echo "🚀 SAC CarRacing Training"
echo "Hardware: RTX 4050 (6GB VRAM)"
echo "======================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Check GPU before starting
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "Running setup tests..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Tests passed!"
    echo "Starting training (Press Ctrl+C to pause)..."
    echo ""
    python train.py
else
    echo "❌ Tests failed. Fix issues first."
    exit 1
fi
