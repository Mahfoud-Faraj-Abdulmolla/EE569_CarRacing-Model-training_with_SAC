## 🚀 Quick Start

### 1. Setup
```bash
# 1. Create virtual environment with Python 3.12
# (Requires python 3.12 to be installed on your system)
uv venv --python 3.12

# 2. Activate environment
source .venv/bin/activate

# 3. Install dependencies
# 'swig' is required for Box2D
uv pip install swig
uv pip install -r requirements.txt
```

### 2. Verify Setup
```bash
python test_setup.py
```

### 3. Train
```bash
./run_training.sh
# OR
python train.py
```

## ⏸️ Pause & Resume

- **Pause:** Ctrl+C (auto-saves checkpoint)
- **Resume:** Run `python train.py` again

## 📊 Monitor Progress

```bash
# Terminal 1: Training
python train.py

# Terminal 2: TensorBoard
tensorboard --logdir=logs
```

Open: `http://localhost:6006`

## 🔧 RTX 4050 Optimizations

- ✅ Batch size: 128 (VRAM-safe)
- ✅ Hidden size: 256 (efficient)
- ✅ Mixed precision training
- ✅ Smaller CNN (64→128→128)
- ✅ Memory buffer: 200k
- ✅ Auto-resume capability

## 📈 Expected Performance

- **Training time:** 24-30 hours (2000 episodes)
- **Target reward:** 750-850+
- **VRAM usage:** ~4-5 GB peak

## 📁 Structure

```
├── train.py           # Main training
├── inference.py       # Run trained model
├── test_setup.py      # Verify setup
├── run_training.sh    # Start script
├── checkpoints/       # Model saves
├── videos/           # Evaluation videos
└── logs/             # TensorBoard
```

## ⚠️ Troubleshooting

**Out of Memory:**
```python
# In train.py, reduce:
BATCH_SIZE = 64        # Down from 128
MEMORY_SIZE = 100000   # Down from 200000
```
