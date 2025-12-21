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

## 🔧 GPU Configurations

### Current: RTX 4050 (6GB VRAM, 16GB RAM) - MAXED!

The default `train.py` is **maxed out** for RTX 4050:

| Parameter | Value | Usage |
|-----------|-------|-------|
| BATCH_SIZE | 1280 | ~85% VRAM |
| HIDDEN_SIZE | 512 | Maximum capacity |
| MEMORY_SIZE | 500,000 | ~80% RAM |
| NUM_EPISODES | 2000 | ~4-5 hours |

### Upgrade: RTX 5060 Ti (16GB VRAM, 32GB RAM)

For RTX 5060 Ti, edit these values in `train.py` (lines 15-27):

```python
# Hyperparameters (MAXED for RTX 5060 Ti - 16GB VRAM, 32GB RAM)
NUM_EPISODES = 2500                    # More episodes for polish
BATCH_SIZE = 2048                      # 16GB VRAM can handle much more
MEMORY_SIZE = 800000                   # 32GB RAM allows massive buffer
HIDDEN_SIZE = 512                      # Better network capacity
INITIAL_EXPLORATION_STEPS = 30000      # More exploration
```

**Expected RTX 5060 Ti Performance:**
- ⚡ Training time: ~5-6 hours (2500 episodes)
- 📈 Target reward: 850-950+
- 🎯 VRAM usage: ~12-14GB (75-85%)
- 💾 RAM usage: ~24-26GB (75-80%)

## ⏸️ Checkpoint System

Training automatically saves checkpoints:
- **On interrupt (Ctrl+C):** Saves `checkpoints/latest.pth`
- **Every 20 episodes:** Evaluates and saves `checkpoints/best_model.pth` if improved
- **Resume:** Just run `python train.py` again - it auto-detects checkpoints

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

**Out of Memory (RTX 4050):**
```python
# In train.py, reduce:
BATCH_SIZE = 768       # Down from 1280
MEMORY_SIZE = 350000   # Down from 500000
HIDDEN_SIZE = 384      # Down from 512
```

**Out of Memory (RTX 5060 Ti):**
```python
# In train.py, reduce:
BATCH_SIZE = 1280      # Down from 2048
MEMORY_SIZE = 600000   # Down from 800000
HIDDEN_SIZE = 384      # Down from 512
```
