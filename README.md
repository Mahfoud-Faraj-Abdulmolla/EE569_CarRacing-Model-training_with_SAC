# CarRacing-v3: SAC Implementation

## Overview
This repository contains an implementation of **Soft Actor-Critic (SAC)** for the CarRacing-v3 environment. The agent learns to drive from raw pixel inputs (84×84 grayscale images) using deep reinforcement learning.

## Features
- **Algorithm:** Soft Actor-Critic (SAC)
- **Input:** 84×84 grayscale pixels (4-frame stack)
- **Output:** Continuous actions (steering, acceleration, brake)
- **Evaluation:** 3 episodes as per assignment requirements
- **Performance:** >700 average reward achieved

## Installation
```bash
pip install -r requirements.txt
Usage
Training
bash
python train.py
Evaluation & Video Recording
bash
python inference.py --checkpoint checkpoints/best_actor.pth --episodes 3 --save-video
Project Structure
text
├── train.py                 # Training script
├── inference.py             # Evaluation script
├── requirements.txt         # Dependencies
├── checkpoints/            # Saved models
│   ├── best_model.pth
│   └── best_actor.pth
├── videos/                 # Recorded runs
│   └── best_run.mp4
├── logs/                   # TensorBoard logs
└── README.md
Model Architecture
CNN Encoder: 3 convolutional layers (96→192→256)

Actor Network: Gaussian policy with automatic entropy tuning

Critic Networks: Twin Q-networks (TD3-style)

Hidden Layers: 1536 units with residual connections

Hyperparameters
Parameter	Value
Learning Rate	8e-5
Batch Size	768
Discount Factor (γ)	0.99
Target Update (τ)	0.005
Replay Buffer Size	3M
Initial Exploration Steps	5000
Results
✅ Assignment Requirement Met: >700 average reward over 3 episodes

📈 Training Logs: Available via TensorBoard

🎥 Best Run Video: videos/best_run.mp4

Requirements Checklist
Pixel input (84×84)

Average reward > 700

3-episode evaluation

Video recording capability

TensorBoard logging

Clean, modular code
