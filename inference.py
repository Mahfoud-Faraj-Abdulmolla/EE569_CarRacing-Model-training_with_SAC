"""
inference.py - Evaluation script for CarRacing-v3 SAC agent
EE569 Deep Learning Assignment
"""

import argparse
import os
import torch
import numpy as np
import cv2
import gymnasium as gym
from collections import deque
from gymnasium.wrappers import RecordVideo
import json

# Import environment wrappers from train.py
class CarRacingImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width), dtype=np.uint8
        )
        self.top_crop = 12
        self.bottom_crop = 96

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cropped = gray[self.top_crop:self.bottom_crop, :]
        resized = cv2.resize(cropped, (self.width, self.height),
                             interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(resized)


class StackFrames(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(stack_size, env.observation_space.shape[0],
                   env.observation_space.shape[1]),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.stack_size):
            self.frames.append(observation)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(render_mode=None, skip_frames=4):
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    env = FrameSkip(env, skip=skip_frames)
    env = CarRacingImageWrapper(env)
    env = StackFrames(env, stack_size=4)
    return env


# Actor Network (same as in train.py)
class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=512):
        super(ActorNetwork, self).__init__()
        self.log_std_min, self.log_std_max = -20, 2

        self.conv1 = torch.nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)

        def conv_output_size(size, kernel, stride):
            return (size - (kernel - 1) - 1) // stride + 1
        
        w = conv_output_size(conv_output_size(conv_output_size(state_dim[2], 8, 4), 4, 2), 3, 1)
        h = conv_output_size(conv_output_size(conv_output_size(state_dim[1], 8, 4), 4, 2), 3, 1)
        linear_input_size = w * h * 64

        self.fc1 = torch.nn.Linear(linear_input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)

        self.mean_layer = torch.nn.Linear(hidden_size, action_dim)
        self.log_std_layer = torch.nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.nn.functional.relu(self.bn1(self.conv1(state)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))

        mean = torch.tanh(self.mean_layer(x))
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, deterministic=True):
        """Get action for inference"""
        with torch.no_grad():
            mean, _ = self.forward(state)
            action = torch.tanh(mean)
        return action.cpu().numpy()[0]


def load_model(checkpoint_path, state_dim, action_dim, device):
    """Load actor model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = ActorNetwork(state_dim, action_dim).to(device)
    
    try:
        # Try to load as state dict
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'actor_state_dict' in checkpoint:
                # Full checkpoint
                model.load_state_dict(checkpoint['actor_state_dict'])
                print(f"✅ Loaded from full checkpoint (episode {checkpoint.get('episode', 'N/A')})")
                if 'eval_reward' in checkpoint:
                    print(f"   Evaluation reward: {checkpoint['eval_reward']:.1f}")
            elif 'state_dict' in checkpoint:
                # Alternative format
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume it's already a state dict
                model.load_state_dict(checkpoint)
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"⚠️ Error loading: {e}")
        print("Trying strict=False...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    
    model.eval()
    return model


def run_episode(env, model, device, episode_num=0, save_video=False, video_dir="./videos"):
    """Run a single evaluation episode"""
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        # Prepare state tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
        
        # Get action
        action = model.get_action(state_tensor, deterministic=True)
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        state = next_state
        steps += 1
        
        # Early stopping if performing very poorly
        if total_reward < -100:
            break
        
        # Max steps
        if steps > 1000:
            break
    
    return total_reward, steps


def evaluate_model(model, device, num_episodes=3, save_video=False, video_dir="./videos"):
    """Evaluate model for specified number of episodes"""
    rewards = []
    steps_list = []
    
    for episode in range(num_episodes):
        # Create environment
        render_mode = 'rgb_array' if save_video else None
        env = make_env(render_mode=render_mode)
        
        # Wrap with video recorder for first episode if saving video
        if save_video and episode == 0:
            os.makedirs(video_dir, exist_ok=True)
            env = RecordVideo(
                env,
                video_dir,
                name_prefix=f"evaluation_episode",
                episode_trigger=lambda x: True
            )
        
        print(f"  Episode {episode + 1}/{num_episodes}...", end="")
        reward, steps = run_episode(env, model, device, episode, save_video and episode == 0, video_dir)
        rewards.append(reward)
        steps_list.append(steps)
        print(f" Reward: {reward:.1f}, Steps: {steps}")
        
        env.close()
    
    return np.array(rewards), np.array(steps_list)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC agent on CarRacing-v3')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_actor.pth',
                        help='Path to model checkpoint (default: checkpoints/best_actor.pth)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of evaluation episodes (default: 3, as per assignment)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save video of the first evaluation episode')
    parser.add_argument('--video-dir', type=str, default='./videos',
                        help='Directory to save videos (default: ./videos)')
    parser.add_argument('--output-file', type=str, default='evaluation_results.json',
                        help='File to save evaluation results (default: evaluation_results.json)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("CAR RACING - EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation episodes: {args.episodes}")
    print(f"Save video: {args.save_video}")
    print("=" * 60)
    
    # Create a temporary environment to get dimensions
    temp_env = make_env()
    state_dim = temp_env.observation_space.shape
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Load model
    print(f"\n📂 Loading model...")
    model = load_model(args.checkpoint, state_dim, action_dim, device)
    
    # Evaluate
    print(f"\n🏁 Running evaluation ({args.episodes} episodes)...")
    rewards, steps = evaluate_model(
        model, device, 
        num_episodes=args.episodes,
        save_video=args.save_video,
        video_dir=args.video_dir
    )
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(steps)
    
    # Save evaluation results
    results = {
        'checkpoint': args.checkpoint,
        'num_episodes': args.episodes,
        'rewards': rewards.tolist(),
        'steps': steps.tolist(),
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_steps': float(mean_steps),
        'device': str(device),
        'assignment_requirement_met': mean_reward > 700
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"Individual rewards: {[f'{r:.1f}' for r in rewards]}")
    print(f"Mean steps per episode: {mean_steps:.0f}")
    
    if mean_reward > 700:
        print("✅ ASSIGNMENT REQUIREMENT MET: Mean reward > 700")
    else:
        print("⚠️  ASSIGNMENT REQUIREMENT NOT MET: Mean reward < 700")
    
    print(f"\n💾 Results saved to: {args.output_file}")
    
    if args.save_video:
        # Find the video file
        video_files = [f for f in os.listdir(args.video_dir) if f.startswith("evaluation_episode")]
        if video_files:
            # Rename to best_run.mp4 as per assignment requirement
            original_video = os.path.join(args.video_dir, video_files[0])
            best_video = os.path.join(args.video_dir, "best_run.mp4")
            
            try:
                os.rename(original_video, best_video)
                print(f"🎥 Video saved as: {best_video}")
            except:
                print(f"🎥 Video saved in: {args.video_dir}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
