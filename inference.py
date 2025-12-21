"""
inference.py - Evaluation and video recording for CarRacing-v3 SAC agent
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
    def __init__(self, env, skip=3):
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


def make_env(render_mode=None, skip_frames=3):
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    env = FrameSkip(env, skip=skip_frames)
    env = CarRacingImageWrapper(env)
    env = StackFrames(env, stack_size=4)
    return env


# Actor Network - يجب أن يكون مطابقاً تماماً لـ train.py
class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=1536):
        super(ActorNetwork, self).__init__()
        self.log_std_min, self.log_std_max = -20, 2

        # البنية الحقيقية من train.py: 96 -> 192 -> 256
        self.conv1 = torch.nn.Conv2d(state_dim[0], 96, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(96, 192, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(192, 256, kernel_size=3, stride=1)

        self.bn1 = torch.nn.BatchNorm2d(96)
        self.bn2 = torch.nn.BatchNorm2d(192)
        self.bn3 = torch.nn.BatchNorm2d(256)

        # نفس الحسابات المستخدمة في train.py
        def convw(size, kernel, stride):
            return (size - (kernel - 1) - 1) // stride + 1

        w = convw(convw(convw(state_dim[2], 5, 2), 3, 2), 3, 1)
        h = convw(convw(convw(state_dim[1], 5, 2), 3, 2), 3, 1)
        linear_input_size = w * h * 256

        self.fc1 = torch.nn.Linear(linear_input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_residual = torch.nn.Linear(hidden_size, hidden_size)

        self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)

        self.mean_layer = torch.nn.Linear(hidden_size, action_dim)
        self.log_std_layer = torch.nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.nn.functional.relu(self.bn1(self.conv1(state)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = torch.nn.functional.relu(self.ln1(self.fc1(x)))
        residual = x
        x = torch.nn.functional.relu(self.ln2(self.fc2(x)))
        x = x + self.fc_residual(residual)

        mean = torch.tanh(self.mean_layer(x)) * 1.5
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, deterministic=True):
        """الحصول على action للاستدلال"""
        with torch.no_grad():
            mean, _ = self.forward(state)
            action = torch.tanh(mean)
        return action.cpu().numpy()[0]


def load_model(checkpoint_path, state_dim, action_dim, device):
    """تحميل نموذج الـactor من checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ActorNetwork(state_dim, action_dim, hidden_size=1536).to(device)

    try:
        # حاول تحميل 
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=True)

        # تحقق من نوع الـcheckpoint
        if isinstance(checkpoint, dict):
            if 'actor_state_dict' in checkpoint:
                # تحميل من checkpoint كامل
                model.load_state_dict(checkpoint['actor_state_dict'])
                print(f"✅ Loaded from full checkpoint (episode {checkpoint.get('episode', 'N/A')})")
                if 'eval_reward' in checkpoint:
                    print(f"   Evaluation reward: {checkpoint['eval_reward']:.1f}")
            elif 'state_dict' in checkpoint:
                # تنسيق بديل
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # افتراض أنه state_dict مباشر
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded from state dict directly")
        else:
            # state_dict مباشر
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded from state dict directly")

    except Exception as e:
        print(f"⚠️  Error loading: {e}")
        print("Trying to load with strict=False...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            print(f"✅ Loaded with strict=False (some layers may be missing)")
        except Exception as e2:
            print(f"❌ Failed to load even with strict=False: {e2}")
            print("Creating empty model...")
            model = ActorNetwork(state_dim, action_dim, hidden_size=1536).to(device)

    model.eval()
    return model


def run_episode(env, model, device, max_steps=1500, render=False, record_video=False):
    """تشغيل حلقة واحدة وإرجاع المكافأة الإجمالية"""
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < max_steps:
        # تحضير state tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0

        # الحصول على action من النموذج
        action = model.get_action(state_tensor, deterministic=True)

        # تنفيذ action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state
        steps += 1

        if render and not record_video:
            env.render()

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC agent on CarRacing-v3')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_actor.pth',
                        help='Path to model checkpoint (default: checkpoints/best_actor.pth)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of evaluation episodes (default: 3, as per assignment)')
    parser.add_argument('--save-video', action='store_true', default=True,
                        help='Save video of the evaluation episodes')
    parser.add_argument('--video-dir', type=str, default='./videos',
                        help='Directory to save videos (default: ./videos)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'], help='Device to use')

    args = parser.parse_args()

    # ضبط الجهاز
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("CAR RACING - SAC EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation episodes: {args.episodes}")
    print(f"Save video: {args.save_video}")
    print("=" * 60)

    # إنشاء بيئة مؤقتة للحصول على الأبعاد
    temp_env = make_env()
    state_dim = temp_env.observation_space.shape
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # تحميل النموذج
    print(f"\n📂 Loading model...")
    model = load_model(args.checkpoint, state_dim, action_dim, device)

    # التقييم
    print(f"\n🏁 Running evaluation ({args.episodes} episodes)...")

    rewards = []
    steps_list = []

    for episode in range(args.episodes):
        # للحلقة الأولى، تسجيل فيديو إذا طُلب
        if args.save_video and episode == 0:
            # إنشاء بيئة مع تسجيل فيديو
            video_env = make_env(render_mode='rgb_array')
            video_env = RecordVideo(
                video_env,
                args.video_dir,
                name_prefix='best_run',
                episode_trigger=lambda x: True
            )

            print(f"Episode {episode + 1}/{args.episodes} (recording video)...", end="")
            reward, steps = run_episode(video_env, model, device, record_video=True)
            video_env.close()

            # التأكد من أن الفيديو يحمل اسم best_run.mp4
            video_files = [f for f in os.listdir(args.video_dir)
                           if f.startswith('best_run') and f.endswith('.mp4')]
            if video_files:
                original = os.path.join(args.video_dir, video_files[0])
                target = os.path.join(args.video_dir, "best_run.mp4")
                if original != target:
                    if os.path.exists(target):
                        os.remove(target)
                    os.rename(original, target)
        else:
            # للبقية، بدون فيديو
            env = make_env()
            print(f"Episode {episode + 1}/{args.episodes}...", end="")
            reward, steps = run_episode(env, model, device, render=False)
            env.close()

        rewards.append(reward)
        steps_list.append(steps)
        print(f" Reward: {reward:.1f}, Steps: {steps}")

    # حساب الإحصائيات
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(steps_list)

    # حفظ نتائج التقييم
    results = {
        'checkpoint': args.checkpoint,
        'num_episodes': args.episodes,
        'rewards': rewards,
        'steps': steps_list,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_steps': float(mean_steps),
        'device': str(device),
        'assignment_requirement_met': bool(mean_reward > 700)
    }

    results_file = os.path.join(args.video_dir if args.save_video else '.', 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # طباعة الملخص
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

    print(f"\n💾 Results saved to: {results_file}")

    if args.save_video:
        video_path = os.path.join(args.video_dir, "best_run.mp4")
        if os.path.exists(video_path):
            print(f"🎥 Video saved as: {video_path}")
        else:
            print(f"⚠️  Video not found at: {video_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
