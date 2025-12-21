import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import cv2
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import sys

# Hyperparameters (MAXED OUT for RTX 4050 - 6GB VRAM, 16GB RAM)
# For RTX 5060 Ti (16GB VRAM, 32GB RAM), see README.md for recommended values
NUM_EPISODES = 2000                    # Full training run
MAX_STEPS_PER_EPISODE = 2000           # Full lap completion
BATCH_SIZE = 1280                      # MAXED: Target ~85% VRAM (was using 74% at 1024)
GAMMA = 0.99
TAU = 0.01                             # Soft target updates
ALPHA_INIT = 0.2
LEARNING_RATE = 3e-4                   # Standard SAC learning rate
MEMORY_SIZE = 500000                   # MAXED: Target ~80% RAM (was using 50% at 350k)
HIDDEN_SIZE = 512                      # MAXED: Better network capacity
INITIAL_EXPLORATION_STEPS = 20000      # Exploration before policy training
EXPLORATION_NOISE = 0.15               # Action noise for exploration


# Path for checkpoints and logs
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
VIDEO_DIR = "./videos"

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_limit):
        super(ActorNetwork, self).__init__()
        
        # Efficient CNN for 16GB VRAM
        self.conv1 = nn.Conv2d(state_dim[0], 64, kernel_size=5, stride=2)   # 96->64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)            # 192->128
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)           # 256→128
        
        # Calculate linear input size based on 84x84 input
        # Conv1: (84-5)/2 + 1 = 40
        # Conv2: (40-3)/2 + 1 = 19
        # Conv3: (19-3)/1 + 1 = 17
        w, h = 17, 17
        linear_input_size = w * h * 128
        
        self.linear1 = nn.Linear(linear_input_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.action_limit = action_limit

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_limit
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_limit * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        
        # Efficient CNN for 16GB VRAM (same as Actor)
        self.conv1 = nn.Conv2d(state_dim[0], 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        
        w, h = 17, 17
        cnn_output_size = w * h * 128
        
        # Q1 architecture
        self.linear1_q1 = nn.Linear(cnn_output_size + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear1_q2 = nn.Linear(cnn_output_size + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        xu = torch.cat([x, action], 1)
        
        # Q1 forward
        x1 = F.relu(self.linear1_q1(xu))
        x1 = F.relu(self.linear2_q1(x1))
        x1 = self.linear3_q1(x1)

        # Q2 forward
        x2 = F.relu(self.linear1_q2(xu))
        x2 = F.relu(self.linear2_q2(x2))
        x2 = self.linear3_q2(x2)

        return x1, x2

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done)

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, action_limit, device):
        self.device = device
        self.action_limit = action_limit
        self.alpha_init = ALPHA_INIT
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, action_limit).to(device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        
        # Mixed precision scalers
        self.scaler_critic = GradScaler('cuda')
        self.scaler_actor = GradScaler('cuda')
        
        self.total_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.actor(state)
            return mean.cpu().data.numpy().flatten()
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()

    def update_parameters(self, memory, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        
        self.total_steps += 1

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.target_critic(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * GAMMA * min_qf_next_target

        # Critic update with Mixed Precision
        with autocast('cuda'):
            current_q1, current_q2 = self.critic(state_batch, action_batch)
            qf1_loss = F.mse_loss(current_q1, next_q_value)
            qf2_loss = F.mse_loss(current_q2, next_q_value)
            critic_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        self.scaler_critic.scale(critic_loss).backward()
        self.scaler_critic.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.scaler_critic.step(self.critic_optimizer)
        self.scaler_critic.update()

        # Actor update with Mixed Precision
        with autocast('cuda'):
            pi, log_pi = self.actor.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        self.scaler_actor.scale(actor_loss).backward()
        self.scaler_actor.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.actor_optimizer)
        self.scaler_actor.update()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha': self.alpha.item()
        }, None

    def save_checkpoint(self, filename, episode, best_reward):
        torch.save({
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'best_reward': best_reward,
            'total_steps': self.total_steps
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.total_steps = checkpoint['total_steps']
        return checkpoint['episode'], checkpoint['best_reward']

def preprocess_state(state):
    # Convert to grayscale and resize to 84x84
    # State shape is (96, 96, 3)
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_state(state)
    if is_new_episode:
        stacked_frames = deque([frame] * 4, maxlen=4)
    else:
        stacked_frames.append(frame)
    return np.stack(stacked_frames, axis=0), stacked_frames

def main():
    # Detect GPU and Enable Optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        print(f"✅ CUDA optimizations enabled")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available. Training will be slow.")

    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    
    action_limit = env.action_space.high[0]
    state_dim = (4, 84, 84)
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim, HIDDEN_SIZE, action_limit, device)
    memory = ReplayMemory(MEMORY_SIZE)
    writer = SummaryWriter(LOG_DIR)
    
    # Auto-resume capability
    LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest.pth")
    start_episode = 0
    best_eval_reward = -float('inf')

    if os.path.exists(LATEST_CHECKPOINT):
        print("🔄 Found checkpoint. Resuming training...")
        try:
            start_episode, best_eval_reward = agent.load_checkpoint(LATEST_CHECKPOINT)
            print(f"   Episode: {start_episode}")
            print(f"   Best reward: {best_eval_reward:.1f}")
            # Ensure we start from the next episode
            start_episode += 1
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("🆕 Starting new training...")
    else:
        print("🆕 Starting new training...")

    try:
        for episode in range(start_episode, NUM_EPISODES):
            state, _ = env.reset()
            state, stacked_frames = stack_frames(None, state, True)
            episode_reward = 0
            
            for step in range(MAX_STEPS_PER_EPISODE):
                if agent.total_steps < INITIAL_EXPLORATION_STEPS and not os.path.exists(LATEST_CHECKPOINT):
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)
                    # Add exploration noise
                    action = action + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
                    action = action.clip(env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                
                # Reward scaling (optional but often helpful for CarRacing)
                # reward = reward / 10.0 
                
                memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # Updated frequency: Every 2 steps + double gradient updates
                if len(memory) > BATCH_SIZE and agent.total_steps % 2 == 0:
                    for _ in range(2):  # Double gradient updates for faster learning
                        update_info, _ = agent.update_parameters(memory, BATCH_SIZE)
                        if update_info:
                            writer.add_scalar('Loss/critic', update_info['critic_loss'], agent.total_steps)
                            writer.add_scalar('Loss/actor', update_info['actor_loss'], agent.total_steps)
                            writer.add_scalar('Alpha', update_info['alpha'], agent.total_steps)

                if done:
                    break

            writer.add_scalar('Reward/train', episode_reward, episode)
            print(f"Episode {episode}/{NUM_EPISODES}, Reward: {episode_reward:.1f}, Steps: {step}")
            
            # Optional Memory Monitoring
            if episode % 10 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"   VRAM: {allocated:.2f}GB / {reserved:.2f}GB")

            # Save latest checkpoint after each episode
            agent.save_checkpoint(LATEST_CHECKPOINT, episode, best_eval_reward)

            # Evaluation and Best Checkpoint Saving (every 20 episodes)
            if episode % 20 == 0:
                avg_reward = 0
                eval_episodes = 3 # Small eval to save time
                
                # Create evaluation environment with video recording
                eval_env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
                eval_env = gym.wrappers.RecordVideo(
                    eval_env, 
                    video_folder=VIDEO_DIR, 
                    episode_trigger=lambda x: True, # Record all eval episodes
                    name_prefix=f"eval_ep_{episode}"
                )

                for _ in range(eval_episodes):
                    eval_state, _ = eval_env.reset()
                    eval_state, eval_frames = stack_frames(None, eval_state, True)
                    eval_reward = 0
                    while True:
                        eval_action = agent.select_action(eval_state, evaluate=True)
                        eval_next_state, r, term, trunc, _ = eval_env.step(eval_action)
                        eval_next_state, eval_frames = stack_frames(eval_frames, eval_next_state, False)
                        eval_reward += r
                        eval_state = eval_next_state
                        if term or trunc:
                            break
                    avg_reward += eval_reward
                
                eval_env.close() # Close to save video
                
                avg_reward /= eval_episodes
                writer.add_scalar('Reward/eval', avg_reward, episode)
                print(f"⭐ Evaluation: {avg_reward:.1f} (Video saved to {VIDEO_DIR})")
                
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    agent.save_checkpoint(os.path.join(CHECKPOINT_DIR, "best_model.pth"), episode, best_eval_reward)
                    print(f"🏆 New best model saved!")

    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted!")
        print("Saving checkpoint...")
        agent.save_checkpoint(LATEST_CHECKPOINT, episode, best_eval_reward)
        print(f"✅ Saved: {LATEST_CHECKPOINT}")
        sys.exit(0)
    finally:
        env.close()
        writer.close()

if __name__ == "__main__":
    main()
