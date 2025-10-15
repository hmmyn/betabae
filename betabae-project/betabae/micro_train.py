"""
MicroBetaBae Training: Sutton's Bitter Lesson Implementation

Key Features:
1. Efficient memory usage (works past 10,000 episodes)
2. Search-based action selection
3. Reflection and replay learning
4. Minimal dependencies (no PyTorch)
"""

import numpy as np
import random
import math
from collections import deque
from pathlib import Path
import json
import time

from betabae.micro_core import MicroBetaBae

class MicroEnvironment:
    """Minimal environment wrapper"""
    def __init__(self, env_name='CartPole-v1'):
        self.env_name = env_name
        if env_name == 'CartPole-v1':
            self.obs_dim = 4
            self.action_dim = 2
            self.max_steps = 500
        else:
            # Default environment
            self.obs_dim = 4
            self.action_dim = 2
            self.max_steps = 100
    
    def reset(self):
        """Reset environment"""
        if self.env_name == 'CartPole-v1':
            # CartPole dynamics
            self.state = np.array([0.0, 0.0, 0.0, 0.0])  # [pos, vel, angle, ang_vel]
            self.step_count = 0
        else:
            # Simple random walk
            self.state = np.random.randn(self.obs_dim) * 0.1
            self.step_count = 0
        
        return self.state.copy()
    
    def step(self, action):
        """Take environment step"""
        self.step_count += 1
        
        if self.env_name == 'CartPole-v1':
            # Simplified CartPole dynamics
            pos, vel, angle, ang_vel = self.state
            
            # Action: 0 = left, 1 = right
            force = 1.0 if action == 1 else -1.0
            
            # Physics
            gravity = 9.8
            masscart = 1.0
            masspole = 0.1
            total_mass = masspole + masscart
            length = 0.5
            polemass_length = masspole * length
            force_mag = 10.0
            tau = 0.02
            
            costheta = math.cos(angle)
            sintheta = math.sin(angle)
            
            temp = (force + polemass_length * ang_vel * ang_vel * sintheta) / total_mass
            thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
            xacc = temp - polemass_length * thetaacc * costheta / total_mass
            
            # Update state
            pos = pos + tau * vel
            vel = vel + tau * xacc
            angle = angle + tau * ang_vel
            ang_vel = ang_vel + tau * thetaacc
            
            self.state = np.array([pos, vel, angle, ang_vel])
            
            # Reward and done
            reward = 1.0
            done = (abs(angle) > 0.2095 or abs(pos) > 2.4 or self.step_count >= self.max_steps)
            
        else:
            # Simple random walk
            self.state += np.random.randn(self.obs_dim) * 0.1 + np.array([action - 0.5, 0, 0, 0])
            reward = 0.0
            done = self.step_count >= self.max_steps
        
        return self.state.copy(), reward, done, {}

def train_micro_betabae(
    env_name='CartPole-v1',
    n_episodes=10000,
    save_every=100,
    output_dir='./micro_outputs',
    d_model=32,
    seq_len=8,
    lr=0.001,
    search_enabled=True,
    reflection_enabled=True
):
    """
    Train MicroBetaBae following Sutton's Bitter Lesson
    
    Key principles:
    1. Computation > Knowledge: Use search for better decisions
    2. Reflection: Learn from mistakes
    3. Scalability: Efficient memory usage
    """
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    env = MicroEnvironment(env_name)
    agent = MicroBetaBae(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        d_model=d_model,
        seq_len=seq_len,
        lr=lr
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    
    print(f"Starting MicroBetaBae training on {env_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Model parameters: {len(agent.params)}")
    print(f"Search enabled: {search_enabled}")
    print(f"Reflection enabled: {reflection_enabled}")
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        # Reset environment
        obs = env.reset()
        history = deque(maxlen=agent.seq_len)
        
        # Initialize history
        for _ in range(agent.seq_len):
            history.append(np.concatenate([obs, np.zeros(env.action_dim)]))
        
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        done = False
        while not done and step_count < env.max_steps:
            # Convert history to list of lists for micrograd
            hist_list = [list(h) for h in history]
            
            # Action selection
            if search_enabled and episode > 100:  # Enable search after initial exploration
                action = agent.search_action(hist_list, env)
            else:
                # Random action for exploration
                action = random.randint(0, env.action_dim - 1)
            
            # Take action
            next_obs, reward, done, _ = env.step(action)
            
            # Learn from experience
            agent.learn(hist_list, action, next_obs, reward)
            
            # Update history
            action_onehot = np.zeros(env.action_dim)
            action_onehot[action] = 1.0
            history.append(np.concatenate([next_obs, action_onehot]))
            
            episode_reward += reward
            episode_loss += agent.loss_log[-1] if agent.loss_log else 0
            step_count += 1
            
            obs = next_obs
        
        # Reflection learning
        if reflection_enabled and episode % 10 == 0:
            agent.reflect()
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        episode_losses.append(episode_loss / max(step_count, 1))
        
        # Progress reporting
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_loss = np.mean(episode_losses[-100:])
            
            elapsed = time.time() - start_time
            episodes_per_sec = episode / elapsed if elapsed > 0 else 0
            
            print(f"Episode {episode:5d}: "
                  f"Reward={avg_reward:6.2f}, "
                  f"Length={avg_length:6.1f}, "
                  f"Loss={avg_loss:6.4f}, "
                  f"Speed={episodes_per_sec:6.2f} ep/s")
        
        # Save model and logs
        if episode % save_every == 0:
            agent.save_logs(episode, output_dir / 'logs')
            
            # Save training statistics
            stats = {
                'episode': episode,
                'rewards': episode_rewards,
                'lengths': episode_lengths,
                'losses': episode_losses,
                'elapsed_time': time.time() - start_time
            }
            
            with open(output_dir / f'stats_ep_{episode:05d}.json', 'w') as f:
                json.dump(stats, f)
    
    # Final save
    agent.save_logs(n_episodes - 1, output_dir / 'logs')
    
    print(f"\nTraining completed!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final average length: {np.mean(episode_lengths[-100:]):.2f}")
    
    return agent, episode_rewards, episode_lengths, episode_losses

def analyze_micro_results(output_dir):
    """Analyze MicroBetaBae training results"""
    output_dir = Path(output_dir)
    
    # Load statistics
    stats_files = sorted(output_dir.glob('stats_ep_*.json'))
    if not stats_files:
        print("No statistics files found!")
        return
    
    # Load final stats
    with open(stats_files[-1], 'r') as f:
        stats = json.load(f)
    
    rewards = stats['rewards']
    lengths = stats['lengths']
    losses = stats['losses']
    
    print(f"\nMicroBetaBae Analysis:")
    print(f"Total episodes: {len(rewards)}")
    print(f"Final average reward: {np.mean(rewards[-100:]):.2f}")
    print(f"Final average length: {np.mean(lengths[-100:]):.2f}")
    print(f"Final average loss: {np.mean(losses[-100:]):.2f}")
    
    # Learning progress
    window = 100
    if len(rewards) >= window:
        early_reward = np.mean(rewards[:window])
        late_reward = np.mean(rewards[-window:])
        improvement = late_reward - early_reward
        
        print(f"Learning improvement: {improvement:.2f}")
        print(f"Early performance: {early_reward:.2f}")
        print(f"Late performance: {late_reward:.2f}")
    
    return stats

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MicroBetaBae')
    parser.add_argument('--env', default='CartPole-v1', help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--output', default='./micro_outputs', help='Output directory')
    parser.add_argument('--d_model', type=int, default=32, help='Model dimension')
    parser.add_argument('--seq_len', type=int, default=8, help='Sequence length')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no_search', action='store_true', help='Disable search')
    parser.add_argument('--no_reflection', action='store_true', help='Disable reflection')
    
    args = parser.parse_args()
    
    # Train the agent
    agent, rewards, lengths, losses = train_micro_betabae(
        env_name=args.env,
        n_episodes=args.episodes,
        output_dir=args.output,
        d_model=args.d_model,
        seq_len=args.seq_len,
        lr=args.lr,
        search_enabled=not args.no_search,
        reflection_enabled=not args.no_reflection
    )
    
    # Analyze results
    analyze_micro_results(args.output)
