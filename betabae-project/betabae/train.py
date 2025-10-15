import torch
import torch.nn.functional as F
from collections import deque
import numpy as np
try:
    # Prefer Gymnasium when available
    import gymnasium as gym  # type: ignore
    GYMN_VERSION = "gymnasium"
except Exception:  # pragma: no cover
    import gym  # type: ignore
    GYMN_VERSION = "gym"
from betabae.core import MinimalAgent
from betabae.logger import EvolutionLogger

def train(agent, env, logger, n_episodes=100, max_steps=100):
    history = deque(maxlen=agent.seq_len)

    for episode in range(n_episodes):
        reset_out = env.reset()
        # Gymnasium: (obs, info), Gym: obs
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        history.clear()
        for _ in range(agent.seq_len):
            history.append(np.concatenate([obs, np.zeros(env.action_space.n)]))

        done = False
        step = 0
        while not done and step < max_steps:
            hist_tensor = torch.from_numpy(np.array(history)).float().unsqueeze(0)
            pred, logits, attn = agent(hist_tensor)

            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            step_out = env.step(action)
            # Gymnasium: (obs, reward, terminated, truncated, info)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                # Old Gym: (obs, reward, done, info)
                next_obs, reward, done, _ = step_out

            actual = torch.from_numpy(next_obs).float().unsqueeze(0)
            action_tensor = torch.LongTensor([action])

            loss = agent.loss(pred, actual, logits, action_tensor)

            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            logger.log_step(
                attn=attn,
                hidden=agent.net.last_hidden,
                error=loss.item(),
                action=action
            )

            action_onehot = np.zeros(env.action_space.n)
            action_onehot[action] = 1.0
            history.append(np.concatenate([next_obs, action_onehot]))

            step += 1

        logger.save(episode)
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = MinimalAgent(obs_dim, action_dim)
    logger = EvolutionLogger('logs/run_001')
    train(agent, env, logger, n_episodes=args.episodes)
