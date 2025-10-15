from pathlib import Path
import numpy as np

class EvolutionLogger:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.attention = []
        self.hidden = []
        self.pred_error = []
        self.actions = []

    def log_step(self, attn, hidden, error, action):
        self.attention.append(attn.detach().cpu().numpy())
        self.hidden.append(hidden.detach().cpu().numpy())
        self.pred_error.append(error)
        self.actions.append(action)

    def save(self, episode):
        np.savez_compressed(
            self.save_dir / f'episode_{episode:05d}.npz',
            attention=np.array(self.attention),
            hidden=np.array(self.hidden),
            pred_error=np.array(self.pred_error),
            actions=np.array(self.actions)
        )
        self.attention = []
        self.hidden = []
        self.pred_error = []
        self.actions = []
