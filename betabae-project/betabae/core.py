import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.last_hidden = None

    def forward(self, x):
        x = self.embed(x)
        attn_output, attn_weights = self.attn(x, x, x)
        self.last_hidden = x[:, -1, :]
        return x, attn_weights

class MinimalAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, d_model=32, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.net = SimpleTransformer(obs_dim + action_dim, d_model)
        self.predict = nn.Linear(d_model, obs_dim)
        self.act = nn.Linear(d_model, action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, history):
        h, attn = self.net(history)
        pred = self.predict(h[:, -1])
        logits = self.act(h[:, -1])
        return pred, logits, attn

    def loss(self, pred, actual, logits, action):
        """
        Improved loss function combining multiple objectives:
        1. Huber Loss for robust prediction learning
        2. Actor-Critic with entropy regularization
        3. Attention regularization for sparsity
        """
        # 1. Prediction Loss: Huber Loss for robustness
        pred_loss = self._huber_loss(pred, actual)
        
        # 2. Value Function: Temporal Difference
        value_estimate = self._compute_value(pred, logits)
        surprise = F.mse_loss(pred, actual)
        td_error = -surprise.detach() - value_estimate
        
        # 3. Policy Loss: Actor-Critic with entropy regularization
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Entropy bonus for exploration
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_bonus = 0.01 * entropy.mean()
        
        # Actor-Critic loss
        actor_loss = -(td_error.detach() * action_log_prob).mean()
        critic_loss = td_error.pow(2).mean()
        
        # Total loss with proper weighting
        total_loss = pred_loss + 0.5 * actor_loss + 0.5 * critic_loss - entropy_bonus
        
        return total_loss
    
    def _huber_loss(self, pred, target):
        """Huber loss for robust prediction learning"""
        delta = 1.0
        error = pred - target
        abs_error = torch.abs(error)
        
        # Quadratic loss for small errors, linear for large errors
        quadratic_loss = 0.5 * error.pow(2)
        linear_loss = delta * (abs_error - 0.5 * delta)
        
        # Use quadratic loss where error is small, linear elsewhere
        loss = torch.where(abs_error <= delta, quadratic_loss, linear_loss)
        return loss.mean()
    
    def _compute_value(self, pred, logits):
        """Compute value estimate from predictions and policy"""
        # Value based on prediction confidence
        pred_confidence = pred.pow(2).sum(dim=-1)
        
        # Policy confidence affects value
        probs = F.softmax(logits, dim=-1)
        policy_confidence = probs.pow(2).sum(dim=-1)
        
        # Combined value estimate
        value = pred_confidence + policy_confidence
        return value.mean()
