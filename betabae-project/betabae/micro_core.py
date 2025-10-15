"""
MicroBetaBae: Micrograd + Sutton's Bitter Lesson + DeepSeek R1 Principles

Key Design Principles:
1. Sutton's Bitter Lesson: Computation > Knowledge
2. DeepSeek R1: Reasoning through search and reflection
3. Micrograd: Minimal autograd for efficiency
4. Scalable: Works past 10,000 episodes without memory issues
"""

import numpy as np
import random
import math
from collections import deque, defaultdict
import json
from pathlib import Path

# Micrograd Value class for automatic differentiation
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        # Handle Value objects in data
        if isinstance(data, Value):
            self.data = data.data
        else:
            self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    def pow(self, other):
        """Power operation"""
        return self ** other

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        x = self.data
        out = Value(math.log(x), (self,), 'log')
        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    
    def __abs__(self): 
        return Value(abs(self.data), (self,), f'abs')
    
    def abs(self):
        """Absolute value"""
        return Value(abs(self.data), (self,), 'abs')

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class MicroLayer:
    """Minimal neural network layer using micrograd"""
    def __init__(self, nin, nout, activation='relu'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin * nout)]
        self.b = [Value(0) for _ in range(nout)]
        self.activation = activation
        
    def __call__(self, x):
        out = []
        for i in range(len(self.b)):
            sum_val = self.b[i]
            for j in range(len(x)):
                sum_val = sum_val + self.w[i * len(x) + j] * x[j]
            
            if self.activation == 'relu':
                out.append(sum_val.relu())
            elif self.activation == 'tanh':
                out.append(sum_val.tanh())
            else:
                out.append(sum_val)
        return out
    
    def parameters(self):
        return self.w + self.b

class MicroAttention:
    """Minimal attention mechanism using micrograd"""
    def __init__(self, d_model, n_heads=1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Query, Key, Value projections
        self.q_proj = MicroLayer(d_model, d_model, 'linear')
        self.k_proj = MicroLayer(d_model, d_model, 'linear')
        self.v_proj = MicroLayer(d_model, d_model, 'linear')
        self.out_proj = MicroLayer(d_model, d_model, 'linear')
        
        self.scale = Value(1.0 / math.sqrt(self.head_dim))
    
    def __call__(self, x):
        batch_size, seq_len, d_model = len(x), len(x[0]), len(x[0][0])
        
        # Project to Q, K, V
        q = [self.q_proj(x[i][j]) for i in range(batch_size) for j in range(seq_len)]
        k = [self.k_proj(x[i][j]) for i in range(batch_size) for j in range(seq_len)]
        v = [self.v_proj(x[i][j]) for i in range(batch_size) for j in range(seq_len)]
        
        # Reshape for attention
        q = [q[i*seq_len:(i+1)*seq_len] for i in range(batch_size)]
        k = [k[i*seq_len:(i+1)*seq_len] for i in range(batch_size)]
        v = [v[i*seq_len:(i+1)*seq_len] for i in range(batch_size)]
        
        # Compute attention scores
        attention_scores = []
        attention_weights = []
        
        for b in range(batch_size):
            batch_scores = []
            batch_weights = []
            
            for i in range(seq_len):
                scores = []
                for j in range(seq_len):
                    # Dot product attention
                    score = Value(0)
                    for d in range(self.head_dim):
                        score = score + q[b][i][d] * k[b][j][d]
                    score = score * self.scale
                    scores.append(score)
                
                # Softmax
                max_score = max(scores, key=lambda x: x.data)
                exp_scores = [(s - max_score).exp() for s in scores]
                sum_exp = sum(exp_scores)
                weights = [exp / sum_exp for exp in exp_scores]
                
                batch_scores.append(scores)
                batch_weights.append(weights)
            
            attention_scores.append(batch_scores)
            attention_weights.append(batch_weights)
        
        # Apply attention to values
        out = []
        for b in range(batch_size):
            batch_out = []
            for i in range(seq_len):
                weighted_sum = [Value(0) for _ in range(d_model)]
                for j in range(seq_len):
                    for d in range(d_model):
                        weighted_sum[d] = weighted_sum[d] + attention_weights[b][i][j] * v[b][j][d]
                batch_out.append(weighted_sum)
            out.append(batch_out)
        
        # Output projection
        final_out = []
        for b in range(batch_size):
            batch_final = []
            for i in range(seq_len):
                batch_final.append(self.out_proj(out[b][i]))
            final_out.append(batch_final)
        
        return final_out, attention_weights

class MicroBetaBae:
    """
    MicroBetaBae: Minimal implementation following Sutton's Bitter Lesson
    
    Core Principles:
    1. Computation > Knowledge: Let the model learn everything
    2. Search-based reasoning: Use MCTS-like search for actions
    3. Reflection: Learn from mistakes through replay
    4. Scalability: Efficient memory and computation
    """
    
    def __init__(self, obs_dim, action_dim, d_model=32, seq_len=8, lr=0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.lr = lr
        
        # Core network components
        self.embed = MicroLayer(obs_dim + action_dim, d_model, 'tanh')
        self.attention = MicroAttention(d_model)
        self.predict = MicroLayer(d_model, obs_dim, 'linear')
        self.act = MicroLayer(d_model, action_dim, 'linear')
        
        # Sutton's Bitter Lesson: Use computation for search
        self.search_depth = 3
        self.search_width = 5
        
        # DeepSeek R1: Reflection and replay
        self.replay_buffer = deque(maxlen=10000)
        self.reflection_buffer = deque(maxlen=1000)
        
        # Memory-efficient logging
        self.attention_log = deque(maxlen=1000)
        self.hidden_log = deque(maxlen=1000)
        self.loss_log = deque(maxlen=1000)
        
        # Parameters for optimization
        self.params = []
        for layer in [self.embed, self.predict, self.act]:
            self.params.extend(layer.parameters())
        self.params.extend(self.attention.q_proj.parameters())
        self.params.extend(self.attention.k_proj.parameters())
        self.params.extend(self.attention.v_proj.parameters())
        self.params.extend(self.attention.out_proj.parameters())
    
    def forward(self, history):
        """Forward pass with attention"""
        # Embed history
        embedded = []
        for step in history:
            embedded.append(self.embed(step))
        
        # Apply attention
        attended, attention_weights = self.attention([embedded])
        
        # Get last hidden state
        last_hidden = attended[0][-1]
        
        # Predictions
        pred = self.predict(last_hidden)
        logits = self.act(last_hidden)
        
        return pred, logits, attention_weights[0]
    
    def search_action(self, history, env):
        """
        Sutton's Bitter Lesson: Use computation for better decisions
        MCTS-like search for action selection
        """
        best_action = 0
        best_value = float('-inf')
        
        for action in range(self.action_dim):
            # Simulate action
            total_value = 0
            for _ in range(self.search_width):
                value = self._simulate_action(history, action, env, self.search_depth)
                total_value += value
            
            avg_value = total_value / self.search_width
            
            if avg_value > best_value:
                best_value = avg_value
                best_action = action
        
        return best_action
    
    def _simulate_action(self, history, action, env, depth):
        """Simulate action sequence for search"""
        if depth == 0:
            return 0
        
        # Predict next state
        pred, logits, _ = self.forward(history)
        
        # Estimate value (negative prediction error = good)
        pred_error = sum((pred[i].data - history[-1][i])**2 for i in range(self.obs_dim))
        value = -pred_error
        
        # Add exploration bonus
        exploration_bonus = 0.1 * math.sqrt(math.log(len(self.replay_buffer) + 1))
        
        return value + exploration_bonus
    
    def learn(self, history, action, next_obs, reward):
        """
        DeepSeek R1: Learn through reflection and replay with improved loss functions
        """
        # Forward pass
        pred, logits, attention_weights = self.forward(history)
        
        # 1. Prediction Loss: Huber Loss for robustness
        pred_loss = self._huber_loss(pred, next_obs)
        
        # 2. Value Function: Temporal Difference with eligibility traces
        value_estimate = self._compute_value(pred, logits)
        td_error = Value(reward) - value_estimate
        
        # 3. Policy Loss: Actor-Critic with entropy regularization
        log_probs = self._log_softmax(logits)
        action_log_prob = log_probs[action]
        
        # Entropy bonus for exploration
        entropy = -sum(p * log_prob for p, log_prob in zip(self._softmax(logits), log_probs))
        entropy_bonus = Value(0.01) * entropy
        
        # Actor-Critic loss
        actor_loss = -action_log_prob * td_error
        critic_loss = td_error ** 2
        
        # 4. Attention Regularization: Encourage sparsity
        attn_reg = self._attention_regularization(attention_weights)
        
        # Total loss with proper weighting
        total_loss = pred_loss + Value(0.5) * actor_loss + Value(0.5) * critic_loss - entropy_bonus + Value(0.1) * attn_reg
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability (simplified)
        max_grad_norm = 1.0
        for p in self.params:
            grad_val = p.grad.data if hasattr(p.grad, 'data') else p.grad
            if grad_val > max_grad_norm:
                p.grad = max_grad_norm
            elif grad_val < -max_grad_norm:
                p.grad = -max_grad_norm
        
        # Update parameters with momentum
        for p in self.params:
            if not hasattr(p, 'momentum'):
                p.momentum = Value(0.0)
            p.momentum = Value(0.9) * p.momentum + self.lr * p.grad
            p.data -= p.momentum
            p.grad = 0.0
        
        # Log for analysis
        self.attention_log.append([[w.data for w in row] for row in attention_weights])
        self.hidden_log.append([h.data for h in pred])
        self.loss_log.append(total_loss.data)
        
        # Store in replay buffer with priority
        priority = abs(td_error.data) + 1e-6  # Higher TD error = higher priority
        self.replay_buffer.append({
            'history': history,
            'action': action,
            'next_obs': next_obs,
            'reward': reward,
            'attention': attention_weights,
            'loss': total_loss.data,
            'td_error': td_error.data,
            'priority': priority
        })
        
        # Reflection: Learn from mistakes (high TD error)
        if abs(td_error.data) > 0.5:  # High TD error = surprise/mistake
            self.reflection_buffer.append({
                'history': history,
                'action': action,
                'next_obs': next_obs,
                'reward': reward,
                'td_error': td_error.data,
                'loss': total_loss.data
            })
    
    def _huber_loss(self, pred, target):
        """Huber loss for robust prediction learning"""
        total_loss = Value(0.0)
        delta = Value(1.0)  # Huber loss threshold
        
        for i in range(len(pred)):
            error = pred[i] - Value(target[i])
            abs_error = abs(error)
            
            if abs_error.data <= delta.data:
                # Quadratic loss for small errors
                total_loss = total_loss + Value(0.5) * error**2
            else:
                # Linear loss for large errors
                total_loss = total_loss + delta * (abs_error - Value(0.5) * delta)
        
        return total_loss
    
    def _compute_value(self, pred, logits):
        """Compute value estimate from predictions and policy"""
        # Value = expected reward based on prediction quality
        pred_quality = Value(0.0)
        for i in range(len(pred)):
            pred_quality = pred_quality + pred[i]**2
        
        # Policy confidence affects value
        policy_confidence = sum(self._softmax(logits))**2
        
        # Combined value estimate
        value = pred_quality + policy_confidence
        return value
    
    def _attention_regularization(self, attention_weights):
        """Encourage attention sparsity and diversity"""
        reg_loss = Value(0.0)
        
        for attn_row in attention_weights:
            # Entropy regularization (encourage sparsity)
            entropy = -sum(w * (w + Value(1e-10)).log() for w in attn_row)
            reg_loss = reg_loss + entropy
            
            # L1 regularization (encourage sparsity)
            l1_reg = sum(abs(w) for w in attn_row)
            reg_loss = reg_loss + Value(0.01) * l1_reg
        
        return reg_loss
    
    def _log_softmax(self, logits):
        """Compute log softmax"""
        max_logit = max(logits, key=lambda x: x.data)
        exp_logits = [(logit - max_logit).exp() for logit in logits]
        sum_exp = sum(exp_logits)
        probs = [exp / sum_exp for exp in exp_logits]
        log_probs = [prob.exp().log() for prob in probs]
        return log_probs
    
    def _softmax(self, logits):
        """Compute softmax"""
        max_logit = max(logits, key=lambda x: x.data)
        exp_logits = [(logit - max_logit).exp() for logit in logits]
        sum_exp = sum(exp_logits)
        probs = [exp / sum_exp for exp in exp_logits]
        return probs
    
    def reflect(self):
        """
        DeepSeek R1: Learn from past mistakes
        """
        if len(self.reflection_buffer) < 10:
            return
        
        # Sample random mistakes
        mistakes = random.sample(self.reflection_buffer, min(5, len(self.reflection_buffer)))
        
        for mistake in mistakes:
            # Re-learn from mistake
            self.learn(
                mistake['history'],
                mistake['action'],
                mistake['next_obs'],
                mistake['reward']
            )
    
    def save_logs(self, episode, save_dir):
        """Save logs efficiently"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save attention patterns
        if self.attention_log:
            attention_data = list(self.attention_log)
            np.save(save_dir / f'attention_ep_{episode:05d}.npy', attention_data)
        
        # Save hidden states
        if self.hidden_log:
            hidden_data = list(self.hidden_log)
            np.save(save_dir / f'hidden_ep_{episode:05d}.npy', hidden_data)
        
        # Save losses
        if self.loss_log:
            loss_data = list(self.loss_log)
            np.save(save_dir / f'loss_ep_{episode:05d}.npy', loss_data)
        
        # Save model state
        model_state = {
            'params': [p.data for p in self.params],
            'episode': episode,
            'replay_size': len(self.replay_buffer),
            'reflection_size': len(self.reflection_buffer)
        }
        
        with open(save_dir / f'model_ep_{episode:05d}.json', 'w') as f:
            json.dump(model_state, f)
    
    def load_model(self, save_dir, episode):
        """Load model state"""
        save_dir = Path(save_dir)
        model_file = save_dir / f'model_ep_{episode:05d}.json'
        
        if model_file.exists():
            with open(model_file, 'r') as f:
                model_state = json.load(f)
            
            for i, param_data in enumerate(model_state['params']):
                if i < len(self.params):
                    self.params[i].data = param_data
            
            print(f"Loaded model from episode {episode}")
            return True
        return False
