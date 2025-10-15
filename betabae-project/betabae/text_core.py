import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_heads = n_heads
        
        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'ln2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        
        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        
        # Token and position embeddings
        tok_emb = self.token_embed(x)  # (B, T, d_model)
        pos_emb = self.pos_embed(torch.arange(T, device=x.device))  # (T, d_model)
        x = tok_emb + pos_emb  # (B, T, d_model)
        
        # Store attention weights for visualization
        attention_weights = []
        
        # Transformer layers
        for layer in self.layers:
            # Self-attention
            residual = x
            attn_out, attn_weights_layer = layer['attn'](layer['ln1'](x), layer['ln1'](x), layer['ln1'](x))
            x = residual + attn_out
            attention_weights.append(attn_weights_layer)
            
            # Feed-forward
            residual = x
            x = x + layer['ff'](layer['ln2'](x))
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.output_proj(x)  # (B, T, vocab_size)
        
        return logits, attention_weights

class TextAgent(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, seq_len=128, lr=3e-4):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        self.net = TextTransformer(vocab_size, d_model, n_layers, n_heads, seq_len)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        """Forward pass returning logits and attention weights"""
        return self.net(x)
    
    def loss(self, logits, targets):
        """
        Improved text generation loss with multiple objectives:
        1. Cross-entropy for next-token prediction
        2. Label smoothing for better generalization
        3. Attention regularization for sparsity
        """
        # Shift targets to align with predictions
        logits = logits[:, :-1, :].contiguous()  # (B, T-1, vocab_size)
        targets = targets[:, 1:].contiguous()     # (B, T-1)
        
        # 1. Cross-entropy loss with label smoothing
        vocab_size = logits.size(-1)
        smooth_loss = self._label_smoothing_loss(logits, targets, vocab_size)
        
        # 2. Perplexity-based regularization
        perplexity_reg = self._perplexity_regularization(logits)
        
        # 3. Attention regularization (if attention weights available)
        attn_reg = self._text_attention_regularization()
        
        # Total loss
        total_loss = smooth_loss + 0.1 * perplexity_reg + 0.01 * attn_reg
        return total_loss
    
    def _label_smoothing_loss(self, logits, targets, vocab_size, smoothing=0.1):
        """Label smoothing for better generalization"""
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        # Label smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -(log_probs * (1 - smoothing) / vocab_size).sum(dim=-1)
        smooth_loss = smooth_loss.mean()
        
        return (1 - smoothing) * ce_loss + smoothing * smooth_loss
    
    def _perplexity_regularization(self, logits):
        """Regularize perplexity to prevent overconfidence"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Perplexity = exp(entropy)
        entropy = -(probs * log_probs).sum(dim=-1)
        perplexity = entropy.exp()
        
        # Penalize very low perplexity (overconfidence)
        target_perplexity = 10.0  # Target perplexity
        reg_loss = F.mse_loss(perplexity, torch.full_like(perplexity, target_perplexity))
        
        return reg_loss
    
    def _text_attention_regularization(self):
        """Regularize attention patterns for text generation"""
        # This would use stored attention weights from forward pass
        # For now, return zero (would need to modify forward pass to store weights)
        return torch.tensor(0.0)
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=None):
        """Generate text from a prompt"""
        self.eval()
        with torch.no_grad():
            # Convert prompt to tensor
            if isinstance(prompt, str):
                # This would need a tokenizer in practice
                prompt_tokens = [ord(c) % self.vocab_size for c in prompt[:self.seq_len]]
            else:
                prompt_tokens = prompt
            
            # Pad or truncate to seq_len
            if len(prompt_tokens) < self.seq_len:
                prompt_tokens = prompt_tokens + [0] * (self.seq_len - len(prompt_tokens))
            else:
                prompt_tokens = prompt_tokens[:self.seq_len]
            
            x = torch.tensor(prompt_tokens).unsqueeze(0)  # (1, seq_len)
            
            generated = x.clone()
            
            for _ in range(max_length):
                # Get logits for next token
                logits, _ = self.forward(x)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Update sequence
                x = torch.cat([x[:, 1:], next_token], dim=1)
                generated = torch.cat([generated, next_token], dim=1)
            
            return generated
