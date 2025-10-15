import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import re

class TextLogger:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Logging buffers
        self.attention_weights = []  # List of attention matrices per step
        self.hidden_states = []      # Last token hidden states
        self.losses = []            # Training losses
        self.perplexities = []      # Perplexity values
        self.generated_samples = [] # Generated text samples
        
    def log_step(self, attention_weights, hidden_state, loss, perplexity=None):
        """Log training step data"""
        # Store attention weights (detach from computation graph)
        self.attention_weights.append([attn.detach().cpu().numpy() for attn in attention_weights])
        
        # Store last token hidden state
        self.hidden_states.append(hidden_state.detach().cpu().numpy())
        
        # Store loss and perplexity
        self.losses.append(loss)
        if perplexity is not None:
            self.perplexities.append(perplexity)
    
    def log_generation(self, sample_text, step):
        """Log generated text sample"""
        self.generated_samples.append({
            'step': step,
            'text': sample_text
        })
    
    def save_epoch(self, epoch):
        """Save all logged data for an epoch"""
        if not self.attention_weights:
            return
            
        # Convert to numpy arrays
        attention_array = np.array(self.attention_weights)  # (steps, layers, heads, T, T)
        hidden_array = np.array(self.hidden_states)        # (steps, d_model)
        losses_array = np.array(self.losses)               # (steps,)
        
        # Save compressed data
        save_data = {
            'attention': attention_array,
            'hidden_states': hidden_array,
            'losses': losses_array,
        }
        
        if self.perplexities:
            save_data['perplexities'] = np.array(self.perplexities)
        
        np.savez_compressed(
            self.save_dir / f'epoch_{epoch:03d}.npz',
            **save_data
        )
        
        # Save generation samples as text
        if self.generated_samples:
            with open(self.save_dir / f'generations_epoch_{epoch:03d}.txt', 'w') as f:
                for sample in self.generated_samples:
                    f.write(f"Step {sample['step']}: {sample['text']}\n")
        
        # Clear buffers
        self.attention_weights = []
        self.hidden_states = []
        self.losses = []
        self.perplexities = []
        self.generated_samples = []

class GitaDataset:
    def __init__(self, csv_path, seq_len=128, vocab_size=256):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Load and preprocess text
        self.text = self._load_and_preprocess(csv_path)
        self.vocab = self._build_vocab()
        self.vocab_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_vocab = {idx: char for char, idx in self.vocab_to_idx.items()}
        
        # Convert text to token sequences
        self.tokens = self._text_to_tokens(self.text)
        
        print(f"Dataset loaded:")
        print(f"  Text length: {len(self.text)} characters")
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  Number of sequences: {len(self.tokens)}")
        print(f"  Sample text: {self.text[:200]}...")
    
    def _load_and_preprocess(self, csv_path):
        """Load Gita text from CSV and preprocess"""
        df = pd.read_csv(csv_path)
        
        # Combine all translations into one text
        text = " ".join(df['translation'].astype(str))
        
        # Basic preprocessing
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def _build_vocab(self):
        """Build character-level vocabulary"""
        # Get unique characters
        chars = sorted(list(set(self.text)))
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        vocab = special_tokens + chars
        
        # Limit vocabulary size
        if len(vocab) > self.vocab_size:
            vocab = vocab[:self.vocab_size]
        
        return vocab
    
    def _text_to_tokens(self, text):
        """Convert text to token sequences"""
        # Convert characters to indices
        tokens = []
        for char in text:
            if char in self.vocab_to_idx:
                tokens.append(self.vocab_to_idx[char])
            else:
                tokens.append(self.vocab_to_idx['<UNK>'])
        
        # Split into sequences
        sequences = []
        for i in range(0, len(tokens) - self.seq_len + 1, self.seq_len // 2):  # 50% overlap
            seq = tokens[i:i + self.seq_len]
            if len(seq) == self.seq_len:
                sequences.append(seq)
        
        return sequences
    
    def get_batch(self, batch_size=8):
        """Get a random batch of sequences"""
        indices = np.random.choice(len(self.tokens), batch_size, replace=False)
        batch = torch.tensor([self.tokens[i] for i in indices], dtype=torch.long)
        return batch
    
    def decode_tokens(self, tokens):
        """Convert token indices back to text"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        
        text = ""
        for token in tokens:
            if token < len(self.idx_to_vocab):
                text += self.idx_to_vocab[token]
        
        return text
    
    def encode_text(self, text):
        """Convert text to token indices"""
        tokens = []
        for char in text:
            if char in self.vocab_to_idx:
                tokens.append(self.vocab_to_idx[char])
            else:
                tokens.append(self.vocab_to_idx['<UNK>'])
        return tokens
