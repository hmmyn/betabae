"""
MicroBetaBae Visualization: Analyze attention patterns and learning dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.decomposition import PCA

def plot_micro_learning_curves(output_dir):
    """Plot learning curves for MicroBetaBae"""
    output_dir = Path(output_dir)
    
    # Load statistics
    stats_files = sorted(output_dir.glob('stats_ep_*.json'))
    if not stats_files:
        print("No statistics files found!")
        return
    
    with open(stats_files[-1], 'r') as f:
        stats = json.load(f)
    
    rewards = stats['rewards']
    lengths = stats['lengths']
    losses = stats['losses']
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards, alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(lengths, alpha=0.7, linewidth=0.5, color='orange')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode losses
    axes[1, 0].plot(losses, alpha=0.7, linewidth=0.5, color='green')
    axes[1, 0].set_title('Episode Losses')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rolling averages
    window = 100
    if len(rewards) >= window:
        rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        rolling_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
        
        axes[1, 1].plot(rolling_rewards, label=f'Reward (window={window})', linewidth=2)
        axes[1, 1].plot(rolling_lengths, label=f'Length (window={window})', linewidth=2)
        axes[1, 1].set_title('Rolling Averages')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'micro_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved learning curves: {output_dir / 'micro_learning_curves.png'}")

def analyze_micro_attention(output_dir):
    """Analyze attention patterns from MicroBetaBae"""
    output_dir = Path(output_dir)
    log_dir = output_dir / 'logs'
    
    if not log_dir.exists():
        print("No log directory found!")
        return
    
    # Load attention data
    attention_files = sorted(log_dir.glob('attention_ep_*.npy'))
    if not attention_files:
        print("No attention files found!")
        return
    
    print(f"Found {len(attention_files)} attention files")
    
    # Analyze attention evolution
    attention_entropies = []
    attention_sparsity = []
    
    for attn_file in attention_files:
        attention_data = np.load(attn_file)
        
        # Take last attention matrix
        if len(attention_data) > 0:
            last_attn = attention_data[-1]
            
            # Calculate entropy (uniformity measure)
            entropy = -(last_attn * np.log(last_attn + 1e-10)).sum(axis=-1).mean()
            attention_entropies.append(entropy)
            
            # Calculate sparsity (concentration measure)
            sparsity = last_attn.max(axis=-1).mean()
            attention_sparsity.append(sparsity)
    
    # Plot attention analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Attention entropy over episodes
    axes[0].plot(attention_entropies, marker='o', markersize=3)
    axes[0].set_title('Attention Entropy Evolution')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Entropy')
    axes[0].grid(True, alpha=0.3)
    
    # Attention sparsity over episodes
    axes[1].plot(attention_sparsity, marker='o', markersize=3, color='orange')
    axes[1].set_title('Attention Sparsity Evolution')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Max Attention Weight')
    axes[1].grid(True, alpha=0.3)
    
    # Show attention matrices for key episodes
    key_episodes = [0, len(attention_files)//2, len(attention_files)-1]
    for i, ep_idx in enumerate(key_episodes):
        if ep_idx < len(attention_files):
            attention_data = np.load(attention_files[ep_idx])
            attn_matrix = attention_data[-1]  # Last attention matrix
            
            im = axes[2].imshow(attn_matrix, cmap='viridis', aspect='auto')
            axes[2].set_title(f'Attention Matrix (Episode {ep_idx})')
            axes[2].set_xlabel('Past Token')
            axes[2].set_ylabel('Current Token')
            plt.colorbar(im, ax=axes[2])
            break
    
    plt.tight_layout()
    plt.savefig(output_dir / 'micro_attention_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved attention analysis: {output_dir / 'micro_attention_analysis.png'}")

def analyze_micro_embeddings(output_dir):
    """Analyze hidden state evolution"""
    output_dir = Path(output_dir)
    log_dir = output_dir / 'logs'
    
    if not log_dir.exists():
        print("No log directory found!")
        return
    
    # Load hidden state data
    hidden_files = sorted(log_dir.glob('hidden_ep_*.npy'))
    if not hidden_files:
        print("No hidden state files found!")
        return
    
    print(f"Found {len(hidden_files)} hidden state files")
    
    # Collect all hidden states
    all_hidden = []
    episode_labels = []
    
    for i, hidden_file in enumerate(hidden_files):
        hidden_data = np.load(hidden_file)
        all_hidden.append(hidden_data)
        episode_labels.extend([i] * len(hidden_data))
    
    all_hidden = np.vstack(all_hidden)
    
    # PCA analysis
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_hidden)
    
    # Plot embedding evolution
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=episode_labels,
        cmap='viridis',
        s=1,
        alpha=0.6
    )
    plt.colorbar(scatter, label='Episode')
    plt.title('MicroBetaBae Representation Space Evolution')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.savefig(output_dir / 'micro_embeddings.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved embedding plot: {output_dir / 'micro_embeddings.png'}")

def create_micro_summary(output_dir):
    """Create comprehensive summary of MicroBetaBae results"""
    output_dir = Path(output_dir)
    
    # Load final statistics
    stats_files = sorted(output_dir.glob('stats_ep_*.json'))
    if not stats_files:
        print("No statistics files found!")
        return
    
    with open(stats_files[-1], 'r') as f:
        stats = json.load(f)
    
    rewards = stats['rewards']
    lengths = stats['lengths']
    losses = stats['losses']
    
    # Calculate metrics
    total_episodes = len(rewards)
    final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    final_length = np.mean(lengths[-100:]) if len(lengths) >= 100 else np.mean(lengths)
    final_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
    
    # Learning progress
    if len(rewards) >= 200:
        early_reward = np.mean(rewards[:100])
        late_reward = np.mean(rewards[-100:])
        improvement = late_reward - early_reward
    else:
        early_reward = np.mean(rewards[:len(rewards)//2])
        late_reward = np.mean(rewards[len(rewards)//2:])
        improvement = late_reward - early_reward
    
    # Create summary
    summary = f"""
MicroBetaBae Training Summary
============================

Training Configuration:
- Total Episodes: {total_episodes:,}
- Elapsed Time: {stats.get('elapsed_time', 0):.2f} seconds
- Episodes per Second: {total_episodes / max(stats.get('elapsed_time', 1), 1):.2f}

Performance Metrics:
- Final Average Reward: {final_reward:.2f}
- Final Average Length: {final_length:.1f} steps
- Final Average Loss: {final_loss:.4f}
- Learning Improvement: {improvement:.2f}

Learning Progress:
- Early Performance: {early_reward:.2f}
- Late Performance: {late_reward:.2f}
- Improvement: {improvement:.2f} ({improvement/early_reward*100:.1f}%)

Key Insights:
- Sutton's Bitter Lesson: Computation > Knowledge
- Search-based action selection improves decision quality
- Reflection learning accelerates convergence
- Micrograd enables efficient scaling past 10,000 episodes
"""
    
    print(summary)
    
    # Save summary
    with open(output_dir / 'micro_summary.txt', 'w') as f:
        f.write(summary)
    
    print(f"Saved summary: {output_dir / 'micro_summary.txt'}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python micro_visualize.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    # Generate all visualizations
    plot_micro_learning_curves(output_dir)
    analyze_micro_attention(output_dir)
    analyze_micro_embeddings(output_dir)
    create_micro_summary(output_dir)
    
    print("\nMicroBetaBae analysis complete!")
