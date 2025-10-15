import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import imageio

def create_text_attention_video(log_dir: Path, output_path: Path, dataset=None):
    """Create video showing attention matrix evolution for text generation"""
    epochs = sorted(log_dir.glob('epoch_*.npz'))
    
    if not epochs:
        print("No epoch files found!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    frames = []
    
    for epoch_file in epochs:
        data = np.load(epoch_file)
        attention = data['attention']  # (steps, layers, heads, T, T)
        
        # Take the last step of the epoch, first layer, first head
        if len(attention.shape) == 5:
            attn_matrix = attention[-1, 0, 0]  # Last step, first layer, first head
        else:
            attn_matrix = attention[-1, 0]  # Last step, first layer
        
        ax.clear()
        im = ax.imshow(attn_matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Extract epoch number from filename
        epoch_num = epoch_file.stem.split('_')[1]
        ax.set_title(f'Attention Matrix - Epoch {epoch_num}')
        ax.set_xlabel('Past Token Position')
        ax.set_ylabel('Current Token Position')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Convert to frame
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]  # Remove alpha channel
        frames.append(frame)
    
    # Create video
    if frames:
        imageio.mimsave(output_path, frames, fps=2)
        print(f"Saved attention video: {output_path}")
    else:
        print("No frames to create video!")

def plot_text_learning_curves(log_dir: Path, output_path: Path):
    """Plot learning curves for text generation"""
    epochs = sorted(log_dir.glob('epoch_*.npz'))
    
    if not epochs:
        print("No epoch files found!")
        return
    
    losses = []
    perplexities = []
    
    for epoch_file in epochs:
        data = np.load(epoch_file)
        losses.extend(data['losses'])
        if 'perplexities' in data:
            perplexities.extend(data['perplexities'])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Loss curve
    axes[0].plot(losses, alpha=0.7, linewidth=0.5)
    axes[0].set_title('Training Loss Evolution')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity curve (if available)
    if perplexities:
        axes[1].plot(perplexities, alpha=0.7, linewidth=0.5, color='orange')
        axes[1].set_title('Perplexity Evolution')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Perplexity')
        axes[1].grid(True, alpha=0.3)
    else:
        # Calculate perplexity from loss
        perplexities_calc = [np.exp(loss) for loss in losses]
        axes[1].plot(perplexities_calc, alpha=0.7, linewidth=0.5, color='orange')
        axes[1].set_title('Perplexity Evolution (calculated from loss)')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Perplexity')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning curves: {output_path}")

def plot_text_embedding_evolution(log_dir: Path, output_path: Path):
    """Plot embedding evolution for text generation"""
    epochs = sorted(log_dir.glob('epoch_*.npz'))
    
    if not epochs:
        print("No epoch files found!")
        return
    
    all_hidden = []
    epoch_labels = []
    
    for i, epoch_file in enumerate(epochs):
        data = np.load(epoch_file)
        hidden = data['hidden_states']  # (steps, d_model)
        all_hidden.append(hidden)
        epoch_labels.extend([i] * len(hidden))
    
    all_hidden = np.vstack(all_hidden)
    
    # PCA to 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_hidden)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=epoch_labels,
        cmap='viridis',
        s=1,
        alpha=0.6
    )
    plt.colorbar(scatter, label='Epoch')
    plt.title('Text Representation Space Evolution')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved embedding plot: {output_path}")

def analyze_text_attention_patterns(log_dir: Path, output_path: Path):
    """Analyze attention patterns in text generation"""
    epochs = sorted(log_dir.glob('epoch_*.npz'))
    
    if not epochs:
        print("No epoch files found!")
        return
    
    # Analyze attention patterns across epochs
    attention_entropies = []
    attention_sparsity = []
    
    for epoch_file in epochs:
        data = np.load(epoch_file)
        attention = data['attention']  # (steps, layers, heads, T, T)
        
        # Take last step of epoch
        last_attention = attention[-1, 0, 0]  # First layer, first head
        
        # Calculate entropy (uniformity measure)
        entropy = -(last_attention * np.log(last_attention + 1e-10)).sum(axis=-1).mean()
        attention_entropies.append(entropy)
        
        # Calculate sparsity (concentration measure)
        sparsity = last_attention.max(axis=-1).mean()
        attention_sparsity.append(sparsity)
    
    # Plot attention analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Attention entropy over epochs
    axes[0].plot(attention_entropies, marker='o', markersize=4)
    axes[0].set_title('Attention Entropy Evolution')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Entropy')
    axes[0].grid(True, alpha=0.3)
    
    # Attention sparsity over epochs
    axes[1].plot(attention_sparsity, marker='o', markersize=4, color='orange')
    axes[1].set_title('Attention Sparsity Evolution')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Max Attention Weight')
    axes[1].grid(True, alpha=0.3)
    
    # Show attention matrices for key epochs
    key_epochs = [0, len(epochs)//2, len(epochs)-1]
    for i, ep_idx in enumerate(key_epochs):
        if ep_idx < len(epochs):
            data = np.load(epochs[ep_idx])
            attn = data['attention'][-1, 0, 0]  # Last step, first layer, first head
            
            im = axes[2].imshow(attn, cmap='viridis', aspect='auto')
            axes[2].set_title(f'Attention Matrix (Epoch {ep_idx})')
            axes[2].set_xlabel('Past Token')
            axes[2].set_ylabel('Current Token')
            plt.colorbar(im, ax=axes[2])
            break
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved attention analysis: {output_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python text_visualize.py <log_dir> <output_dir>")
        sys.exit(1)
    
    log_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(exist_ok=True)
    
    # Generate all visualizations
    create_text_attention_video(log_dir, output_dir / 'text_attention.mp4')
    plot_text_learning_curves(log_dir, output_dir / 'text_learning_curves.png')
    plot_text_embedding_evolution(log_dir, output_dir / 'text_embeddings.png')
    analyze_text_attention_patterns(log_dir, output_dir / 'text_attention_analysis.png')
