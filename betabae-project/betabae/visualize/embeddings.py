import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_embedding_evolution(log_dir: Path, output_path: Path):
    episodes = sorted(log_dir.glob('episode_*.npz'))
    all_hidden = []
    episode_labels = []

    for i, ep_file in enumerate(episodes):
        data = np.load(ep_file)
        hidden = data['hidden']  # (steps, 1, d_model)
        hidden = np.squeeze(hidden)  # (steps, d_model)
        all_hidden.append(hidden)
        episode_labels.extend([i] * len(hidden))

    all_hidden = np.vstack(all_hidden)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_hidden)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=episode_labels,
        cmap='viridis',
        s=2,
        alpha=0.5
    )
    plt.colorbar(scatter, label='Episode')
    plt.title('Representation Space Evolution')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(output_path, dpi=100)
    print(f"Saved plot: {output_path}")

if __name__ == '__main__':
    import sys
    log_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    plot_embedding_evolution(log_dir, output_path)
