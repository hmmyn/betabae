import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import imageio

def create_attention_video(log_dir: Path, output_path: Path):
    episodes = sorted(log_dir.glob('episode_*.npz'))
    fig, ax = plt.subplots(figsize=(6, 6))
    frames = []

    for ep_file in episodes:
        data = np.load(ep_file)
        attn = data['attention']  # (steps, 1, T, T)
        attn_matrix = attn[-1, 0]  # Last step, first head

        ax.clear()
        im = ax.imshow(attn_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Episode {ep_file.stem.split("_")[1]}')
        ax.set_xlabel('Past Token')
        ax.set_ylabel('Current Token')

        fig.canvas.draw()
        # Convert to RGB array
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]  # Remove alpha channel
        frames.append(frame)

    imageio.mimsave(output_path, frames, fps=5)
    print(f"Saved video: {output_path}")

if __name__ == '__main__':
    import sys
    log_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    create_attention_video(log_dir, output_path)
