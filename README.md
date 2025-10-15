BetaBae
=====

Overview
--------
BetaBae is a compact research codebase for experimenting with lightweight agent architectures that combine sequence modeling (a small transformer-style module) with simple actor-critic style learning. The codebase includes two parallel modalities: "micro" (compact agents and micro-experiments) and "text" components (text-oriented models and visualizations). The repository contains training scripts, logging utilities, small visualization helpers, and analysis notebooks for result exploration.

Core idea / ideation
--------------------
- MinimalAgent: a minimalist sequence-aware agent built from a tiny transformer (single-head multihead attention + linear embeddings). The agent ingests history vectors composed of concatenated observations and action one-hot vectors, and predicts next-observations and action logits.
- Multi-objective loss: the training loop mixes a prediction objective (robust Huber-style loss) with actor-critic style policy/value learning and a small entropy bonus to encourage exploration. The design blends model-based prediction and policy learning into a unified loss to shape representations.
- Modality split: The codebase separates concerns between micro-scale experiments (quick RL-style environments like CartPole) and text-oriented work (with separate core/train/visualize modules).

Repository layout (key files)
-----------------------------
- `betabae/`
  - `__init__.py` — package initializer.
  - `core.py` / `micro_core.py` / `text_core.py` — core model and agent implementations.
  - `train.py`, `micro_train.py`, `text_train.py` — training scripts and entrypoints.
  - `logger.py`, `text_logger.py` — logging utilities; `visualize/` contains visualization helpers (`attention.py`, `embeddings.py`).
- `logs/` — folder for episodic logs and run outputs (e.g. `logs/run_001/episode_XXXXX.npz`).
- `outputs/`, `betabae_vibe_res`, `micro_million/` — experiment outputs and result directories.
- Notebooks: `explore_results.ipynb`, `micro_analysis.ipynb`, `text_analysis.ipynb` for interactive analysis.

Quick usage
-----------
(These are examples; details depend on your environment and installed packages.)

To run the default micro-training (CartPole):

```bash
python betabae/train.py --env CartPole-v1 --episodes 100
```

or directly:

```bash
python betabae/micro_train.py
```

The training scripts create a logger (for example `EvolutionLogger('logs/run_001')`) and write episodic `.npz` files into a run directory.

Key implementation notes discovered during review
------------------------------------------------
- MinimalAgent (from `betabae/core.py`) is implemented using a `SimpleTransformer` with `nn.MultiheadAttention` and linear heads for prediction and action logits. The architecture is intentionally small (d_model default 32, single attention head).
- The loss function mixes Huber-style prediction loss with an Actor-Critic style policy/value loss and a small entropy bonus.
- Training scripts support both Gymnasium and older Gym APIs by detecting return shapes from `env.reset()` and `env.step()`.

Known issues / immediate TODOs
----------------------------
- A runtime error was observed in a recent micro-training run (from `micro_training.log`):
  - TypeError: comparison between a `Value` and a `float` in `micro_core.py` during gradient clipping or similar (line references present in log). This indicates some code is attempting `if p.grad > max_grad_norm` rather than comparing a numeric value (should use p.grad.norm() or similar and guard against None).
- There's no top-level `README.md` prior to this addition (now created) and I didn't find an explicit `requirements.txt` or `pyproject.toml` listed at the root. Ensure environment dependencies are documented.

Suggestions & next steps
------------------------
1. Reproducibility
   - Add a `requirements.txt` or `pyproject.toml` listing the key dependencies (torch, gymnasium/gym, numpy, etc.).
   - Ensure `.gitignore` excludes large experiment outputs and virtual envs (`.venv/`, `outputs/`, `logs/`).

2. Fixes & robustness
   - Inspect and fix the gradient comparison bug in `micro_core.py` (use safe checks for `p.grad` and use `.norm()` for gradient magnitude checks).
   - Add small unit tests that import core components and run a single forward/backward step on a tiny random batch.

3. Documentation & onboarding
   - Provide a quickstart in the README with a reproducible command and expected outputs.
   - Document the format of saved `.npz` episode files and logger expectations.

4. Project hygiene
   - Add minimal CI (GitHub Actions) that installs dependencies and runs the smoke tests.
   - Consider adding explicit model checkpointing and metadata (git sha, hyperparameters) to saved runs.

Contact / ownership
-------------------
If you'd like, I can now:
- Fix the TypeError observed in `micro_core.py` and add a small unit test around the fix.
- Create `requirements.txt` and a minimal `setup.cfg`/`pyproject.toml`.
- Expand the README with API-level docs (function/class descriptions) after deeper file-by-file review.

Completion
----------
I created this `README.md` at the project root to capture ideation, layout, usage, and immediate next steps. I can iterate on it with more detail if you want me to open specific files and extract docstrings and example commands.