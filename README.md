# v60

My submission for the **Partcl/HRT Macro Placement Challenge 2026** — an analytical, GPU-accelerated macro placer that positions hard and soft macros on a chip canvas to minimize the proxy cost `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion`.

The submission lives in [`submissions/v60/`](submissions/v60/). For the original challenge documentation, see [`README_upstream.md`](README_upstream.md).

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge

# Initialize TILOS MacroPlacement submodule (required for evaluation)
git submodule update --init external/MacroPlacement

# Install the package and all dependencies
uv sync
```

### Run the submission

```bash
# Single benchmark
uv run evaluate submissions/v60/v60_placer.py -b ibm01

# All 17 IBM benchmarks with a comparison table
uv run evaluate submissions/v60/v60_placer.py --all

# Visualize the result
uv run evaluate submissions/v60/v60_placer.py -b ibm01 --vis
```

The entry point is `v60_placer.py`; it pulls in `v60_engine.py` and `v60_kernels.py` from the same folder. Do not run the engine or kernels files directly.

## How It Works

v60 is a **multi-level analytical placer** — clustering + batched gradient descent + refinement. It places macros by gradient descent on a batch of differentiable loss terms (wirelength, density, overlap, congestion), running many restarts in parallel on the GPU and keeping the best overlap-free result.

The pipeline has three layers:

1. **Engine** (`v60_engine.py`) — a spectral-clustering + multi-stage gradient placer. It clusters hard macros with K-means on the 2D Fiedler embedding of the netlist (the two strongest connectivity axes). A short *Stage 0* places the cluster super-macros; *Stage 1* explores freely from those cluster centers; *Stage 2* spreads macros out under density, overlap, and congestion penalties. Each restart is legalized to zero hard-macro overlap.

2. **Basin-hopping** — perturbs the best placement and re-runs *Stage 2*, guided by a promising-seed priority queue and a visited-basin tabu list, to escape local minima.

3. **Soft-only polish** — freezes the hard macros and re-optimizes just the soft macros for a final wirelength/density/congestion gain.

Key details: the congestion term is a faithful differentiable port of the TILOS L-shape router; many hyperparameters auto-scale with canvas size (tuned via Optuna sweeps); seeds are always ranked by the real `compute_proxy_cost`, never the proxy loss; and a runtime guard thins expensive work on large designs to stay within the 1-hour-per-benchmark budget.

## Runtime & Reproducibility

- **CPU and GPU.** The algorithm runs on either — `device='auto'` picks CUDA when available and falls back to CPU otherwise. A GPU is strongly recommended, since the many parallel restarts are what make the wall-clock budget comfortable.
- **Non-deterministic by default.** This is intentional: the default mode keeps TF32 and the fast (non-deterministic) CUDA kernels enabled, which is the recommended setting for evaluation.
- **Determinism is opt-in.** Setting `deterministic=True` on `v60_Placer` in `v60_placer.py` pins every RNG and disables non-deterministic kernels, giving bit-identical placements across runs — but it costs roughly **1.3–2× runtime**.
- **Other tradeoffs.** `torch.compile` is used automatically when the environment supports it and falls back to eager execution otherwise; Stage 2 uses bf16 autocast on GPU for speed; and the runtime guard trades a little solution quality on the largest designs to stay safely under the 1-hour cap.

## License

Licensed under the Apache License 2.0 — see [`LICENSE.md`](LICENSE.md) for details.
