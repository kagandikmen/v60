# v60

**v60** is a multi-stage, analytical, GPU-accelerated macro placement algorithm. Given a netlist of hard and soft macros and a fixed-size canvas, it produces 2D positions for every macro that minimize the proxy cost `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion` — a weighted sum of the three quality terms most commonly used in placement: total wire routing length, packing uniformity, and routing congestion. The weights are the ones fixed by the challenge.

Macro placement sits at the very front of a digital VLSI design flow: before standard cells can be placed and routed, the large macros (memories, custom blocks) have to be positioned on the die. Their placement disproportionately drives the quality of everything downstream, and the search space is large, near-discrete (hard macros must not overlap), and full of local minima — so the problem has long been a target for both classical analytical placers and more recent learning-based approaches.

v60 was originally built for the [Partcl/HRT Macro Placement Challenge 2026](https://github.com/partcleda/macro-place-challenge-2026) and submitted as an entry there. Development has continued beyond the submission deadline: this version of the codebase adds a real-proxy coordinate-descent refinement stage (`IncrementalEval` + CD polish) and a relaxed legal-pick criterion that lets downstream stages clean up small residual overlap rather than discarding otherwise-good seeds.

The code lives in [`submissions/v60/`](submissions/v60/) (the directory layout is inherited from the challenge repository). For the original challenge README, see [`README_upstream.md`](README_upstream.md).

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026

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

The entry point is `v60_placer.py`; it pulls in `v60_engine.py`, `v60_kernels.py`, and `v60_incremental_eval.py` from the same folder. Do not run those modules directly.

## How It Works

The macro placement problem is to assign 2D positions to a set of hard and soft macros on a bounded canvas such that a weighted sum of wirelength, placement density, and routing congestion is minimized. v60 approaches this as a continuous optimization problem: all three terms are made differentiable and gradient descent is run directly on macro coordinates.

**Starting point: spectral clustering.** Before any gradient step, v60 partitions the hard macros into spatial clusters using the Fiedler embedding of the netlist graph — the 2D projection onto the eigenvectors corresponding to the two smallest non-zero eigenvalues of the graph Laplacian. This embedding has a useful property: macros that are heavily interconnected end up close together in the embedding regardless of their initial positions, so K-means on this space naturally recovers groups that belong together in the floorplan. Each cluster is then collapsed into a super-macro whose size is the bounding box of its members, giving the optimizer a coarse but topologically meaningful starting skeleton.

**The three-stage gradient engine.** From those cluster positions, the engine runs three successive gradient descent phases. *Stage 0* is a short global pass that places the super-macros; it establishes a rough floorplan before individual macros are freed. *Stage 1* then expands the super-macros back into their constituent hard macros and optimizes freely — this is the main exploration phase, where the optimizer is given enough room to move macros far from their cluster centers if the wirelength objective calls for it. *Stage 2* finishes by activating the density, overlap, and congestion penalties at full strength, driving hard-macro overlap towards zero while balancing the other cost terms.

**Why many restarts?** The optimization landscape is highly non-convex — the congestion term alone introduces many local minima, and different initializations can settle into very different final arrangements. Rather than betting on a single trajectory, v60 runs 32 independent restarts in parallel on the GPU, each from a different random initialization, and scores every result against the real `compute_proxy_cost`. The lowest-proxy result whose hard-macro overlap falls within a small tolerance (a fixed fraction of the median hard-macro area) is carried forward; the downstream stages are trusted to clean up any small residual overlap rather than discarding an otherwise good seed for it.

**Escaping local minima: basin-hopping.** After the three-stage gradient engine, v60 applies a basin-hopping wrapper around the best placement found so far. A Gaussian perturbation is applied to macro positions, Stage 2 is re-run from the perturbed state, and the result is accepted if it improves the real proxy cost. A tabu list tracks visited basins (identified by both spatial displacement and proxy cost similarity) to avoid re-exploring the same region, and a priority queue biases perturbation towards seeds that have previously yielded improvements. The hop loop terminates early if no improvement is found for several consecutive attempts.

**Soft-only polish.** Next, v60 freezes the hard macros and runs another batch of gradient descent restarts on the soft macros only. Hard macros are already legal and well-placed at this point; freeing them risks breaking that arrangement for marginal gain. With them fixed, the soft macros can converge tightly around the hard macro skeleton, recovering whatever wirelength, density, and congestion headroom the earlier stages left behind.

**Final refinement: coordinate descent on the real proxy.** The gradient-based stages optimize a differentiable *surrogate* of the proxy cost — sufficiently faithful to drive the placement to a good neighborhood, but not bit-identical to the official scorer. To close that gap, v60 finishes with a single-macro coordinate descent pass directly on the real `compute_proxy_cost`. For each movable macro, eight directional offsets at several step sizes are tried; the top few candidates by a cheap wirelength + density delta are then fully evaluated against the real proxy and the best improving move is committed. The reason this is tractable at all is an incremental scorer — `IncrementalEval` — that maintains per-net wirelength bounding boxes, per-cell density, and per-cell routing-congestion grids, updating only the cells touched by each candidate move. It is validated bit-perfect against the official PLC scorer to within float32 noise (~1e-7). Hard macros participate too, with candidate moves gated by an axis-aligned bounding-box check against the other hard macros to preserve legality.

A few implementation details worth noting: the congestion term used during gradient descent is a differentiable port of the TILOS L-shape router; most geometric hyperparameters (step sizes, penalty weights, grid resolutions) auto-scale with canvas area based on Optuna sweeps across the benchmark suite; and a runtime guard progressively thins the more expensive work on the largest designs to stay within the one-hour-per-benchmark budget.

## Runtime & Reproducibility

- **CPU and GPU.** The algorithm runs on either — `device='auto'` picks CUDA when available and falls back to CPU otherwise. A GPU is strongly recommended, since the many parallel restarts are what make the wall-clock budget comfortable.
- **Non-deterministic by default.** This is intentional: the default mode keeps TF32 and the fast (non-deterministic) CUDA kernels enabled, which is the recommended setting for evaluation.
- **Determinism is opt-in.** Setting `deterministic=True` on `v60_Placer` in `v60_placer.py` pins every RNG and disables non-deterministic kernels, giving bit-identical placements across runs — but it costs roughly **1.3–2× runtime**.
- **Other tradeoffs.** `torch.compile` is used automatically when the environment supports it and falls back to eager execution otherwise; Stage 2 uses bf16 autocast on GPU for speed; and the runtime guard trades a little solution quality on the largest designs to stay safely under the 1-hour cap.

## License

Licensed under the Apache License 2.0 — see [`LICENSE.md`](LICENSE.md) for details.
