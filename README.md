# v60

**v60** is a multi-stage, analytical, GPU-accelerated macro placement algorithm. Given a netlist of hard and soft macros and a fixed-size canvas, it produces 2D positions for every macro that minimize the proxy cost `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion` — a weighted sum of the three quality terms most commonly used in placement: total wire routing length, packing uniformity, and routing congestion. The weights are the ones fixed by the challenge.

Macro placement sits at the very front of a digital VLSI design flow: before standard cells can be placed and routed, the large macros (memories, custom blocks) have to be positioned on the die. Their placement disproportionately drives the quality of everything downstream, and the search space is large, near-discrete (hard macros must not overlap), and full of local minima — so the problem has long been a target for both classical analytical placers and more recent learning-based approaches.

v60 was originally built for the [Partcl/HRT Macro Placement Challenge 2026](https://github.com/partcleda/macro-place-challenge-2026), which scores entries on the proxy cost above across the 17 ICCAD04 IBM benchmarks. The code lives in [`submissions/v60/`](submissions/v60/).

## Results

Across the 17 ICCAD04 IBM benchmarks (the 18-design suite minus ibm05, which contains no macros and is excluded by the challenge), v60 reaches an **average proxy cost of 0.9196** — **36.9 % below the RePlAce baseline** (1.4578) and **56.7 % below Simulated Annealing** (2.1251) — with **zero macro overlaps on every design**. Lower proxy is better.

| Benchmark | v60 proxy | SA | RePlAce | vs SA | vs RePlAce | Overlaps |
|-----------|----------:|-------:|--------:|------:|-----------:|:--------:|
| ibm01 | 0.7493 | 1.3166 | 0.9976 | +43.1 % | +24.9 % | 0 |
| ibm02 | 0.9017 | 1.9072 | 1.8370 | +52.7 % | +50.9 % | 0 |
| ibm03 | 0.8559 | 1.7401 | 1.3222 | +50.8 % | +35.3 % | 0 |
| ibm04 | 0.8349 | 1.5037 | 1.3024 | +44.5 % | +35.9 % | 0 |
| ibm06 | 0.9691 | 2.5057 | 1.6187 | +61.3 % | +40.1 % | 0 |
| ibm07 | 0.9237 | 2.0229 | 1.4633 | +54.3 % | +36.9 % | 0 |
| ibm08 | 0.9633 | 1.9239 | 1.4285 | +49.9 % | +32.6 % | 0 |
| ibm09 | 0.7177 | 1.3875 | 1.1194 | +48.3 % | +35.9 % | 0 |
| ibm10 | 0.8600 | 2.1108 | 1.5009 | +59.3 % | +42.7 % | 0 |
| ibm11 | 0.8065 | 1.7111 | 1.1774 | +52.9 % | +31.5 % | 0 |
| ibm12 | 0.9724 | 2.8261 | 1.7261 | +65.6 % | +43.7 % | 0 |
| ibm13 | 0.8294 | 1.9141 | 1.3355 | +56.7 % | +37.9 % | 0 |
| ibm14 | 1.0169 | 2.2750 | 1.5436 | +55.3 % | +34.1 % | 0 |
| ibm15 | 0.9839 | 2.3000 | 1.5159 | +57.2 % | +35.1 % | 0 |
| ibm16 | 0.9943 | 2.2337 | 1.4780 | +55.5 % | +32.7 % | 0 |
| ibm17 | 1.1681 | 3.6726 | 1.6446 | +68.2 % | +29.0 % | 0 |
| ibm18 | 1.0859 | 2.7755 | 1.7722 | +60.9 % | +38.7 % | 0 |
| **Average** | **0.9196** | 2.1251 | 1.4578 | **+56.7 %** | **+36.9 %** | **0** |

**Set `deterministic=True` on `v60_Placer` in [`submissions/v60/v60_placer.py`](submissions/v60/v60_placer.py) and you will reproduce this table bit-for-bit** — these are the exact figures from such a run. The default (non-deterministic) mode is faster and lands within run-to-run noise (~1.4 %) of these numbers.

The SA and RePlAce baselines are the published results reported in [*An Updated Assessment of Reinforcement Learning for Macro Placement*](https://doi.org/10.1109/TCAD.2025.3644293) (IEEE), measured through the same TILOS MacroPlacement evaluator that scores v60.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kagandikmen/v60.git
cd v60

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

The macro placement problem is to assign 2D positions to a set of hard and soft macros on a bounded canvas such that a weighted sum of wirelength, placement density, and routing congestion is minimized. v60 works in two phases. The first is *global placement* on the GPU, driven by gradient descent. Gradient descent needs a differentiable objective, but the official cost is not one: it is computed by ranking the busiest grid cells and routing each net onto a discrete grid — operations that carry no useful gradient. So this phase instead optimizes a smooth, differentiable surrogate of the three cost terms, running gradient descent directly on macro coordinates to reach a strong overall arrangement. The second phase is *refinement*: local-search passes that work on the exact cost. Local search only ever compares finished placements, so it needs no gradients and can call the official scorer directly — closing the gap the surrogate leaves behind.

### Global placement

**Starting point: spectral clustering.** Before any gradient step, v60 partitions the hard macros into spatial clusters using the Fiedler embedding of the netlist graph — the 2D projection onto the eigenvectors corresponding to the two smallest non-zero eigenvalues of the graph Laplacian. This embedding has a useful property: macros that are heavily interconnected end up close together in the embedding regardless of their initial positions, so K-means on this space naturally recovers groups that belong together in the floorplan. Each cluster is then collapsed into a super-macro whose size is the bounding box of its members, giving the optimizer a coarse but topologically meaningful starting skeleton.

**The three-stage gradient engine.** From those cluster positions, the engine runs three successive gradient descent phases. *Stage 0* is a short global pass that places the super-macros; it establishes a rough floorplan before individual macros are freed. *Stage 1* then expands the super-macros back into their constituent hard macros and optimizes freely — this is the main exploration phase, where the optimizer is given enough room to move macros far from their cluster centers if the wirelength objective calls for it. *Stage 2* finishes by activating the density, overlap, and congestion penalties at full strength, driving hard-macro overlap towards zero while balancing the other cost terms.

**Why many restarts?** The optimization landscape is highly non-convex — the congestion term alone introduces many local minima, and different initializations can settle into very different final arrangements. Rather than betting on a single trajectory, v60 runs 32 independent restarts in parallel on the GPU, each from a different random initialization, and scores every result against the real `compute_proxy_cost`. The lowest-proxy result whose hard-macro overlap falls within a small tolerance (a fixed fraction of the median hard-macro area) is carried forward; the downstream stages are trusted to clean up any small residual overlap rather than discarding an otherwise good seed for it.

**Escaping local minima: basin-hopping.** After the three-stage gradient engine, v60 applies a basin-hopping wrapper around the best placement found so far. A Gaussian perturbation is applied to macro positions, Stage 2 is re-run from the perturbed state, and the result is accepted if it improves the real proxy cost. A tabu list tracks visited basins (identified by both spatial displacement and proxy cost similarity) to avoid re-exploring the same region, and a priority queue biases perturbation towards seeds that have previously yielded improvements. The hop loop terminates early if no improvement is found for several consecutive attempts.

**Soft-only polish.** Next, v60 freezes the hard macros and runs another batch of gradient descent restarts on the soft macros only. Hard macros are already legal and well-placed at this point; freeing them risks breaking that arrangement for marginal gain. With them fixed, the soft macros can converge tightly around the hard macro skeleton, recovering whatever wirelength, density, and congestion headroom the earlier stages left behind.

### Refinement

Scoring a single candidate move on the full official cost — re-routing every net and re-scoring the whole grid — would be far too slow for the thousands of moves these passes try. `IncrementalEval` makes it practical: it returns exactly the official cost, but after a trial move it recomputes only the parts that actually change — the few nets whose wirelength shifts, and the grid cells whose density or congestion moves — leaving the rest of the placement untouched.

**Coordinate descent.** The first refinement pass nudges one macro at a time. For each movable macro, v60 generates candidate offsets along sixteen evenly-spaced directions across a wide range of step sizes, evaluates each one against the cost, and commits the single best improving move; it sweeps over all macros repeatedly until the gains taper off. Hard macros participate too, with each candidate move checked against the other hard macros so the zero-overlap guarantee is never broken.

**Pair swaps.** Single-macro moves cannot reach one particular configuration: two macros each sitting where the other would do better, so that no individual move improves the cost but exchanging the two does. v60 therefore finishes with two swap passes — one over hard macros, one over soft. For each macro it considers exchanging positions with the partners it shares the most connections with (and, for hard macros, its nearest spatial neighbours), and commits any swap that improves the placement. Hard-macro swaps are held to the same overlap check as coordinate descent.

A few implementation details worth noting: the congestion term used during gradient descent is a differentiable port of the TILOS L-shape router; most geometric hyperparameters (step sizes, penalty weights, grid resolutions) auto-scale with canvas area based on Optuna sweeps across the benchmark suite; and a runtime guard progressively thins the more expensive work on the largest designs to stay within the one-hour-per-benchmark budget.

## Runtime & Reproducibility

- **CPU and GPU.** The algorithm runs on either — `device='auto'` picks CUDA when available and falls back to CPU otherwise. A GPU is strongly recommended, since the many parallel restarts are what make the wall-clock budget comfortable.
- **Non-deterministic by default.** This is intentional: the default mode keeps TF32 and the fast (non-deterministic) CUDA kernels enabled, which is the recommended setting for evaluation.
- **Determinism is opt-in.** Setting `deterministic=True` on `v60_Placer` in `v60_placer.py` pins every RNG and disables non-deterministic kernels, giving bit-identical placements across runs — this is the mode that reproduces the [Results](#results) table exactly. It costs roughly **1.3–2× runtime** over the default.
- **Other tradeoffs.** `torch.compile` is used automatically when the environment supports it and falls back to eager execution otherwise; Stage 2 uses bf16 autocast on GPU for speed; the refinement passes parallelize their candidate evaluations across CPU cores, so more cores shorten the refinement phase; and the runtime guard trades a little solution quality on the largest designs to stay safely under the 1-hour cap.

## License

Licensed under the Apache License 2.0 — see [`LICENSE.md`](LICENSE.md) for details.
