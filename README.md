# v60

**v60** is a multi-stage, analytical, GPU-accelerated macro placement algorithm. Given a netlist of hard and soft macros and a fixed-size canvas, it produces 2D positions for every macro that minimize the proxy cost `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion` — a weighted sum of the three quality terms most commonly used in placement: total wire routing length, packing uniformity, and routing congestion.

Macro placement sits at the very front of a digital VLSI design flow: before standard cells can be placed and routed, the large macros (memories, custom blocks) have to be positioned on the die. Their placement disproportionately drives the quality of everything downstream, and the search space is large, near-discrete (hard macros must not overlap), and full of local minima — so the problem has long been a target for both classical analytical placers and more recent learning-based approaches.

v60 was originally built for the [Partcl/HRT Macro Placement Challenge 2026](https://github.com/partcleda/macro-place-challenge-2026). v60 lives in [`submissions/v60/`](submissions/v60/), whereas the rest of the repository simply mirrors the challenge repository.

## Results*

Across the 17 ICCAD04 IBM benchmarks (the 18-design suite minus ibm05, which contains no macros and is excluded by the challenge), v60 reaches an average proxy cost of 0.8946 with no macro overlaps on any design.

| Benchmark | v60 proxy | SA | RePlAce | vs SA | vs RePlAce | Overlaps | Runtime |
|-----------|----------:|-------:|--------:|------:|-----------:|:--------:|--------:|
| ibm01 | 0.7354 | 1.3166 | 0.9976 | +44.1% | +26.3% | 0 | 550 s |
| ibm02 | 0.8855 | 1.9072 | 1.8370 | +53.6% | +51.8% | 0 | 1340 s |
| ibm03 | 0.8178 | 1.7401 | 1.3222 | +53.0% | +38.1% | 0 | 1920 s |
| ibm04 | 0.8284 | 1.5037 | 1.3024 | +44.9% | +36.4% | 0 | 1660 s |
| ibm06 | 0.9523 | 2.5057 | 1.6187 | +62.0% | +41.2% | 0 | 1620 s |
| ibm07 | 0.8822 | 2.0229 | 1.4633 | +56.4% | +39.7% | 0 | 2050 s |
| ibm08 | 0.9466 | 1.9239 | 1.4285 | +50.8% | +33.7% | 0 | 2190 s |
| ibm09 | 0.7054 | 1.3875 | 1.1194 | +49.2% | +37.0% | 0 | 2160 s |
| ibm10 | 0.8190 | 2.1108 | 1.5009 | +61.2% | +45.4% | 0 | 4810 s |
| ibm11 | 0.7691 | 1.7111 | 1.1774 | +55.1% | +34.7% | 0 | 3170 s |
| ibm12 | 0.9534 | 2.8261 | 1.7261 | +66.3% | +44.8% | 0 | 4790 s |
| ibm13 | 0.8129 | 1.9141 | 1.3355 | +57.5% | +39.1% | 0 | 2460 s |
| ibm14 | 1.0506 | 2.2750 | 1.5436 | +53.8% | +31.9% | 0 | 4570 s |
| ibm15 | 0.9536 | 2.3000 | 1.5159 | +58.5% | +37.1% | 0 | 3600 s |
| ibm16 | 0.9347 | 2.2337 | 1.4780 | +58.2% | +36.8% | 0 | 5320 s |
| ibm17 | 1.1249 | 3.6726 | 1.6446 | +69.4% | +31.6% | 0 | 5300 s |
| ibm18 | 1.0358 | 2.7755 | 1.7722 | +62.7% | +41.6% | 0 | 3460 s |
| **Average** | **0.8946** | 2.1251 | 1.4578 | **+57.9%** | **+38.6%** | **0** | **3000 s** |

*\*Run with `deterministic=True` on an NVIDIA RTX 6000 Ada (48 GB) paired with an AMD EPYC 75F3.*

Set `deterministic=True` on `v60_Placer` in [`submissions/v60/v60_placer.py`](submissions/v60/v60_placer.py) to closely reproduce this table. The default (non-deterministic) mode is faster and lands within the same run-to-run noise margin.

The SA and RePlAce baselines are the published results reported in [*An Updated Assessment of Reinforcement Learning for Macro Placement*](https://doi.org/10.1109/TCAD.2025.3644293).

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

The macro placement problem is to assign 2D positions to a set of hard and soft macros on a bounded canvas such that the downstream PPA outcome is as good as possible. In our case, these goals are represented as a weighted sum of wirelength, placement density, and routing congestion. In this simplification, the placement algorithm works through minimizing this value called *proxy cost* or simply *proxy*, which is defined as `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion`.

v60 works in two phases. The first is *global placement* performed on the GPU, driven by gradient descent. (Theoretically it can be performed on the CPU as well, but orders of magnitude slower. See [Runtime & Reproducibility](#runtime--reproducibility).) Gradient descent needs a differentiable objective, but the official cost is not one: it is computed by ranking the busiest grid cells and routing each net onto a discrete grid — operations that carry no useful gradient. So this phase instead optimizes a smooth, differentiable surrogate of the three cost terms, running gradient descent directly on macro coordinates to reach a strong overall arrangement. The second phase is *refinement*: local-search passes that work on the exact cost. Local search only ever compares finished placements, so it needs no gradients and can call the official scorer directly — closing the gap the surrogate leaves behind.

### Global placement

**Starting point: spectral clustering.** Before any gradient step, v60 partitions the hard macros into spatial clusters using the Fiedler embedding of the netlist graph — the 2D projection onto the eigenvectors corresponding to the two smallest non-zero eigenvalues of the graph Laplacian. This embedding has a useful property: macros that are heavily interconnected end up close together in the embedding regardless of their initial positions, so K-means on this space naturally recovers groups that belong together in the floorplan. Each cluster is then collapsed into a super-macro whose size is the bounding box of its members, giving the optimizer a coarse but topologically meaningful starting skeleton.

**The three-stage gradient engine.** From those cluster positions, the engine runs three successive gradient descent phases. *Stage 0* is a short global pass that places the super-macros; it establishes a rough floorplan before individual macros are freed. *Stage 1* then expands the super-macros back into their constituent hard macros and optimizes freely — this is the main exploration phase, where the optimizer is given enough room to move macros far from their cluster centers if the wirelength objective calls for it. *Stage 2* finishes by activating the density, overlap, and congestion penalties at full strength, driving hard-macro overlap towards zero while balancing the other cost terms.

**Why many restarts?** The optimization landscape is highly non-convex — the congestion term alone introduces many local minima, and different initializations can settle into very different final arrangements. This is why v60 runs 64 independent restarts in parallel rather than betting on a single trajectory. Each restart starts from a different random initialization, and every restart is scored on the exact proxy cost at the end of Stage 2. The lowest-proxy result whose hard-macro overlap falls within a small tolerance (a fixed fraction of the median hard-macro area) is carried forward; the downstream stages are trusted to clean up any small residual overlap rather than discarding an otherwise good seed for it. (This means that there is a minuscule but indeed present chance you end up with an invalid placement at the very end; but I have not once seen this happen with the current default values. If absolutely necessary, the tolerance threshold can still be adjusted.)

**Escaping local minima: basin-hopping.** After the three-stage gradient engine, v60 applies a basin-hopping wrapper around the best placement found so far. A Gaussian perturbation is applied to macro positions, Stage 2 is re-run from the perturbed state, and the result is accepted if it improves the real proxy cost. A tabu list tracks visited basins (identified by both spatial displacement and proxy cost similarity) to avoid re-exploring the same region, and a priority queue biases perturbation towards seeds that have previously yielded improvements. The hop loop terminates early if no improvement is found for several consecutive attempts.

**Soft-only polish.** Next, v60 freezes the hard macros and runs another batch of gradient descent restarts on the soft macros only. Hard macros are already legal and well-placed at this point; freeing them risks breaking that arrangement for marginal gain. With them fixed, the soft macros can converge tightly around the hard macro skeleton, recovering whatever wirelength, density, and congestion headroom the earlier stages left behind.

### Refinement

**Coordinate descent.** The first refinement pass nudges one macro at a time. For each movable macro, v60 generates candidate offsets along sixteen evenly-spaced directions across a wide range of step sizes, evaluates each one against the cost, and commits the single best improving move; it sweeps over all macros repeatedly until the gains taper off. Hard macros participate too, with each candidate move checked against the other hard macros so the zero-overlap guarantee is never broken.

Scoring a single candidate move on the full official cost — re-routing every net and re-scoring the whole grid — would be far too slow for the thousands of moves coordinate descent and the following refinement stages try. `IncrementalEval` makes it practical: it returns exactly the official cost, but after a trial move it recomputes only the parts that actually change — the few nets whose wirelength shifts, and the grid cells whose density or congestion moves. This drastically reduces the amount of proxy computation needed during refinement.

**Pair swaps.** Single-macro moves cannot reach one particular configuration: two macros each sitting where the other would do better, so that no individual move improves the cost but exchanging the two does. v60 therefore finishes with two swap passes — one over hard macros, one over soft. For each macro it considers exchanging positions with the partners it shares the most connections with (and, for hard macros, its nearest spatial neighbours), and commits any swap that improves the placement. Hard-macro swaps are held to the same overlap check as the one in coordinate descent.

**Basin-hopping on the exact cost.** Coordinate descent and the swap passes converge to a placement that no single move or pairwise exchange can improve. To escape that local minimum, v60 applies basin-hopping once more, now on the exact cost: each hop adds Gaussian noise to a small set of soft macros that are the endpoints of nets crossing the worst 5% of routing cells. The algorithm then re-runs coordinate descent over the entire placement, and keeps the result only if it scores better. Every hop tries a ladder of eight noise scales (sigma values) in parallel and keeps the best outcome, letting the perturbation strength adapt to the design and to where the descent stands. Hard macros are never perturbed, though the re-descent may still move them legally.

## Runtime & Reproducibility

- The algorithm runs on either — `device='auto'` picks CUDA when available and falls back to CPU otherwise. A GPU is strongly recommended, since the many parallel restarts are what make the wall-clock budget comfortable.
- The algorithm is nondeterministic by default. Setting `deterministic=True` in `v60_placer.py` pins every RNG and switches to deterministic kernels wherever PyTorch provides them. The end-to-end results should then closely replicate those in the [Results](#results) table, but runtime would be higher.
- `torch.compile` is used automatically when the environment supports it and falls back to eager execution otherwise.
- Stage 2 uses bf16 autocast on GPU for speed. No meaningful proxy deterioration was observed.
- The refinement passes parallelize their candidate evaluations across CPU cores, so more CPU cores shorten the refinement phase.

## License

Licensed under the Apache License 2.0 — see [`LICENSE.md`](LICENSE.md) for details.
