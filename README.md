# v60

**v60** is a multi-stage, analytical, GPU-accelerated macro placement algorithm. Given a netlist of hard and soft macros and a fixed-size canvas, it produces 2D positions for every macro with the main goal of achieving the best possible final PPA outcome. v60 does this under a reasonable runtime budget by optimizing rather for the simplified formula `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion`. This formula, named *proxy cost* or simply *proxy*, is a weighted sum of the three quality terms that directly affect the final PPA outcome: total wirelength, packing uniformity, and routing congestion. See [How It Works](#how-it-works) for more detail about v60's inner workings.

Macro placement sits at the very front of a digital VLSI design flow: before standard cells can be placed and routed, the large macros (memories, custom blocks) have to be positioned on the die. Their placement disproportionately drives the quality of everything downstream, and the search space is large, near-discrete (hard macros must not overlap), and full of local minima. As a result, this problem has long been a target of VLSI research, which in the process came up with solutions varying from rather classical approaches like simulated annealing to modern learning-based approaches like reinforcement learning. v60 is another attempt at this long-studied optimization problem that involves a multi-stage, analytical, mixed approach that heavily makes use of GPU computing.

v60 can be configured to run in one of two available modes: full mode and flash mode. In full mode, the entire pipeline is run to achieve the lowest possible proxy cost. In flash mode, several of the more expensive stages are trimmed or dropped: the refinement basin-hop and both pair-swap passes are skipped, the gradient engine runs fewer restarts with shorter descents, and coordinate descent stops earlier. This sacrifices a little proxy for a remarkably shorter runtime. Full mode is the default mode; to switch to flash mode see [Runtime & Reproducibility](#runtime--reproducibility).

v60 was originally built for the [Partcl/HRT Macro Placement Challenge 2026](https://github.com/partcleda/macro-place-challenge-2026). v60 lives in [`submissions/v60/`](submissions/v60/), whereas the rest of the repository simply mirrors the challenge repository.

## Results

Across the 17 ICCAD04 IBM benchmarks (the 18-design suite minus ibm05, which contains no macros), v60 reaches an average proxy cost of 0.8946 in full mode and 0.9379 in flash mode, with no macro overlaps on any design.

| Benchmark | Full mode proxy | Flash mode proxy | SA | RePlAce | Full mode runtime | Flash mode runtime |
|-----------|-----------:|-----------:|-------:|--------:|-------------:|-------------:|
| ibm01 | 0.7354 | 0.7484 | 1.3166 | 0.9976 | 550 s | 240 s |
| ibm02 | 0.8855 | 0.9263 | 1.9072 | 1.8370 | 1340 s | 310 s |
| ibm03 | 0.8178 | 0.8812 | 1.7401 | 1.3222 | 1920 s | 330 s |
| ibm04 | 0.8284 | 0.8762 | 1.5037 | 1.3024 | 1660 s | 350 s |
| ibm06 | 0.9523 | 0.9891 | 2.5057 | 1.6187 | 1620 s | 330 s |
| ibm07 | 0.8822 | 0.9196 | 2.0229 | 1.4633 | 2050 s | 430 s |
| ibm08 | 0.9466 | 1.0070 | 1.9239 | 1.4285 | 2190 s | 500 s |
| ibm09 | 0.7054 | 0.7236 | 1.3875 | 1.1194 | 2160 s | 440 s |
| ibm10 | 0.8190 | 0.8654 | 2.1108 | 1.5009 | 4810 s | 1120 s |
| ibm11 | 0.7691 | 0.8086 | 1.7111 | 1.1774 | 3170 s | 650 s |
| ibm12 | 0.9534 | 0.9967 | 2.8261 | 1.7261 | 4790 s | 990 s |
| ibm13 | 0.8129 | 0.8683 | 1.9141 | 1.3355 | 2460 s | 580 s |
| ibm14 | 1.0506 | 1.0515 | 2.2750 | 1.5436 | 4570 s | 960 s |
| ibm15 | 0.9536 | 0.9964 | 2.3000 | 1.5159 | 3600 s | 710 s |
| ibm16 | 0.9347 | 0.9948 | 2.2337 | 1.4780 | 5320 s | 1040 s |
| ibm17 | 1.1249 | 1.1751 | 3.6726 | 1.6446 | 5300 s | 1240 s |
| ibm18 | 1.0358 | 1.1157 | 2.7755 | 1.7722 | 3460 s | 700 s |
| **Average** | **0.8946** | **0.9379** | 2.1251 | 1.4578 | **3000 s** | **640 s** |

*\*Run with `deterministic=True` on an NVIDIA RTX 6000 Ada (48 GB) paired with an AMD EPYC 75F3, using Python 3.11.10, PyTorch 2.10.0+cu128 (CUDA 12.8, cuDNN 9.10.02), and NVIDIA driver 550.127.05.*

Set `deterministic=True` in [`submissions/v60/v60_placer.py`](submissions/v60/v60_placer.py) to exactly reproduce this table, in case you are operating on the same HW/SW stack. The default (non-deterministic) setting is faster and lands within the same run-to-run noise margin. Running on different hardware should still land you within the same noise margin (at least in the final average) but be aware that runtime will most likely differ. Running without a GPU is not recommended as it drastically increases the runtime.

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

The entry point of the algorithm is `v60_placer.py`, which pulls in `v60_engine.py`, `v60_kernels.py`, and `v60_incremental_eval.py` from the same folder. Do not run those modules directly.

## How It Works

The macro placement problem is to assign 2D positions to a set of hard and soft macros on a bounded canvas such that the downstream PPA outcome is as good as possible. In our case, these goals are represented as a weighted sum of wirelength, placement density, and routing congestion. In this simplification, the placement algorithm works through minimizing this value called *proxy cost* or simply *proxy*, which is defined as `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion`. This simplification is most importantly done to keep the runtime under control. Sadly, optimizing using the real PPA values is still an utter impracticality as this would take orders of magnitude longer.

v60 works in two phases. The first is *global placement* performed on the GPU, driven by gradient descent. (Theoretically it can be performed on the CPU as well, but this is not recommended due to the exorbitant runtime involved in doing so. See [Runtime & Reproducibility](#runtime--reproducibility).) An issue arising with this premise is that gradient descent needs a differentiable objective, but the official cost is not one. So this phase instead optimizes a smooth, differentiable surrogate of the three cost terms, running gradient descent directly on macro coordinates to reach a strong overall arrangement. The second phase is *refinement*, which includes various local-search passes that work on the exact proxy cost. These local-search passes only ever compare finished placements, so they need no gradients and can call the official scorer directly — closing the gap the surrogate leaves behind.

### Global placement

**Starting point: spectral clustering.** Before any gradient step, v60 partitions the hard macros into spatial clusters using the Fiedler embedding of the netlist graph — the 2D projection onto the eigenvectors corresponding to the two smallest non-zero eigenvalues of the graph Laplacian. This embedding has a useful property: macros that are heavily interconnected end up close together in the embedding regardless of their initial positions, so K-means on this space naturally recovers groups that belong together in the floorplan. Each cluster is then collapsed into a super-macro whose size is the bounding box of its members, giving the optimizer a coarse but topologically meaningful starting skeleton.

**The three-stage gradient engine.** From those cluster positions, the engine runs three successive gradient descent phases. *Stage 0* is a short global pass that places the super-macros; it establishes a rough floorplan before individual macros are freed. *Stage 1* then expands the super-macros back into their constituent hard macros and optimizes freely. Stage 1 is the main exploration phase, because its loss term does not include any density term or overlap penalties, which gives the optimizer enough room to move macros far from their cluster centers if the wirelength objective calls for it. *Stage 2* finishes by activating the density, overlap, and congestion penalties at full strength, driving hard-macro overlap towards zero while balancing the other cost terms.

**Why many restarts?** The optimization landscape is highly non-convex. The congestion term alone introduces many local minima, and different initializations can settle into very different final arrangements. This is why v60 runs 64 independent restarts in parallel (16 in flash mode) rather than betting on a single trajectory. Each restart starts from a different random initialization, and every restart is scored on the exact proxy cost at the end of Stage 2. The lowest-proxy result whose hard-macro overlap falls within a small tolerance (a fixed fraction of the median hard-macro area) is carried forward, and the downstream stages are trusted to clean up any small residual overlap rather than discarding an otherwise good seed for it. (This means that there is a minuscule but indeed present chance you end up with an invalid placement at the very end; but I have not once seen this happen with the current default values. If absolutely necessary, the tolerance threshold can be adjusted.)

**Escaping local minima: basin-hopping.** After the three-stage gradient engine, v60 applies a basin-hopping wrapper around the best placement found so far. This goes as follows: A Gaussian perturbation is applied to macro positions, Stage 2 is re-run from the perturbed state, and the result is accepted if it improves the real proxy cost. A tabu list tracks visited basins (identified by both spatial displacement and proxy cost similarity) to avoid re-exploring the same region, and a priority queue biases perturbation towards seeds that have previously yielded improvements. The hop loop terminates early if no improvement is found for several consecutive attempts.

Basin-hop stops earlier in flash mode, since later hops deliver diminishing returns.

One important detail: Empirical evidence shows that the basin-hop is also a very effective legalizer. Therefore, basin-hop is repeated continuously if no overlap-free placement is yet achieved.

**Soft-only polish.** Next, v60 freezes the hard macros and runs another batch of gradient descent restarts on the soft macros only. Hard macros are already legal and well-placed at this point; freeing them risks breaking that arrangement for marginal gain. With them fixed, the soft macros can converge tightly around the hard macro skeleton, recovering whatever wirelength, density, and congestion headroom the earlier stages left behind.

Soft polish is run with fewer restarts in flash mode due to diminishing returns involved in increasing its number of restarts.

### Refinement

**Coordinate descent.** In this stage, the algorithm nudges one macro at a time. For each movable macro, v60 generates candidate offsets along sixteen evenly-spaced directions across a wide range of step sizes, evaluates each one against the cost, and commits the single best improving move. The algorithm sweeps over all macros repeatedly until the gains taper off. While displacing hard macros, each candidate move is checked against the other hard macros so the coordinate descent preserves the zero-overlap placement delivered to it by the global placement.

Scoring a single candidate move on the full official cost — re-routing every net and re-scoring the whole grid — would be far too slow for the thousands of moves coordinate descent and the following refinement stages try. `IncrementalEval` makes it practical: it returns exactly the official cost, but after a trial move it recomputes only the parts that actually change — the few nets whose wirelength shifts, and the grid cells whose density or congestion moves. This drastically reduces the amount of proxy computation needed during refinement.

Coordinate descent is stopped earlier in flash mode, since later sweeps deliver diminishing returns.

**Pair swaps.** Single-macro moves cannot reach one particular configuration: two macros each sitting where the other would do better, so that no individual move improves the cost but exchanging the two does. v60 therefore finishes with two swap passes — one over hard macros, one over soft. For each macro it considers exchanging positions with the partners it shares the most connections with (and, for hard macros, its nearest spatial neighbours), and commits any swap that improves the placement. Hard-macro swaps are held to the same overlap check as the one in coordinate descent.

Both swap passes are skipped in flash mode, since they move the proxy only marginally.

**Basin-hopping on the exact cost.** Coordinate descent and the swap passes converge to a placement that no single move or pairwise exchange can improve. To escape that local minimum, v60 applies basin-hopping once more, but now on the exact cost: each hop adds Gaussian noise to a small set of soft macros that are the endpoints of nets crossing the worst 5% of routing cells. This is due to the empirical observation that at this later stage there is not much left to improve other than congestion.

After the Gaussian perturbation, the algorithm re-runs coordinate descent over the entire placement, and keeps the result only if it scores better. Every hop tries a ladder of eight noise scales (sigma values) in parallel and keeps the best outcome, letting the perturbation strength adapt to the design and to where the descent stands. Hard macros are never perturbed, though the re-descent may still move them legally.

The final basin-hop is not run in flash mode due to its runtime-heavy nature.

## Full mode vs. flash mode

v60 can run in two modes:

- Full mode: runs the entire pipeline with the aim of achieving the lowest proxy cost. This is the default mode.
- Flash mode: only runs the highest-ROI stages of the pipeline. This includes dropping the refinement basin-hop and both pair-swap passes, running 16 restarts instead of 64 with shorter Stage 1 and Stage 2 descents, a single GPU basin-hop instead of two, a lighter soft-only polish, and an earlier coordinate-descent stop. Trades some proxy for a significantly better runtime.

| Stage | Full | Flash |
|-------|:----:|:----:|
| Spectral clustering + three-stage engine | yes (64 restarts) | yes (16 restarts, shorter Stage 1/2) |
| GPU basin-hopping | yes (up to 2 hops) | yes (1 hop) |
| Soft-only polish | yes (16 restarts) | yes (8 restarts) |
| Coordinate descent | yes | yes (earlier stop) |
| Pair swaps (hard + soft) | yes | — |
| Refinement basin-hopping | yes | — |

## Runtime & Reproducibility

- The algorithm can run without a GPU: `device='auto'` picks CUDA when available and falls back to CPU otherwise. A GPU is strongly recommended though, since the many parallel restarts are what make the wall-clock budget comfortable.
- The algorithm is nondeterministic by default. Setting `deterministic=True` in [`v60_placer.py`](submissions/v60/v60_placer.py) pins every RNG and switches to deterministic kernels wherever PyTorch provides them. The end-to-end results should then exactly replicate those in the [Results](#results) table, if run on the same hardware with the same tool versions. Keep in mind that determinism costs some runtime.
- v60 runs in full mode by default. To switch to flash mode, set `mode='flash'` in [`v60_placer.py`](submissions/v60/v60_placer.py). Refer to [Full mode vs. flash mode](#full-mode-vs-flash-mode) for what each mode runs.
- `torch.compile` is used automatically when the environment supports it and falls back to eager execution otherwise.
- Stage 2 uses bf16 autocast on GPU for speed. No proxy deterioration was observed.
- The refinement passes parallelize their candidate evaluations across CPU cores, so more CPU cores shorten the refinement phase.

## License

Licensed under the Apache License 2.0 — see [`LICENSE.md`](LICENSE.md) for details.
