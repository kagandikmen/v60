"""
Analytical Placer v60 — orchestrator.

This is the file to run. v60_placer.py drives the placement engine
(v60_engine.py, which clusters hard macros via K-means on the 2D Fiedler
embedding and runs the three-stage gradient descent), then layers several
refinement stages on top of the engine's best seeds:

  - a basin-hopping wrapper (perturb-and-reminimize),
  - a soft-only Stage 2 polish (hard macros frozen),
  - real-proxy coordinate-descent (CD) polish, and
  - hard-hard and soft-soft pair-swap polish.

A multi-candidate path runs the top engine seeds through these stages and
keeps the best final placement.

The engine lives in v60_engine.py.

Usage:
    uv run evaluate submissions/v60/v60_placer.py -b ibm01
    uv run evaluate submissions/v60/v60_placer.py --all
"""

import math
import multiprocessing as mp
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch

# Enable TF32 ASAP — before any fp32 matmul has a chance to fire the
# "TensorFloat32 ... not enabled" warning. Also set in v60_kernels,
# but doing it here too guarantees the flag is on regardless of which
# entry point is hit first (orchestrator, cohort, or sweep subprocess).
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True

if osp.dirname(osp.abspath(__file__)) not in sys.path:
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from macro_place.benchmark import Benchmark
from macro_place._plc import PlacementCost
from macro_place.objective import compute_proxy_cost

from v60_engine import v60_Engine
from v60_kernels   import (
    _basin_hop, _congestion_work_tier,
    _extract_raw, _parse_plc_routing_params, _run_batch,
)
from v60_incremental_eval import (
    IncrementalEval, WEIGHT_WL, WEIGHT_DENSITY, WEIGHT_CONG,
)


def _set_deterministic(seed: int = 0) -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic     = True
    torch.backends.cudnn.benchmark         = False
    torch.backends.cuda.matmul.allow_tf32  = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _set_fast_nondeterministic() -> None:
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


class v60_Placer:
    """
    v60 orchestrator: runs the engine, then a basin-hop wrapper, a
    soft-only Stage 2 polish, real-proxy CD polish, and pair-swap polish.

    The engine clusters hard macros via K-means on the 2D Fiedler
    embedding (see v60_engine.py).
    """

    def __init__(
        self,
        # ── Restart count for the cohort ──────────────────────────────────
        num_restarts: int = 32,

        # ── Multi-level config ─────────────────────────────────────────────
        num_clusters             = 'auto',
        cluster_jitter_frac: float = 0.30,
        cluster_margin_frac: float = 0.50,
        cluster_seed:        int   = 0,

        # ── Stage 0 ────────────────────────────────────────────────────────
        stage0_steps:           int   = 1500,
        stage0_lr:              float = 1.0,
        stage0_lambda_density:  float = 200.0,
        stage0_lambda_overlap:  float = 500.0,
        stage0_gamma_start:     float = 2.0,
        stage0_gamma_end:       float = 0.3,
        stage0_target_density:  float = 0.7,

        # ── Stage 1/2 feature toggles ──────────────────────────────────────
        use_hh_overlap:            bool = True,
        use_ss_overlap:            bool = True,
        use_soft_degree_inflation: bool = False,

        # ── Common runtime ────────────────────────────────────────────────
        plc_root: str = "external/MacroPlacement/Testcases/ICCAD04",
        device: str   = 'auto',
        verbose: bool = True,
        deterministic: bool = False,   # production default: keep TF32 / fast kernels on
        seed: int           = 0,
        # ── Basin-hopping wrapper (v60). After the cohort run, perturb
        # the running-best placement and re-run Stage 2 from it, with a
        # promising-seed priority queue + visited-basin tabu list (see
        # v60_kernels._basin_hop). Each hop is a Stage-2-only batch, so the
        # added cost is roughly basin_hop runs of ~num_steps_s2 each. Set
        # basin_hop=False to skip it and return the plain cohort result.
        # Defaults follow the ibm01 v60_ibm01.log post-mortem (May 2026):
        #   - sigma_set retuned to the productive band (σ≥0.05 destroyed every
        #     placement); the cohort batch is stratified across this band so
        #     best-of-B picks the helpful sigma per seed.
        #   - tabu paired with a |Δproxy| gate so the 0.79xx cluster gets
        #     flagged as the same basin and we stop spawning around it.
        #   - improve_quota ends the hop loop after 3 non-improving hops to
        #     hand the freed budget to final_explore.
        #   - B per hop is small (8) so the freed wall-time goes to more hop
        #     attempts (higher max_hops / final_explore).
        basin_hop:                  bool  = True,
        basin_hop_max_hops:         int   = 24,
        basin_hop_final_explore:    int   = 0,
        basin_hop_sigma_set                = (0.015, 0.025, 0.035),  # productive band only
        basin_hop_tabu_eps:         float = 0.01,                    # spatial: mean macro disp / scale
        basin_hop_tabu_proxy_eps:   float = 0.005,                   # cost gate: |Δproxy| (0 to disable)
        basin_hop_improve_quota:    int   = 1,                       # stop after N non-improving (under-threshold) hops (0 = full budget)
        # v60: only reductions >= this fraction of the current incumbent count
        # as improvements for the quota / sigma-push logic. Tiny gains are
        # still accepted as the new incumbent but increment no_improve so the
        # loop escalates / ends on time. 0.0 = legacy behaviour.
        basin_hop_min_improve_frac: float = 0.005,                   # 0.5% of current proxy
        basin_hop_stratify:         bool  = False,                   # split B across sigma_set per hop
        basin_hop_restarts:         int   = 8,                       # small B per hop + more hops
        # Runtime guard for large/routability-heavy benchmarks. 'auto' keeps
        # ibm01-style behavior but caps expensive Stage-2 reruns on big netlists.
        congestion_runtime_mode: str = 'auto',
        # -- v60 soft-only polish -------------------------------------------------
        soft_polish_enabled: bool = True,
        soft_polish_restarts: int = 16,
        # The L-driven knobs default to 'auto' -> resolved per-benchmark from the
        # canvas size in _resolve_soft_polish (see the v60 Optuna sweep re-fit).
        # An explicit number/tuple still overrides 'auto' (used by the sweep).
        soft_polish_steps='auto',
        soft_polish_lr='auto',
        soft_polish_lambda_cong='auto',
        soft_polish_lambda_density='auto',
        soft_polish_lambda_hs: float = 8.0,             # fixed: no L-structure in the sweep
        soft_polish_lambda_ss='auto',
        soft_polish_jitter_fracs='auto',
        soft_polish_use_degree_inflation: bool = False,
        soft_polish_verbose: bool = True,
        # -- v60 CD polish (real-proxy local search after soft polish) ----------
        # Single-macro coordinate descent over soft macros, using
        # IncrementalEval to evaluate the real proxy cost per candidate move.
        # Cheap WL+density delta picks the best candidate per macro; the full
        # proxy is then re-evaluated (incremental cong cache) to confirm the
        # move actually improves the real objective before committing. Catches
        # the small surrogate-vs-real gaps soft polish leaves behind.
        cd_polish_enabled: bool = True,
        cd_polish_sweeps: int = 15,
        cd_polish_step_frac: float = 0.01,        # candidate offset = step_frac * 0.5 * (W+H)
        cd_polish_step_set: tuple = tuple(2.0**i for i in range(-3, 16)),  # multipliers of base step: 2^-3 .. 2^15
        cd_polish_num_directions: int = 16,       # evenly-spaced unit vectors per macro candidate scan
        cd_polish_min_improve: float = 1e-7,      # absolute proxy improvement to accept a move
        cd_polish_patience: int = 2,              # stop after this many consecutive zero-move sweeps
        # Early-stop a CD run once a full sweep reduces the proxy by less than
        # this fraction of the pre-sweep proxy. Complements patience: patience
        # catches "no moves at all", this catches "moves that barely help".
        cd_polish_min_sweep_improve_frac: float = 0.001,   # 0.1% relative
        cd_polish_include_hard: bool = True,      # also move hard macros (with overlap legality check)
        cd_polish_verbose: bool = True,
        # -- v60 pair-swap polish (hard-hard position swaps after CD polish) ---
        # Coordinate descent over PAIRS of hard macros: for each hard macro
        # m_i, consider swapping positions with each of its k-nearest hard
        # macros. The cheap filter is the sum of single-macro WL+density
        # deltas (approximate — misses interaction on shared nets); the top-K
        # candidates are then full-eval'd on the real proxy and the best
        # improving swap (if any) is committed. Catches the moves that
        # single-macro CD misses: configurations where A and B both want
        # each other's spot but neither move alone improves.
        pair_swap_enabled: bool = True,
        pair_swap_sweeps: int = 30,
        pair_swap_k_neighbors: int = 24,          # k-nearest hard macros (spatial)
        pair_swap_k_co_net: int = 8,              # top-K hard macros by shared-net count
        pair_swap_min_improve: float = 1e-7,
        pair_swap_patience: int = 3,
        pair_swap_verbose: bool = True,
        # -- v60 soft-soft pair-swap polish ------------------------------------
        # Same algorithm as hard pair-swap, applied to movable soft macros.
        # No AABB checks for soft sources (soft macros legally overlap in
        # v60's representation); a hard partner still gets AABB-checked at
        # the source's old position. Co-net partners only — spatial nearest
        # scales poorly to thousands of soft macros. Every AABB-legal
        # partner is full-eval'd on the real proxy (no cheap WL+density
        # pre-filter), so keep k_co_net small.
        soft_pair_swap_enabled: bool = True,
        soft_pair_swap_sweeps:  int  = 5,
        soft_pair_swap_k_co_net: int = 4,
        soft_pair_swap_min_improve: float = 1e-7,
        soft_pair_swap_patience: int = 2,
        soft_pair_swap_verbose: bool = True,
        # -- v60 multi-candidate post-Stage-2 refinement -----------------------
        # Run the post-Stage-2 pipeline for the top-N engine seeds, not just the
        # winner, with separate widths for the (expensive, GPU) and (cheap,
        # parallelizable, CPU) halves:
        #   - post_stage2_n_gpu: how many top engine seeds get the GPU side
        #     (basin-hop + soft polish). Each is runtime-heavy, so keep small.
        #   - post_stage2_n_cpu: how many legal candidates from the pooled GPU
        #     output get the CPU side (CD + pair-swap + soft-pair-swap). The
        #     GPU stages emit their top legal candidates (soft-polish restarts +
        #     basin-hop incumbents); the global top-n_cpu of that pool are
        #     polished, and the best final wins. CPU side is ~independent per
        #     candidate, so this widens cheaply (parallel) for little runtime.
        # Defaults (1, 1) reproduce the original single-winner pipeline exactly.
        post_stage2_n_gpu: int = 1,
        post_stage2_n_cpu: int = 8,
        # Run the n_cpu CPU-downstream chains in parallel processes (GIL makes
        # threads useless for the Python-bound CD/swap loops). Falls back to a
        # sequential loop if the pool can't be created. max_workers caps the
        # process count ('auto' = min(n_cpu, os.cpu_count())).
        post_stage2_cpu_parallel: bool = True,
        post_stage2_cpu_max_workers = 'auto',
        # -- Stage 2 seed picker overlap tolerance ------------------------------
        # Threshold = ratio * median(hard_macro_area), floored at 1e-9.
        # ratio=0 → strict (legal-only). ratio=0.1 lets seeds with up to ~10%
        # of one typical macro's area in cumulative overlap count as "legal"
        # for the lowest-proxy-legal pick. Downstream stages (basin-hop, soft
        # polish, CD) then have a chance to legalize the residual.
        stage2_overlap_tol_ratio: float = 0.5,
    ):
        self.num_restarts = int(num_restarts)
        self.num_clusters         = num_clusters
        self.cluster_jitter_frac  = float(cluster_jitter_frac)
        self.cluster_margin_frac  = float(cluster_margin_frac)
        self.cluster_seed         = int(cluster_seed)
        self.stage0_steps          = int(stage0_steps)
        self.stage0_lr             = float(stage0_lr)
        self.stage0_lambda_density = float(stage0_lambda_density)
        self.stage0_lambda_overlap = float(stage0_lambda_overlap)
        self.stage0_gamma_start    = float(stage0_gamma_start)
        self.stage0_gamma_end      = float(stage0_gamma_end)
        self.stage0_target_density = float(stage0_target_density)
        self.use_hh_overlap            = use_hh_overlap
        self.use_ss_overlap            = use_ss_overlap
        self.use_soft_degree_inflation = use_soft_degree_inflation
        self.plc_root      = plc_root
        self.device        = device
        self.verbose       = verbose
        self.deterministic = deterministic
        self.seed          = seed
        self.basin_hop                = bool(basin_hop)
        self.basin_hop_max_hops       = int(basin_hop_max_hops)
        self.basin_hop_final_explore  = int(basin_hop_final_explore)
        self.basin_hop_sigma_set      = tuple(float(s) for s in basin_hop_sigma_set)
        self.basin_hop_tabu_eps       = float(basin_hop_tabu_eps)
        self.basin_hop_tabu_proxy_eps = float(basin_hop_tabu_proxy_eps)
        self.basin_hop_improve_quota  = int(basin_hop_improve_quota)
        self.basin_hop_min_improve_frac = float(basin_hop_min_improve_frac)
        self.basin_hop_stratify       = bool(basin_hop_stratify)
        self.basin_hop_restarts       = int(basin_hop_restarts)
        self.congestion_runtime_mode  = str(congestion_runtime_mode)
        self.soft_polish_enabled = bool(soft_polish_enabled)
        self.soft_polish_restarts = int(soft_polish_restarts)
        # 'auto' kept as-is; explicit values coerced. Resolved in _resolve_soft_polish.
        self.soft_polish_steps = soft_polish_steps if soft_polish_steps == 'auto' else int(soft_polish_steps)
        self.soft_polish_lr = soft_polish_lr if soft_polish_lr == 'auto' else float(soft_polish_lr)
        self.soft_polish_lambda_cong = (soft_polish_lambda_cong if soft_polish_lambda_cong == 'auto'
                                        else float(soft_polish_lambda_cong))
        self.soft_polish_lambda_density = (soft_polish_lambda_density if soft_polish_lambda_density == 'auto'
                                           else float(soft_polish_lambda_density))
        self.soft_polish_lambda_hs = float(soft_polish_lambda_hs)
        self.soft_polish_lambda_ss = (soft_polish_lambda_ss if soft_polish_lambda_ss == 'auto'
                                      else float(soft_polish_lambda_ss))
        self.soft_polish_jitter_fracs = (soft_polish_jitter_fracs if soft_polish_jitter_fracs == 'auto'
                                         else tuple(float(x) for x in soft_polish_jitter_fracs))
        self.soft_polish_use_degree_inflation = bool(soft_polish_use_degree_inflation)
        self.soft_polish_verbose = bool(soft_polish_verbose)
        self.cd_polish_enabled    = bool(cd_polish_enabled)
        self.cd_polish_sweeps     = int(cd_polish_sweeps)
        self.cd_polish_step_frac  = float(cd_polish_step_frac)
        self.cd_polish_step_set   = tuple(float(s) for s in cd_polish_step_set)
        self.cd_polish_num_directions = max(1, int(cd_polish_num_directions))
        self.cd_polish_min_improve = float(cd_polish_min_improve)
        self.cd_polish_patience    = int(cd_polish_patience)
        self.cd_polish_min_sweep_improve_frac = float(cd_polish_min_sweep_improve_frac)
        self.cd_polish_include_hard = bool(cd_polish_include_hard)
        self.cd_polish_verbose    = bool(cd_polish_verbose)
        self.pair_swap_enabled       = bool(pair_swap_enabled)
        self.pair_swap_sweeps        = int(pair_swap_sweeps)
        self.pair_swap_k_neighbors   = max(1, int(pair_swap_k_neighbors))
        self.pair_swap_k_co_net      = max(0, int(pair_swap_k_co_net))
        self.pair_swap_min_improve   = float(pair_swap_min_improve)
        self.pair_swap_patience      = int(pair_swap_patience)
        self.pair_swap_verbose       = bool(pair_swap_verbose)
        self.soft_pair_swap_enabled        = bool(soft_pair_swap_enabled)
        self.soft_pair_swap_sweeps         = int(soft_pair_swap_sweeps)
        self.soft_pair_swap_k_co_net       = max(1, int(soft_pair_swap_k_co_net))
        self.soft_pair_swap_min_improve    = float(soft_pair_swap_min_improve)
        self.soft_pair_swap_patience       = int(soft_pair_swap_patience)
        self.soft_pair_swap_verbose        = bool(soft_pair_swap_verbose)
        self.post_stage2_n_gpu = max(1, int(post_stage2_n_gpu))
        self.post_stage2_n_cpu = max(1, int(post_stage2_n_cpu))
        self.post_stage2_cpu_parallel = bool(post_stage2_cpu_parallel)
        self.post_stage2_cpu_max_workers = post_stage2_cpu_max_workers
        self.stage2_overlap_tol_ratio = float(stage2_overlap_tol_ratio)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _soft_log(self, msg):
        if self.soft_polish_verbose:
            self._log(msg)

    def _cd_log(self, msg):
        if self.cd_polish_verbose:
            self._log(msg)

    def _swap_log(self, msg):
        if self.pair_swap_verbose:
            self._log(msg)

    def _soft_swap_log(self, msg):
        if self.soft_pair_swap_verbose:
            self._log(msg)

    def _resolve_device(self) -> str:
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device

    def _proxy_cost(self, pos: torch.Tensor, benchmark: Benchmark, plc: PlacementCost) -> float:
        nM  = int(benchmark.num_macros)
        sub = pos[:nM] if pos.shape[0] > nM else pos
        try:
            costs = compute_proxy_cost(sub.to(torch.float32), benchmark, plc)
            return float(costs['proxy_cost'])
        except Exception as exc:
            self._log(f"  PLC eval failed: {exc}")
            return float('inf')

    def _proxy_and_overlap(self, pos: torch.Tensor, benchmark: Benchmark,
                           plc: PlacementCost) -> tuple:
        """Return (proxy_cost, total_overlap_area). Overlap is NaN on failure
        so the caller's legality test (overlap <= 1e-9) treats it as illegal."""
        nM  = int(benchmark.num_macros)
        sub = pos[:nM] if pos.shape[0] > nM else pos
        try:
            costs = compute_proxy_cost(sub.to(torch.float32), benchmark, plc)
            return (float(costs['proxy_cost']),
                    float(costs.get('total_overlap_area', float('nan'))))
        except Exception as exc:
            self._log(f"  PLC eval failed: {exc}")
            return float('inf'), float('nan')

    def _hard_frac(self, benchmark: Benchmark) -> float:
        nH = int(benchmark.num_hard_macros)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        hw = benchmark.macro_sizes.numpy()[:nH, 0] / 2.0
        hh = benchmark.macro_sizes.numpy()[:nH, 1] / 2.0
        hard_area   = 4.0 * float((hw * hh).sum())
        canvas_area = cw * ch
        return hard_area / max(canvas_area, 1e-12)

    @staticmethod
    def _congestion_runtime_scale(benchmark: Benchmark) -> float:
        """0..1 runtime-pressure score for routability-heavy designs."""
        return _congestion_work_tier(benchmark) / 4.0

    def _runtime_guard_enabled(self) -> bool:
        mode = self.congestion_runtime_mode.lower()
        return mode not in ('off', 'false', '0', 'none')

    def _cap_for_congestion_runtime(self, benchmark: Benchmark, value: int, caps: tuple) -> int:
        if not self._runtime_guard_enabled():
            return int(value)
        pressure = self._congestion_runtime_scale(benchmark)
        if pressure >= 1.0:
            return min(int(value), int(caps[0]))
        if pressure >= 0.75:
            return min(int(value), int(caps[1]))
        if pressure >= 0.5:
            return min(int(value), int(caps[2]))
        if pressure >= 0.25:
            return min(int(value), int(caps[3]))
        return int(value)

    @staticmethod
    def _auto_soft_polish_cong_interval(benchmark: Benchmark) -> int:
        """Soft-polish cong_eval_interval — floors at 2 even on small designs."""
        return _congestion_work_tier(benchmark) + 2

    def _make_cohort(self, cls, num_restarts):
        return cls(
            num_restarts              = num_restarts,
            num_clusters              = self.num_clusters,
            cluster_jitter_frac       = self.cluster_jitter_frac,
            cluster_margin_frac       = self.cluster_margin_frac,
            cluster_seed              = self.cluster_seed,
            stage0_steps              = self.stage0_steps,
            stage0_lr                 = self.stage0_lr,
            stage0_lambda_density     = self.stage0_lambda_density,
            stage0_lambda_overlap     = self.stage0_lambda_overlap,
            stage0_gamma_start        = self.stage0_gamma_start,
            stage0_gamma_end          = self.stage0_gamma_end,
            stage0_target_density     = self.stage0_target_density,
            use_hh_overlap            = self.use_hh_overlap,
            use_ss_overlap            = self.use_ss_overlap,
            use_soft_degree_inflation = self.use_soft_degree_inflation,
            plc_root                  = self.plc_root,
            device                    = self.device,
            verbose                   = self.verbose,
            deterministic             = self.deterministic,
            seed                      = self.seed,
            stage2_overlap_tol_ratio  = self.stage2_overlap_tol_ratio,
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()
        if self.deterministic:
            _set_deterministic(self.seed)
        else:
            _set_fast_nondeterministic()

        device_str = self._resolve_device()
        hard_frac = self._hard_frac(benchmark)
        self._log(
            f"[v60 {benchmark.name}] engine  hard_frac={hard_frac:.3f}  "
            f"×{self.num_restarts}  device={device_str}  num_clusters={self.num_clusters}  "
            f"jitter={self.cluster_jitter_frac:.3f}  "
            f"s0_steps={self.stage0_steps}  s0_lr={self.stage0_lr:.2f}"
        )

        # v60: single engine run.
        cohort_elapsed = {}
        cohorts = {'engine': self._make_cohort(v60_Engine, self.num_restarts)}
        t = time.time()
        pos_pick = cohorts['engine'].place(benchmark)
        cohort_elapsed['engine'] = round(time.time() - t, 3)

        netlist  = osp.join(self.plc_root, benchmark.name, "netlist.pb.txt")
        init_plc = osp.join(self.plc_root, benchmark.name, "initial.plc")

        all_results = [('engine', pos_pick)]

        if not osp.exists(netlist):
            for tag, pos in all_results:
                if pos is not None:
                    self._log(f"[v60 {benchmark.name}] no netlist; returning {tag}")
                    return pos
            return None

        plc = PlacementCost(netlist)
        if osp.exists(init_plc):
            plc.restore_placement(init_plc, ifInital=True, ifReadComment=True)

        scored = []
        for tag, pos in all_results:
            if pos is None:
                scored.append((tag, float('inf'), float('nan'), pos))
            else:
                pr, ov = self._proxy_and_overlap(pos, benchmark, plc)
                scored.append((tag, pr, ov, pos))
        # v60: prefer the best *legal* (overlap-free) cohort result; fall back
        # to lowest proxy only when no cohort produced a legal placement.
        legal = [c for c in scored if c[2] <= 1e-9]
        best_tag, best_proxy, _best_ov, best_pos = min(
            legal if legal else scored, key=lambda c: c[1]
        )

        proxy_str = '  '.join(f'{tag}={p:.4f}' for tag, p, _, _ in scored)
        self._log(
            f"[v60 {benchmark.name}] {proxy_str}  winner={best_tag}  "
            f"best={best_proxy:.4f}  total {time.time()-t0:.1f}s"
        )

        # ── Multi-candidate post-Stage-2 refinement ──────────────────────
        # GPU side (basin-hop + soft polish) runs for the top-n_gpu engine
        # seeds and emits a pool of legal candidates; the CPU side (CD +
        # pair-swap + soft-pair-swap) then polishes the top-n_cpu of that pool
        # (in parallel processes when enabled), and the best final wins.
        if (best_pos is not None and math.isfinite(best_proxy)
                and best_tag in cohorts):
            winner_cohort = cohorts[best_tag]
            nM      = int(benchmark.num_macros)
            mov_idx = np.where(benchmark.get_movable_mask().numpy())[0]
            scale   = 0.5 * (float(benchmark.canvas_width) + float(benchmark.canvas_height))

            seed_pool = list(getattr(winner_cohort, '_last_seed_pool', None) or [])
            if not seed_pool:
                seed_pool = [{'pos': best_pos[:nM].detach().cpu().numpy().astype(np.float64),
                              'proxy': float(best_proxy), 'overlap_area': 0.0}]
            n_gpu = min(self.post_stage2_n_gpu, len(seed_pool))
            self._log(
                f"[v60 {benchmark.name}] post-Stage-2: n_gpu={n_gpu}  "
                f"n_cpu={self.post_stage2_n_cpu}  (engine seed pool={len(seed_pool)})"
            )

            # GPU side: build the legal candidate pool across the top-n_gpu seeds.
            cand_pool = []
            for gi in range(n_gpu):
                seed = seed_pool[gi]
                cand_pool += self._gpu_side(
                    seed['pos'], seed['proxy'], benchmark, plc, winner_cohort,
                    mov_idx, scale, t0, label=f"{best_tag}#{gi}")

            # Rank best-first, dedup by proxy, take top-n_cpu for the CPU side.
            cand_pool.sort(key=lambda c: c['proxy'])
            seen, deduped = set(), []
            for c in cand_pool:
                key = round(c['proxy'], 9)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(c)
            cpu_cands = deduped[: self.post_stage2_n_cpu]
            self._log(
                f"[v60 {benchmark.name}] CPU side: polishing {len(cpu_cands)} "
                f"candidate(s)  (gpu pool={len(cand_pool)})  total {time.time()-t0:.1f}s"
            )

            results = self._run_cpu_side(cpu_cands, benchmark, plc)
            for (rpos, rproxy, rtag) in results:
                if rproxy < best_proxy - 1e-9:
                    best_pos, best_proxy, best_tag = rpos, rproxy, rtag
            self._log(
                f"[v60 {benchmark.name}] post-Stage-2 done: best={best_proxy:.4f}  "
                f"tag={best_tag}  total {time.time()-t0:.1f}s"
            )

        return best_pos

    def _gpu_side(self, seed_pos_np, seed_proxy, benchmark, plc, winner_cohort,
                  mov_idx, scale, t0, label):
        """GPU-side refinement (basin-hop + soft polish) from one engine seed.

        Returns a list of legal candidate dicts {pos: tensor, proxy: float,
        tag: str}: the basin-hop incumbent plus the top soft-polish legal
        restarts. These feed the pooled top-n_cpu selection for the CPU side.
        """
        nM = int(benchmark.num_macros)
        cur_t = benchmark.macro_positions.clone()
        cur_t[:nM] = torch.tensor(seed_pos_np, dtype=cur_t.dtype)
        cur_proxy = float(seed_proxy)
        cur_tag = label

        # ── Basin-hop from this seed ─────────────────────────────────────
        if (self.basin_hop and winner_cohort is not None
                and math.isfinite(cur_proxy)):
            Bhop = (self.basin_hop_restarts if self.basin_hop_restarts > 0
                    else int(getattr(winner_cohort, 'num_restarts', 16)))
            Bhop = self._cap_for_congestion_runtime(benchmark, Bhop, caps=(5, 6, 7, 8))
            rng  = np.random.default_rng(self.seed if self.deterministic else None)
            try:
                ic      = compute_proxy_cost(cur_t[:nM].to(torch.float32), benchmark, plc)
                init_ov = float(ic.get('total_overlap_area', float('nan')))
            except Exception:
                init_ov = float('nan')
            initial = {
                'pos':          cur_t[:nM].detach().cpu().numpy().astype(np.float32),
                'proxy':        cur_proxy,
                'overlap_area': init_ov,
            }

            def run_from_init(init_b_nmov_2):
                p = winner_cohort.place(benchmark, init_positions=init_b_nmov_2)
                metrics = getattr(winner_cohort, '_last_run_metrics', None) or []
                if metrics:
                    legal_m = [m for m in metrics
                               if float(m.get('overlap_area', float('nan'))) <= 1e-9]
                    mm = min(legal_m if legal_m else metrics,
                             key=lambda m: m.get('score_proxy', float('inf')))
                    pr = float(mm.get('score_proxy', float('inf')))
                    ov = float(mm.get('overlap_area', float('nan')))
                    if not math.isfinite(pr):
                        pr = self._proxy_cost(p, benchmark, plc)
                else:
                    pr = self._proxy_cost(p, benchmark, plc)
                    ov = float('nan')
                return {'pos': p[:nM].detach().cpu().numpy().astype(np.float32),
                        'proxy': pr, 'overlap_area': ov}

            self._log(
                f"[v60 {benchmark.name}] basin-hop ({label}): B={Bhop}  "
                f"max_hops={self.basin_hop_max_hops}  improve_quota={self.basin_hop_improve_quota}"
            )
            bh = _basin_hop(
                run_from_init, initial, mov_idx, scale, Bhop, rng,
                max_hops=self.basin_hop_max_hops,
                final_explore_n=self.basin_hop_final_explore,
                sigma_set=self.basin_hop_sigma_set,
                tabu_eps=self.basin_hop_tabu_eps,
                tabu_proxy_eps=self.basin_hop_tabu_proxy_eps,
                improve_quota=self.basin_hop_improve_quota,
                min_improve_frac=self.basin_hop_min_improve_frac,
                stratify=self.basin_hop_stratify,
                log=self._log,
            )
            if bh['proxy'] < cur_proxy - 1e-9:
                self._log(f"[v60 {benchmark.name}] basin-hop ({label}) improved "
                          f"{cur_proxy:.4f} -> {bh['proxy']:.4f}")
                out_t      = benchmark.macro_positions.clone()
                out_t[:nM] = torch.tensor(bh['pos'], dtype=out_t.dtype)
                cur_t      = out_t
                cur_proxy  = float(bh['proxy'])
                cur_tag    = f"{label}+bh"

        candidates = [{'pos': cur_t, 'proxy': cur_proxy, 'tag': cur_tag}]

        # ── Soft polish; pull its top legal restarts into the pool ───────
        if self.soft_polish_enabled and math.isfinite(cur_proxy):
            base_costs = compute_proxy_cost(cur_t[:nM].to(torch.float32), benchmark, plc)
            self._soft_log(
                f"[v60 {benchmark.name}] soft polish ({cur_tag}) start: "
                f"proxy={base_costs['proxy_cost']:.4f}"
            )
            self._last_soft_pool = []
            _polished, _pc = self._soft_only_polish(cur_t, benchmark, plc)
            pool = getattr(self, '_last_soft_pool', []) or []
            for (pr, pos_np) in pool[: self.post_stage2_n_cpu]:
                cand_t = benchmark.macro_positions.clone()
                cand_t[:nM] = torch.tensor(pos_np, dtype=cand_t.dtype)
                candidates.append({'pos': cand_t, 'proxy': float(pr),
                                   'tag': f"{cur_tag}+soft"})

        return candidates

    def _cpu_side(self, cand_pos, cand_proxy, cand_tag, benchmark, plc, t0=None):
        """CPU-side refinement (CD polish -> hard pair-swap -> soft pair-swap)
        from one candidate. Returns (pos tensor, proxy, tag). Self-contained so
        it can run in a worker process."""
        if t0 is None:
            t0 = time.time()
        best_pos   = cand_pos
        best_proxy = float(cand_proxy)
        best_tag   = cand_tag

        if (self.cd_polish_enabled and best_pos is not None
                and math.isfinite(best_proxy) and plc is not None):
            cd_out, cd_costs = self._cd_polish(best_pos, benchmark, plc)
            if cd_costs is not None and float(cd_costs['proxy_cost']) < best_proxy - 1e-9:
                best_pos   = cd_out
                best_proxy = float(cd_costs['proxy_cost'])
                best_tag   = f"{best_tag}+cd"

        if (self.pair_swap_enabled and best_pos is not None
                and math.isfinite(best_proxy) and plc is not None):
            ps_out, ps_costs = self._pair_swap_polish(best_pos, benchmark, plc)
            if ps_costs is not None and float(ps_costs['proxy_cost']) < best_proxy - 1e-9:
                best_pos   = ps_out
                best_proxy = float(ps_costs['proxy_cost'])
                best_tag   = f"{best_tag}+swap"

        if (self.soft_pair_swap_enabled and best_pos is not None
                and math.isfinite(best_proxy) and plc is not None):
            sps_out, sps_costs = self._pair_swap_polish_soft(best_pos, benchmark, plc)
            if sps_costs is not None and float(sps_costs['proxy_cost']) < best_proxy - 1e-9:
                best_pos   = sps_out
                best_proxy = float(sps_costs['proxy_cost'])
                best_tag   = f"{best_tag}+softswap"

        return best_pos, best_proxy, best_tag

    def _run_cpu_side(self, cpu_cands, benchmark, plc):
        """Run the CPU downstream on each candidate; return a list of
        (pos tensor, proxy, tag).

        Parallelism uses a *fork*-context multiprocessing.Process per candidate
        (not ProcessPoolExecutor): with fork the target closure and all inputs
        (self, benchmark, plc) are inherited through the fork — never pickled —
        which sidesteps the module-identity pickling failure that ProcessPool
        hits under the path-loaded submission. Only the numpy result returns
        over the Queue. Children are CPU-only (CD/swap stages don't touch CUDA),
        so the parent's CUDA context is irrelevant to them. A child that raises
        reports it (sends a None result) and that candidate is re-run
        sequentially in the parent. There is no wall-clock guard: the parent
        blocks until every child of a wave reports."""
        nM = int(benchmark.num_macros)
        if not cpu_cands:
            return []

        def _seq_one(c):
            return self._cpu_side(c['pos'], c['proxy'], c['tag'], benchmark, plc)

        if (not self.post_stage2_cpu_parallel) or len(cpu_cands) == 1:
            return [_seq_one(c) for c in cpu_cands]

        if self.post_stage2_cpu_max_workers == 'auto':
            max_workers = min(len(cpu_cands), os.cpu_count() or 1)
        else:
            max_workers = max(1, min(int(self.post_stage2_cpu_max_workers), len(cpu_cands)))

        try:
            ctx = mp.get_context('fork')
        except (ValueError, RuntimeError) as exc:
            self._log(f"[v60 {benchmark.name}] fork context unavailable ({exc}); "
                      f"CPU side sequential")
            return [_seq_one(c) for c in cpu_cands]

        def _child(idx, cand, q):
            # Inherited via fork; never pickled. Only the result is sent back.
            try:
                pos, proxy, tag = self._cpu_side(
                    cand['pos'], cand['proxy'], cand['tag'], benchmark, plc)
                q.put((idx, pos[:nM].detach().cpu().numpy().astype(np.float64),
                       float(proxy), tag))
            except Exception as exc:  # report; parent re-runs this idx
                q.put((idx, None, None, repr(exc)))

        results = [None] * len(cpu_cands)
        failed = []
        n = len(cpu_cands)
        self._log(
            f"[v60 {benchmark.name}] CPU side: {n} candidate(s), "
            f"fork pool max_workers={max_workers}"
        )
        for ws in range(0, n, max_workers):
            wave = list(range(ws, min(ws + max_workers, n)))
            q = ctx.Queue()
            procs = {}
            for idx in wave:
                p = ctx.Process(target=_child, args=(idx, cpu_cands[idx], q),
                                daemon=True)
                p.start()
                procs[idx] = p
            # Each child sends exactly one message (result or error sentinel),
            # so block for exactly len(wave) messages — no wall-clock guard.
            for _ in wave:
                idx, pos_np, proxy, tag = q.get()
                if pos_np is None:
                    self._log(f"[v60 {benchmark.name}] CPU child {idx} errored: {tag}")
                    failed.append(idx)
                else:
                    t = benchmark.macro_positions.clone()
                    t[:nM] = torch.tensor(pos_np, dtype=t.dtype)
                    results[idx] = (t, float(proxy), tag)
            for p in procs.values():
                p.join()

        # Re-run any errored candidates sequentially in the parent.
        for idx in failed:
            results[idx] = _seq_one(cpu_cands[idx])

        return [r for r in results if r is not None]

    @staticmethod
    def _canvas_L(benchmark: Benchmark) -> float:
        """Canvas size metric: rms of the two side lengths."""
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        return math.sqrt(cw * cw + ch * ch) / math.sqrt(2.0)

    def _resolve_soft_polish(self, benchmark: Benchmark) -> dict:
        """Resolve the 'auto' soft-polish knobs from the canvas size.

        Re-fit from the v60 Optuna sweeps (ibm01/02/06/18). The loss weights
        scale with canvas size L: a single linear ramp `sat` maps L=23
        (ibm01-scale) -> 0 and L=65 (ibm18-scale) -> 1. The mid-size benches
        (ibm02/06) are insensitive to these knobs (|Spearman| ~ 0.1-0.25) while
        ibm18 is highly sensitive (~0.8), so a ramp that is accurate at the
        extremes and approximate in the middle is the right trade-off.
        An explicit (non-'auto') ctor value overrides the formula.
        """
        L = self._canvas_L(benchmark)
        sat = min(1.0, max(0.0, (L - 23.0) / 42.0))

        steps = self.soft_polish_steps
        if steps == 'auto':
            # delayed ramp: small/mid show no step preference, only ibm18 needs more
            knee = min(1.0, max(0.0, (L - 33.0) / 32.0))
            steps = int(round(3500.0 + 3000.0 * knee))

        lr = self.soft_polish_lr
        if lr == 'auto':
            lr = 0.48 + 1.34 * sat

        lc = self.soft_polish_lambda_cong
        if lc == 'auto':
            lc = 52000.0 * (1.0 + 1.5 * sat)

        ld = self.soft_polish_lambda_density
        if ld == 'auto':
            ld = 280.0 + 380.0 * sat

        lss = self.soft_polish_lambda_ss
        if lss == 'auto':
            lss = 10.0 + 98.0 * sat

        jitter = self.soft_polish_jitter_fracs
        if jitter == 'auto':
            jitter_max = 0.080 + 0.038 * sat
            jitter = self._jitter_schedule(jitter_max, self.soft_polish_restarts)

        return {
            'steps': int(steps),
            'lr': float(lr),
            'lambda_cong': float(lc),
            'lambda_density': float(ld),
            'lambda_hs': float(self.soft_polish_lambda_hs),
            'lambda_ss': float(lss),
            'jitter_fracs': tuple(float(x) for x in jitter),
        }

    @staticmethod
    def _jitter_schedule(max_frac: float, restarts: int) -> tuple:
        """Monotone jitter schedule: restart 0 stays at 0, the rest ramp to max_frac."""
        restarts = max(1, int(restarts))
        if restarts == 1:
            return (0.0,)
        max_frac = max(0.0, float(max_frac))
        vals = [0.0]
        for i in range(1, restarts):
            t = i / max(1, restarts - 1)
            vals.append(max_frac * (t ** 1.35))
        return tuple(vals)

    def _make_soft_polish_params(self, benchmark: Benchmark) -> dict:
        sp = self._resolve_soft_polish(benchmark)
        params = {
            'num_steps_s1': 0,
            'lr_s1': 1.0,
            'gamma_s1_start': 2.149,
            'gamma_s1_end': 0.351,
            'lambda_cong_s1': 0.0,
            'num_steps_s2': sp['steps'],
            'lr_s2': sp['lr'],
            'gamma_s2_start': 0.36,
            'gamma_s2_end': 0.020,
            'lambda_density_end': sp['lambda_density'],
            'lambda_hs_overlap_start': sp['lambda_hs'],
            'lambda_hs_overlap_end': sp['lambda_hs'],
            'lambda_hh_overlap_start': 0.0,
            'lambda_hh_overlap_end': 0.0,
            'lambda_ss_overlap_start': sp['lambda_ss'],
            'lambda_ss_overlap_end': sp['lambda_ss'],
            'overlap_ramp_frac': 0.15,
            'soft_inflation_scale': 4.0 if self.soft_polish_use_degree_inflation else 0.0,
            'soft_inflation_alpha': 0.5,
            'soft_inflation_max': 25.0,
            'soft_inflation_clamp': 0.20,
            'lambda_cong_s2': sp['lambda_cong'],
            'gamma_rudy': 0.1,
            'congestion_threshold': 0.0,
            'cong_eps': 1e-3,
            'use_cong_v2': True,
            'cong_snap_sigma': 0.22,
            'cong_edge_chunk': 65536,
            'cong_eval_interval_s2': self._auto_soft_polish_cong_interval(benchmark),
            'cong_sigma_l': 0.41,
            'cong_abu_frac': 0.05,
            'lambda_cong_size_scale': False,
            'cong_scale': 1.0,
            'use_preconditioner': False,
            's2_bf16': True,
            'safety_gap': 0.001,
            'inflation_factor': 1.0,
            'target_density': 0.7,
            'cluster_jitter_frac': 0.0,
        }
        params.update(_parse_plc_routing_params(
            osp.join(self.plc_root, benchmark.name, 'initial.plc')
        ))
        return params

    def _soft_only_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc: PlacementCost,
    ):
        nM = int(benchmark.num_macros)
        nH = int(benchmark.num_hard_macros)
        nS = nM - nH
        if nS <= 0:
            costs = compute_proxy_cost(placement[:nM].to(torch.float32), benchmark, plc)
            return placement, costs

        raw = _extract_raw(benchmark)
        base_np = placement[:nM].detach().cpu().numpy().astype(np.float64)
        raw['macro_positions'] = base_np.copy()

        original_movable = raw['movable_mask'].copy()
        soft_mask = np.zeros_like(original_movable, dtype=bool)
        soft_mask[nH:nM] = original_movable[nH:nM]
        raw['movable_mask'] = soft_mask
        soft_indices = np.where(soft_mask)[0]
        if soft_indices.size == 0:
            costs = compute_proxy_cost(placement[:nM].to(torch.float32), benchmark, plc)
            return placement, costs

        B = max(1, self.soft_polish_restarts)
        rng = np.random.default_rng(self.seed if self.deterministic else None)
        scale = 0.5 * (float(benchmark.canvas_width) + float(benchmark.canvas_height))
        jitter_fracs = self._resolve_soft_polish(benchmark)['jitter_fracs']
        init = np.repeat(base_np[soft_indices][None, :, :], B, axis=0).astype(np.float32)
        for b in range(1, B):
            frac = jitter_fracs[b % len(jitter_fracs)]
            if frac <= 0.0:
                continue
            init[b] += rng.normal(0.0, frac * scale, size=init[b].shape).astype(np.float32)
            init[b, :, 0] = np.clip(init[b, :, 0], 0.0, float(benchmark.canvas_width))
            init[b, :, 1] = np.clip(init[b, :, 1], 0.0, float(benchmark.canvas_height))

        params = self._make_soft_polish_params(benchmark)
        device = self.device if self.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._soft_log(
            f"[v60 {benchmark.name}] soft-only Stage2: B={B} nS={int(soft_indices.size)} "
            f"steps={params['num_steps_s2']} lr={params['lr_s2']:.3f} "
            f"cong_every={params.get('cong_eval_interval_s2', 1)} "
            f"lc={params['lambda_cong_s2']:.0f} ld={params['lambda_density_end']:.0f} "
            f"hs={params['lambda_hs_overlap_end']:.0f} ss={params['lambda_ss_overlap_end']:.0f} "
            f"deg_infl={self.soft_polish_use_degree_inflation}"
        )

        all_pos, _all_metrics = _run_batch(
            raw, params, B, device,
            init_positions=init,
            cluster_data=None,
        )

        best_pos = None
        best_costs = None
        best_proxy = float('inf')
        best_legal_pos = None
        best_legal_costs = None
        best_legal_proxy = float('inf')
        legal_restarts = []   # (proxy, pos_np) for overlap-free restarts
        for i, pos_np in enumerate(all_pos):
            costs = compute_proxy_cost(torch.tensor(pos_np, dtype=torch.float32), benchmark, plc)
            proxy = float(costs['proxy_cost'])
            ovlp = float(costs.get('total_overlap_area', float('nan')))
            self._soft_log(
                f"  soft seed {i}: proxy={proxy:.4f}  "
                f"wl={costs['wirelength_cost']:.3f} den={costs['density_cost']:.3f} "
                f"cong={costs['congestion_cost']:.3f} ovlp={ovlp:.4g}"
            )
            if proxy < best_proxy:
                best_proxy = proxy
                best_pos = pos_np
                best_costs = costs
            # v60: track the best *legal* (overlap-free) soft restart too
            if ovlp <= 1e-9:
                legal_restarts.append((proxy, pos_np))
                if proxy < best_legal_proxy:
                    best_legal_proxy = proxy
                    best_legal_pos = pos_np
                    best_legal_costs = costs

        # Stash the ranked legal-restart pool (best-first) so the multi-candidate
        # post-Stage-2 path can pull the top-K, not just the single best.
        legal_restarts.sort(key=lambda t: t[0])
        self._last_soft_pool = legal_restarts

        # prefer the best legal restart; fall back to best-proxy if none legal
        if best_legal_pos is not None:
            if best_legal_proxy > best_proxy + 1e-9:
                self._soft_log(
                    f"  [legal-pick] best legal soft proxy={best_legal_proxy:.4f} "
                    f"over lower illegal proxy={best_proxy:.4f}"
                )
            best_pos = best_legal_pos
            best_costs = best_legal_costs

        if best_pos is None:
            best_pos = base_np
            best_costs = compute_proxy_cost(torch.tensor(best_pos, dtype=torch.float32), benchmark, plc)

        out = placement.clone()
        out[:nM] = torch.tensor(best_pos, dtype=out.dtype)
        return out, best_costs

    def _cd_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc: PlacementCost,
    ):
        """Single-macro coordinate descent over movable macros (soft and,
        if `cd_polish_include_hard`, hard too) using IncrementalEval.

        Strategy per macro:
          1. Enumerate candidate offsets (cd_polish_num_directions evenly-spaced
             directions × cd_polish_step_set step sizes). For hard macros,
             candidates that would overlap another hard macro are rejected via a
             vectorised AABB intersection check (~5 µs).
          2. Full-eval EVERY candidate on the real proxy (tentatively commit,
             read proxy incl. congestion, revert), and track the candidate with
             the best actual proxy improvement.
          3. If the best improvement exceeds `cd_polish_min_improve`, commit
             that candidate; else skip the macro this sweep.

        No cheap WL+density pre-filter: a measured ceiling test showed the
        old "top-K by WL+density delta" filter systematically discarded
        congestion-reducing moves — moves that worsen WL+density but improve
        the real proxy via congestion never reached the full eval. On the
        cong-dominated benches that is exactly where the gains are, so every
        candidate is now evaluated on the real proxy directly. (The full
        congestion eval is cheap enough after the vectorised IncrementalEval
        cong path; the cost is bounded by trimming step_set / num_directions.)

        Hard-macro inclusion: hard macros have outsize cong impact (their
        blockage shifts whole rows/columns of routing demand) but are also
        constrained by no-overlap legality. We enforce legality by skipping
        any candidate move that would create a hard-hard AABB intersection.
        Soft macros are not subject to the overlap check (they're virtual
        cluster proxies that legally overlap stuff in the placement).

        Sweeps stop after `cd_polish_patience` consecutive zero-move sweeps,
        or after `cd_polish_sweeps` complete.
        """
        nM = int(benchmark.num_macros)
        nH = int(benchmark.num_hard_macros)
        nS = nM - nH
        if nS <= 0 or not self.cd_polish_enabled:
            return placement, None

        movable = benchmark.get_movable_mask().cpu().numpy().astype(bool)
        if self.cd_polish_include_hard:
            target_idx = np.where(movable[:nM])[0]
        else:
            target_idx = np.where(movable[:nM] & (np.arange(nM) >= nH))[0]
        if target_idx.size == 0:
            return placement, None

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        base_step = self.cd_polish_step_frac * 0.5 * (cw + ch)
        # N evenly-spaced unit vectors around the circle (N = cd_polish_num_directions).
        # Unit-length so each candidate has the same Euclidean step magnitude
        # regardless of angle.
        N_dirs = self.cd_polish_num_directions
        dirs = [(math.cos(2.0 * k * math.pi / N_dirs),
                 math.sin(2.0 * k * math.pi / N_dirs))
                for k in range(N_dirs)]
        step_mults = self.cd_polish_step_set if self.cd_polish_step_set else (1.0,)

        # Build IncrementalEval from current placement.
        e = IncrementalEval(benchmark, plc=plc)
        base_pos = placement[:nM].detach().cpu().numpy().astype(np.float64)
        e.set_placement(base_pos)

        # Initial real proxy from the same caches the CD loop will be reading.
        cur_proxy = float(e.proxy(include_cong=True))
        # Count targets split into soft/hard for the start-of-run log.
        n_hard_targets = int(np.sum(target_idx < nH))
        n_soft_targets = int(target_idx.size - n_hard_targets)
        self._cd_log(
            f"[v60 {benchmark.name}] CD polish start: proxy={cur_proxy:.6f}  "
            f"nS={n_soft_targets} nH={n_hard_targets} (include_hard={self.cd_polish_include_hard})  "
            f"sweeps={self.cd_polish_sweeps}  "
            f"step_frac={self.cd_polish_step_frac:.4f}  "
            f"step_set={self.cd_polish_step_set}  full-eval (no cheap filter)"
        )
        t0 = time.time()

        # Hard-macro AABB cache (xlo, xhi, ylo, yhi). Used to gate candidate
        # hard-macro moves so they don't overlap any other hard macro.
        hard_aabb = np.zeros((nH, 4), dtype=np.float64)
        for m in range(nH):
            cx = float(e.macro_pos[m, 0]); cy = float(e.macro_pos[m, 1])
            hw = float(e.macro_w[m]) * 0.5; hh = float(e.macro_h[m]) * 0.5
            hard_aabb[m, 0] = cx - hw
            hard_aabb[m, 1] = cx + hw
            hard_aabb[m, 2] = cy - hh
            hard_aabb[m, 3] = cy + hh

        rng = np.random.default_rng(self.seed if self.deterministic else None)
        total_moved = 0
        zero_streak = 0   # consecutive zero-move sweeps for patience-based stop

        for sweep in range(self.cd_polish_sweeps):
            proxy_before_sweep = cur_proxy
            order = target_idx[rng.permutation(len(target_idx))]
            moved_this_sweep = 0
            tested_this_sweep = 0
            active_macros = 0   # macros with >= 1 legal candidate this sweep
            no_improve = 0      # macros whose candidates didn't beat min_improve
            for m_i in order:
                m_i = int(m_i)
                cur_x = float(e.macro_pos[m_i, 0])
                cur_y = float(e.macro_pos[m_i, 1])
                is_hard = (m_i < nH)
                if is_hard:
                    hw_i = float(e.macro_w[m_i]) * 0.5
                    hh_i = float(e.macro_h[m_i]) * 0.5

                # Enumerate candidate positions (directions × step sizes). For
                # hard macros, reject any that would overlap another hard macro
                # (AABB intersection, vectorised). No cheap WL+density filter:
                # every candidate is full-evaluated on the real proxy below.
                # Dedup by clipped position: any step large enough to overshoot
                # the canvas clips to the same corner/edge, so many of the large
                # multipliers produce identical positions — evaluate each unique
                # position once (exact: duplicates have identical proxy).
                cands = []  # list of (new_x, new_y)
                seen = set()
                for mult in step_mults:
                    s = base_step * mult
                    for (dx, dy) in dirs:
                        new_x = float(np.clip(cur_x + dx * s, 0.0, cw))
                        new_y = float(np.clip(cur_y + dy * s, 0.0, ch))
                        if new_x == cur_x and new_y == cur_y:
                            continue
                        key = (new_x, new_y)
                        if key in seen:
                            continue
                        seen.add(key)
                        tested_this_sweep += 1
                        if is_hard:
                            # Vectorised hard-hard AABB overlap check.
                            new_xlo = new_x - hw_i; new_xhi = new_x + hw_i
                            new_ylo = new_y - hh_i; new_yhi = new_y + hh_i
                            no_overlap = (
                                (new_xhi <= hard_aabb[:, 0]) |
                                (new_xlo >= hard_aabb[:, 1]) |
                                (new_yhi <= hard_aabb[:, 2]) |
                                (new_ylo >= hard_aabb[:, 3])
                            )
                            no_overlap[m_i] = True  # ignore self
                            if not np.all(no_overlap):
                                continue
                        cands.append((new_x, new_y))

                if not cands:
                    continue
                active_macros += 1

                # Score every candidate on the real proxy WITHOUT committing,
                # via the incremental cong scorer (no per-candidate re-smooth,
                # no commit/revert churn); commit only the winning move. The
                # per-macro base proxy is exact (cached assembled grids); each
                # candidate proxy matches a full recompute to float-reorder.
                cur_wl  = e.compute_wl_cost()
                cur_den = e.compute_density_cost()
                base_proxy = (WEIGHT_WL * cur_wl + WEIGHT_DENSITY * cur_den
                              + WEIGHT_CONG * e.cong_cost())
                best_new_proxy = base_proxy
                best_new_xy = None
                for new_xy in cands:
                    new_proxy = e.proxy_for_move(m_i, new_xy, cur_wl, cur_den)
                    if new_proxy < best_new_proxy:
                        best_new_proxy = new_proxy
                        best_new_xy = new_xy

                if best_new_xy is not None and \
                        (base_proxy - best_new_proxy) >= self.cd_polish_min_improve:
                    # Commit the winning candidate.
                    final_st = e.delta_for_move(m_i, best_new_xy, include_cong=False)
                    e.commit_move(m_i, best_new_xy, final_st)
                    cur_proxy = best_new_proxy
                    moved_this_sweep += 1
                    # Refresh the AABB so subsequent overlap checks see the new
                    # position. (Soft macros aren't in hard_aabb so no-op there.)
                    if is_hard:
                        hard_aabb[m_i, 0] = best_new_xy[0] - hw_i
                        hard_aabb[m_i, 1] = best_new_xy[0] + hw_i
                        hard_aabb[m_i, 2] = best_new_xy[1] - hh_i
                        hard_aabb[m_i, 3] = best_new_xy[1] + hh_i
                else:
                    no_improve += 1

            total_moved += moved_this_sweep
            sweep_rel_improve = ((proxy_before_sweep - cur_proxy)
                                 / max(abs(proxy_before_sweep), 1e-12))
            self._cd_log(
                f"  CD sweep {sweep+1}/{self.cd_polish_sweeps}: "
                f"tested={tested_this_sweep} active={active_macros} "
                f"moved={moved_this_sweep} no_improve={no_improve} "
                f"proxy={cur_proxy:.6f} sweep_improve={sweep_rel_improve*100:.3f}%  "
                f"elapsed={time.time()-t0:.1f}s"
            )
            # Early stop: this sweep reduced the proxy by less than the
            # required fraction. Catches the long tail of marginal sweeps
            # (the zero-move patience check below only fires on no moves at all).
            if sweep_rel_improve < self.cd_polish_min_sweep_improve_frac:
                self._cd_log(
                    f"  CD: sweep improvement {sweep_rel_improve*100:.3f}% < "
                    f"{self.cd_polish_min_sweep_improve_frac*100:.3f}% threshold, ending."
                )
                break
            if moved_this_sweep == 0:
                zero_streak += 1
                if zero_streak >= max(1, self.cd_polish_patience):
                    self._cd_log(
                        f"  CD: {zero_streak} consecutive zero-move sweep(s); "
                        f"patience={self.cd_polish_patience} hit, ending."
                    )
                    break
                else:
                    self._cd_log(
                        f"  CD: zero moves this sweep "
                        f"({zero_streak}/{self.cd_polish_patience}); continuing."
                    )
            else:
                zero_streak = 0

        # Return polished placement.
        out = placement.clone()
        out[:nM] = torch.tensor(e.macro_pos, dtype=out.dtype)
        # Also return a dict resembling compute_proxy_cost's output, computed
        # from the IncrementalEval caches we already have populated.
        br = e.proxy_breakdown(include_cong=True)
        costs = {
            'proxy_cost':      br['proxy_cost'],
            'wirelength_cost': br['wirelength_cost'],
            'density_cost':    br['density_cost'],
            'congestion_cost': br['congestion_cost'],
        }
        self._cd_log(
            f"[v60 {benchmark.name}] CD polish done: "
            f"total_moved={total_moved} proxy={cur_proxy:.6f}  "
            f"wl={costs['wirelength_cost']:.3f} den={costs['density_cost']:.3f} "
            f"cong={costs['congestion_cost']:.3f}  elapsed={time.time()-t0:.1f}s"
        )
        return out, costs

    def _pair_swap_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc: PlacementCost,
    ):
        """Pair-swap coordinate descent over hard macros.

        For each movable hard macro m_i, consider swapping positions with
        each of its k-nearest hard-macro neighbors. The cheap filter is
        the sum of the two single-macro WL+density deltas — approximate,
        since it ignores the interaction on nets that touch BOTH macros
        (their shared-net bbox changes are double-counted, not composed),
        but cheap enough to use as a candidate ranker. The top-K cheap
        candidates are then fully re-evaluated against the real proxy
        (commit both moves, eval, revert both moves), and the best
        improving swap (if any) is committed.

        Why this complements single-macro CD: single-macro CD misses
        configurations where macro A is at the spot B 'wants' and vice
        versa — neither move alone is an improvement, but swapping is.
        Restricting to hard-hard swaps keeps the legality check tractable:
        the swap is gated by AABB checks ensuring each macro at its new
        position doesn't collide with any OTHER hard macro (the swap
        partner is excluded from the check since it has moved out of the
        way).

        Sweeps stop after `pair_swap_patience` consecutive zero-swap
        sweeps, or after `pair_swap_sweeps` complete. Within a sweep, a
        macro that has already participated in a committed swap is
        skipped for the rest of that sweep.
        """
        nM = int(benchmark.num_macros)
        nH = int(benchmark.num_hard_macros)
        if nH < 2 or not self.pair_swap_enabled:
            return placement, None

        movable = benchmark.get_movable_mask().cpu().numpy().astype(bool)
        target_idx = np.where(movable[:nH])[0]
        if target_idx.size < 2:
            return placement, None

        # Build IncrementalEval from current placement.
        e = IncrementalEval(benchmark, plc=plc)
        base_pos = placement[:nM].detach().cpu().numpy().astype(np.float64)
        e.set_placement(base_pos)
        cur_proxy = float(e.proxy(include_cong=True))

        self._swap_log(
            f"[v60 {benchmark.name}] pair-swap start: proxy={cur_proxy:.6f}  "
            f"nT={int(target_idx.size)}  sweeps={self.pair_swap_sweeps}  "
            f"k_neighbors={self.pair_swap_k_neighbors}  k_co_net={self.pair_swap_k_co_net}  "
            f"multi_swap=on"
        )
        t0 = time.time()

        # Hard-macro AABB cache (xlo, xhi, ylo, yhi).
        hard_aabb = np.zeros((nH, 4), dtype=np.float64)
        for m in range(nH):
            cx = float(e.macro_pos[m, 0]); cy = float(e.macro_pos[m, 1])
            hw = float(e.macro_w[m]) * 0.5; hh = float(e.macro_h[m]) * 0.5
            hard_aabb[m, 0] = cx - hw
            hard_aabb[m, 1] = cx + hw
            hard_aabb[m, 2] = cy - hh
            hard_aabb[m, 3] = cy + hh

        # Co-net partner counts: co_net_count[i, j] is the number of nets
        # whose pin set covers both hard macros i and j. Used as a second
        # partner-selection axis alongside spatial proximity — many pairs
        # that are net-coupled are spatially distant and would never be
        # considered by k-nearest alone.
        co_net_count = None
        if self.pair_swap_k_co_net > 0:
            co_net_count = np.zeros((nH, nH), dtype=np.int32)
            pin_owner = e.pin_owner
            for pin_idxs in e.net_pins:
                pin_idxs = np.asarray(pin_idxs)
                if pin_idxs.size < 2:
                    continue
                owners = pin_owner[pin_idxs]
                hard_on_net = np.unique(owners[owners < nH])
                if hard_on_net.size < 2:
                    continue
                for ii in range(hard_on_net.size):
                    for jj in range(ii + 1, hard_on_net.size):
                        i = int(hard_on_net[ii]); j = int(hard_on_net[jj])
                        co_net_count[i, j] += 1
                        co_net_count[j, i] += 1
            nonzero_pairs = int(np.count_nonzero(co_net_count) // 2)
            self._swap_log(
                f"  pair-swap co-net precomp: {nonzero_pairs} hard-hard pairs "
                f"share ≥1 net (k_co_net={self.pair_swap_k_co_net})"
            )
        all_others_base = np.array(
            [j for j in target_idx], dtype=int,
        )

        rng = np.random.default_rng(self.seed if self.deterministic else None)
        total_swaps = 0
        zero_streak = 0

        for sweep in range(self.pair_swap_sweeps):
            order = target_idx[rng.permutation(len(target_idx))]
            swaps_this_sweep = 0
            cands_evaluated  = 0
            aabb_pass        = 0
            full_evals       = 0

            for m_i in order:
                m_i = int(m_i)
                pos_i = e.macro_pos[m_i].copy()
                hw_i = float(e.macro_w[m_i]) * 0.5
                hh_i = float(e.macro_h[m_i]) * 0.5

                # Partner pool: union of (a) k-nearest spatial neighbors and
                # (b) top-K macros by shared-net count. The once-per-sweep
                # lockout is intentionally absent — a macro can participate
                # in multiple committed swaps per sweep, with min_improve > 0
                # preventing oscillation.
                avail = all_others_base[all_others_base != m_i]
                if avail.size == 0:
                    continue
                other_pos = e.macro_pos[avail]
                dists = np.linalg.norm(other_pos - pos_i, axis=1)
                k = min(self.pair_swap_k_neighbors, avail.size)
                if k < avail.size:
                    spatial = avail[np.argpartition(dists, k - 1)[:k]]
                else:
                    spatial = avail
                if co_net_count is not None and self.pair_swap_k_co_net > 0:
                    row = co_net_count[m_i]
                    if row.max() > 0:
                        K_co = min(self.pair_swap_k_co_net, nH - 1)
                        top_idx = np.argpartition(row, -K_co)[-K_co:]
                        co_partners = top_idx[row[top_idx] > 0]
                        nearest = np.unique(np.concatenate([spatial, co_partners]))
                        nearest = nearest[nearest != m_i]
                    else:
                        nearest = spatial
                else:
                    nearest = spatial

                # Collect AABB-legal swap candidates. The WL+density cheap
                # filter was dropped after empirical evidence on ibm06: cong
                # is the dominant cost term, and cong-improving swaps with
                # neutral-or-positive WL+density were being rejected before
                # full-eval. We now full-eval every legal swap with one of
                # the k-nearest hard macros.
                cands = []  # list of (m_j, pos_j)
                for m_j in nearest:
                    m_j = int(m_j)
                    pos_j = e.macro_pos[m_j].copy()
                    hw_j = float(e.macro_w[m_j]) * 0.5
                    hh_j = float(e.macro_h[m_j]) * 0.5
                    cands_evaluated += 1

                    # AABB legality: i at pos_j and j at pos_i must not
                    # overlap any OTHER hard macro. Exclude {i, j} since
                    # they swap out of each other's way.
                    xlo_i = pos_j[0] - hw_i; xhi_i = pos_j[0] + hw_i
                    ylo_i = pos_j[1] - hh_i; yhi_i = pos_j[1] + hh_i
                    no_ov_i = (
                        (xhi_i <= hard_aabb[:, 0]) |
                        (xlo_i >= hard_aabb[:, 1]) |
                        (yhi_i <= hard_aabb[:, 2]) |
                        (ylo_i >= hard_aabb[:, 3])
                    )
                    no_ov_i[m_i] = True; no_ov_i[m_j] = True
                    if not np.all(no_ov_i):
                        continue
                    xlo_j = pos_i[0] - hw_j; xhi_j = pos_i[0] + hw_j
                    ylo_j = pos_i[1] - hh_j; yhi_j = pos_i[1] + hh_j
                    no_ov_j = (
                        (xhi_j <= hard_aabb[:, 0]) |
                        (xlo_j >= hard_aabb[:, 1]) |
                        (yhi_j <= hard_aabb[:, 2]) |
                        (ylo_j >= hard_aabb[:, 3])
                    )
                    no_ov_j[m_i] = True; no_ov_j[m_j] = True
                    if not np.all(no_ov_j):
                        continue
                    # The pair themselves can't overlap each other at their
                    # new positions either.
                    pair_ok = (
                        (xhi_i <= xlo_j) or (xlo_i >= xhi_j) or
                        (yhi_i <= ylo_j) or (ylo_i >= yhi_j)
                    )
                    if not pair_ok:
                        continue

                    cands.append((m_j, pos_j))

                if not cands:
                    continue
                aabb_pass += 1

                # Full real-proxy evaluation of each AABB-legal swap. Commit
                # both moves, evaluate, then revert both — order of revert
                # doesn't matter for final state since the moves are disjoint.
                best_new_proxy = cur_proxy
                best_j = None
                best_pos_j = None
                for (m_j, pos_j) in cands:
                    full_evals += 1
                    pj = (float(pos_j[0]), float(pos_j[1]))
                    pi = (float(pos_i[0]), float(pos_i[1]))
                    st_i = e.delta_for_move(m_i, pj, include_cong=False)
                    e.commit_move(m_i, pj, st_i)
                    st_j = e.delta_for_move(m_j, pi, include_cong=False)
                    e.commit_move(m_j, pi, st_j)
                    new_proxy = float(e.proxy(include_cong=True))
                    if new_proxy < best_new_proxy:
                        best_new_proxy = new_proxy
                        best_j = m_j
                        best_pos_j = pos_j
                    # Revert.
                    rev_j = e.delta_for_move(m_j, pj, include_cong=False)
                    e.commit_move(m_j, pj, rev_j)
                    rev_i = e.delta_for_move(m_i, pi, include_cong=False)
                    e.commit_move(m_i, pi, rev_i)

                if best_j is not None and \
                        (cur_proxy - best_new_proxy) >= self.pair_swap_min_improve:
                    pj = (float(best_pos_j[0]), float(best_pos_j[1]))
                    pi = (float(pos_i[0]), float(pos_i[1]))
                    st_i = e.delta_for_move(m_i, pj, include_cong=False)
                    e.commit_move(m_i, pj, st_i)
                    st_j = e.delta_for_move(best_j, pi, include_cong=False)
                    e.commit_move(best_j, pi, st_j)
                    cur_proxy = best_new_proxy
                    swaps_this_sweep += 1
                    # Refresh AABB for both swapped macros.
                    hw_b = float(e.macro_w[best_j]) * 0.5
                    hh_b = float(e.macro_h[best_j]) * 0.5
                    hard_aabb[m_i, 0] = pj[0] - hw_i
                    hard_aabb[m_i, 1] = pj[0] + hw_i
                    hard_aabb[m_i, 2] = pj[1] - hh_i
                    hard_aabb[m_i, 3] = pj[1] + hh_i
                    hard_aabb[best_j, 0] = pi[0] - hw_b
                    hard_aabb[best_j, 1] = pi[0] + hw_b
                    hard_aabb[best_j, 2] = pi[1] - hh_b
                    hard_aabb[best_j, 3] = pi[1] + hh_b

            total_swaps += swaps_this_sweep
            self._swap_log(
                f"  pair-swap sweep {sweep+1}/{self.pair_swap_sweeps}: "
                f"cands={cands_evaluated} aabb_pass={aabb_pass} "
                f"full_evals={full_evals} swaps={swaps_this_sweep} "
                f"proxy={cur_proxy:.6f}  elapsed={time.time()-t0:.1f}s"
            )
            if swaps_this_sweep == 0:
                zero_streak += 1
                if zero_streak >= max(1, self.pair_swap_patience):
                    self._swap_log(
                        f"  pair-swap: {zero_streak} consecutive zero-swap sweep(s); "
                        f"patience={self.pair_swap_patience} hit, ending."
                    )
                    break
            else:
                zero_streak = 0

        out = placement.clone()
        out[:nM] = torch.tensor(e.macro_pos, dtype=out.dtype)
        br = e.proxy_breakdown(include_cong=True)
        costs = {
            'proxy_cost':      br['proxy_cost'],
            'wirelength_cost': br['wirelength_cost'],
            'density_cost':    br['density_cost'],
            'congestion_cost': br['congestion_cost'],
        }
        self._swap_log(
            f"[v60 {benchmark.name}] pair-swap done: total_swaps={total_swaps} "
            f"proxy={cur_proxy:.6f}  "
            f"wl={costs['wirelength_cost']:.3f} den={costs['density_cost']:.3f} "
            f"cong={costs['congestion_cost']:.3f}  elapsed={time.time()-t0:.1f}s"
        )
        return out, costs

    def _pair_swap_polish_soft(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc: PlacementCost,
    ):
        """Pair-swap coordinate descent over SOFT macros.

        Mirrors `_pair_swap_polish` for hard macros but with two key
        differences:
          1. No AABB legality check. Soft macros are cluster proxies that
             legally overlap in v60's representation; any swap is
             geometrically valid.
          2. Co-net partners only — spatial k-nearest scales poorly to
             the thousands of soft macros typical of these benchmarks.
             For each source soft macro i, partners are the top-K macros
             (soft or hard) that share the most nets with i.
          3. No cheap WL+density pre-filter: every AABB-legal partner is
             full-eval'd on the real proxy. This catches cong-affecting
             swaps that a WL+density filter would reject (most swaps
             don't move cong much, but the ones that do are exactly the
             ones we don't want to throw away on a cheap heuristic).
             Cost grows linearly with k_co_net, so keep that small.

        Multi-swap-per-sweep is on; min_improve > 0 blocks oscillation.
        """
        nM = int(benchmark.num_macros)
        nH = int(benchmark.num_hard_macros)
        nS = nM - nH
        if nS < 2 or not self.soft_pair_swap_enabled:
            return placement, None

        movable = benchmark.get_movable_mask().cpu().numpy().astype(bool)
        target_idx = np.where(movable[:nM] & (np.arange(nM) >= nH))[0]
        if target_idx.size < 2:
            return placement, None

        e = IncrementalEval(benchmark, plc=plc)
        base_pos = placement[:nM].detach().cpu().numpy().astype(np.float64)
        e.set_placement(base_pos)
        cur_proxy = float(e.proxy(include_cong=True))

        self._soft_swap_log(
            f"[v60 {benchmark.name}] soft-pair-swap start: proxy={cur_proxy:.6f}  "
            f"nT={int(target_idx.size)}  sweeps={self.soft_pair_swap_sweeps}  "
            f"k_co_net={self.soft_pair_swap_k_co_net}  multi_swap=on  filter=real-proxy"
        )
        t0 = time.time()

        # Co-net counts over ALL macros (soft can be net-coupled to hard or
        # other soft; partners can be either). [nM, nM] symmetric.
        co_net_count = np.zeros((nM, nM), dtype=np.int32)
        pin_owner = e.pin_owner
        for pin_idxs in e.net_pins:
            pin_idxs = np.asarray(pin_idxs)
            if pin_idxs.size < 2:
                continue
            owners = pin_owner[pin_idxs]
            macros_on_net = np.unique(owners[owners < nM])
            if macros_on_net.size < 2:
                continue
            for ii in range(macros_on_net.size):
                for jj in range(ii + 1, macros_on_net.size):
                    a = int(macros_on_net[ii]); b = int(macros_on_net[jj])
                    co_net_count[a, b] += 1
                    co_net_count[b, a] += 1
        nonzero_pairs = int(np.count_nonzero(co_net_count) // 2)
        self._soft_swap_log(
            f"  soft-pair-swap co-net precomp: {nonzero_pairs} macro-macro pairs "
            f"share ≥1 net"
        )

        # Hard-macro AABB cache — needed for safety check when a soft source's
        # partner is a hard macro. The hard macro must not overlap any OTHER
        # hard macro at its new (= soft source's old) position. Soft source
        # at hard's old position is free since soft has no overlap restriction.
        hard_aabb = np.zeros((nH, 4), dtype=np.float64) if nH > 0 else None
        if hard_aabb is not None:
            for m in range(nH):
                cx = float(e.macro_pos[m, 0]); cy = float(e.macro_pos[m, 1])
                hw = float(e.macro_w[m]) * 0.5; hh = float(e.macro_h[m]) * 0.5
                hard_aabb[m, 0] = cx - hw
                hard_aabb[m, 1] = cx + hw
                hard_aabb[m, 2] = cy - hh
                hard_aabb[m, 3] = cy + hh

        K_co  = self.soft_pair_swap_k_co_net
        rng   = np.random.default_rng(self.seed if self.deterministic else None)
        total_swaps = 0
        zero_streak = 0

        for sweep in range(self.soft_pair_swap_sweeps):
            order = target_idx[rng.permutation(len(target_idx))]
            swaps_this_sweep = 0
            cands_evaluated  = 0
            aabb_pass        = 0
            full_evals       = 0

            for m_i in order:
                m_i = int(m_i)
                pos_i = e.macro_pos[m_i].copy()

                # Top-K co-net partners (any macro, soft or hard).
                row = co_net_count[m_i]
                if row.max() == 0:
                    continue
                K = min(K_co, nM - 1)
                top = np.argpartition(row, -K)[-K:]
                partners = top[row[top] > 0]
                partners = partners[partners != m_i]
                if partners.size == 0:
                    continue

                # Collect AABB-legal partners (no cheap filter — every legal
                # candidate is full-eval'd below).
                cands = []  # (m_j, pos_j, is_hard_j)
                for m_j in partners:
                    m_j = int(m_j)
                    pos_j = e.macro_pos[m_j].copy()
                    is_hard_j = (m_j < nH)
                    cands_evaluated += 1

                    # If partner is hard, AABB-check the hard macro at the
                    # soft source's position against every other hard macro.
                    # The soft at the hard's old position is unconstrained.
                    if is_hard_j and hard_aabb is not None:
                        hw_j = float(e.macro_w[m_j]) * 0.5
                        hh_j = float(e.macro_h[m_j]) * 0.5
                        xlo = pos_i[0] - hw_j; xhi = pos_i[0] + hw_j
                        ylo = pos_i[1] - hh_j; yhi = pos_i[1] + hh_j
                        no_ov = (
                            (xhi <= hard_aabb[:, 0]) |
                            (xlo >= hard_aabb[:, 1]) |
                            (yhi <= hard_aabb[:, 2]) |
                            (ylo >= hard_aabb[:, 3])
                        )
                        no_ov[m_j] = True  # self
                        if not np.all(no_ov):
                            continue

                    cands.append((m_j, pos_j, is_hard_j))

                if not cands:
                    continue
                aabb_pass += 1

                # Full real-proxy eval of every AABB-legal partner.
                best_new_proxy = cur_proxy
                best = None  # (m_j, pos_j, is_hard_j)
                for (m_j, pos_j, is_hard_j) in cands:
                    full_evals += 1
                    pj = (float(pos_j[0]), float(pos_j[1]))
                    pi = (float(pos_i[0]), float(pos_i[1]))
                    st_i = e.delta_for_move(m_i, pj, include_cong=False)
                    e.commit_move(m_i, pj, st_i)
                    st_j = e.delta_for_move(m_j, pi, include_cong=False)
                    e.commit_move(m_j, pi, st_j)
                    new_proxy = float(e.proxy(include_cong=True))
                    if new_proxy < best_new_proxy:
                        best_new_proxy = new_proxy
                        best = (m_j, pos_j, is_hard_j)
                    rev_j = e.delta_for_move(m_j, pj, include_cong=False)
                    e.commit_move(m_j, pj, rev_j)
                    rev_i = e.delta_for_move(m_i, pi, include_cong=False)
                    e.commit_move(m_i, pi, rev_i)

                if best is not None and \
                        (cur_proxy - best_new_proxy) >= self.soft_pair_swap_min_improve:
                    m_j, pos_j, is_hard_j = best
                    pj = (float(pos_j[0]), float(pos_j[1]))
                    pi = (float(pos_i[0]), float(pos_i[1]))
                    st_i = e.delta_for_move(m_i, pj, include_cong=False)
                    e.commit_move(m_i, pj, st_i)
                    st_j = e.delta_for_move(m_j, pi, include_cong=False)
                    e.commit_move(m_j, pi, st_j)
                    cur_proxy = best_new_proxy
                    swaps_this_sweep += 1
                    # If the partner was a hard macro, refresh its AABB row.
                    if is_hard_j and hard_aabb is not None:
                        hw_j = float(e.macro_w[m_j]) * 0.5
                        hh_j = float(e.macro_h[m_j]) * 0.5
                        hard_aabb[m_j, 0] = pi[0] - hw_j
                        hard_aabb[m_j, 1] = pi[0] + hw_j
                        hard_aabb[m_j, 2] = pi[1] - hh_j
                        hard_aabb[m_j, 3] = pi[1] + hh_j

            total_swaps += swaps_this_sweep
            self._soft_swap_log(
                f"  soft-pair-swap sweep {sweep+1}/{self.soft_pair_swap_sweeps}: "
                f"cands={cands_evaluated} aabb_pass={aabb_pass} "
                f"full_evals={full_evals} swaps={swaps_this_sweep} "
                f"proxy={cur_proxy:.6f}  elapsed={time.time()-t0:.1f}s"
            )
            if swaps_this_sweep == 0:
                zero_streak += 1
                if zero_streak >= max(1, self.soft_pair_swap_patience):
                    self._soft_swap_log(
                        f"  soft-pair-swap: {zero_streak} consecutive zero-swap sweep(s); "
                        f"patience={self.soft_pair_swap_patience} hit, ending."
                    )
                    break
            else:
                zero_streak = 0

        out = placement.clone()
        out[:nM] = torch.tensor(e.macro_pos, dtype=out.dtype)
        br = e.proxy_breakdown(include_cong=True)
        costs = {
            'proxy_cost':      br['proxy_cost'],
            'wirelength_cost': br['wirelength_cost'],
            'density_cost':    br['density_cost'],
            'congestion_cost': br['congestion_cost'],
        }
        self._soft_swap_log(
            f"[v60 {benchmark.name}] soft-pair-swap done: total_swaps={total_swaps} "
            f"proxy={cur_proxy:.6f}  "
            f"wl={costs['wirelength_cost']:.3f} den={costs['density_cost']:.3f} "
            f"cong={costs['congestion_cost']:.3f}  elapsed={time.time()-t0:.1f}s"
        )
        return out, costs


def place(benchmark: Benchmark) -> torch.Tensor:
    return v60_Placer().place(benchmark)
