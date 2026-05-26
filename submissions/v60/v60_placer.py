"""
Analytical Placer v60 — v59 set plus soft-only polish

This is the file to run. v60_placer.py is the orchestrator: it drives the
v60 engine (v60_engine.py), then a basin-hop wrapper and a soft-only
Stage 2 polish on top.

v60 starts from the v59 single-cohort orchestrator, then freezes hard macros
and re-optimizes only soft macros from the base placement. The engine
clusters hard macros via K-means on the 2D Fiedler embedding.

Wall-time savings vs v57: ~3x for the base run. v60 adds a local
soft-only Stage 2 polish after the base placement.

The engine lives in v60_engine.py.
No v60 module imports across version boundaries.

Usage:
    uv run evaluate submissions/examples/v60_placer.py -b ibm01
    uv run evaluate submissions/examples/v60_placer.py --all
"""

import math
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
    _DiagLogger, _resolve_log_path, _basin_hop, _congestion_work_tier,
    _extract_raw, _parse_plc_routing_params, _run_batch,
)
from v60_incremental_eval import IncrementalEval


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
    v60 orchestrator: runs the v60 engine, then the inherited basin-hop
    wrapper plus a soft-only Stage 2 polish on top.

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
        # ── "Logging final boss". When log_dir is set, a single per-benchmark
        # JSONL is opened by the orchestrator and threaded into the cohort so
        # every Stage 0/1/2 step, per-seed final position, legalization
        # delta and per-seed score lands in one trace. Default OFF.
        log_dir:           str = None,
        log_every_n_steps: int = 50,
        # Per-step per-seed full hard-macro positions in the JSONL trace.
        # Costs ~50MB/run on 24 seeds × 246 macros × 100 sampled steps.
        # The cheap centroid+spread aggregates always log either way.
        log_positions_per_step: bool = False,   # v60 debug
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
        #   - B per hop dropped from inherited 32 to 8; max_hops/final_explore
        #     bumped to use the freed wall-time on more hop attempts.
        basin_hop:                  bool  = True,
        basin_hop_max_hops:         int   = 24,
        basin_hop_final_explore:    int   = 0,
        basin_hop_sigma_set                = (0.015, 0.025, 0.035),  # productive band only
        basin_hop_tabu_eps:         float = 0.01,                    # spatial: mean macro disp / scale
        basin_hop_tabu_proxy_eps:   float = 0.005,                   # cost gate: |Δproxy| (0 to disable)
        basin_hop_improve_quota:    int   = 2,                       # stop after N non-improving hops (0 = full budget)
        # v60: only reductions >= this fraction of the current incumbent count
        # as improvements for the quota / sigma-push logic. Tiny gains are
        # still accepted as the new incumbent but increment no_improve so the
        # loop escalates / ends on time. 0.0 = legacy behaviour.
        basin_hop_min_improve_frac: float = 0.005,                   # 0.5% of current proxy
        basin_hop_stratify:         bool  = False,                   # split B across sigma_set per hop
        basin_hop_restarts:         int   = 8,                       # was 0 (inherit cohort B); now small B + more hops
        # Runtime guard for large/routability-heavy benchmarks. 'auto' keeps
        # ibm01-style behavior but caps expensive Stage-2 reruns on big netlists.
        congestion_runtime_mode: str = 'auto',
        # -- v60 soft-only polish -------------------------------------------------
        soft_polish_enabled: bool = True,
        soft_polish_restarts: int = 32,
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
        cd_polish_step_set: tuple = (0.25, 0.5, 1.0, 2.0),   # multipliers of base step
        cd_polish_top_k: int = 8,                 # full-eval the top-K WL+density candidates per macro
        cd_polish_min_improve: float = 1e-7,      # absolute proxy improvement to accept a move
        cd_polish_patience: int = 2,              # stop after this many consecutive zero-move sweeps
        cd_polish_verbose: bool = True,
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
        self.log_dir                = log_dir
        self.log_every_n_steps      = int(log_every_n_steps)
        self.log_positions_per_step = bool(log_positions_per_step)
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
        self.cd_polish_top_k       = int(cd_polish_top_k)
        self.cd_polish_min_improve = float(cd_polish_min_improve)
        self.cd_polish_patience    = int(cd_polish_patience)
        self.cd_polish_verbose    = bool(cd_polish_verbose)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _soft_log(self, msg):
        if self.soft_polish_verbose:
            self._log(msg)

    def _cd_log(self, msg):
        if self.cd_polish_verbose:
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
        # log_dir intentionally NOT passed — the orchestrator owns one logger
        # for the whole run and threads it into each cohort via place(...,
        # diag_logger=...). log_every_n_steps still goes through so cohorts
        # know the cadence.
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
            log_dir                   = None,
            log_every_n_steps         = self.log_every_n_steps,
            log_positions_per_step    = self.log_positions_per_step,
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

        # ── Diagnostic logger (one file per benchmark per run).
        # When log_dir is None, logger is no-op.
        log_path   = _resolve_log_path(self.log_dir, benchmark.name, tag='v60')
        diag_logger = _DiagLogger(log_path)
        if diag_logger.active:
            self._log(f"[v60 {benchmark.name}] diag log: {log_path}")
            diag_logger.log(
                'run_start',
                benchmark=benchmark.name, device=str(device_str),
                cohort_pick={'tag': 'engine', 'hard_frac': float(hard_frac)},
                num_restarts={'engine': int(self.num_restarts)},
                config={
                    'num_clusters': self.num_clusters,
                    'cluster_jitter_frac': float(self.cluster_jitter_frac),
                    'cluster_margin_frac': float(self.cluster_margin_frac),
                    'cluster_seed':        int(self.cluster_seed),
                    'stage0_steps':        int(self.stage0_steps),
                    'stage0_lr':           float(self.stage0_lr),
                    'stage0_lambda_density': float(self.stage0_lambda_density),
                    'stage0_lambda_overlap': float(self.stage0_lambda_overlap),
                    'stage0_gamma_start':    float(self.stage0_gamma_start),
                    'stage0_gamma_end':      float(self.stage0_gamma_end),
                    'stage0_target_density': float(self.stage0_target_density),
                    'use_hh_overlap':            bool(self.use_hh_overlap),
                    'use_ss_overlap':            bool(self.use_ss_overlap),
                    'use_soft_degree_inflation': bool(self.use_soft_degree_inflation),
                    'deterministic':             bool(self.deterministic),
                    'seed':                      int(self.seed),
                    'log_every_n_steps':         int(self.log_every_n_steps),
                    'log_positions_per_step':    bool(self.log_positions_per_step),
                },
                benchmark_stats={
                    'num_macros':      int(benchmark.num_macros),
                    'num_hard_macros': int(benchmark.num_hard_macros),
                    'num_soft_macros': int(benchmark.num_macros - benchmark.num_hard_macros),
                    'num_nets':        int(len(benchmark.net_nodes)),
                    'canvas_width':    float(benchmark.canvas_width),
                    'canvas_height':   float(benchmark.canvas_height),
                    'grid_rows':       int(benchmark.grid_rows),
                    'grid_cols':       int(benchmark.grid_cols),
                },
            )

        # v60: single engine run.
        cohort_elapsed = {}
        cohorts = {'engine': self._make_cohort(v60_Engine, self.num_restarts)}
        t = time.time()
        pos_pick = cohorts['engine'].place(benchmark, diag_logger=diag_logger)
        cohort_elapsed['engine'] = round(time.time() - t, 3)

        netlist  = osp.join(self.plc_root, benchmark.name, "netlist.pb.txt")
        init_plc = osp.join(self.plc_root, benchmark.name, "initial.plc")

        all_results = [('engine', pos_pick)]

        if not osp.exists(netlist):
            for tag, pos in all_results:
                if pos is not None:
                    self._log(f"[v60 {benchmark.name}] no netlist; returning {tag}")
                    if diag_logger.active:
                        diag_logger.log('run_end', benchmark=benchmark.name,
                                        elapsed_s=round(time.time() - t0, 3),
                                        winner=tag, reason='no_netlist',
                                        cohort_elapsed=cohort_elapsed)
                        diag_logger.close()
                    return pos
            if diag_logger.active:
                diag_logger.log('run_end', benchmark=benchmark.name,
                                elapsed_s=round(time.time() - t0, 3),
                                winner=None, reason='no_results')
                diag_logger.close()
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

        # ── Basin-hopping: perturb the running best and re-minimise (Stage 2
        #    only) with a promising-seed priority queue + visited-basin tabu.
        if (self.basin_hop and best_pos is not None and best_tag in cohorts
                and math.isfinite(best_proxy)):
            winner  = cohorts[best_tag]
            nM      = int(benchmark.num_macros)
            mov_idx = np.where(benchmark.get_movable_mask().numpy())[0]
            scale   = 0.5 * (float(benchmark.canvas_width) + float(benchmark.canvas_height))
            Bhop    = (self.basin_hop_restarts if self.basin_hop_restarts > 0
                       else int(getattr(winner, 'num_restarts', 16)))
            Bhop    = self._cap_for_congestion_runtime(benchmark, Bhop, caps=(5, 6, 7, 8))
            rng     = np.random.default_rng(self.seed if self.deterministic else None)
            try:
                ic      = compute_proxy_cost(best_pos[:nM].to(torch.float32), benchmark, plc)
                init_ov = float(ic.get('total_overlap_area', float('nan')))
            except Exception:
                init_ov = float('nan')
            initial = {
                'pos':          best_pos[:nM].detach().cpu().numpy().astype(np.float32),
                'proxy':        float(best_proxy),
                'overlap_area': init_ov,
            }

            def run_from_init(init_b_nmov_2):
                p = winner.place(benchmark, init_positions=init_b_nmov_2, diag_logger=diag_logger)
                metrics = getattr(winner, '_last_run_metrics', None) or []
                if metrics:
                    # match the cohort placer's best-legal pick: lowest proxy
                    # among overlap-free seeds, else lowest proxy overall
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
                f"[v60 {benchmark.name}] basin-hop: winner={best_tag}  B={Bhop}  "
                f"max_hops={self.basin_hop_max_hops}  final_explore={self.basin_hop_final_explore}  "
                f"sigmas={self.basin_hop_sigma_set}  stratify={self.basin_hop_stratify}  "
                f"tabu_eps={self.basin_hop_tabu_eps}  tabu_proxy_eps={self.basin_hop_tabu_proxy_eps}  "
                f"improve_quota={self.basin_hop_improve_quota}  "
                f"min_improve_frac={self.basin_hop_min_improve_frac}"
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
            if bh['proxy'] < best_proxy - 1e-9:
                self._log(f"[v60 {benchmark.name}] basin-hop improved "
                          f"{best_proxy:.4f} -> {bh['proxy']:.4f}")
                out_t      = benchmark.macro_positions.clone()
                out_t[:nM] = torch.tensor(bh['pos'], dtype=out_t.dtype)
                best_pos   = out_t
                best_proxy = float(bh['proxy'])
                best_tag   = f"{best_tag}+bh"
            else:
                self._log(f"[v60 {benchmark.name}] basin-hop: no improvement "
                          f"over {best_proxy:.4f}")
            if diag_logger.active:
                diag_logger.log('basin_hop_end', benchmark=benchmark.name,
                                winner=best_tag, best_proxy=float(best_proxy),
                                B=int(Bhop), max_hops=int(self.basin_hop_max_hops),
                                final_explore=int(self.basin_hop_final_explore))

        if self.soft_polish_enabled and best_pos is not None and math.isfinite(best_proxy):
            base_costs = compute_proxy_cost(best_pos[:int(benchmark.num_macros)].to(torch.float32), benchmark, plc)
            self._soft_log(
                f"[v60 {benchmark.name}] soft polish start: proxy={base_costs['proxy_cost']:.4f}  "
                f"wl={base_costs['wirelength_cost']:.3f} den={base_costs['density_cost']:.3f} "
                f"cong={base_costs['congestion_cost']:.3f}"
            )
            polished, polish_costs = self._soft_only_polish(best_pos, benchmark, plc)
            polished_proxy = float(polish_costs['proxy_cost'])
            if polished_proxy < best_proxy - 1e-9:
                self._soft_log(
                    f"[v60 {benchmark.name}] soft polish done: proxy "
                    f"{best_proxy:.4f} -> {polished_proxy:.4f}  "
                    f"wl={polish_costs['wirelength_cost']:.3f} den={polish_costs['density_cost']:.3f} "
                    f"cong={polish_costs['congestion_cost']:.3f}  total {time.time()-t0:.1f}s"
                )
                best_pos = polished
                best_proxy = polished_proxy
                best_tag = f"{best_tag}+soft"
            else:
                self._soft_log(
                    f"[v60 {benchmark.name}] soft polish: no improvement over "
                    f"{best_proxy:.4f} (got {polished_proxy:.4f}) — kept pre-polish result  "
                    f"total {time.time()-t0:.1f}s"
                )

        if (self.cd_polish_enabled and best_pos is not None
                and math.isfinite(best_proxy) and plc is not None):
            cd_out, cd_costs = self._cd_polish(best_pos, benchmark, plc)
            if cd_costs is not None:
                cd_proxy = float(cd_costs['proxy_cost'])
                if cd_proxy < best_proxy - 1e-9:
                    self._cd_log(
                        f"[v60 {benchmark.name}] CD polish improved proxy "
                        f"{best_proxy:.4f} -> {cd_proxy:.4f}  total {time.time()-t0:.1f}s"
                    )
                    best_pos   = cd_out
                    best_proxy = cd_proxy
                    best_tag   = f"{best_tag}+cd"
                else:
                    self._cd_log(
                        f"[v60 {benchmark.name}] CD polish: no improvement over "
                        f"{best_proxy:.4f} (got {cd_proxy:.4f}) — kept pre-CD result  "
                        f"total {time.time()-t0:.1f}s"
                    )

        if diag_logger.active:
            diag_logger.log(
                'result',
                benchmark=benchmark.name,
                scores={tag: (None if p == float('inf') else float(p))
                        for tag, p, _, _ in scored},
                winner=best_tag,
                best_proxy=float(best_proxy) if best_proxy != float('inf') else None,
                cohort_elapsed=cohort_elapsed,
            )
            diag_logger.log(
                'run_end',
                benchmark=benchmark.name,
                elapsed_s=round(time.time() - t0, 3),
                winner=best_tag,
                best_proxy=float(best_proxy) if best_proxy != float('inf') else None,
            )
            diag_logger.close()
        return best_pos


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
            'log_every_n_steps': 10**9,
            'log_positions_per_step': False,
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
        B = self._cap_for_congestion_runtime(benchmark, B, caps=(12, 16, 20, 24))
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
            diag_logger=None,
            cohort_tag='soft_polish',
        )

        best_pos = None
        best_costs = None
        best_proxy = float('inf')
        best_legal_pos = None
        best_legal_costs = None
        best_legal_proxy = float('inf')
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
            if ovlp <= 1e-9 and proxy < best_legal_proxy:
                best_legal_proxy = proxy
                best_legal_pos = pos_np
                best_legal_costs = costs

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
        """Single-macro coordinate descent over soft macros using IncrementalEval.

        Strategy per macro:
          1. For each candidate offset (8 directions × cd_polish_step_set),
             compute the WL+density delta via `delta_for_move(include_cong=False)`
             (cheap, ~0.1 ms each).
          2. Take the top-K WL+density-improving candidates (most-negative
             delta_wl + 0.5·delta_density).
          3. For each of those K, tentatively commit, re-evaluate the *real*
             proxy (incremental cong cache, ~4 ms), and revert. Track the
             candidate with the best actual proxy improvement.
          4. If the best real-proxy improvement exceeds `cd_polish_min_improve`,
             commit that candidate; else skip the macro this sweep.

        Why top-K instead of top-1: on cong-sensitive benches, the
        WL+density-best candidate often has bad cong (and gets rejected).
        Meanwhile the #2 or #3 WL+density candidate may be slightly worse
        on WL+density but enough better on cong to be a net improvement.
        Top-K = 8 catches these without exploding the cost — only candidates
        with delta_wlden < 0 are eligible, so weak macros stay cheap.

        Sweeps stop after `cd_polish_patience` consecutive zero-move sweeps,
        or after `cd_polish_sweeps` complete.
        """
        nM = int(benchmark.num_macros)
        nH = int(benchmark.num_hard_macros)
        nS = nM - nH
        if nS <= 0 or not self.cd_polish_enabled:
            return placement, None

        movable = benchmark.get_movable_mask().cpu().numpy().astype(bool)
        soft_idx = np.where(movable[:nM] & (np.arange(nM) >= nH))[0]
        if soft_idx.size == 0:
            return placement, None

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        base_step = self.cd_polish_step_frac * 0.5 * (cw + ch)
        # 8-direction offsets per step multiplier. Diagonals normalized so the
        # diagonal step has the same Euclidean length as an axis step (×1/√2).
        dirs = []
        diag = 1.0 / math.sqrt(2.0)
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1),
                         (-diag, -diag), (-diag, diag),
                         ( diag, -diag), ( diag, diag)]:
            dirs.append((dx, dy))
        step_mults = self.cd_polish_step_set if self.cd_polish_step_set else (1.0,)

        # Build IncrementalEval from current placement.
        e = IncrementalEval(benchmark, plc=plc)
        base_pos = placement[:nM].detach().cpu().numpy().astype(np.float64)
        e.set_placement(base_pos)

        # Initial real proxy from the same caches the CD loop will be reading.
        cur_proxy = float(e.proxy(include_cong=True))
        self._cd_log(
            f"[v60 {benchmark.name}] CD polish start: proxy={cur_proxy:.6f}  "
            f"nS={int(soft_idx.size)}  sweeps={self.cd_polish_sweeps}  "
            f"step_frac={self.cd_polish_step_frac:.4f}  "
            f"step_set={self.cd_polish_step_set}  top_k={self.cd_polish_top_k}"
        )
        t0 = time.time()

        rng = np.random.default_rng(self.seed if self.deterministic else None)
        total_moved = 0
        zero_streak = 0   # consecutive zero-move sweeps for patience-based stop

        for sweep in range(self.cd_polish_sweeps):
            order = soft_idx[rng.permutation(len(soft_idx))]
            moved_this_sweep = 0
            tested_this_sweep = 0
            wl_den_pos = 0    # candidates with delta_wlden <= 0 (worth trying full eval)
            cong_rejected = 0 # full-eval moves that didn't beat min_improve
            for m_i in order:
                m_i = int(m_i)
                cur_x = float(e.macro_pos[m_i, 0])
                cur_y = float(e.macro_pos[m_i, 1])

                # Cheap scan: collect all WL+density-improving candidates.
                # WL weight is 1.0, density weight is 0.5 in the real proxy.
                cands = []  # list of (d_wlden, (new_x, new_y))
                for mult in step_mults:
                    s = base_step * mult
                    for (dx, dy) in dirs:
                        new_x = float(np.clip(cur_x + dx * s, 0.0, cw))
                        new_y = float(np.clip(cur_y + dy * s, 0.0, ch))
                        if new_x == cur_x and new_y == cur_y:
                            continue
                        tested_this_sweep += 1
                        st = e.delta_for_move(m_i, (new_x, new_y), include_cong=False)
                        d_wlden = st['delta_wl'] + 0.5 * st['delta_density']
                        if d_wlden < 0.0:
                            cands.append((d_wlden, (new_x, new_y)))

                if not cands:
                    continue
                wl_den_pos += 1

                # Top-K by WL+density delta, ascending (most negative first).
                cands.sort(key=lambda t: t[0])
                cands = cands[: max(1, self.cd_polish_top_k)]

                # Full-eval each candidate by commit + proxy + revert.
                # Track the one with the best real-proxy improvement.
                best_new_proxy = cur_proxy
                best_new_xy = None
                for (_dw, new_xy) in cands:
                    st = e.delta_for_move(m_i, new_xy, include_cong=False)
                    e.commit_move(m_i, new_xy, st)
                    new_proxy = float(e.proxy(include_cong=True))
                    if new_proxy < best_new_proxy:
                        best_new_proxy = new_proxy
                        best_new_xy = new_xy
                    # Revert to original. cur_x/cur_y were captured before any
                    # commits in this macro's evaluation, so the revert is exact.
                    revert = e.delta_for_move(m_i, (cur_x, cur_y), include_cong=False)
                    e.commit_move(m_i, (cur_x, cur_y), revert)

                if best_new_xy is not None and \
                        (cur_proxy - best_new_proxy) >= self.cd_polish_min_improve:
                    # Commit the best candidate (state recomputed since we
                    # reverted after each eval).
                    final_st = e.delta_for_move(m_i, best_new_xy, include_cong=False)
                    e.commit_move(m_i, best_new_xy, final_st)
                    cur_proxy = best_new_proxy
                    moved_this_sweep += 1
                else:
                    cong_rejected += 1

            total_moved += moved_this_sweep
            self._cd_log(
                f"  CD sweep {sweep+1}/{self.cd_polish_sweeps}: "
                f"tested={tested_this_sweep} wlden_neg={wl_den_pos} "
                f"moved={moved_this_sweep} cong_rej={cong_rejected} "
                f"proxy={cur_proxy:.6f}  elapsed={time.time()-t0:.1f}s"
            )
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


def place(benchmark: Benchmark) -> torch.Tensor:
    return v60_Placer().place(benchmark)
