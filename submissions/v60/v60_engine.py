"""
v60 engine — multi-level analytical placer (K-means on 2D Fiedler embedding)

This is the v60 placement engine, driven by the v60_placer.py orchestrator
(run that, not this file directly).

Hard macros are clustered via K-means on the 2D Fiedler embedding
(eigenvectors 2 and 3 of the normalised Laplacian) — the two strongest
connectivity axes. Cluster shapes are determined by spatial layout in
those two axes: macros that are connectivity-similar in the 1st-2nd
Fiedler components but differ on weaker axes get pulled together.
Stage 0 then refines the cluster centres via mini-analytical placement.

Engine-specific parts:
    _spectral_cluster_macros_fiedler2d, _build_cluster_data_fiedler2d.

Everything else (loss kernels, Stage 0 helpers, Stage 1/2 runner, cong
diagnostic) lives in `v60_kernels`. The class below is fully
standalone — it does not inherit from anything.
"""

import os.path as osp
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

from v60_kernels import (
    _set_deterministic,
    _set_fast_nondeterministic,
    _total_overlap,
    _extract_raw,
    _parse_plc_routing_params,
    _quiet_plc,
    _run_batch,
    _congestion_work_tier,
    _build_macro_macro_adjacency,
    _kmeans,
    _spectral_embed_to_canvas,
    _run_stage0,
    _dump_cong_diagnostic_multi,
)
from v60_incremental_eval import IncrementalEval


# ══════════════════════════════════════════════════════════════════════════════
#  Cohort-specific: Fiedler-2D K-means cluster builder + Stage 0
# ══════════════════════════════════════════════════════════════════════════════

def _spectral_cluster_macros_fiedler2d(A: np.ndarray, K: int, seed: int = 0):
    """
    K-means on the 2D Fiedler embedding (eigenvectors 2 and 3 of the
    normalised Laplacian). Returns (labels[nH], embed_2d[nH, 2]).
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    if K >= n or n < 3:
        return np.arange(n, dtype=np.int64), np.zeros((n, 2), dtype=np.float32)

    deg = A.sum(axis=1) + 1e-9
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    L_norm = np.eye(n, dtype=np.float32) - (D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :])
    L_norm = 0.5 * (L_norm + L_norm.T)
    eigvals, eigvecs = np.linalg.eigh(L_norm)

    embed_2d      = eigvecs[:, 1:3].astype(np.float32)
    norms         = np.linalg.norm(embed_2d, axis=1, keepdims=True) + 1e-9
    embed_2d_norm = embed_2d / norms

    rng = np.random.RandomState(seed)
    labels, _ = _kmeans(embed_2d_norm, K, rng)
    return labels.astype(np.int64), embed_2d


def _build_cluster_data_fiedler2d(benchmark: Benchmark, K: int,
                                    margin_frac: float, seed: int,
                                    stage0_steps: int = 1500,
                                    stage0_lr: float = 1.0,
                                    stage0_lambda_density: float = 200.0,
                                    stage0_lambda_overlap: float = 500.0,
                                    stage0_gamma_start: float = 2.0,
                                    stage0_gamma_end: float = 0.3,
                                    stage0_target_density: float = 0.7):
    """Fiedler-2D K-means clustering + Stage 0 super-macro placement."""
    nH = int(benchmark.num_hard_macros)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    A  = _build_macro_macro_adjacency(benchmark, nH)
    labels, embed_2d = _spectral_cluster_macros_fiedler2d(A, K, seed=seed)
    embed_canvas = _spectral_embed_to_canvas(embed_2d, cw, ch, margin_frac=margin_frac)

    K_eff = int(labels.max()) + 1 if nH > 0 else 0
    init_centers = np.zeros((K_eff, 2), dtype=np.float32)
    for k in range(K_eff):
        members = embed_canvas[labels == k]
        if len(members) > 0:
            init_centers[k] = members.mean(axis=0)
        else:
            init_centers[k] = (cw * 0.5, ch * 0.5)

    centers = _run_stage0(
        benchmark, labels, K_eff, init_centers,
        num_steps=stage0_steps, lr=stage0_lr,
        lambda_density=stage0_lambda_density,
        lambda_overlap=stage0_lambda_overlap,
        gamma_start=stage0_gamma_start,
        gamma_end=stage0_gamma_end,
        target_density=stage0_target_density,
    )
    return {
        'K':              K_eff,
        'labels':         labels,
        'centers_canvas': centers,
        'embed_canvas':   embed_canvas,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Placer
# ══════════════════════════════════════════════════════════════════════════════

class v60_Engine:
    """
    v60 placement engine (multi-level, fully standalone — no inheritance).

    The clustering step uses K-means on the 2D Fiedler embedding.

    Canvas-aware resolvers fire when the corresponding arg is 'auto'
    (default for all five; all anchored at L=23 / ibm01-tuned values, so
    ibm01 behavior is preserved exactly):
        num_steps_s1 ('auto') = 5000 + round(170*max(0,L-23.0))
        num_steps_s2 ('auto') = 5000 + round(100*max(0,L-23)), clamped [5000, ∞)
        lr_s1        ('auto') = 1.15 + 1.65*(1-exp(-(L-23)/4)),  clamped [1.0, 3.0]
        lr_s2        ('auto') = 0.185 * (L/23)^0.5,         clamped [0.18, 0.45]
        gamma_s2_end          = 0.045 fixed (auto formula dropped — see ctor note)
    """

    def __init__(
        self,
        # ── Multi-level config ─────────────────────────────────────────────
        num_clusters         = 'auto',
        cluster_jitter_frac: float = 0.30,
        cluster_margin_frac: float = 0.50,
        cluster_seed: int          = 0,
        # ── Stage 0 ──────────────────────────────────────────────────────────
        stage0_steps:           int   = 1500,
        stage0_lr:              float = 1.0,
        stage0_lambda_density:  float = 200.0,
        stage0_lambda_overlap:  float = 500.0,
        stage0_gamma_start:     float = 2.0,
        stage0_gamma_end:       float = 0.3,
        stage0_target_density:  float = 0.7,
        # ── Stage 1 / Stage 2 (defaults tuned on the ibm01 best) ───────────
        num_steps_s1          = 'auto',
        # Multipliers on the RESOLVED Stage-1 / Stage-2 step counts (after the
        # 'auto' ramp). 1.0 = no change; the placer's mode preset drives these
        # (e.g. flash mode shortens the gradient descents). Applied in
        # _resolve_num_steps_s1/s2, so a fixed num_steps is scaled too.
        num_steps_s1_coeff: float = 1.0,
        num_steps_s2_coeff: float = 1.0,
        lr_s1                 = 'auto',     # rescaled formula, anchored 1.15 at L=23.
        gamma_s1_start: float = 2.149,
        gamma_s1_end: float   = 0.351,
        lambda_cong_s1: float = 8000.0,   # v60 (2026-05-20): sweep top-20 median at ibm01.
        num_steps_s2          = 'auto',     # v60: 5000 baseline + 100/unit-L bump for big canvases.
        lr_s2                 = 'auto',     # v60: rescaled formula, anchored 0.185 at L=23.
        gamma_s2_start: float = 0.498,
        # v60 (2026-05-21): the 0.031*(23/L)^0.5 auto formula was dropped — the
        # sweep showed the gamma_s2_end optimum is non-monotone in L (ibm01
        # ~0.055, ibm02 ~0.008, ibm18 ~0.043), so a monotone formula cannot fit
        # it. Weak knob (|r| <= 0.26); fixed 0.045 serves ibm01 + ibm18. Pass
        # gamma_s2_end='auto' to restore the (dormant) formula.
        gamma_s2_end          = 0.045,
        # v60 (2026-05-21): bumped 547.1 -> 650 toward the ibm02 sweep optimum
        # (~848, the only bench with a real density signal); ibm18's signal is
        # flat so the bump costs nothing there. Weak knob, scattered optimum.
        lambda_density_end: float = 650.0,
        lambda_hs_overlap_start: float = 124.5,
        lambda_hs_overlap_end:   float = 589.5,
        lambda_hh_overlap_start: float = 118.1,
        lambda_hh_overlap_end:   float = 2876.7,
        lambda_ss_overlap_start: float = 10.4,
        lambda_ss_overlap_end:   float = 351.0,
        overlap_ramp_frac: float = 0.6,
        # ── Soft inflation by degree
        soft_inflation_scale: float = 4.0,
        soft_inflation_alpha: float = 0.5,
        soft_inflation_max:   float = 25.0,
        soft_inflation_clamp: float = 0.20,
        # ── Feature toggles
        use_hh_overlap: bool             = True,
        use_ss_overlap: bool             = True,
        use_soft_degree_inflation: bool  = False,
        # v60 (2026-05-20): lambda_cong_s2 set to the sweep top-20 median at
        # ibm01 (~9.9k). The size-scale block lifts it for bigger canvases.
        lambda_cong_s2: float = 10000.0,
        gamma_rudy: float           = 0.1,
        congestion_threshold: float = 0.0,
        cong_eps: float             = 1e-3,
        # ── Congestion proxy knobs.
        # use_cong_v2=True -> faithful star-from-driver port + box-blur (v60
        # default); v2 cong magnitude differs from v1 so lambda_cong_s1/s2 may
        # need re-tuning. cong_sigma_l is v1-only.
        use_cong_v2:     bool  = True,
        cong_snap_sigma: float = 0.22,
        cong_edge_chunk: int   = 65536,  # 8x v60 default (8192). For ibm01 (~12k edges) -> 1 chunk; ibm10/14 still chunk safely.
        cong_eval_interval_s1 = 'auto',
        cong_eval_interval_s2 = 'auto',
        cong_eval_interval_tiers_s1: tuple = (1, 2, 2, 3, 4),
        cong_eval_interval_tiers_s2: tuple = (1, 2, 3, 4, 5),
        cong_sigma_l:  float = 0.41,
        cong_abu_frac: float = 0.029,   # v60: ibm01-tuned (loss-side; metric still uses 0.05).
        lambda_cong_size_scale: bool = True,   # v60: auto-scale lc_s1/lc_s2 by fast-saturating exp in L (sweep-fit 2026-05-20).
        cong_scale:    float = 1.0,
        # ── Jacobi preconditioner
        use_preconditioner: bool = False,
        # ── Stage-2 acceleration (no mid-stage pruning).
        s2_bf16: bool               = True,
        safety_gap: float = 0.001,
        inflation_factor: float = 1.0,    # v60 debug: no hard-macro inflation
        target_density: float = 0.7,
        plc_root: str     = "external/MacroPlacement/Testcases/ICCAD04",
        num_restarts: int = 128,
        device: str       = 'auto',
        verbose: bool     = True,
        deterministic: bool = False,   # production default: keep TF32 / fast kernels on
        seed: int           = 0,
        # ── Cong diagnostic
        dump_diagnostic_path: str = None,
        dump_diagnostic_seeds: int = 3,
        dump_diagnostic_top_k_cells:    int = 30,
        dump_diagnostic_top_k_per_cell: int = 5,
        # ── Stage-2 seed picker overlap tolerance.
        # Threshold under which a seed counts as "legal" for the lowest-proxy-
        # legal pick. Computed as ratio * median(hard_macro_area), with a 1e-9
        # absolute floor. ratio=0.0 (default) → strict 1e-9 behavior. Loosen
        # if you trust the downstream stages (basin-hop, soft polish, CD) to
        # legalize small residual overlaps; e.g. ratio=0.1 gives ~0.1 µm²
        # tolerance on ibm02 (median hard macro ~1 µm²).
        stage2_overlap_tol_ratio: float = 0.0,
    ):
        # Multi-level
        self.num_clusters        = num_clusters
        self.cluster_jitter_frac = float(cluster_jitter_frac)
        self.cluster_margin_frac = float(cluster_margin_frac)
        self.cluster_seed        = int(cluster_seed)
        # Stage 0
        self.stage0_steps           = int(stage0_steps)
        self.stage0_lr              = float(stage0_lr)
        self.stage0_lambda_density  = float(stage0_lambda_density)
        self.stage0_lambda_overlap  = float(stage0_lambda_overlap)
        self.stage0_gamma_start     = float(stage0_gamma_start)
        self.stage0_gamma_end       = float(stage0_gamma_end)
        self.stage0_target_density  = float(stage0_target_density)
        # Stage 1 / Stage 2
        self.num_steps_s1       = num_steps_s1
        self.num_steps_s1_coeff = float(num_steps_s1_coeff)
        self.num_steps_s2_coeff = float(num_steps_s2_coeff)
        self.lr_s1              = lr_s1
        self.gamma_s1_start     = gamma_s1_start
        self.gamma_s1_end       = gamma_s1_end
        self.lambda_cong_s1     = lambda_cong_s1
        self.num_steps_s2       = num_steps_s2
        self.lr_s2              = lr_s2
        self.gamma_s2_start     = gamma_s2_start
        self.gamma_s2_end       = gamma_s2_end
        self.lambda_density_end = lambda_density_end
        self.lambda_hs_overlap_start = lambda_hs_overlap_start
        self.lambda_hs_overlap_end   = lambda_hs_overlap_end
        self.lambda_hh_overlap_start = lambda_hh_overlap_start
        self.lambda_hh_overlap_end   = lambda_hh_overlap_end
        self.lambda_ss_overlap_start = lambda_ss_overlap_start
        self.lambda_ss_overlap_end   = lambda_ss_overlap_end
        self.overlap_ramp_frac       = overlap_ramp_frac
        self.soft_inflation_scale    = soft_inflation_scale
        self.soft_inflation_alpha    = soft_inflation_alpha
        self.soft_inflation_max      = soft_inflation_max
        self.soft_inflation_clamp    = soft_inflation_clamp
        self.use_hh_overlap            = use_hh_overlap
        self.use_ss_overlap            = use_ss_overlap
        self.use_soft_degree_inflation = use_soft_degree_inflation
        self.lambda_cong_s2     = lambda_cong_s2
        self.gamma_rudy           = gamma_rudy
        self.congestion_threshold = congestion_threshold
        self.cong_eps             = cong_eps
        self.use_cong_v2          = bool(use_cong_v2)
        self.cong_snap_sigma      = float(cong_snap_sigma)
        self.cong_edge_chunk      = int(cong_edge_chunk)
        self.cong_eval_interval_s1 = cong_eval_interval_s1
        self.cong_eval_interval_s2 = cong_eval_interval_s2
        self.cong_eval_interval_tiers_s1 = tuple(int(v) for v in cong_eval_interval_tiers_s1)
        self.cong_eval_interval_tiers_s2 = tuple(int(v) for v in cong_eval_interval_tiers_s2)
        if (len(self.cong_eval_interval_tiers_s1) != 5 or
                any(v < 1 for v in self.cong_eval_interval_tiers_s1)):
            raise ValueError('cong_eval_interval_tiers_s1 must contain five positive integers')
        if (len(self.cong_eval_interval_tiers_s2) != 5 or
                any(v < 1 for v in self.cong_eval_interval_tiers_s2)):
            raise ValueError('cong_eval_interval_tiers_s2 must contain five positive integers')
        self.cong_sigma_l         = float(cong_sigma_l)
        self.cong_abu_frac        = float(cong_abu_frac)
        self.lambda_cong_size_scale = bool(lambda_cong_size_scale)
        self.cong_scale           = float(cong_scale)
        self.use_preconditioner   = bool(use_preconditioner)
        self.s2_bf16              = s2_bf16
        self.safety_gap           = safety_gap
        self.inflation_factor     = inflation_factor
        self.target_density       = target_density
        self.plc_root       = plc_root
        self.num_restarts   = num_restarts
        self.device         = device
        self.verbose        = verbose
        self.deterministic  = deterministic
        self.seed           = seed
        # Diagnostic
        self.dump_diagnostic_path           = dump_diagnostic_path
        self.dump_diagnostic_seeds          = int(dump_diagnostic_seeds)
        self.dump_diagnostic_top_k_cells    = int(dump_diagnostic_top_k_cells)
        self.dump_diagnostic_top_k_per_cell = int(dump_diagnostic_top_k_per_cell)
        # Stage-2 picker overlap tolerance
        self.stage2_overlap_tol_ratio = float(stage2_overlap_tol_ratio)

    def _resolve_device(self) -> str:
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device

    def _params_dict(self) -> dict:
        return {
            'num_steps_s1': self.num_steps_s1, 'lr_s1': self.lr_s1,
            'gamma_s1_start': self.gamma_s1_start, 'gamma_s1_end': self.gamma_s1_end,
            'lambda_cong_s1': self.lambda_cong_s1,
            'num_steps_s2': self.num_steps_s2, 'lr_s2': self.lr_s2,
            'gamma_s2_start': self.gamma_s2_start, 'gamma_s2_end': self.gamma_s2_end,
            'lambda_density_end': self.lambda_density_end,
            'lambda_hs_overlap_start': self.lambda_hs_overlap_start,
            'lambda_hs_overlap_end':   self.lambda_hs_overlap_end,
            'lambda_hh_overlap_start': self.lambda_hh_overlap_start if self.use_hh_overlap else 0.0,
            'lambda_hh_overlap_end':   self.lambda_hh_overlap_end   if self.use_hh_overlap else 0.0,
            'lambda_ss_overlap_start': self.lambda_ss_overlap_start if self.use_ss_overlap else 0.0,
            'lambda_ss_overlap_end':   self.lambda_ss_overlap_end   if self.use_ss_overlap else 0.0,
            'overlap_ramp_frac':       self.overlap_ramp_frac,
            'soft_inflation_scale':    self.soft_inflation_scale if self.use_soft_degree_inflation else 0.0,
            'soft_inflation_alpha':    self.soft_inflation_alpha,
            'soft_inflation_max':      self.soft_inflation_max,
            'soft_inflation_clamp':    self.soft_inflation_clamp,
            'lambda_cong_s2': self.lambda_cong_s2,
            'gamma_rudy': self.gamma_rudy, 'congestion_threshold': self.congestion_threshold,
            'cong_eps': self.cong_eps,
            'use_cong_v2':     self.use_cong_v2,
            'cong_snap_sigma': self.cong_snap_sigma,
            'cong_edge_chunk': self.cong_edge_chunk,
            'cong_eval_interval_s1': self.cong_eval_interval_s1,
            'cong_eval_interval_s2': self.cong_eval_interval_s2,
            'cong_sigma_l':  self.cong_sigma_l,
            'cong_abu_frac': self.cong_abu_frac,
            'lambda_cong_size_scale': self.lambda_cong_size_scale,
            'cong_scale':    self.cong_scale,
            'use_preconditioner': self.use_preconditioner,
            's2_bf16': self.s2_bf16,
            'safety_gap': self.safety_gap, 'inflation_factor': self.inflation_factor,
            'target_density': self.target_density,
            'cluster_jitter_frac': self.cluster_jitter_frac,
        }

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _canvas_L(benchmark: Benchmark) -> float:
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        return (cw * cw + ch * ch) ** 0.5 / (2.0 ** 0.5)

    def _apply_num_steps_coeff(self, ns: int, coeff: float, src: str) -> tuple:
        if coeff == 1.0:
            return int(ns), src
        return max(1, int(round(ns * coeff))), f'{src}×{coeff:g}'

    def _resolve_num_steps_s1(self, benchmark: Benchmark) -> tuple:
        if self.num_steps_s1 != 'auto':
            return self._apply_num_steps_coeff(
                int(self.num_steps_s1), self.num_steps_s1_coeff, 'fixed')
        L = self._canvas_L(benchmark)
        # v60 (2026-05-18): previous diagonal scaling was too aggressive on
        # larger IBM canvases. Anchor ibm01 at 5000 steps and ibm18 at ~12000.
        # v60 (2026-05-21): unified onto the canvas L metric (rms of the two
        # side lengths) shared by every other size formula here, so elongated
        # (non-square) canvases scale consistently. The IBM benches are all
        # near-square (L == avg_dim), so this moves no IBM result.
        # Stage-1 ramp slightly increased beyond the pre-2026-06-09 slope.
        ns = 5000 + int(round(170.0 * max(0.0, L - 23.0)))
        return self._apply_num_steps_coeff(
            ns, self.num_steps_s1_coeff, f'auto(L={L:.2f})')

    # v60 (2026-05-15): the lr / gamma resolvers were originally calibrated
    # for a different baseline than the ibm01-tuned fixed defaults; that's
    # why they were left disabled. Rescaled here so each formula returns the
    # current fixed default at L=23 (ibm01) — i.e. ibm01 behavior is
    # preserved exactly when defaults switch to 'auto'. Scaling above L=23
    # follows a power law in (L/23) or (23/L); clamps widened on the bench-
    # side direction so big canvases get the intended scaling instead of
    # saturating immediately.
    def _resolve_lr_s1(self, benchmark: Benchmark) -> tuple:
        if self.lr_s1 != 'auto':
            return float(self.lr_s1), 'fixed'
        L  = self._canvas_L(benchmark)
        # v60 (2026-05-20): the old formula 1.34*(23/L)^0.4 DECREASED lr with
        # canvas size. The ibm01/02/18 Optuna sweep proved this is backwards:
        # the lr_s1 optimum RISES with L (top-20 medians ibm01 L=23 -> ~1.20,
        # ibm02 L=32.5 -> ~2.53, ibm18 L=65 -> ~2.75) and then saturates. With
        # v60's shorter Stage 1 budget a big canvas covers more ground per
        # step, so it needs a higher lr, not lower. Fast-saturating exp with
        # tau=4: no benchmark sits between L=23 and L=32.5, so the curve only
        # needs to fit ibm01 and the near-saturated cluster (L>=32.5).
        lr = 1.15 + 1.65 * (1.0 - np.exp(-max(0.0, L - 23.0) / 4.0))
        lr = max(1.0, min(3.0, lr))
        return lr, f'auto(L={L:.2f})'

    def _resolve_lr_s2(self, benchmark: Benchmark) -> tuple:
        if self.lr_s2 != 'auto':
            return float(self.lr_s2), 'fixed'
        L  = self._canvas_L(benchmark)
        # lr_s2 increases with L (sqrt slope): bigger canvases have farther
        # soft-macro distances to travel in spreading. Anchored at L=23.
        lr = 0.185 * (max(L, 1e-6) / 23.0) ** 0.5
        lr = max(0.18, min(0.45, lr))
        return lr, f'auto(L={L:.2f})'

    def _resolve_gamma_s2_end(self, benchmark: Benchmark) -> tuple:
        if self.gamma_s2_end != 'auto':
            return float(self.gamma_s2_end), 'fixed'
        L = self._canvas_L(benchmark)
        # gamma_s2_end decreases with L (sqrt slope): bigger canvases have
        # more nets to resolve, sharper LSE at end of s2 helps decisive
        # finish. Anchored at L=23.
        v = 0.031 * (23.0 / max(L, 1e-6)) ** 0.5
        v = max(0.014, min(0.05, v))
        return v, f'auto(L={L:.2f})'

    def _resolve_num_steps_s2(self, benchmark: Benchmark) -> tuple:
        if self.num_steps_s2 != 'auto':
            return self._apply_num_steps_coeff(
                int(self.num_steps_s2), self.num_steps_s2_coeff, 'fixed')
        L = self._canvas_L(benchmark)
        # num_steps_s2 increases with L linearly above ibm01 baseline.
        # Modest slope (half the s1 cap-style bump) since s2 is the
        # spreading stage and saturation is documented at 5000 for small
        # benches. Upper cap removed for symmetry with s1 (no current
        # bench hits 11000 anyway — ibm16 at L=81 only reaches 10808).
        # Stage-2 ramp fully restored to the pre-2026-06-09 slope.
        ns = 5000 + int(round(100.0 * max(0.0, L - 23.0)))
        ns = max(5000, ns)
        return self._apply_num_steps_coeff(
            ns, self.num_steps_s2_coeff, f'auto(L={L:.2f})')

    def _resolve_num_clusters(self, benchmark: Benchmark) -> tuple:
        nH = int(benchmark.num_hard_macros)
        if self.num_clusters != 'auto':
            k = max(2, min(int(self.num_clusters), max(2, nH)))
            return k, 'fixed'
        k = max(8, min(64, int(round(nH ** 0.5))))
        k = min(k, max(2, nH))
        return k, f'auto(sqrt(nH={nH}))'

    def _resolve_cong_interval(self, value, benchmark: Benchmark, stage: int) -> tuple:
        """Resolve cong_eval_interval_s{1,2}.

        'auto' selects from the configured five-entry tier table using
        _congestion_work_tier. Values above 1 thin the expensive congestion-
        gradient calls; the ×interval multiplier in _run_batch keeps roughly
        the same integrated congestion force. Final seed ranking still uses compute_proxy_cost.
        """
        if value != 'auto':
            return max(1, int(value)), 'fixed'
        tier = _congestion_work_tier(benchmark)
        intervals = (self.cong_eval_interval_tiers_s1 if stage == 1
                     else self.cong_eval_interval_tiers_s2)
        interval = intervals[tier]
        n = int(benchmark.num_nets)
        cells = int(benchmark.grid_rows) * int(benchmark.grid_cols)
        return interval, f'auto(nets={n}, cells={cells})'

    def place(self, benchmark: Benchmark, init_positions=None) -> torch.Tensor:
        """
        If `init_positions` is provided, Stage 1 is skipped. Otherwise:
        Fiedler-2D K-means cluster the hard macros, run Stage 0 to place
        the K super-macros, init each individual macro at its cluster
        centre + jitter, then run Stage 1 + Stage 2.
        """
        t0 = time.time()
        if self.deterministic:
            _set_deterministic(self.seed)
        else:
            _set_fast_nondeterministic()
        device_str = self._resolve_device()
        raw        = _extract_raw(benchmark)
        params     = self._params_dict()
        params.update(_parse_plc_routing_params(
            osp.join(self.plc_root, benchmark.name, 'initial.plc')
        ))

        ns1_val,   ns1_tag   = self._resolve_num_steps_s1(benchmark)
        ns2_val,   ns2_tag   = self._resolve_num_steps_s2(benchmark)
        lr_s1_val, lr_s1_tag = self._resolve_lr_s1(benchmark)
        lr_s2_val, lr_s2_tag = self._resolve_lr_s2(benchmark)
        gs2e_val,  gs2e_tag  = self._resolve_gamma_s2_end(benchmark)
        ci_s1_val, ci_s1_tag = self._resolve_cong_interval(self.cong_eval_interval_s1, benchmark, stage=1)
        ci_s2_val, ci_s2_tag = self._resolve_cong_interval(self.cong_eval_interval_s2, benchmark, stage=2)
        params['num_steps_s1'] = ns1_val
        params['num_steps_s2'] = ns2_val
        params['lr_s1']        = lr_s1_val
        params['lr_s2']        = lr_s2_val
        params['gamma_s2_end'] = gs2e_val
        params['cong_eval_interval_s1'] = ci_s1_val
        params['cong_eval_interval_s2'] = ci_s2_val

        # ── v60 (2026-05-20): auto-scale congestion lambdas by canvas size.
        #    Re-fit to the ibm01/02/18 Optuna sweep (top-20-median
        #    targets, not just the single winners). The old auto_ns1/5000
        #    scaler only reached ~2.4x on ibm18 and badly under-scaled cong.
        #    Top-20-median per-bench optima:
        #        lc_s1 :  ibm01 ~8.0k   ibm02 ~72k   ibm18 ~77k
        #        lc_s2 :  ibm01 ~9.9k   ibm02 ~66k   ibm18 ~75k
        #    Both rise steeply off ibm01 then saturate by ibm02; lc_s1 needs
        #    more scaling than lc_s2 — a single shared factor cannot fit both.
        #    Fast-saturating exp (tau=4); with bases lc_s1=8000, lc_s2=10000
        #    the scale is 1.0 at L=23 (ibm01 lands on its own top-20 median).
        if self.lambda_cong_size_scale:
            _L = self._canvas_L(benchmark)
            _sat = 1.0 - float(np.exp(-max(0.0, _L - 23.0) / 4.0))
            scale_s1 = 1.0 + 8.625 * _sat  # L=23 ->1.0, ~32.5 ->~8.8, ~65 ->~9.6
            scale_s2 = 1.0 + 6.5   * _sat  # L=23 ->1.0, ~32.5 ->~6.9, ~65 ->~7.5
            params['lambda_cong_s1'] = float(params['lambda_cong_s1']) * scale_s1
            params['lambda_cong_s2'] = float(params['lambda_cong_s2']) * scale_s2
            params['_lambda_cong_scale_s1'] = float(scale_s1)
            params['_lambda_cong_scale_s2'] = float(scale_s2)
            size_scale = scale_s1   # for the log line below
        else:
            size_scale = 1.0
            params['_lambda_cong_scale_s1'] = 1.0
            params['_lambda_cong_scale_s2'] = 1.0

        cluster_data = None
        cluster_tag  = ''
        if init_positions is None:
            K, K_tag = self._resolve_num_clusters(benchmark)
            t_cluster = time.time()
            cluster_data = _build_cluster_data_fiedler2d(
                benchmark, K=K,
                margin_frac=self.cluster_margin_frac,
                seed=self.cluster_seed,
                stage0_steps=self.stage0_steps,
                stage0_lr=self.stage0_lr,
                stage0_lambda_density=self.stage0_lambda_density,
                stage0_lambda_overlap=self.stage0_lambda_overlap,
                stage0_gamma_start=self.stage0_gamma_start,
                stage0_gamma_end=self.stage0_gamma_end,
                stage0_target_density=self.stage0_target_density,
            )
            cluster_tag = (f'fiedler2d K={cluster_data["K"]} ({K_tag})  '
                           f'jitter={self.cluster_jitter_frac:.3f}  '
                           f'margin={self.cluster_margin_frac:.2f}  '
                           f's0={self.stage0_steps}st  lr_s0={self.stage0_lr:.2f}  '
                           f'cluster_t={time.time() - t_cluster:.1f}s')

        nS = raw['nM'] - raw['nH']
        if init_positions is None:
            B = self.num_restarts
            self._log(
                f"[v60_engine {benchmark.name}] {B} restarts  "
                f"device={device_str}  nH={raw['nH']} nS={nS}  "
                f"hh={self.use_hh_overlap} ss={self.use_ss_overlap} deg_infl={self.use_soft_degree_inflation}  "
                f"bf16={self.s2_bf16}  "
                f"ramp_frac={self.overlap_ramp_frac}  "
                f"s1={ns1_val}steps ({ns1_tag})  s2={ns2_val}steps ({ns2_tag})  lc_scale={size_scale:.2f} (lc_s1={params['lambda_cong_s1']:.0f} lc_s2={params['lambda_cong_s2']:.0f})  "
                f"cong_every=({ci_s1_val},{ci_s2_val}) ({ci_s1_tag}; {ci_s2_tag})  "
                f"lr_s1={lr_s1_val:.4f} ({lr_s1_tag})  "
                f"lr_s2={lr_s2_val:.4f} ({lr_s2_tag})  "
                f"γs2e={gs2e_val:.4f} ({gs2e_tag})  "
                f"cluster: {cluster_tag}"
            )
        else:
            B = int(init_positions.shape[0])
            self._log(
                f"[v60_engine {benchmark.name}] place_from_init: {B} restarts (Stage 1 skipped)  "
                f"device={device_str}  nH={raw['nH']} nS={nS}  "
                f"hh={self.use_hh_overlap} ss={self.use_ss_overlap} deg_infl={self.use_soft_degree_inflation}  "
                f"bf16={self.s2_bf16}  "
                f"ramp_frac={self.overlap_ramp_frac}  s2={ns2_val}steps ({ns2_tag})  "
                f"cong_every={ci_s2_val} ({ci_s2_tag})"
            )

        all_pos, all_metrics = _run_batch(
            raw, params, B, device_str,
            init_positions=init_positions,
            cluster_data=cluster_data,
        )
        for m in all_metrics:
            m.setdefault('score_proxy',   float('nan'))
            m.setdefault('score_wl',      float('nan'))
            m.setdefault('score_density', float('nan'))
            m.setdefault('score_cong',    float('nan'))
            m.setdefault('overlap_area',  float('nan'))

        netlist    = osp.join(self.plc_root, benchmark.name, "netlist.pb.txt")
        best_proxy = float('inf')
        best_pos   = all_pos[0]
        best_legal_proxy = float('inf')
        best_legal_pos   = None
        plc        = None

        # Stage-2 picker overlap tolerance: ratio × median(hard_macro_area),
        # floored at 1e-9. ratio=0 (default) → strict legal-pick. Loosening
        # this lets downstream stages (basin-hop, soft polish, CD) start from
        # a lower-proxy near-legal seed and legalize the residual overlap.
        # NB: raw['hw_np'] / hh_np are HALF-widths (size/2), so full area =
        # (2·hw)·(2·hh) = 4·hw·hh.
        if self.stage2_overlap_tol_ratio > 0.0 and raw['nH'] > 0:
            hw = raw['hw_np'][:raw['nH']]
            hh = raw['hh_np'][:raw['nH']]
            hard_areas = 4.0 * hw * hh
            median_hard_area = float(np.median(hard_areas))
            stage2_ovlp_tol = max(1e-9,
                                  self.stage2_overlap_tol_ratio * median_hard_area)
            self._log(
                f"[v60_engine {benchmark.name}] stage2 legal-pick tol="
                f"{stage2_ovlp_tol:.4g} "
                f"(ratio={self.stage2_overlap_tol_ratio} × "
                f"median_hard_area={median_hard_area:.4g})"
            )
        else:
            stage2_ovlp_tol = 1e-9

        # v60: per-seed pool (pos, proxy, overlap) so the orchestrator can pull
        # the top-N seeds — not just the single winner — for multi-candidate
        # post-Stage-2 refinement. Populated in the scoring loop below.
        seed_pool = []

        if osp.exists(netlist):
            init_plc = osp.join(self.plc_root, benchmark.name, "initial.plc")
            with _quiet_plc():
                plc = PlacementCost(netlist)
                if osp.exists(init_plc):
                    plc.restore_placement(init_plc, ifInital=True, ifReadComment=True)

            # Ranking every seed through compute_proxy_cost repeatedly invokes
            # the pure-Python PLC router. On the large IBM designs that costs
            # tens of seconds per seed. IncrementalEval rebuilds the same exact
            # proxy state in under a second after a one-time setup, and matches
            # PLC to float noise (~1e-7), so reuse one scorer across the cohort.
            try:
                seed_eval = IncrementalEval(benchmark, plc=plc)
            except Exception as exc:
                seed_eval = None
                self._log(
                    f"  [seed-rank] IncrementalEval init failed ({exc}); "
                    f"falling back to PLC scoring"
                )

            for i, pos_np in enumerate(all_pos):
                ovlp = _total_overlap(pos_np, raw['nH'],
                                      raw['hw_np'][:raw['nH']], raw['hh_np'][:raw['nH']])
                try:
                    if seed_eval is not None:
                        seed_eval.set_placement(
                            np.asarray(pos_np, dtype=np.float64)
                        )
                        costs = seed_eval.proxy_breakdown(include_cong=True)
                    else:
                        costs = compute_proxy_cost(
                            torch.tensor(pos_np, dtype=torch.float32), benchmark, plc,
                        )
                    proxy = costs['proxy_cost']
                    all_metrics[i].update({
                        'score_proxy':   float(proxy),
                        'score_wl':      float(costs['wirelength_cost']),
                        'score_density': float(costs['density_cost']),
                        'score_cong':    float(costs['congestion_cost']),
                        'overlap_area':  float(ovlp),
                    })
                    self._log(
                        f"  seed {i}: proxy={proxy:.4f}  "
                        f"wl={costs['wirelength_cost']:.3f}  "
                        f"den={costs['density_cost']:.3f}  "
                        f"cong={costs['congestion_cost']:.3f}  "
                        f"ovlp_area={ovlp:.4f}"
                    )
                    seed_pool.append({
                        'pos':          pos_np,
                        'proxy':        float(proxy),
                        'overlap_area': float(ovlp),
                    })
                    if proxy < best_proxy:
                        best_proxy = proxy
                        best_pos   = pos_np
                    # v60: track the best *legal* seed (overlap below tolerance).
                    if ovlp <= stage2_ovlp_tol and proxy < best_legal_proxy:
                        best_legal_proxy = float(proxy)
                        best_legal_pos   = pos_np
                except Exception as exc:
                    self._log(f"  seed {i}: PLC eval failed ({exc})")
                    all_metrics[i]['overlap_area'] = float(ovlp)

        # v60: a Stage-2 seed with no hard-macro overlap is preferred over a
        # lower-proxy illegal seed; fall back to best-proxy only if none legal.
        if best_legal_pos is not None:
            if best_legal_proxy > best_proxy + 1e-9:
                self._log(
                    f"  [legal-pick] best legal proxy={best_legal_proxy:.4f} "
                    f"over lower illegal proxy={best_proxy:.4f}"
                )
            best_proxy = best_legal_proxy
            best_pos   = best_legal_pos

        self._last_run_metrics = all_metrics
        # Expose the seed pool (used by the orchestrator's multi-candidate
        # post-Stage-2 path). Ranked best-first: legal seeds (overlap below
        # the picker tolerance) by proxy, then the rest by proxy.
        seed_pool.sort(key=lambda s: (s['overlap_area'] > stage2_ovlp_tol, s['proxy']))
        self._last_seed_pool = seed_pool

        if self.dump_diagnostic_path and plc is not None:
            try:
                _dump_cong_diagnostic_multi(
                    self.dump_diagnostic_path,
                    raw, params, all_pos,
                    scores=[m.get('score_proxy', float('inf')) for m in all_metrics],
                    benchmark=benchmark, plc=plc,
                    num_seeds=self.dump_diagnostic_seeds,
                    top_k_cells=self.dump_diagnostic_top_k_cells,
                    top_k_per_cell=self.dump_diagnostic_top_k_per_cell,
                )
            except Exception as exc:
                self._log(f"[v60_engine {benchmark.name}] cong diagnostic FAILED: {exc}")

        self._log(
            f"[v60_engine {benchmark.name}] best proxy={best_proxy:.4f}  "
            f"total {time.time()-t0:.1f}s"
        )

        nM  = benchmark.num_macros
        out = benchmark.macro_positions.clone()
        out[:nM] = torch.tensor(best_pos, dtype=out.dtype)
        return out


# ── entry point ───────────────────────────────────────────────────────────────

def place(benchmark: Benchmark) -> torch.Tensor:
    return v60_Engine().place(benchmark)
