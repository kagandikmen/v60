"""
v60 shared kernels and helpers.

Hosts the placement primitives shared by the v60 engine and the
orchestrator's soft-polish stage:
loss kernels (WAWL, density, hard-soft / hard-hard / soft-soft overlap,
official L-shape congestion, RUDY congestion), legalization, the bf16
torch.compile wrappers, the per-net Jacobi preconditioner, the Stage 0
super-macro placer, the multi-level helpers (spectral cluster, K-means,
embed-to-canvas), the cong diagnostic dump, and the Stage 1 + Stage 2
batched runner `_run_batch`.

The v60 engine (v60_engine.py) defines its own `_build_cluster_data*`
function (K-means on the 2D Fiedler embedding), imports from this module,
and defines its full class with constructor, resolvers, _params_dict,
and place().

Not directly runnable — import via v60_engine.py / v60_placer.py.
"""

import os
import os.path as osp
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.optim as optim

if osp.dirname(osp.abspath(__file__)) not in sys.path:
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from macro_place.benchmark import Benchmark
from macro_place._plc import PlacementCost
from macro_place.objective import compute_proxy_cost

# Enable TF32 matmul + cuDNN TF32 explicitly. The modern API
# `set_float32_matmul_precision('high')` should be equivalent to setting
# `allow_tf32=True`, but on some PyTorch versions torch.compile's inductor
# backend checks the legacy `allow_tf32` attribute directly and warns
# "TensorFloat32 ... available but not enabled" even with the modern API
# set. Setting both silences the warning and ensures TF32 is actually used
# (1.5-3x speedup on Ampere+ GPUs, ~3 decimal places of precision —
# acceptable for our gradient signal).
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True


def _congestion_work_tier(benchmark) -> int:
    """Routability-runtime tier (0..4) from netlist size × placement grid.

    work = num_nets × grid_cells. Drives the v60 congestion runtime guard:
    a higher tier means sparser congestion-gradient evaluation and smaller
    basin-hop / soft-polish batches. Defined here so the orchestrator and
    the v60 engine share one threshold ladder.
    """
    nets  = int(benchmark.num_nets)
    cells = int(benchmark.grid_rows) * int(benchmark.grid_cols)
    work  = nets * max(1, cells)
    if work >= 160_000_000:
        return 4
    if work >= 80_000_000:
        return 3
    if work >= 30_000_000:
        return 2
    if nets >= 20_000:
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  Determinism toggle
# ══════════════════════════════════════════════════════════════════════════════

def _set_deterministic(seed: int = 0) -> None:
    """
    Pin every RNG and disable non-deterministic CUDA kernels so two runs on
    the same hardware/PyTorch version produce bit-identical placements.

    Costs ~1.3-2x runtime. Idempotent — safe to call multiple times.
    """
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
    """
    Restore the fast math path used for production/eval runs.

    `_set_deterministic()` intentionally disables TF32 and deterministic
    algorithm choices can stick globally inside a long-lived Python process.
    Call this on non-deterministic runs so a prior debug run does not leave
    the process in the slow lane.
    """
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Legalization
# ══════════════════════════════════════════════════════════════════════════════

def _split_push_legalize(pos_np, nH, cw, ch, hwH, hhH, movable_mask, gap=0.001, max_passes=500):
    hx = pos_np[:nH, 0].copy()
    hy = pos_np[:nH, 1].copy()
    eps = 1e-6
    clamp_x = lambda i, v: float(min(max(v, hwH[i]), cw - hwH[i]))
    clamp_y = lambda i, v: float(min(max(v, hhH[i]), ch - hhH[i]))

    for _ in range(max_passes):
        changed = False
        for i in range(nH):
            dx = np.abs(hx - hx[i])
            dy = np.abs(hy - hy[i])
            ox_arr = np.maximum(0.0, (hwH + hwH[i]) - dx)
            oy_arr = np.maximum(0.0, (hhH + hhH[i]) - dy)
            area = ox_arr * oy_arr
            area[i] = 0.0
            if not (area > 0).any():
                continue
            j = int(area.argmax())
            mi, mj = bool(movable_mask[i]), bool(movable_mask[j])
            if not (mi or mj):
                continue
            ox, oy = float(ox_arr[j]), float(oy_arr[j])
            if ox <= oy:
                push = ox + gap + eps
                ci, cj = float(hx[i]), float(hx[j])
                d = 1.0 if ci >= cj else -1.0
                if mi and mj:
                    h = push / 2.0
                    ni = clamp_x(i, ci + d * h); nj = clamp_x(j, cj - d * h)
                    li = abs(ci + d * h - ni);   lj = abs(cj - d * h - nj)
                    if li > 1e-12: nj = clamp_x(j, cj - d * (h + li))
                    if lj > 1e-12: ni = clamp_x(i, ci + d * (h + lj))
                elif mi: ni, nj = clamp_x(i, ci + d * push), cj
                else:    ni, nj = ci, clamp_x(j, cj - d * push)
                if abs(ni - hx[i]) > 1e-12: hx[i] = ni; changed = True
                if abs(nj - hx[j]) > 1e-12: hx[j] = nj; changed = True
            else:
                push = oy + gap + eps
                ci, cj = float(hy[i]), float(hy[j])
                d = 1.0 if ci >= cj else -1.0
                if mi and mj:
                    h = push / 2.0
                    ni = clamp_y(i, ci + d * h); nj = clamp_y(j, cj - d * h)
                    li = abs(ci + d * h - ni);   lj = abs(cj - d * h - nj)
                    if li > 1e-12: nj = clamp_y(j, cj - d * (h + li))
                    if lj > 1e-12: ni = clamp_y(i, ci + d * (h + lj))
                elif mi: ni, nj = clamp_y(i, ci + d * push), cj
                else:    ni, nj = ci, clamp_y(j, cj - d * push)
                if abs(ni - hy[i]) > 1e-12: hy[i] = ni; changed = True
                if abs(nj - hy[j]) > 1e-12: hy[j] = nj; changed = True
        if not changed:
            break

    out = pos_np.copy()
    out[:nH, 0] = hx
    out[:nH, 1] = hy
    return out


def _total_overlap(pos_np, nH, hwH, hhH):
    hx, hy = pos_np[:nH, 0], pos_np[:nH, 1]
    total = 0.0
    for i in range(nH - 1):
        dx = np.abs(hx[i + 1:] - hx[i])
        dy = np.abs(hy[i + 1:] - hy[i])
        ox = np.maximum(0.0, (hwH[i + 1:] + hwH[i]) - dx)
        oy = np.maximum(0.0, (hhH[i + 1:] + hhH[i]) - dy)
        total += float((ox * oy).sum())
    return total


# ══════════════════════════════════════════════════════════════════════════════
#  Batched differentiable losses
# ══════════════════════════════════════════════════════════════════════════════

def _lse_per_net_batch(v, net_ids_t, num_nets, gamma):
    B = v.shape[0]
    vg = v / gamma

    offsets   = torch.arange(B, device=v.device, dtype=torch.long).unsqueeze(1) * num_nets
    net_ids_b = (net_ids_t.unsqueeze(0) + offsets).reshape(-1)
    vg_flat   = vg.reshape(-1)

    net_max_flat = v.new_full((B * num_nets,), -1e30)
    net_max_flat.scatter_reduce_(0, net_ids_b, vg_flat.detach(), reduce='amax', include_self=True)

    shifted      = vg_flat - net_max_flat[net_ids_b]
    sum_exp_flat = v.new_zeros(B * num_nets).scatter_add(0, net_ids_b, torch.exp(shifted))

    return (net_max_flat + torch.log(sum_exp_flat.clamp(min=1e-30))).reshape(B, num_nets)


def _wa_wirelength_batch(pos_batch, node_ids_t, net_ids_t, num_nets, net_weights_t, gamma):
    B        = pos_batch.shape[0]
    xy       = pos_batch[:, node_ids_t, :]
    per_net  = pos_batch.new_zeros(B, num_nets)
    for dim in range(2):
        v       = xy[:, :, dim]
        per_net = per_net + _lse_per_net_batch( v, net_ids_t, num_nets, gamma) * gamma
        per_net = per_net + _lse_per_net_batch(-v, net_ids_t, num_nets, gamma) * gamma
    return (per_net * net_weights_t).sum(dim=1)


def _density_loss_batch(pos_macro_batch, hw_t, hh_t, cw, ch, G_rows, G_cols, target_density=0.7):
    cell_w = cw / G_cols
    cell_h = ch / G_rows
    dev, dtype = pos_macro_batch.device, pos_macro_batch.dtype

    x_lo = (pos_macro_batch[:, :, 0] - hw_t).unsqueeze(2)
    x_hi = (pos_macro_batch[:, :, 0] + hw_t).unsqueeze(2)
    y_lo = (pos_macro_batch[:, :, 1] - hh_t).unsqueeze(2)
    y_hi = (pos_macro_batch[:, :, 1] + hh_t).unsqueeze(2)

    cx_lo = (torch.arange(G_cols, device=dev, dtype=dtype) * cell_w)
    cx_hi = cx_lo + cell_w
    cy_lo = (torch.arange(G_rows, device=dev, dtype=dtype) * cell_h)
    cy_hi = cy_lo + cell_h

    ov_x = torch.clamp(torch.minimum(x_hi, cx_hi) - torch.maximum(x_lo, cx_lo), min=0.0)
    ov_y = torch.clamp(torch.minimum(y_hi, cy_hi) - torch.maximum(y_lo, cy_lo), min=0.0)

    density = torch.bmm(ov_y.transpose(1, 2), ov_x) / (cell_w * cell_h)
    return (torch.clamp(density - target_density, min=0.0) ** 2).sum(dim=(1, 2))


def _hard_soft_overlap_loss_batch(pos_macro_batch, hw_t, hh_t, nH, nM):
    if nH == nM or nH == 0:
        return torch.zeros(pos_macro_batch.shape[0], device=pos_macro_batch.device, dtype=pos_macro_batch.dtype)

    pos_h = pos_macro_batch[:, :nH, :]
    pos_s = pos_macro_batch[:, nH:nM, :]

    hw_h = hw_t[:nH]
    hh_h = hh_t[:nH]
    hw_s = hw_t[nH:nM]
    hh_s = hh_t[nH:nM]

    dx = torch.abs(pos_s[:, :, 0].unsqueeze(2) - pos_h[:, :, 0].unsqueeze(1))
    dy = torch.abs(pos_s[:, :, 1].unsqueeze(2) - pos_h[:, :, 1].unsqueeze(1))

    sum_hw = hw_s.unsqueeze(1) + hw_h.unsqueeze(0)
    sum_hh = hh_s.unsqueeze(1) + hh_h.unsqueeze(0)

    ox = torch.clamp(sum_hw.unsqueeze(0) - dx, min=0.0)
    oy = torch.clamp(sum_hh.unsqueeze(0) - dy, min=0.0)

    return (ox * oy).sum(dim=(1, 2))


def _pairwise_overlap_loss_batch(pos_sub_batch, hw_sub, hh_sub, weight_pair=None):
    """
    Pairwise overlap area within a single macro group (hard-hard or soft-soft).
    pos_sub_batch: (B, n, 2). hw_sub/hh_sub: (n,). weight_pair: optional (n, n).
    Returns (B,). Uses upper triangle (i<j) to avoid double-counting and self.
    """
    B, n, _ = pos_sub_batch.shape
    if n < 2:
        return torch.zeros(B, device=pos_sub_batch.device, dtype=pos_sub_batch.dtype)

    dx = torch.abs(pos_sub_batch[:, :, 0].unsqueeze(2) - pos_sub_batch[:, :, 0].unsqueeze(1))
    dy = torch.abs(pos_sub_batch[:, :, 1].unsqueeze(2) - pos_sub_batch[:, :, 1].unsqueeze(1))

    sum_hw = hw_sub.unsqueeze(1) + hw_sub.unsqueeze(0)
    sum_hh = hh_sub.unsqueeze(1) + hh_sub.unsqueeze(0)

    ox = torch.clamp(sum_hw.unsqueeze(0) - dx, min=0.0)
    oy = torch.clamp(sum_hh.unsqueeze(0) - dy, min=0.0)
    area = ox * oy

    triu_mask = torch.triu(
        torch.ones(n, n, device=pos_sub_batch.device, dtype=torch.bool),
        diagonal=1,
    )
    if weight_pair is not None:
        area = area * weight_pair.unsqueeze(0)

    return (area * triu_mask.unsqueeze(0)).sum(dim=(1, 2))


def _hard_hard_overlap_loss_batch(pos_macro_batch, hw_t, hh_t, nH):
    if nH < 2:
        return torch.zeros(pos_macro_batch.shape[0], device=pos_macro_batch.device, dtype=pos_macro_batch.dtype)
    pos_h = pos_macro_batch[:, :nH, :]
    return _pairwise_overlap_loss_batch(pos_h, hw_t[:nH], hh_t[:nH])


def _soft_soft_overlap_loss_batch(pos_macro_batch, hw_t, hh_t, nH, nM, weight_pair=None):
    nS = nM - nH
    if nS < 2:
        return torch.zeros(pos_macro_batch.shape[0], device=pos_macro_batch.device, dtype=pos_macro_batch.dtype)
    pos_s = pos_macro_batch[:, nH:nM, :]
    return _pairwise_overlap_loss_batch(pos_s, hw_t[nH:nM], hh_t[nH:nM], weight_pair)


def _parse_plc_routing_params(initial_plc_path):
    """
    Parse routing-related fields from initial.plc header. Returns a dict with
    keys hroutes_per_micron, vroutes_per_micron, hrouting_alloc, vrouting_alloc,
    smooth_range. Falls back to zeros if a field is missing.
    """
    out = {
        'hroutes_per_micron': 0.0,
        'vroutes_per_micron': 0.0,
        'hrouting_alloc':     0.0,
        'vrouting_alloc':     0.0,
        'smooth_range':       0,
    }
    try:
        with open(initial_plc_path, 'r') as fh:
            for line in fh:
                if not line.startswith('#'):
                    break
                low = line.lower()
                if 'routes per micron' in low:
                    parts = line.replace(',', ' ').split()
                    for tag, key in (('hor', 'hroutes_per_micron'),
                                     ('ver', 'vroutes_per_micron')):
                        if tag in parts:
                            i = parts.index(tag)
                            for p in parts[i+1:]:
                                if p == ':' or p == '':
                                    continue
                                try:
                                    out[key] = float(p)
                                    break
                                except ValueError:
                                    continue
                elif 'routes used by macros' in low:
                    parts = line.replace(',', ' ').split()
                    for tag, key in (('hor', 'hrouting_alloc'),
                                     ('ver', 'vrouting_alloc')):
                        if tag in parts:
                            i = parts.index(tag)
                            for p in parts[i+1:]:
                                if p == ':' or p == '':
                                    continue
                                try:
                                    out[key] = float(p)
                                    break
                                except ValueError:
                                    continue
                elif 'smoothing factor' in low:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            out['smooth_range'] = int(float(parts[-1].strip()))
                        except ValueError:
                            pass
    except FileNotFoundError:
        pass
    return out


def _official_congestion_batch(
    pos_batch, node_ids_t, net_ids_t, num_nets, net_weights_t,
    cw, ch, G_rows, G_cols, gamma,
    hw_t, hh_t, nH,
    hroutes_per_micron, vroutes_per_micron,
    hrouting_alloc, vrouting_alloc,
    abu_frac: float = 0.05, eps: float = 1e-3,
    sigma_l: float = 1.0, cong_scale: float = 1.0,
):
    """
    Differentiable approximation of plc_client_os.get_congestion_cost():

      * Per-net V/H demand from an L-shape routing approximation. Each
        net's H demand is concentrated on a single (soft) row centred at
        the bbox y-midpoint, then spread along bbox_w via the smooth
        column-overlap term. V demand is concentrated on a single (soft)
        column centred at the bbox x-midpoint, spread along bbox_h. The
        row/col indicators are normalised soft Gaussians with sigma=1
        cell, keeping the gradient flowing through pin positions.
      * Hard-macro V/H blockage: each hard macro contributes its area
        coverage (m_ov_x * m_ov_y) divided by the perpendicular cell
        extent, weighted by vrouting_alloc / hrouting_alloc. This is the
        term RUDY missed entirely.
      * Concat V_total + H_total per cell and return the top-abu_frac
        mean (the official aggregator is ABU at frac=0.05).

    Two simplifications vs the official remain: no linear smoothing pass;
    soft Gaussian indicator instead of an exact L/T branching choice.
    Returns [B].
    """
    B = pos_batch.shape[0]
    cell_w = cw / G_cols
    cell_h = ch / G_rows
    dev, dtype = pos_batch.device, pos_batch.dtype
    grid_v_routes = cell_w * vroutes_per_micron
    grid_h_routes = cell_h * hroutes_per_micron
    # sigma_l (cell units) controls H-row / V-col soft pulse width.
    # Smaller sigma_l = sharper indicator = closer to discrete L-shape routing.
    # Default 1.0 was too smooth — diagnostic showed proxy underreports cong
    # by ~40% vs. real ABU. Try 0.3-0.5 for sharper proxy.

    xy = pos_batch[:, node_ids_t, :]
    x, y = xy[:, :, 0], xy[:, :, 1]
    lse_px = _lse_per_net_batch( x, net_ids_t, num_nets, gamma)
    lse_nx = _lse_per_net_batch(-x, net_ids_t, num_nets, gamma)
    lse_py = _lse_per_net_batch( y, net_ids_t, num_nets, gamma)
    lse_ny = _lse_per_net_batch(-y, net_ids_t, num_nets, gamma)
    wa_x_max =  lse_px * gamma
    wa_x_min = -lse_nx * gamma
    wa_y_max =  lse_py * gamma
    wa_y_min = -lse_ny * gamma

    cx_lo = torch.arange(G_cols, device=dev, dtype=dtype) * cell_w
    cx_hi = cx_lo + cell_w
    cy_lo = torch.arange(G_rows, device=dev, dtype=dtype) * cell_h
    cy_hi = cy_lo + cell_h
    cx_mid = cx_lo + cell_w * 0.5
    cy_mid = cy_lo + cell_h * 0.5

    ov_x = torch.clamp(
        torch.minimum(wa_x_max.unsqueeze(2), cx_hi) - torch.maximum(wa_x_min.unsqueeze(2), cx_lo),
        min=0.0,
    )  # [B, num_nets, G_cols]
    ov_y = torch.clamp(
        torch.minimum(wa_y_max.unsqueeze(2), cy_hi) - torch.maximum(wa_y_min.unsqueeze(2), cy_lo),
        min=0.0,
    )  # [B, num_nets, G_rows]

    # L-shape soft indicators centred at bbox midpoints
    cx_center = (wa_x_max + wa_x_min) * 0.5  # [B, num_nets]
    cy_center = (wa_y_max + wa_y_min) * 0.5
    d_col = (cx_mid.view(1, 1, -1) - cx_center.unsqueeze(2)) / cell_w  # [B, N, G_cols]
    d_row = (cy_mid.view(1, 1, -1) - cy_center.unsqueeze(2)) / cell_h  # [B, N, G_rows]
    raw_col = torch.exp(-(d_col ** 2) / (2.0 * sigma_l * sigma_l))
    raw_row = torch.exp(-(d_row ** 2) / (2.0 * sigma_l * sigma_l))
    col_indicator = raw_col / raw_col.sum(dim=2, keepdim=True).clamp(min=1e-12)
    row_indicator = raw_row / raw_row.sum(dim=2, keepdim=True).clamp(min=1e-12)

    # H demand at (r, c): w * row_indicator[r] * (ov_x[c] / cell_w)
    #   total per net = w * 1 * bbox_w / cell_w  (cell-count units of demand)
    # V demand at (r, c): w * col_indicator[c] * (ov_y[r] / cell_h)
    #   total per net = w * bbox_h / cell_h
    nw = net_weights_t.view(1, -1, 1)  # [1, N, 1]
    H_per = nw * row_indicator           # [B, N, G_rows]
    V_per = nw * col_indicator           # [B, N, G_cols]
    ov_x_norm = ov_x / cell_w            # [B, N, G_cols]
    ov_y_norm = ov_y / cell_h            # [B, N, G_rows]

    H_demand = torch.bmm(H_per.transpose(1, 2), ov_x_norm)  # [B, G_rows, G_cols]
    V_demand = torch.bmm(ov_y_norm.transpose(1, 2), V_per)  # [B, G_rows, G_cols]

    H_demand = H_demand / grid_h_routes
    V_demand = V_demand / grid_v_routes

    if nH > 0:
        mh_x = hw_t[:nH]
        mh_y = hh_t[:nH]
        mp = pos_batch[:, :nH, :]
        m_x_lo = mp[:, :, 0] - mh_x
        m_x_hi = mp[:, :, 0] + mh_x
        m_y_lo = mp[:, :, 1] - mh_y
        m_y_hi = mp[:, :, 1] + mh_y
        m_ov_x = torch.clamp(
            torch.minimum(m_x_hi.unsqueeze(2), cx_hi) - torch.maximum(m_x_lo.unsqueeze(2), cx_lo),
            min=0.0,
        )  # [B, nH, G_cols]
        m_ov_y = torch.clamp(
            torch.minimum(m_y_hi.unsqueeze(2), cy_hi) - torch.maximum(m_y_lo.unsqueeze(2), cy_lo),
            min=0.0,
        )  # [B, nH, G_rows]
        macro_cov = torch.bmm(m_ov_y.transpose(1, 2), m_ov_x)  # [B, G_rows, G_cols]
        V_demand = V_demand + macro_cov * (vrouting_alloc / cell_h / grid_v_routes)
        H_demand = H_demand + macro_cov * (hrouting_alloc / cell_w / grid_h_routes)

    flat = torch.cat(
        [V_demand.reshape(B, -1), H_demand.reshape(B, -1)],
        dim=1,
    )
    k = max(1, int(flat.shape[1] * abu_frac))
    topk_vals, _ = torch.topk(flat, k=k, dim=1)
    # cong_scale: post-hoc calibration multiplier. Diagnostic showed real
    # cong is ~1.4x our proxy on v60 placements; cong_scale > 1.0 lifts
    # the cong term toward parity (degenerate with lambda_cong but useful
    # as a calibration anchor or in the diagnostic comparison).
    return topk_vals.mean(dim=1) * cong_scale


def _rudy_congestion_batch(pos_batch, node_ids_t, net_ids_t, num_nets, net_weights_t,
                            cw, ch, G_rows, G_cols, gamma_rudy, cong_thr=0.0, eps=1e-3):
    B = pos_batch.shape[0]
    cell_w = cw / G_cols
    cell_h = ch / G_rows
    dev, dtype = pos_batch.device, pos_batch.dtype

    xy = pos_batch[:, node_ids_t, :]
    x, y = xy[:, :, 0], xy[:, :, 1]

    lse_px = _lse_per_net_batch( x, net_ids_t, num_nets, gamma_rudy)
    lse_nx = _lse_per_net_batch(-x, net_ids_t, num_nets, gamma_rudy)
    lse_py = _lse_per_net_batch( y, net_ids_t, num_nets, gamma_rudy)
    lse_ny = _lse_per_net_batch(-y, net_ids_t, num_nets, gamma_rudy)

    wa_x_max =  lse_px * gamma_rudy
    wa_x_min = -lse_nx * gamma_rudy
    wa_y_max =  lse_py * gamma_rudy
    wa_y_min = -lse_ny * gamma_rudy

    bbox_area = torch.clamp((wa_x_max - wa_x_min) * (wa_y_max - wa_y_min), min=eps)
    demand = net_weights_t / bbox_area

    cx_lo = torch.arange(G_cols, device=dev, dtype=dtype) * cell_w
    cx_hi = cx_lo + cell_w
    cy_lo = torch.arange(G_rows, device=dev, dtype=dtype) * cell_h
    cy_hi = cy_lo + cell_h

    ov_x = torch.clamp(
        torch.minimum(wa_x_max.unsqueeze(2), cx_hi) - torch.maximum(wa_x_min.unsqueeze(2), cx_lo),
        min=0.0,
    )
    ov_y = torch.clamp(
        torch.minimum(wa_y_max.unsqueeze(2), cy_hi) - torch.maximum(wa_y_min.unsqueeze(2), cy_lo),
        min=0.0,
    )

    D = torch.bmm(
        (demand.unsqueeze(2) * ov_y).transpose(1, 2),
        ov_x,
    )

    return (torch.clamp(D - cong_thr, min=0.0) ** 2).sum(dim=(1, 2))


# ══════════════════════════════════════════════════════════════════════════════
#  Faithful differentiable congestion port (_official_congestion_batch_v2)
# ──────────────────────────────────────────────────────────────────────────────
#  An earlier, cruder congestion proxy put a single soft H stripe at each net's
#  bbox y-midpoint (and a V stripe at the x-midpoint), which systematically
#  under-counts multi-pin nets and ignores the box-blur smoothing pass — a
#  diagnostic showed it under-reports real ABU cong by ~40%. This port replaces
#  it with a near-faithful copy of plc_client_os.get_routing()/get_congestion_cost():
#    * every net is a star from its driver pin; each (driver, sink) edge is an
#      L-route — H demand along the driver's row over the columns spanned by
#      [d_x, s_x], V demand along the sink's column over the rows spanned by
#      [d_y, s_y] (this is exactly the >3-pin star decomposition and the 2-pin
#      case of the real router);
#    * the floor()-snapped anchor row/column is approximated by a temperature
#      softmax over cells (width snap_sigma, in cell units) so gradients flow
#      through pin positions;
#    * net demand is normalised by grid_{h,v}_routes, then box-blurred by the
#      same per-axis linear operator as __smooth_routing_cong (M_row / M_col);
#    * hard-macro blockage area is added UNSMOOTHED, mirroring get_routing();
#    * aggregator is ABU at abu_frac over all 2*G cells (the official is 0.05).
#  Pin positions = owner-macro centre + per-pin offset (hard-macro pins carry
#  real offsets via macro_pin_offsets; soft macros / ports carry none).


def _build_cong_smooth_matrix(G: int, smooth_range: int, device, dtype):
    """Linear operator mirroring plc_client_os.__smooth_routing_cong's per-axis
    box-blur. Returns [G, G] row-stochastic M with
        M[src, dst] = 1 / gcell_cnt(src)   if  lp(src) <= dst <= rp(src)  else 0
    where lp/rp = clamp(src ± smooth_range, 0, G-1), gcell_cnt = rp - lp + 1.
    smooth_range <= 0 -> identity (no smoothing)."""
    if smooth_range is None or smooth_range <= 0:
        return torch.eye(G, device=device, dtype=dtype)
    M = torch.zeros(G, G, device=device, dtype=dtype)
    for s in range(G):
        lp = max(0, s - smooth_range)
        rp = min(G - 1, s + smooth_range)
        M[s, lp:rp + 1] = 1.0 / float(rp - lp + 1)
    return M


def _build_cong_v2_data(raw: dict):
    """Static per-edge / per-pin arrays for _official_congestion_batch_v2,
    built from raw['net_pin_nodes_list'] (driver-first pin list per net) and
    raw['macro_pin_offsets_list'] (hard-macro pin offsets). Returns None if
    pin-level connectivity is unavailable (caller falls back to v1)."""
    npn = raw.get('net_pin_nodes_list')
    if not npn:
        return None
    nH = int(raw['nH'])
    mpo = raw.get('macro_pin_offsets_list') or []
    net_w = np.asarray(raw['net_weights'], dtype=np.float64)
    pin_owner = []          # index into the [nM + nP] node-position array
    pin_off   = []          # [P, 2] offset of pin from its owner's centre
    e_drv, e_snk, e_w = [], [], []
    for k, pins in enumerate(npn):
        if pins is None or len(pins) < 2:
            continue
        base = len(pin_owner)
        for owner, slot in pins:
            owner = int(owner); slot = int(slot)
            pin_owner.append(owner)
            if (owner < nH and owner < len(mpo) and mpo[owner] is not None
                    and len(mpo[owner]) > slot):
                off = mpo[owner][slot]
                pin_off.append([float(off[0]), float(off[1])])
            else:
                pin_off.append([0.0, 0.0])
        w = float(net_w[k]) if k < len(net_w) else 1.0
        for j in range(1, len(pins)):
            e_drv.append(base)
            e_snk.append(base + j)
            e_w.append(w)
    if not e_drv:
        return None
    return {
        'pin_owner': np.asarray(pin_owner, dtype=np.int64),
        'pin_off':   np.asarray(pin_off,   dtype=np.float64).reshape(-1, 2),
        'e_drv':     np.asarray(e_drv, dtype=np.int64),
        'e_snk':     np.asarray(e_snk, dtype=np.int64),
        'e_w':       np.asarray(e_w,   dtype=np.float64),
    }


def _official_congestion_batch_v2(
    pos_batch,                                  # [B, nM + nP, 2]
    pin_owner_t, pin_off_t,                     # [P] long, [P, 2]
    e_drv_t, e_snk_t, e_w_t,                    # [E] long, [E] long, [E]
    cw, ch, G_rows, G_cols,
    hw_t, hh_t, nH,
    hroutes_per_micron, vroutes_per_micron,
    hrouting_alloc, vrouting_alloc,
    M_row, M_col,                               # [G_rows, G_rows], [G_cols, G_cols]
    row_c, col_c,                               # cell centres in grid units
    cx_lo, cx_hi, cy_lo, cy_hi,                 # cell bounds in canvas units
    abu_frac: float = 0.05,
    snap_sigma: float = 0.5,
    cong_scale: float = 1.0,
    edge_chunk: int = 8192,
):
    """Differentiable, near-faithful port of get_congestion_cost(). Returns [B].

    Remaining approximations vs. the real router: (1) softmax instead of an
    exact floor() pin->cell snap; (2) cell demand spread by sub-cell overlap
    fraction rather than +weight per crossed cell; (3) 3-pin nets use the star
    decomposition rather than the dedicated L/T optimisation; (4) macro blockage
    uses overlap area in place of the per-edge x_dist/y_dist + partial-overlap
    edge corrections."""
    B = pos_batch.shape[0]
    dev, dtype = pos_batch.device, pos_batch.dtype
    cell_w = cw / G_cols
    cell_h = ch / G_rows
    grid_v_routes = max(cell_w * vroutes_per_micron, 1e-30)
    grid_h_routes = max(cell_h * hroutes_per_micron, 1e-30)
    inv2s2 = 1.0 / (2.0 * snap_sigma * snap_sigma)

    # Static grid coordinate tensors are precomputed once in _run_batch and
    # passed in, avoiding per-step arange allocation inside this hot kernel.

    # Pin coordinates: owner centre + per-pin offset.
    pin_xy = pos_batch.index_select(1, pin_owner_t) + pin_off_t.unsqueeze(0)   # [B, P, 2]

    H_demand = pos_batch.new_zeros(B, G_rows, G_cols)
    V_demand = pos_batch.new_zeros(B, G_rows, G_cols)

    E = int(e_drv_t.shape[0])
    step = max(1, int(edge_chunk))
    for c0 in range(0, E, step):
        c1 = min(E, c0 + step)
        di = e_drv_t[c0:c1]; si = e_snk_t[c0:c1]; wi = e_w_t[c0:c1].view(1, -1, 1)
        drv = pin_xy.index_select(1, di)            # [B, e, 2]
        snk = pin_xy.index_select(1, si)
        dx, dy = drv[..., 0], drv[..., 1]           # [B, e]
        sx, sy = snk[..., 0], snk[..., 1]

        # H demand: anchor row ~ floor(dy / cell_h), columns spanning [dx, sx].
        drv_row_u  = dy / cell_h
        anchor_row = torch.softmax(
            -((row_c.view(1, 1, -1) - drv_row_u.unsqueeze(-1)) ** 2) * inv2s2, dim=-1)   # [B,e,G_rows]
        x_lo = torch.minimum(dx, sx).unsqueeze(-1)
        x_hi = torch.maximum(dx, sx).unsqueeze(-1)
        ov_col = torch.clamp(torch.minimum(x_hi, cx_hi) - torch.maximum(x_lo, cx_lo), min=0.0) / cell_w  # [B,e,G_cols]
        Hw = anchor_row * wi                          # [B,e,G_rows]
        H_demand = H_demand + torch.bmm(Hw.transpose(1, 2), ov_col)     # [B,G_rows,G_cols]

        # V demand: anchor column ~ floor(sx / cell_w), rows spanning [dy, sy].
        snk_col_u  = sx / cell_w
        anchor_col = torch.softmax(
            -((col_c.view(1, 1, -1) - snk_col_u.unsqueeze(-1)) ** 2) * inv2s2, dim=-1)   # [B,e,G_cols]
        y_lo = torch.minimum(dy, sy).unsqueeze(-1)
        y_hi = torch.maximum(dy, sy).unsqueeze(-1)
        ov_row = torch.clamp(torch.minimum(y_hi, cy_hi) - torch.maximum(y_lo, cy_lo), min=0.0) / cell_h  # [B,e,G_rows]
        Vw = anchor_col * wi                          # [B,e,G_cols]
        V_demand = V_demand + torch.bmm(ov_row.transpose(1, 2), Vw)     # [B,G_rows,G_cols]

    H_demand = H_demand / grid_h_routes
    V_demand = V_demand / grid_v_routes

    # Box-blur the net demand (the real router smooths net demand only).
    # H smooths along rows:    out[b,p,c] = sum_r M_row[r,p] * H[b,r,c]
    H_demand = torch.einsum('rp,brc->bpc', M_row, H_demand)
    # V smooths along columns: out[b,r,p] = sum_c V[b,r,c] * M_col[c,p]
    V_demand = torch.einsum('brc,cp->brp', V_demand, M_col)

    # Hard-macro blockage (added unsmoothed, mirroring get_routing()).
    if nH > 0:
        mp = pos_batch[:, :nH, :]
        mx, my = mp[..., 0], mp[..., 1]
        hw = hw_t[:nH]; hh = hh_t[:nH]
        m_ov_x = torch.clamp(torch.minimum((mx + hw).unsqueeze(-1), cx_hi)
                             - torch.maximum((mx - hw).unsqueeze(-1), cx_lo), min=0.0)   # [B,nH,G_cols]
        m_ov_y = torch.clamp(torch.minimum((my + hh).unsqueeze(-1), cy_hi)
                             - torch.maximum((my - hh).unsqueeze(-1), cy_lo), min=0.0)   # [B,nH,G_rows]
        macro_cov = torch.bmm(m_ov_y.transpose(1, 2), m_ov_x)                            # [B,G_rows,G_cols]
        V_demand = V_demand + macro_cov * (vrouting_alloc / cell_h / grid_v_routes)
        H_demand = H_demand + macro_cov * (hrouting_alloc / cell_w / grid_h_routes)

    flat = torch.cat([V_demand.reshape(B, -1), H_demand.reshape(B, -1)], dim=1)
    k = max(1, int(flat.shape[1] * abu_frac))
    topk_vals, _ = torch.topk(flat, k=k, dim=1)
    return topk_vals.mean(dim=1) * cong_scale


# Compiled versions.
#
# v60 (2026-05-20): torch.compile's inductor/triton backend imports
# pkg_resources (i.e. setuptools) at first-call compile time. In an
# environment without setuptools that import fails *deep inside* the
# optimisation loop — long after this module imported cleanly — so a
# plain try/except around the torch.compile() calls below would not
# catch it (torch.compile() returns a lazy wrapper; the failure is on
# first invocation).
#
# To keep v60 free of any hard setuptools dependency, probe for setuptools
# up front. If it is absent, skip torch.compile entirely and run the eager
# functions. Behaviour with setuptools present is unchanged. Override the
# auto-probe with V60_DISABLE_COMPILE=1 (force eager) if desired.
import importlib.util as _ilu

_compile_disabled_env = os.environ.get('V60_DISABLE_COMPILE', '') not in ('', '0')
_have_setuptools      = _ilu.find_spec('setuptools') is not None
_use_compile          = _have_setuptools and not _compile_disabled_env


def _maybe_compile(fn):
    """torch.compile(fn) when the environment can support it, else eager fn."""
    if not _use_compile:
        return fn
    try:
        return torch.compile(fn, dynamic=True)
    except Exception:
        return fn


if not _use_compile:
    _why = 'V60_DISABLE_COMPILE set' if _compile_disabled_env else 'setuptools not found'
    print(f"[v60_kernels] {_why}; running eager (no torch.compile, no setuptools dependency).")

_wa_wl_bc     = _maybe_compile(_wa_wirelength_batch)
_density_bc   = _maybe_compile(_density_loss_batch)
_hs_ovlp_bc   = _maybe_compile(_hard_soft_overlap_loss_batch)
_hh_ovlp_bc   = _maybe_compile(_hard_hard_overlap_loss_batch)
_ss_ovlp_bc   = _maybe_compile(_soft_soft_overlap_loss_batch)
_rudy_bc      = _maybe_compile(_rudy_congestion_batch)
_off_cong_bc  = _maybe_compile(_official_congestion_batch)
_off_cong_v2_compiled = (
    torch.compile(_official_congestion_batch_v2, dynamic=True) if _use_compile else None
)

_off_cong_v2_compile_disabled = False


def _off_cong_v2_bc(*args, **kwargs):
    """Compiled v2 congestion with one-shot fallback on unsupported backends."""
    global _off_cong_v2_compile_disabled
    if _off_cong_v2_compiled is not None and not _off_cong_v2_compile_disabled:
        try:
            return _off_cong_v2_compiled(*args, **kwargs)
        except Exception as exc:
            _off_cong_v2_compile_disabled = True
            warnings.warn(
                f"torch.compile failed for _official_congestion_batch_v2; "
                f"falling back to eager v2 congestion ({exc})",
                RuntimeWarning,
            )
    return _official_congestion_batch_v2(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  Soft-macro degrees (fanin/fanout per soft cluster)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_soft_degrees(benchmark: Benchmark, nH: int, nM: int) -> np.ndarray:
    """Returns a length-nS array of degrees (number of nets each soft is on)."""
    nS = nM - nH
    if nS <= 0:
        return np.zeros(0, dtype=np.float32)
    deg = np.zeros(nS, dtype=np.float32)
    for nodes in benchmark.net_nodes:
        for v in nodes.tolist():
            if nH <= v < nM:
                deg[v - nH] += 1.0
    return deg


# ══════════════════════════════════════════════════════════════════════════════
#  Jacobi preconditioner (DREAMPlace-style, optional)
# ══════════════════════════════════════════════════════════════════════════════
#
# Per-node preconditioner v_i = sum_{e ∈ E_i} 1/(|e| - 1) where E_i is the
# set of nets touching node i and |e| is the net's pin count. High-degree
# nodes (many nets, especially small ones) accumulate larger v_i and so
# get smaller per-step displacements after preconditioning.
#
# We apply this in NORMALIZED form: v_inv = (1/v_i) / mean(1/v_i), so the
# mean gradient scaling is 1.0 and existing lr_s1 / lr_s2 / lambdas keep
# their calibration. Only the relative per-node step sizes change.
#
# Caveat: applied to total gradient, not just WL. For density / cong /
# overlap terms the per-node gradient isn't structurally tied to v_i, so
# preconditioning rescales them by netlist topology (a topology prior).
# The hope is that this prior is benign or mildly helpful since macros
# with more nets generally need finer adjustments.
#
# Disabled by default. Enable via use_preconditioner=True on the placer.

def _compute_preconditioner(raw: dict) -> np.ndarray:
    """
    Returns v[nM + nP] preconditioner values per node:
        v_i = sum over nets e ∈ E_i of 1/(|e| - 1)
    where |e| is the number of pins on net e. Single-pin nets are skipped
    (their contribution is undefined).
    """
    nM = int(raw['nM'])
    nP = int(raw['nP'])
    nT = nM + nP
    node_ids = np.asarray(raw['node_ids'], dtype=np.int64)
    net_ids  = np.asarray(raw['net_ids'],  dtype=np.int64)
    num_nets = int(raw['num_nets'])

    e_size = np.bincount(net_ids, minlength=num_nets).astype(np.float64)
    # 1 / (|e| - 1), zero for |e| <= 1
    e_inv  = np.zeros(num_nets, dtype=np.float64)
    valid  = e_size > 1
    e_inv[valid] = 1.0 / (e_size[valid] - 1.0)

    contrib_per_pin = e_inv[net_ids]                     # [num_pins]
    v_per_node = np.zeros(nT, dtype=np.float64)
    np.add.at(v_per_node, node_ids, contrib_per_pin)
    return v_per_node


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-level placement: clustering + cluster-aware init
# ══════════════════════════════════════════════════════════════════════════════
#
# Hard macros are clustered into K groups by netlist connectivity (spectral
# clustering on the normalized Laplacian of the macro-macro adjacency,
# weighted by shared net count). Each cluster's spatial centre is taken
# from the spectral 2D embedding (eigenvectors 2 and 3) scaled to canvas
# coordinates.
#
# At initialisation, every hard macro starts at its cluster's centre
# (plus a small per-macro jitter). Connected macros that fall in the
# same cluster start clustered together, breaking the random/quadratic-
# init topology that traps the optimiser at the cong=1.14 floor.
#
# We deliberately do NOT enforce a hard cluster boundary in Stage 1/2 —
# the loss is the standard placement loss. Clustering only changes where macros
# START. The optimiser is free to dissolve cluster boundaries during
# refinement; we just give it a connectivity-respecting initial
# topology to descend from.

def _build_macro_macro_adjacency(benchmark: Benchmark, nH: int) -> np.ndarray:
    """
    Build [nH, nH] symmetric adjacency where A[i,j] = sum of net weights
    over nets connecting hard macros i and j (clique model).
    """
    A = np.zeros((nH, nH), dtype=np.float32)
    if nH <= 1:
        return A
    net_weights = benchmark.net_weights.numpy() if hasattr(benchmark.net_weights, 'numpy') else \
                  np.asarray(benchmark.net_weights)
    for k_net, nodes in enumerate(benchmark.net_nodes):
        ns = [int(v) for v in nodes.tolist() if int(v) < nH]
        if len(ns) < 2:
            continue
        w = float(net_weights[k_net]) if k_net < len(net_weights) else 1.0
        # clique-model contribution: divide by (n-1) so each net contributes
        # ~unit weight per macro regardless of fanout.
        contrib = w / max(len(ns) - 1, 1)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                u, v = ns[i], ns[j]
                A[u, v] += contrib
                A[v, u] += contrib
    return A


def _kmeans_pp_init(data: np.ndarray, K: int, rng: np.random.RandomState) -> np.ndarray:
    """K-means++ centroid initialisation."""
    n, d = data.shape
    centroids = np.zeros((K, d), dtype=data.dtype)
    centroids[0] = data[rng.randint(n)]
    for k in range(1, K):
        d_min = np.min(
            np.linalg.norm(data[:, None, :] - centroids[None, :k, :], axis=-1),
            axis=1,
        )
        probs = d_min ** 2
        s = probs.sum()
        if s <= 0:
            centroids[k] = data[rng.randint(n)]
        else:
            probs = probs / s
            centroids[k] = data[rng.choice(n, p=probs)]
    return centroids


def _kmeans(data: np.ndarray, K: int, rng: np.random.RandomState, max_iter: int = 50):
    """Plain K-means with K-means++ init. Returns (labels[n], centroids[K, d])."""
    n, d = data.shape
    if K >= n:
        return np.arange(n, dtype=np.int64), data.copy()
    centroids = _kmeans_pp_init(data, K, rng)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists  = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1)
        new_lab = dists.argmin(axis=1)
        if np.array_equal(new_lab, labels):
            break
        labels = new_lab
        for k in range(K):
            members = data[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
    return labels, centroids


def _spectral_cluster_macros(
    A: np.ndarray, K: int, embed_dim: int = None, seed: int = 0,
):
    """
    Spectral clustering of macros via the normalized symmetric Laplacian.
    Returns (labels[nH], embed_2d[nH, 2]) where embed_2d uses eigenvectors
    2 and 3 (the Fiedler vector and its successor) — these give a 2D layout
    that respects connectivity. The labels come from K-means on a K-dim
    embedding (eigenvectors 2..K+1).
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    if K >= n or n < 3:
        return np.arange(n, dtype=np.int64), np.zeros((n, 2), dtype=np.float32)

    deg = A.sum(axis=1) + 1e-9
    D_inv_sqrt = 1.0 / np.sqrt(deg)
    L_norm = np.eye(n, dtype=np.float32) - (D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :])
    L_norm = 0.5 * (L_norm + L_norm.T)  # symmetrise vs. fp noise

    # Eigh gives ascending eigenvalues
    eigvals, eigvecs = np.linalg.eigh(L_norm)

    # 2D embedding for spatial cluster centres (skip eigvec 0 which is constant)
    embed_2d = eigvecs[:, 1:3].astype(np.float32)  # [n, 2]

    # K-dim embedding for K-means clustering
    if embed_dim is None:
        embed_dim = K
    embed_K = eigvecs[:, 1:1 + embed_dim].astype(np.float32)
    norms = np.linalg.norm(embed_K, axis=1, keepdims=True) + 1e-9
    embed_K = embed_K / norms

    rng = np.random.RandomState(seed)
    labels, _ = _kmeans(embed_K, K, rng)
    return labels.astype(np.int64), embed_2d


def _spectral_embed_to_canvas(embed_2d: np.ndarray, cw: float, ch: float,
                               margin_frac: float = 0.10) -> np.ndarray:
    """
    Linearly scale a 2D spectral embedding into canvas coordinates,
    leaving a ``margin_frac`` border around the canvas perimeter.
    Returns [n, 2] points where every coordinate is in
    [margin_frac * cw, (1 - margin_frac) * cw] (similarly for y).
    """
    n = embed_2d.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = np.zeros_like(embed_2d)
    x_lo, x_hi = margin_frac * cw, (1.0 - margin_frac) * cw
    y_lo, y_hi = margin_frac * ch, (1.0 - margin_frac) * ch
    for axis, (lo, hi) in enumerate([(x_lo, x_hi), (y_lo, y_hi)]):
        v = embed_2d[:, axis]
        v_min, v_max = float(v.min()), float(v.max())
        if v_max - v_min < 1e-9:
            out[:, axis] = (lo + hi) * 0.5
        else:
            out[:, axis] = lo + (v - v_min) / (v_max - v_min) * (hi - lo)
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 0: mini-analytical super-macro placement
# ══════════════════════════════════════════════════════════════════════════════
#
# Given cluster labels, build a K-super-macro problem and place those
# super-macros via mini-analytical (WAWL + density + pairwise overlap),
# using the embedding-derived centres as init. The optimised K positions
# replace a plain "spectral embedding linearly scaled to canvas" init,
# which was never tuned by any objective.
#
# Soft macros are skipped at Stage 0 (only super-clusters of hard macros +
# fixed ports participate). Stage 1+2 place the softs.

def _build_super_net_list(benchmark: Benchmark, labels: np.ndarray,
                           nH: int, nM: int, K: int):
    """
    Aggregate netlist into super-nets between clusters and ports.

    Super-pin id space:
        [0, K)         — cluster ids
        [K, K + nP)    — port ids (port j → super id K + j)
    Soft-macro pins are skipped. A net contributes a super-net only if it
    touches ≥2 distinct super-pins.
    """
    nP = int(benchmark.port_positions.shape[0])
    net_weights = (benchmark.net_weights.numpy()
                   if hasattr(benchmark.net_weights, 'numpy')
                   else np.asarray(benchmark.net_weights))
    super_node_ids = []
    super_net_ids  = []
    super_weights  = []
    new_id = 0
    for k_net, nodes in enumerate(benchmark.net_nodes):
        ns = nodes.tolist()
        super_pins = set()
        for v in ns:
            vi = int(v)
            if vi < nH:
                super_pins.add(int(labels[vi]))
            elif vi >= nM:
                super_pins.add(K + (vi - nM))
            # soft macros (nH <= vi < nM): skipped at Stage 0
        if len(super_pins) < 2:
            continue
        for sp in super_pins:
            super_node_ids.append(int(sp))
            super_net_ids.append(new_id)
        super_weights.append(float(net_weights[k_net]) if k_net < len(net_weights) else 1.0)
        new_id += 1
    return {
        'node_ids':    super_node_ids,
        'net_ids':     super_net_ids,
        'weights':     super_weights,
        'num_nets':    new_id,
        'nP':          nP,
    }


def _compute_super_sizes(benchmark: Benchmark, labels: np.ndarray,
                          K: int, nH: int):
    """
    Per-cluster super-macro size: square with side = sqrt(total area of
    member hard macros). Returns hw[K], hh[K] half-extents.
    """
    macro_sizes = (benchmark.macro_sizes.numpy()
                   if hasattr(benchmark.macro_sizes, 'numpy')
                   else np.asarray(benchmark.macro_sizes))
    hw = np.zeros(K, dtype=np.float32)
    hh = np.zeros(K, dtype=np.float32)
    for k in range(K):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue
        total_area = float((macro_sizes[members, 0] * macro_sizes[members, 1]).sum())
        side = np.sqrt(max(total_area, 1e-12))
        hw[k] = side / 2.0
        hh[k] = side / 2.0
    return hw, hh


def _run_stage0(benchmark: Benchmark, labels: np.ndarray, K: int,
                init_centers: np.ndarray,
                num_steps: int = 1500,
                lr: float = 1.0,
                lambda_density: float = 200.0,
                lambda_overlap: float = 500.0,
                gamma_start: float = 2.0,
                gamma_end: float = 0.3,
                target_density: float = 0.7) -> np.ndarray:
    """
    Mini-analytical placement of K super-macros. Returns optimised
    [K, 2] super-macro positions in canvas coordinates.
    """
    nH = int(benchmark.num_hard_macros)
    nM = int(benchmark.num_macros)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    G_rows = int(benchmark.grid_rows)
    G_cols = int(benchmark.grid_cols)

    if K < 2:
        if init_centers is None or init_centers.shape[0] != K:
            out = np.full((K, 2), [cw * 0.5, ch * 0.5], dtype=np.float32)
        else:
            out = init_centers.astype(np.float32).copy()
        return out

    super_data = _build_super_net_list(benchmark, labels, nH, nM, K)
    hw_arr, hh_arr = _compute_super_sizes(benchmark, labels, K, nH)
    nP = super_data['nP']

    if super_data['num_nets'] == 0:
        out = init_centers.astype(np.float32).copy()
        return out

    # Stage 0 is tiny — CPU avoids GPU sync overhead.
    device = torch.device('cpu')
    dtype  = torch.float32

    node_ids_t = torch.tensor(super_data['node_ids'], dtype=torch.long, device=device)
    net_ids_t  = torch.tensor(super_data['net_ids'],  dtype=torch.long, device=device)
    nw_t       = torch.tensor(super_data['weights'],  dtype=dtype, device=device)
    num_nets   = super_data['num_nets']
    hw_t       = torch.tensor(hw_arr, dtype=dtype, device=device)
    hh_t       = torch.tensor(hh_arr, dtype=dtype, device=device)

    init = init_centers.astype(np.float32).copy()
    init[:, 0] = np.clip(init[:, 0], hw_arr, cw - hw_arr)
    init[:, 1] = np.clip(init[:, 1], hh_arr, ch - hh_arr)
    super_pos = torch.nn.Parameter(torch.tensor(init, dtype=dtype, device=device))

    if nP > 0:
        port_np = (benchmark.port_positions.numpy()
                   if hasattr(benchmark.port_positions, 'numpy')
                   else np.asarray(benchmark.port_positions))
        fixed_ports = torch.tensor(port_np, dtype=dtype, device=device)
    else:
        fixed_ports = None

    opt   = optim.Adam([super_pos], lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, num_steps, eta_min=lr * 0.01)
    x_lo = hw_t
    x_hi = torch.full_like(hw_t, cw) - hw_t
    y_lo = hh_t
    y_hi = torch.full_like(hh_t, ch) - hh_t

    for step in range(num_steps):
        g = gamma_start * (gamma_end / gamma_start) ** (step / max(num_steps - 1, 1))
        opt.zero_grad()
        if fixed_ports is not None:
            pos_full = torch.cat([super_pos, fixed_ports], dim=0)
        else:
            pos_full = super_pos
        pos_b = pos_full.unsqueeze(0)
        sub_b = pos_b[:, :K, :]
        wl    = _wa_wirelength_batch(pos_b, node_ids_t, net_ids_t, num_nets, nw_t, g)
        den   = _density_loss_batch(sub_b, hw_t, hh_t, cw, ch, G_rows, G_cols, target_density)
        ovlp  = _pairwise_overlap_loss_batch(sub_b, hw_t, hh_t)
        loss  = wl + lambda_density * den + lambda_overlap * ovlp
        loss.sum().backward()
        opt.step(); sched.step()
        with torch.no_grad():
            super_pos.data[:, 0].clamp_(x_lo, x_hi)
            super_pos.data[:, 1].clamp_(y_lo, y_hi)

    out = super_pos.detach().cpu().numpy().astype(np.float32)
    return out


# The v60 engine's `_build_cluster_data*` functions (Fiedler-2D clustering)
# live in v60_engine.py — everything they call (adjacency, spectral
# embedding, Stage 0) is exposed here.


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark serialisation
# ══════════════════════════════════════════════════════════════════════════════

def _extract_raw(benchmark: Benchmark) -> dict:
    nets = benchmark.net_nodes
    node_ids, net_ids = [], []
    for k, nodes in enumerate(nets):
        ns = nodes.tolist()
        node_ids.extend(ns)
        net_ids.extend([k] * len(ns))
    nP = int(benchmark.port_positions.shape[0])
    nH = int(benchmark.num_hard_macros)
    nM = int(benchmark.num_macros)
    return {
        'name':            benchmark.name,
        'cw':              float(benchmark.canvas_width),
        'ch':              float(benchmark.canvas_height),
        'nH':              nH,
        'nM':              nM,
        'nP':              nP,
        'G_rows':          int(benchmark.grid_rows),
        'G_cols':          int(benchmark.grid_cols),
        'hw_np':           benchmark.macro_sizes.numpy()[:, 0] / 2.0,
        'hh_np':           benchmark.macro_sizes.numpy()[:, 1] / 2.0,
        'movable_mask':    benchmark.get_movable_mask().numpy(),
        'macro_positions': benchmark.macro_positions.numpy(),
        'port_positions':  (benchmark.port_positions.numpy()
                            if nP > 0 else np.zeros((0, 2), dtype=np.float64)),
        'node_ids':        node_ids,
        'net_ids':         net_ids,
        'net_weights':     benchmark.net_weights.numpy().tolist(),
        'num_nets':        int(len(nets)),
        'soft_degrees':    _compute_soft_degrees(benchmark, nH, nM),
        # Pin-level connectivity for the v60 faithful congestion port. Each
        # net_pin_nodes_list[k] is a driver-first list of [owner, slot] pairs;
        # macro_pin_offsets_list[h] is the [num_pins_h, 2] offset table for
        # hard macro h. Empty lists if the loader didn't populate them (v1
        # congestion is used in that case).
        'net_pin_nodes_list': (
            [npn.tolist() for npn in benchmark.net_pin_nodes]
            if getattr(benchmark, 'net_pin_nodes', None) else []),
        'macro_pin_offsets_list': (
            [mpo.numpy().astype(np.float64) for mpo in benchmark.macro_pin_offsets]
            if getattr(benchmark, 'macro_pin_offsets', None) else []),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Congestion diagnostic (offline analysis of the L-shape proxy)
# ══════════════════════════════════════════════════════════════════════════════
#
# The differentiable proxy in _official_congestion_batch returns a single
# scalar (top-5% ABU mean), but for diagnosis we want the full per-cell
# breakdown plus per-net and per-macro attribution at each saturating
# cell. The helpers below mirror the kernel's arithmetic in pure numpy
# and add the breakdown.
#
# Why this exists: placements all converge to cong ≈ 1.13 on
# ibm01 regardless of init. The diagnostic answers:
#   * Are the same cells always saturating? (structural chokepoint?)
#   * Is the saturation net demand or macro blockage?
#   * Which macros / nets are responsible at the top cells?
# Run via dump_diagnostic_path on v60_Engine.

def _compute_cong_breakdown(raw: dict, params: dict, pos_np: np.ndarray) -> dict:
    """
    Pure-numpy mirror of _official_congestion_batch returning the V and H
    demand grids decomposed into net vs macro contributions, plus the
    intermediates needed to attribute demand at individual cells.

    Args:
        pos_np: [num_nodes, 2] (or [nM, 2] — ports auto-appended from raw).

    Returns dict with grids and per-net / per-macro intermediates.
    """
    cw, ch         = raw['cw'], raw['ch']
    G_rows, G_cols = raw['G_rows'], raw['G_cols']
    nH, nM, nP     = raw['nH'], raw['nM'], raw['nP']
    cell_w = cw / G_cols
    cell_h = ch / G_rows

    hpm = float(params.get('hroutes_per_micron', 0.0))
    vpm = float(params.get('vroutes_per_micron', 0.0))
    hra = float(params.get('hrouting_alloc',     0.0))
    vra = float(params.get('vrouting_alloc',     0.0))
    grid_v_routes = max(cell_w * vpm, 1e-30)
    grid_h_routes = max(cell_h * hpm, 1e-30)
    gamma     = float(params.get('gamma_rudy',     0.1))
    sigma_l   = float(params.get('cong_sigma_l',   1.0))
    abu_frac  = float(params.get('cong_abu_frac',  0.05))
    cong_scale = float(params.get('cong_scale',    1.0))

    # Reconstruct full positions (macros + ports) if caller passed only macros.
    pos = np.asarray(pos_np, dtype=np.float64)
    if pos.shape[0] < nM + nP:
        pos_full = np.zeros((nM + nP, 2), dtype=np.float64)
        pos_full[:nM] = pos[:nM]
        if nP > 0:
            pos_full[nM:] = raw['port_positions']
        pos = pos_full

    node_ids = np.asarray(raw['node_ids'], dtype=np.int64)
    net_ids  = np.asarray(raw['net_ids'],  dtype=np.int64)
    nw       = np.asarray(raw['net_weights'], dtype=np.float64)
    num_nets = int(raw['num_nets'])
    hw_np    = np.asarray(raw['hw_np'][:nH], dtype=np.float64)
    hh_np    = np.asarray(raw['hh_np'][:nH], dtype=np.float64)

    # Per-pin xy then per-net LSE bbox (mirrors _lse_per_net_batch arithmetic).
    xy_pins = pos[node_ids]
    x_pins, y_pins = xy_pins[:, 0], xy_pins[:, 1]

    def _lse_per_net(v):
        # log-sum-exp(v / gamma) per net, numerically stable; returns [num_nets].
        net_max = np.full(num_nets, -np.inf, dtype=np.float64)
        np.maximum.at(net_max, net_ids, v / gamma)
        shifted = v / gamma - net_max[net_ids]
        sum_exp = np.zeros(num_nets, dtype=np.float64)
        np.add.at(sum_exp, net_ids, np.exp(shifted))
        return net_max + np.log(np.maximum(sum_exp, 1e-30))

    wa_x_max =  _lse_per_net( x_pins) * gamma
    wa_x_min = -_lse_per_net(-x_pins) * gamma
    wa_y_max =  _lse_per_net( y_pins) * gamma
    wa_y_min = -_lse_per_net(-y_pins) * gamma

    cx_lo = np.arange(G_cols, dtype=np.float64) * cell_w
    cx_hi = cx_lo + cell_w
    cy_lo = np.arange(G_rows, dtype=np.float64) * cell_h
    cy_hi = cy_lo + cell_h
    cx_mid = cx_lo + cell_w * 0.5
    cy_mid = cy_lo + cell_h * 0.5

    ov_x = np.clip(
        np.minimum(wa_x_max[:, None], cx_hi[None, :])
        - np.maximum(wa_x_min[:, None], cx_lo[None, :]),
        0.0, None,
    )  # [num_nets, G_cols]
    ov_y = np.clip(
        np.minimum(wa_y_max[:, None], cy_hi[None, :])
        - np.maximum(wa_y_min[:, None], cy_lo[None, :]),
        0.0, None,
    )  # [num_nets, G_rows]

    cx_center = (wa_x_max + wa_x_min) * 0.5
    cy_center = (wa_y_max + wa_y_min) * 0.5
    d_col = (cx_mid[None, :] - cx_center[:, None]) / cell_w
    d_row = (cy_mid[None, :] - cy_center[:, None]) / cell_h
    raw_col = np.exp(-(d_col ** 2) / (2.0 * sigma_l * sigma_l))
    raw_row = np.exp(-(d_row ** 2) / (2.0 * sigma_l * sigma_l))
    col_indicator = raw_col / np.maximum(raw_col.sum(axis=1, keepdims=True), 1e-12)
    row_indicator = raw_row / np.maximum(raw_row.sum(axis=1, keepdims=True), 1e-12)

    H_per     = nw[:, None] * row_indicator   # [N, G_rows]
    V_per     = nw[:, None] * col_indicator   # [N, G_cols]
    ov_x_norm = ov_x / cell_w                  # [N, G_cols]
    ov_y_norm = ov_y / cell_h                  # [N, G_rows]

    # Net demand per cell, mirroring the bmm in _official_congestion_batch.
    H_demand_net = np.einsum('nr,nc->rc', H_per,     ov_x_norm) / grid_h_routes
    V_demand_net = np.einsum('nr,nc->rc', ov_y_norm, V_per)     / grid_v_routes

    # Macro blockage.
    if nH > 0:
        m_pos = pos[:nH]
        m_x_lo = m_pos[:, 0] - hw_np
        m_x_hi = m_pos[:, 0] + hw_np
        m_y_lo = m_pos[:, 1] - hh_np
        m_y_hi = m_pos[:, 1] + hh_np
        m_ov_x = np.clip(
            np.minimum(m_x_hi[:, None], cx_hi[None, :])
            - np.maximum(m_x_lo[:, None], cx_lo[None, :]),
            0.0, None,
        )  # [nH, G_cols]
        m_ov_y = np.clip(
            np.minimum(m_y_hi[:, None], cy_hi[None, :])
            - np.maximum(m_y_lo[:, None], cy_lo[None, :]),
            0.0, None,
        )  # [nH, G_rows]
        macro_cov      = m_ov_y.T @ m_ov_x                           # [G_rows, G_cols]
        V_demand_macro = macro_cov * (vra / cell_h / grid_v_routes)
        H_demand_macro = macro_cov * (hra / cell_w / grid_h_routes)
    else:
        m_ov_x         = np.zeros((0, G_cols))
        m_ov_y         = np.zeros((0, G_rows))
        V_demand_macro = np.zeros((G_rows, G_cols))
        H_demand_macro = np.zeros((G_rows, G_cols))

    V_demand_total = V_demand_net + V_demand_macro
    H_demand_total = H_demand_net + H_demand_macro

    flat = np.concatenate([V_demand_total.ravel(), H_demand_total.ravel()])
    k = max(1, int(len(flat) * abu_frac))
    topk_idx     = np.argpartition(flat, -k)[-k:]
    topk_vals    = flat[topk_idx]
    topk_mean    = float(topk_vals.mean()) * cong_scale
    abu_thresh   = float(topk_vals.min())  # pre-scale; for diagnostic ranking

    return {
        'V_total': V_demand_total, 'V_net': V_demand_net, 'V_macro': V_demand_macro,
        'H_total': H_demand_total, 'H_net': H_demand_net, 'H_macro': H_demand_macro,
        'col_indicator': col_indicator, 'row_indicator': row_indicator,
        'ov_x_norm': ov_x_norm, 'ov_y_norm': ov_y_norm,
        'm_ov_x': m_ov_x, 'm_ov_y': m_ov_y,
        'wa_x_max': wa_x_max, 'wa_x_min': wa_x_min,
        'wa_y_max': wa_y_max, 'wa_y_min': wa_y_min,
        'nw': nw,
        'cell_w': cell_w, 'cell_h': cell_h,
        'grid_v_routes': grid_v_routes, 'grid_h_routes': grid_h_routes,
        'vrouting_alloc': vra, 'hrouting_alloc': hra,
        'topk_mean': topk_mean, 'abu_threshold': abu_thresh,
    }


def _dump_cong_diagnostic_one(out_path: str, raw: dict, params: dict,
                                pos_np: np.ndarray, *,
                                benchmark=None, plc=None,
                                top_k_cells: int = 30,
                                top_k_per_cell: int = 5):
    """
    Write a single-placement cong diagnostic report.

    Sections:
      1. Header (bench, grid, routing params).
      2. Differentiable proxy value (matches _official_congestion_batch).
      3. Real proxy_cost components (if benchmark+plc given).
      4. Top-K cells: per-cell V vs H, net% vs macro%.
      5. Per top cell: top contributing nets and macros.
      6. Aggregated per-macro contribution to top-K cells.
      7. Aggregated per-net contribution to top-K cells.
    """
    grid = _compute_cong_breakdown(raw, params, pos_np)

    G_rows, G_cols = raw['G_rows'], raw['G_cols']
    nH    = raw['nH']
    nM    = raw['nM']
    nP    = raw['nP']
    nw    = grid['nw']
    cell_w, cell_h = grid['cell_w'], grid['cell_h']
    grid_v_routes  = grid['grid_v_routes']
    grid_h_routes  = grid['grid_h_routes']
    vra            = grid['vrouting_alloc']
    hra            = grid['hrouting_alloc']
    H_per_row      = nw[:, None] * grid['row_indicator']
    V_per_col      = nw[:, None] * grid['col_indicator']

    # Top-K cells across V+H.
    flat_V = grid['V_total'].ravel()
    flat_H = grid['H_total'].ravel()
    flat   = np.concatenate([flat_V, flat_H])
    K      = min(top_k_cells, len(flat))
    top_idx = np.argpartition(flat, -K)[-K:]
    top_idx = top_idx[np.argsort(-flat[top_idx])]

    cells = []
    for idx in top_idx:
        if idx < flat_V.size:
            r, c = int(idx // G_cols), int(idx % G_cols)
            dim, total = 'V', float(grid['V_total'][r, c])
            net_part   = float(grid['V_net'][r, c])
            macro_part = float(grid['V_macro'][r, c])
        else:
            j = int(idx - flat_V.size)
            r, c = j // G_cols, j % G_cols
            dim, total = 'H', float(grid['H_total'][r, c])
            net_part   = float(grid['H_net'][r, c])
            macro_part = float(grid['H_macro'][r, c])
        cells.append((dim, r, c, total, net_part, macro_part))

    macro_agg = np.zeros(nH, dtype=np.float64)
    net_agg   = np.zeros(grid['col_indicator'].shape[0], dtype=np.float64)

    lines = []
    L = lines.append
    bench_name = benchmark.name if benchmark is not None else '?'
    L(f"# Congestion diagnostic ({bench_name})")
    L(f"# grid: {G_rows} rows x {G_cols} cols ({G_rows * G_cols} cells per dim; ABU-5 over V+H = top {max(1, int(2 * G_rows * G_cols * 0.05))})")
    L(f"# nH={nH}  nM={nM}  nP={nP}  num_nets={grid['col_indicator'].shape[0]}")
    L(f"# vrouting_alloc={vra:.4f}  hrouting_alloc={hra:.4f}")
    L(f"# grid_v_routes={grid_v_routes:.4f}  grid_h_routes={grid_h_routes:.4f}")
    L(f"")
    L(f"# Differentiable proxy (top-5% mean of V+H):  {grid['topk_mean']:.4f}")
    L(f"# ABU threshold (min top-5%):                 {grid['abu_threshold']:.4f}")
    if plc is not None and benchmark is not None:
        try:
            costs = compute_proxy_cost(
                torch.tensor(pos_np, dtype=torch.float32), benchmark, plc,
            )
            L(f"# Real cong score (compute_proxy_cost):       {float(costs['congestion_cost']):.4f}")
            L(f"# Real proxy_cost:                            {float(costs['proxy_cost']):.4f}")
            L(f"# Real wirelength_cost:                       {float(costs['wirelength_cost']):.4f}")
            L(f"# Real density_cost:                          {float(costs['density_cost']):.4f}")
        except Exception as exc:
            L(f"# Real cong score: FAILED ({exc})")
    L(f"")

    L(f"# === Top-{K} congested cells (sorted by total demand desc) ===")
    L(f"# rank  dim  (r, c)    total    net     macro    net%   macro%")
    for i, (dim, r, c, total, net_part, macro_part) in enumerate(cells):
        net_pct   = 100.0 * net_part   / max(total, 1e-12)
        macro_pct = 100.0 * macro_part / max(total, 1e-12)
        L(f"# {i+1:>3}    {dim}   ({r:>2},{c:>2})  {total:7.4f}  {net_part:7.4f}  {macro_part:7.4f}  {net_pct:5.1f}%  {macro_pct:5.1f}%")
    L(f"")

    L(f"# === Top-{top_k_per_cell} contributors per top cell ===")
    for i, (dim, r, c, total, net_part, macro_part) in enumerate(cells):
        L(f"# Cell #{i+1} {dim} ({r},{c})  total={total:.4f}  net={net_part:.4f}  macro={macro_part:.4f}")
        if dim == 'V':
            net_contribs = V_per_col[:, c] * grid['ov_y_norm'][:, r] / grid_v_routes
        else:
            net_contribs = H_per_row[:, r] * grid['ov_x_norm'][:, c] / grid_h_routes
        kk = min(top_k_per_cell, net_contribs.size)
        top_n_idx = np.argpartition(net_contribs, -kk)[-kk:]
        top_n_idx = top_n_idx[np.argsort(-net_contribs[top_n_idx])]
        L(f"#   nets:")
        for n in top_n_idx:
            v = float(net_contribs[int(n)])
            if v > 1e-9:
                L(f"#     net {int(n):>5}  contrib={v:.4f}  weight={float(nw[int(n)]):.3f}  bbox=({grid['wa_x_min'][int(n)]:.1f},{grid['wa_y_min'][int(n)]:.1f})-({grid['wa_x_max'][int(n)]:.1f},{grid['wa_y_max'][int(n)]:.1f})")
                net_agg[int(n)] += v

        if nH > 0:
            mc = grid['m_ov_y'][:, r] * grid['m_ov_x'][:, c]  # [nH] coverage area
            if dim == 'V':
                macro_contribs = mc * (vra / cell_h / grid_v_routes)
            else:
                macro_contribs = mc * (hra / cell_w / grid_h_routes)
            kk = min(top_k_per_cell, nH)
            top_m_idx = np.argpartition(macro_contribs, -kk)[-kk:]
            top_m_idx = top_m_idx[np.argsort(-macro_contribs[top_m_idx])]
            L(f"#   macros:")
            for m in top_m_idx:
                v = float(macro_contribs[int(m)])
                if v > 1e-9:
                    L(f"#     macro {int(m):>4}  contrib={v:.4f}  pos=({float(pos_np[int(m),0]):.1f},{float(pos_np[int(m),1]):.1f})  size=({2*float(raw['hw_np'][int(m)]):.1f}x{2*float(raw['hh_np'][int(m)]):.1f})")
                    macro_agg[int(m)] += v
        L(f"")

    L(f"# === Top {min(20, nH)} macros by aggregated contribution to top-{K} cells ===")
    if nH > 0:
        ranked = np.argsort(-macro_agg)[:20]
        for i, m in enumerate(ranked):
            if macro_agg[int(m)] <= 1e-9:
                break
            L(f"# {i+1:>3}  macro {int(m):>4}  agg={macro_agg[int(m)]:.4f}  pos=({float(pos_np[int(m),0]):.1f},{float(pos_np[int(m),1]):.1f})  size=({2*float(raw['hw_np'][int(m)]):.1f}x{2*float(raw['hh_np'][int(m)]):.1f})")
    L(f"")

    L(f"# === Top 20 nets by aggregated contribution to top-{K} cells ===")
    ranked = np.argsort(-net_agg)[:20]
    for i, n in enumerate(ranked):
        if net_agg[int(n)] <= 1e-9:
            break
        L(f"# {i+1:>3}  net {int(n):>5}  agg={net_agg[int(n)]:.4f}  weight={float(nw[int(n)]):.3f}  bbox=({grid['wa_x_min'][int(n)]:.1f},{grid['wa_y_min'][int(n)]:.1f})-({grid['wa_x_max'][int(n)]:.1f},{grid['wa_y_max'][int(n)]:.1f})")
    L(f"")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'top_cell_indices': set(int(i) for i in top_idx),
        'topk_mean': grid['topk_mean'],
    }


def _dump_cong_diagnostic_multi(out_base: str, raw: dict, params: dict,
                                  pos_np_list, *, scores=None,
                                  benchmark=None, plc=None,
                                  num_seeds: int = 3,
                                  top_k_cells: int = 30,
                                  top_k_per_cell: int = 5):
    """
    Dump per-placement cong diagnostics for the top-`num_seeds` placements
    (sorted by `scores` ascending), plus a chokepoint-stability summary
    showing which top-cells repeat across the dumped placements.

    Files written:
        {out_base}_seed{i}.txt   per-placement (i in 0..num_seeds-1)
        {out_base}_summary.txt   stability summary (jaccard + cell counts)
    """
    n_total = len(pos_np_list)
    if n_total == 0:
        return
    if scores is not None:
        order = np.argsort([float(s) if s is not None else float('inf') for s in scores])
        pos_np_list = [pos_np_list[i] for i in order]
        scores      = [scores[i]      for i in order]
    K = min(num_seeds, n_total)

    per_seed = []
    for i in range(K):
        path_i = f"{out_base}_seed{i}.txt"
        info = _dump_cong_diagnostic_one(
            path_i, raw, params, pos_np_list[i],
            benchmark=benchmark, plc=plc,
            top_k_cells=top_k_cells,
            top_k_per_cell=top_k_per_cell,
        )
        per_seed.append(info)
        print(f"[v60_kernels] cong diagnostic seed {i} → {path_i}  (proxy mean={info['topk_mean']:.4f})")

    summary_path = f"{out_base}_summary.txt"
    G_rows, G_cols = raw['G_rows'], raw['G_cols']
    flat_V_size    = G_rows * G_cols
    counter        = {}
    for info in per_seed:
        for cell_idx in info['top_cell_indices']:
            counter[cell_idx] = counter.get(cell_idx, 0) + 1

    sorted_cells = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    lines = []
    L = lines.append
    L(f"# Cong chokepoint stability across {K} placements (top-{top_k_cells} cells each)")
    if scores is not None:
        scores_fmt = ', '.join(f'{s:.4f}' for s in scores[:K])
        L(f"# Scores (ascending): [{scores_fmt}]")
    means_fmt = ', '.join(f"{info['topk_mean']:.4f}" for info in per_seed)
    L(f"# Diff-proxy means:   [{means_fmt}]")
    L(f"")
    L(f"# Cells in top-{top_k_cells} of any seed, sorted by occurrence (count >= 2 only)")
    L(f"# rank  count/{K}  dim  (r, c)")
    for i, (cell_idx, cnt) in enumerate(sorted_cells):
        if cnt < 2:
            break
        if cell_idx < flat_V_size:
            r, c = cell_idx // G_cols, cell_idx % G_cols
            dim = 'V'
        else:
            j = cell_idx - flat_V_size
            r, c = j // G_cols, j % G_cols
            dim = 'H'
        L(f"# {i+1:>3}   {cnt}/{K}     {dim}   ({r:>2},{c:>2})")
    L(f"")
    L(f"# Pairwise jaccard overlap of top-{top_k_cells} cell sets:")
    for i in range(K):
        for j in range(i + 1, K):
            inter = len(per_seed[i]['top_cell_indices'] & per_seed[j]['top_cell_indices'])
            union = len(per_seed[i]['top_cell_indices'] | per_seed[j]['top_cell_indices'])
            jacc  = inter / max(union, 1)
            L(f"#   seed{i} vs seed{j}: |∩|={inter:>3}/{top_k_cells}  jaccard={jacc:.2f}")
    L(f"")

    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"[v60_kernels] cong diagnostic summary → {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Batched runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_batch(raw: dict, params: dict, B_init: int, device_str: str,
               init_positions=None, cluster_data=None) -> list:
    """
    If `init_positions` is provided ([B, n_mov, 2]), Stage 1 is skipped
    (positions feed straight into Stage 2). Used by basin-hop and the
    soft-polish stage to re-run Stage 2 from given positions.

    If `cluster_data` is provided (and init_positions is None), hard macros
    are initialised at their cluster's spatial centre (from
    cluster_data['centers_canvas'][cluster_data['labels'][i]]) plus a
    per-restart per-macro jitter. Soft macros remain randomly initialised.
    Stage 1 + Stage 2 still run normally — clustering only changes WHERE
    the optimiser starts.

    cluster_data shape:
        {'K': int, 'labels': [nH], 'centers_canvas': [K, 2]}
    """
    device = torch.device(device_str)
    dtype  = torch.float32

    cw, ch         = raw['cw'], raw['ch']
    nH, nM, nP     = raw['nH'], raw['nM'], raw['nP']
    G_rows, G_cols = raw['G_rows'], raw['G_cols']
    hw_np          = raw['hw_np']
    hh_np          = raw['hh_np']
    movable_np     = raw['movable_mask']
    movable_idx    = np.where(movable_np)[0]
    num_nets       = raw['num_nets']
    soft_deg_np    = raw['soft_degrees']
    nS             = nM - nH

    node_ids_t    = torch.tensor(raw['node_ids'],    dtype=torch.long,  device=device)
    net_ids_t     = torch.tensor(raw['net_ids'],     dtype=torch.long,  device=device)
    net_weights_t = torch.tensor(raw['net_weights'], dtype=dtype,       device=device)
    hw_t          = torch.tensor(hw_np,              dtype=dtype,       device=device)
    hh_t          = torch.tensor(hh_np,              dtype=dtype,       device=device)
    movable_idx_t = torch.tensor(movable_idx.tolist(), dtype=torch.long, device=device)
    n_mov         = len(movable_idx)

    # ── Routing parameters (parsed from initial.plc) ────────────────────────
    hpm = float(params.get('hroutes_per_micron', 0.0))
    vpm = float(params.get('vroutes_per_micron', 0.0))
    hra = float(params.get('hrouting_alloc',     0.0))
    vra = float(params.get('vrouting_alloc',     0.0))

    # ── Hard inflation ──────────────────────────────────────────────────────
    inflation = params.get('inflation_factor', 1.1)
    soft_inflation_scale = params.get('soft_inflation_scale', 4.0)
    soft_inflation_alpha = params.get('soft_inflation_alpha', 0.5)
    soft_inflation_max   = params.get('soft_inflation_max',   25.0)
    soft_inflation_clamp = params.get('soft_inflation_clamp', 0.20)  # fraction of canvas side

    hw_t_inf = hw_t.clone()
    hh_t_inf = hh_t.clone()
    hw_t_inf[:nH] = hw_t[:nH] * inflation
    hh_t_inf[:nH] = hh_t[:nH] * inflation

    # ── Soft inflation by degree ─────────────────────────────────────────────
    ss_pair_weight = None
    if soft_inflation_scale > 0 and nS > 0 and soft_deg_np.size > 0:
        d_max = float(soft_deg_np.max()) if soft_deg_np.max() > 0 else 1.0
        d_norm = np.clip(soft_deg_np / d_max, 0.0, 1.0)
        soft_factor = 1.0 + soft_inflation_scale * (d_norm ** soft_inflation_alpha)
        soft_factor = np.clip(soft_factor, 1.0, soft_inflation_max)
        side_scale = np.sqrt(soft_factor).astype(np.float32)
        # Cap per-side at clamp * canvas
        new_hw_s = np.minimum(hw_np[nH:nM] * side_scale, soft_inflation_clamp * cw).astype(np.float32)
        new_hh_s = np.minimum(hh_np[nH:nM] * side_scale, soft_inflation_clamp * ch).astype(np.float32)
        hw_t_inf[nH:nM] = torch.tensor(new_hw_s, dtype=dtype, device=device)
        hh_t_inf[nH:nM] = torch.tensor(new_hh_s, dtype=dtype, device=device)

        # Per-pair weight matrix for soft-soft overlap: w_ij = (deg_i + deg_j) / (2 * d_max)
        # Normalised so an "average pair" has weight ~1.
        d_t = torch.tensor(soft_deg_np / d_max, dtype=dtype, device=device)
        ss_pair_weight = (d_t.unsqueeze(0) + d_t.unsqueeze(1)).clamp(min=0.1)

    hw_mov  = torch.tensor([hw_np[i] if i < nH else 0.0 for i in movable_idx], dtype=dtype, device=device)
    hh_mov  = torch.tensor([hh_np[i] if i < nH else 0.0 for i in movable_idx], dtype=dtype, device=device)
    x_lo_mov = hw_mov
    x_hi_mov = torch.full_like(hw_mov, cw) - hw_mov
    y_lo_mov = hh_mov
    y_hi_mov = torch.full_like(hh_mov, ch) - hh_mov

    base_np = np.zeros((nM + nP, 2), dtype=np.float64)
    base_np[:nM] = raw['macro_positions']
    if nP > 0:
        base_np[nM:] = raw['port_positions']
    fixed_bg = torch.tensor(base_np, dtype=dtype, device=device)

    if init_positions is None:
        B = B_init

        # ── Cluster-aware init for hard macros (multi-level) ──────────────
        # If cluster_data is provided, each hard macro starts at its
        # cluster's centre + a per-restart, per-macro jitter (jitter_frac of
        # canvas side). Soft macros stay random in canvas. The standard
        # Stage 1 ghost phase then refines from this clustered topology.
        cluster_centers = None
        cluster_labels  = None
        cluster_jitter  = float(params.get('cluster_jitter_frac', 0.04))
        if cluster_data is not None:
            cluster_centers = cluster_data['centers_canvas']  # [K, 2]
            cluster_labels  = cluster_data['labels']          # [nH]

        init_list = []
        for seed in range(B):
            rng     = np.random.RandomState(seed)
            seed_mv = []
            for i in movable_idx:
                if i < nM:
                    lo_x = hw_np[i] if i < nH else 0.0
                    hi_x = cw - (hw_np[i] if i < nH else 0.0)
                    lo_y = hh_np[i] if i < nH else 0.0
                    hi_y = ch - (hh_np[i] if i < nH else 0.0)

                    if cluster_centers is not None and i < nH:
                        # Hard macro: place at cluster centre + jitter
                        cx, cy = cluster_centers[cluster_labels[i]]
                        jx = rng.uniform(-cluster_jitter * cw, cluster_jitter * cw)
                        jy = rng.uniform(-cluster_jitter * ch, cluster_jitter * ch)
                        x = float(np.clip(cx + jx, max(lo_x, 0.0), min(hi_x, cw)))
                        y = float(np.clip(cy + jy, max(lo_y, 0.0), min(hi_y, ch)))
                        seed_mv.append([x, y])
                    else:
                        # Random init (soft macros, or no cluster data)
                        seed_mv.append([rng.uniform(max(lo_x, 0.0), min(hi_x, cw)),
                                         rng.uniform(max(lo_y, 0.0), min(hi_y, ch))])
                else:
                    seed_mv.append(base_np[i].tolist())
            init_list.append(seed_mv)

        movable_params = torch.nn.Parameter(
            torch.tensor(init_list, dtype=dtype, device=device)
        )
    else:
        # Stage 1 was done by the caller (orchestrator). Use as-is.
        if isinstance(init_positions, torch.Tensor):
            ip = init_positions.detach().to(device=device, dtype=dtype).contiguous().clone()
        else:
            ip = torch.tensor(np.asarray(init_positions), dtype=dtype, device=device)
        movable_params = torch.nn.Parameter(ip)
        B = movable_params.shape[0]

    def _make_scatter_idx(b):
        b_idx = torch.arange(b, device=device).view(b, 1).expand(b, n_mov).reshape(-1)
        m_idx = movable_idx_t.unsqueeze(0).expand(b, -1).reshape(-1)
        return b_idx, m_idx

    b_idx_full, m_idx_full = _make_scatter_idx(B)

    movable_is_prefix = bool(
        n_mov <= (nM + nP) and np.array_equal(movable_idx, np.arange(n_mov))
    )
    fixed_tail = fixed_bg[n_mov:].unsqueeze(0) if movable_is_prefix else None

    def make_full_pos(params_b, b_idx, m_idx):
        b = params_b.shape[0]
        if movable_is_prefix:
            if fixed_tail.shape[1] == 0:
                return params_b
            return torch.cat((params_b, fixed_tail.expand(b, -1, -1)), dim=1)
        base = fixed_bg.unsqueeze(0).expand(b, -1, -1).clone()
        return base.index_put((b_idx, m_idx), params_b.reshape(-1, 2))

    def project(param):
        with torch.no_grad():
            param.data[:, :, 0].clamp_(x_lo_mov, x_hi_mov)
            param.data[:, :, 1].clamp_(y_lo_mov, y_hi_mov)

    def gamma_at(step, total, g_start, g_end):
        return g_start * (g_end / g_start) ** (step / max(total - 1, 1))

    g_rudy   = params['gamma_rudy']
    cong_thr = params['congestion_threshold']
    cong_eps = params['cong_eps']
    c_sigma  = float(params.get('cong_sigma_l',  1.0))
    c_abu    = float(params.get('cong_abu_frac', 0.05))
    c_scale  = float(params.get('cong_scale',    1.0))

    # ── Congestion kernel selector ──────────────────────────────────────────
    #   use_cong_v2=True  -> the faithful star-from-driver port with the
    #     __smooth_routing_cong box-blur (default for v60). Falls back to the
    #     v1 bbox-midpoint proxy if pin-level connectivity is unavailable.
    #   cong_fn(pos_b) is the single entry point used by Stage 1 / Stage 2 /
    #   the final evaluation below.
    use_cong_v2 = bool(params.get('use_cong_v2', True))
    cong_v2_data = _build_cong_v2_data(raw) if use_cong_v2 else None
    if use_cong_v2 and cong_v2_data is None:
        use_cong_v2 = False
    if use_cong_v2:
        pin_owner_t = torch.tensor(cong_v2_data['pin_owner'], dtype=torch.long, device=device)
        pin_off_t   = torch.tensor(cong_v2_data['pin_off'],   dtype=dtype,      device=device)
        e_drv_t     = torch.tensor(cong_v2_data['e_drv'], dtype=torch.long, device=device)
        e_snk_t     = torch.tensor(cong_v2_data['e_snk'], dtype=torch.long, device=device)
        e_w_t       = torch.tensor(cong_v2_data['e_w'],   dtype=dtype,      device=device)
        sr          = int(params.get('smooth_range', 0))
        M_row_t     = _build_cong_smooth_matrix(G_rows, sr, device, dtype)
        M_col_t     = _build_cong_smooth_matrix(G_cols, sr, device, dtype)
        cell_w_t    = float(cw) / float(G_cols)
        cell_h_t    = float(ch) / float(G_rows)
        col_c_t     = torch.arange(G_cols, device=device, dtype=dtype) + 0.5
        row_c_t     = torch.arange(G_rows, device=device, dtype=dtype) + 0.5
        cx_lo_t     = torch.arange(G_cols, device=device, dtype=dtype) * cell_w_t
        cx_hi_t     = cx_lo_t + cell_w_t
        cy_lo_t     = torch.arange(G_rows, device=device, dtype=dtype) * cell_h_t
        cy_hi_t     = cy_lo_t + cell_h_t
        snap_sigma  = float(params.get('cong_snap_sigma', 0.5))
        edge_chunk  = int(params.get('cong_edge_chunk', 8192))

        def cong_fn(pos_b):
            return _off_cong_v2_bc(
                pos_b, pin_owner_t, pin_off_t, e_drv_t, e_snk_t, e_w_t,
                cw, ch, G_rows, G_cols, hw_t, hh_t, nH,
                hpm, vpm, hra, vra, M_row_t, M_col_t,
                row_c_t, col_c_t, cx_lo_t, cx_hi_t, cy_lo_t, cy_hi_t,
                abu_frac=c_abu, snap_sigma=snap_sigma, cong_scale=c_scale,
                edge_chunk=edge_chunk)
    else:
        def cong_fn(pos_b):
            return _off_cong_bc(pos_b, node_ids_t, net_ids_t, num_nets, net_weights_t,
                                cw, ch, G_rows, G_cols, g_rudy, hw_t, hh_t, nH,
                                hpm, vpm, hra, vra, c_abu, cong_eps, c_sigma, c_scale)

    # ── Jacobi preconditioner (optional). Computes (1/v_i)/mean(1/v_i) over
    #    movable nodes — applied as in-place gradient scaling between
    #    backward() and opt.step() in both Stage 1 and Stage 2.
    use_precond = bool(params.get('use_preconditioner', False))
    v_inv_t     = None
    if use_precond:
        v_per_node = _compute_preconditioner(raw)        # [nM + nP], float64
        v_mov      = v_per_node[movable_idx]              # [n_mov]
        v_inv_np   = 1.0 / np.maximum(v_mov, 1e-9)
        # Normalize so mean=1 → preserves overall lr / lambda calibration.
        v_inv_np   = v_inv_np / max(v_inv_np.mean(), 1e-12)
        v_inv_t    = torch.tensor(
            v_inv_np, dtype=dtype, device=device,
        ).view(1, n_mov, 1)

    if init_positions is None:
        # ── Stage 1 (ghost phase, free exploration) ──────────────────────────
        ns1    = params['num_steps_s1']
        lc_s1  = params['lambda_cong_s1']
        cong_interval_s1 = max(1, int(params.get('cong_eval_interval_s1', 1)))
        opt1   = optim.Adam([movable_params], lr=params['lr_s1'])
        sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, ns1, eta_min=params['lr_s1'] * 0.01)

        for step in range(ns1):
            g = gamma_at(step, ns1, params['gamma_s1_start'], params['gamma_s1_end'])
            opt1.zero_grad()
            pos_b  = make_full_pos(movable_params, b_idx_full, m_idx_full)
            wl_b   = _wa_wl_bc(pos_b, node_ids_t, net_ids_t, num_nets, net_weights_t, g)
            eval_cong = (lc_s1 != 0.0) and (step % cong_interval_s1 == 0 or step == ns1 - 1)
            if eval_cong:
                cong_b = cong_fn(pos_b)
                cong_loss_b = cong_b * float(cong_interval_s1)
            else:
                cong_b = wl_b.new_zeros(wl_b.shape)
                cong_loss_b = cong_b
            loss_seed_s1 = wl_b + lc_s1 * cong_loss_b           # per-seed [B]
            loss_seed_s1.sum().backward()
            if v_inv_t is not None and movable_params.grad is not None:
                movable_params.grad.mul_(v_inv_t)
            opt1.step(); sched1.step(); project(movable_params)

        # No Stage-1 pruning. We never rank seeds with the differentiable
        # proxy — the only valid ranker is compute_proxy_cost, which runs
        # once per seed at the end (in the cohort placer).

    # ── Stage 2 (spreading + repulsion + congestion) ─────────────────────────
    ns2     = params['num_steps_s2']
    lc_s2   = params['lambda_cong_s2']
    cong_interval_s2 = max(1, int(params.get('cong_eval_interval_s2', 1)))
    ld_end  = params['lambda_density_end']

    ld_hs_start = params['lambda_hs_overlap_start']
    ld_hs_end   = params['lambda_hs_overlap_end']
    ld_hh_start = params['lambda_hh_overlap_start']
    ld_hh_end   = params['lambda_hh_overlap_end']
    ld_ss_start = params['lambda_ss_overlap_start']
    ld_ss_end   = params['lambda_ss_overlap_end']
    ramp_frac   = params.get('overlap_ramp_frac', 0.6)  # reach end value at ramp_frac * ns2

    hh_active = (ld_hh_start != 0.0) or (ld_hh_end != 0.0)
    ss_active = (ld_ss_start != 0.0) or (ld_ss_end != 0.0)

    target_density = params['target_density']
    opt2    = optim.Adam([movable_params], lr=params['lr_s2'])
    sched2  = optim.lr_scheduler.CosineAnnealingLR(opt2, ns2, eta_min=params['lr_s2'] * 0.005)

    use_bf16     = bool(params.get('s2_bf16', False)) and (device.type == 'cuda')
    autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16)

    def linear_ramp(step, total, start, end, frac):
        if frac <= 0.0:
            return float(end)
        progress = min(1.0, (step / max(total - 1, 1)) / frac)
        return float(start + (end - start) * progress)

    # No mid-stage pruning. Every seed runs all the way through Stage 2 and
    # is ranked at the end by compute_proxy_cost — never by the
    # differentiable proxy.

    for step in range(ns2):
        g      = gamma_at(step, ns2, params['gamma_s2_start'], params['gamma_s2_end'])
        lam_d  = ld_end * (step / max(ns2 - 1, 1))
        lam_hs = linear_ramp(step, ns2, ld_hs_start, ld_hs_end, ramp_frac)
        lam_hh = linear_ramp(step, ns2, ld_hh_start, ld_hh_end, ramp_frac) if hh_active else 0.0
        lam_ss = linear_ramp(step, ns2, ld_ss_start, ld_ss_end, ramp_frac) if ss_active else 0.0
        opt2.zero_grad()
        with autocast_ctx:
            pos_b       = make_full_pos(movable_params, b_idx_full, m_idx_full)
            pos_macro_b = pos_b[:, :nM, :]
            wl_b      = _wa_wl_bc(pos_b, node_ids_t, net_ids_t, num_nets, net_weights_t, g)
            den_b     = _density_bc(pos_macro_b, hw_t_inf, hh_t_inf, cw, ch, G_rows, G_cols, target_density)
            hs_ovlp_b = _hs_ovlp_bc(pos_macro_b, hw_t, hh_t, nH, nM)
            eval_cong = (lc_s2 != 0.0) and (step % cong_interval_s2 == 0 or step == ns2 - 1)
            if eval_cong:
                cong_b = cong_fn(pos_b)
                cong_loss_b = cong_b * float(cong_interval_s2)
            else:
                cong_b = wl_b.new_zeros(wl_b.shape)
                cong_loss_b = cong_b
            hh_ovlp_b = _hh_ovlp_bc(pos_macro_b, hw_t, hh_t, nH) if hh_active else None
            ss_ovlp_b = _ss_ovlp_bc(pos_macro_b, hw_t_inf, hh_t_inf, nH, nM, ss_pair_weight) if ss_active else None
            loss = (wl_b
                    + lam_d  * den_b
                    + lam_hs * hs_ovlp_b
                    + lc_s2  * cong_loss_b)
            if hh_active:
                loss = loss + lam_hh * hh_ovlp_b
            if ss_active:
                loss = loss + lam_ss * ss_ovlp_b
        loss.sum().backward()
        if v_inv_t is not None and movable_params.grad is not None:
            movable_params.grad.mul_(v_inv_t)
        opt2.step(); sched2.step(); project(movable_params)

    with torch.no_grad():
        final_b = make_full_pos(movable_params, b_idx_full, m_idx_full)
        final_np = final_b.cpu().numpy()[:, :nM, :].astype(np.float64)

        g_final          = params['gamma_s2_end']
        lam_d_final      = ld_end
        lam_hs_final     = ld_hs_end
        pos_macro_final  = final_b[:, :nM, :]
        wl_final   = _wa_wl_bc(final_b, node_ids_t, net_ids_t, num_nets, net_weights_t, g_final)
        den_final  = _density_bc(pos_macro_final, hw_t_inf, hh_t_inf, cw, ch, G_rows, G_cols, target_density)
        hs_final   = _hs_ovlp_bc(pos_macro_final, hw_t, hh_t, nH, nM)
        cong_final = cong_fn(final_b)
        if hh_active:
            hh_final     = _hh_ovlp_bc(pos_macro_final, hw_t, hh_t, nH)
            lam_hh_final = ld_hh_end
        else:
            hh_final     = torch.zeros_like(wl_final)
            lam_hh_final = 0.0
        if ss_active:
            ss_final     = _ss_ovlp_bc(pos_macro_final, hw_t_inf, hh_t_inf, nH, nM, ss_pair_weight)
            lam_ss_final = ld_ss_end
        else:
            ss_final     = torch.zeros_like(wl_final)
            lam_ss_final = 0.0
        total_final = (wl_final
                       + lam_d_final  * den_final
                       + lam_hs_final * hs_final
                       + lc_s2        * cong_final
                       + lam_hh_final * hh_final
                       + lam_ss_final * ss_final)
        wl_arr   = wl_final.cpu().numpy()
        den_arr  = den_final.cpu().numpy()
        hs_arr   = hs_final.cpu().numpy()
        hh_arr   = hh_final.cpu().numpy()
        ss_arr   = ss_final.cpu().numpy()
        cong_arr = cong_final.cpu().numpy()
        tot_arr  = total_final.cpu().numpy()

    hwH = hw_np[:nH]
    hhH = hh_np[:nH]

    results = []
    metrics = []
    for b in range(B):
        legalized = _split_push_legalize(final_np[b], nH, cw, ch, hwH, hhH, movable_np[:nH], params['safety_gap'])
        results.append(legalized)
        metrics.append({
            'internal_wl':         float(wl_arr[b]),
            'internal_density':    float(den_arr[b]),
            'internal_hs_overlap': float(hs_arr[b]),
            'internal_hh_overlap': float(hh_arr[b]),
            'internal_ss_overlap': float(ss_arr[b]),
            'internal_cong':       float(cong_arr[b]),
            'internal_total':      float(tot_arr[b]),
            'gamma_final':              float(g_final),
            'lambda_density_final':     float(lam_d_final),
            'lambda_hs_overlap_final':  float(lam_hs_final),
            'lambda_hh_overlap_final':  float(lam_hh_final),
            'lambda_ss_overlap_final':  float(lam_ss_final),
            'lambda_cong_final':        float(lc_s2),
        })

    return results, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  v60: basin-hopping wrapper (perturb-and-reminimize with a promising-seed
#       priority queue + visited-basin tabu list)
# ──────────────────────────────────────────────────────────────────────────────
#  Layered ABOVE the multi-restart Stage-2 minimiser. The orchestrator runs the
#  normal Stage-0/1/2 pipeline once (the "initial basin"), then this driver
#  repeatedly: pops the queued seed with the best parent proxy, perturbs its
#  movable macros B ways (Gaussian, per-row std stratified across sigma_set when
#  stratify=True — so a single hop covers the productive σ band), re-runs
#  Stage 2 from there, and records the result. An improving result resets the
#  no-improve counter and spawns fresh perturbations of itself; a non-improving
#  result still spawns ONE small perturbation (the "take the least-bad move
#  when stuck" rule) but queued behind better seeds. A result whose movable
#  displacement / scale < tabu_eps OR whose |Δproxy| from any visited result
#  < tabu_proxy_eps is treated as the same minimum and does not spawn. After
#  the hop budget, the top final_explore_n still-queued seeds are each run
#  once. improve_quota > 0 ends the hop loop early after that many consecutive
#  non-improving hops (0 = run the full budget).


def _basin_hop(run_from_init, initial, movable_idx, scale, B, rng, *,
               max_hops: int = 24,
               final_explore_n: int = 8,
               sigma_set=(0.015, 0.025, 0.035),
               tabu_eps: float = 0.01,
               tabu_proxy_eps: float = 0.005,
               improve_quota: int = 3,
               min_improve_frac: float = 0.008,
               stratify: bool = True,
               log=print):
    """
    run_from_init(init[B, n_mov, 2] float32 ndarray) -> dict with keys
        'pos'          : [nM, 2] ndarray (best-of-batch placement)
        'proxy'        : float    (real proxy of that placement)
        'overlap_area' : float    (hard-macro overlap area; <=1e-9 == legal)
    initial: the same dict for the already-run initial (Stage-0/1/2) basin.
    movable_idx: int ndarray, indices into the [nM] axis of the movable macros.
    scale: positive float — perturbation std is sigma * scale per coordinate
           (e.g. 0.5 * (canvas_w + canvas_h)).
    B: batch size (restarts) per hop.   rng: numpy Generator.

    sigma_set: perturb std (as a fraction of `scale`) for the productive band.
        When stratify=True (default), the B-row cohort is split across these
        sigmas (round-robin), so the best-of-B selects whichever sigma helped
        for this seed — no schedule needed, no destructive kicks if the set
        excludes them.
    tabu_eps / tabu_proxy_eps: a result is flagged "same basin" if its mean
        macro displacement / scale < tabu_eps OR |Δproxy| from any visited
        result < tabu_proxy_eps. Two complementary signals: spatial proximity
        and cost-equivalence. Pass tabu_proxy_eps=0.0 to disable the cost gate.
    improve_quota: end the hop loop early after that many consecutive non-
        improving hops (0 = run the full budget). Useful when σ has been
        de-tuned to the productive band — improvement stops fast.
    min_improve_frac: minimum fractional proxy reduction that counts as a
        "meaningful improvement" for the quota / sigma-push logic. A hop that
        reduces proxy by less than this fraction of the current incumbent
        still becomes the new incumbent (we accept the small gain), but it
        counts as a non-improving hop for `improve_quota` and the PQ push
        falls back to the smallest σ. 0.0 = legacy behaviour (any reduction
        counts). Default 0.008 (0.8%).

    Returns the best legal result if any, else the best result overall.
    """
    import heapq
    incumbent      = dict(initial)
    best_legal     = dict(initial) if initial.get('overlap_area', 1.0) <= 1e-9 else None
    visited        = [np.asarray(initial['pos'])]
    visited_proxy  = [float(initial['proxy'])]
    counter        = [0]
    pq             = []

    # Per-row sigma assignment for the stratified batch: cycle sigma_set across
    # the B restarts so each hop covers the productive band in one shot. The
    # cohort's best-of-B then naturally picks whichever sigma helped.
    sig_vec = np.asarray(
        [sigma_set[i % len(sigma_set)] for i in range(B)],
        dtype=np.float32,
    )  # [B]

    def _push(parent_pos, parent_proxy, sigmas):
        # In stratify mode the popped sigma is irrelevant (the batch covers
        # the whole sigma_set), so we collapse the multi-sigma push into one
        # entry per parent.
        push_sigmas = (sigma_set[0],) if stratify else sigmas
        for s in push_sigmas:
            heapq.heappush(pq, (float(parent_proxy), counter[0], np.asarray(parent_pos), float(s)))
            counter[0] += 1

    def _is_dup(pos, proxy):
        pm = np.asarray(pos)[movable_idx]
        for v, vp in zip(visited, visited_proxy):
            d = float(np.mean(np.linalg.norm(pm - v[movable_idx], axis=1))) / max(scale, 1e-12)
            if d < tabu_eps:
                return True
            if tabu_proxy_eps > 0.0 and abs(float(proxy) - float(vp)) < tabu_proxy_eps:
                return True
        return False

    def _perturb_and_run(parent_pos, sigma):
        base_mov = np.asarray(parent_pos)[movable_idx].astype(np.float32)          # [n_mov, 2]
        if stratify:
            noise = (rng.standard_normal((B, base_mov.shape[0], 2)).astype(np.float32)
                     * (sig_vec[:, None, None] * scale))
        else:
            noise = (rng.standard_normal((B, base_mov.shape[0], 2)).astype(np.float32)
                     * (float(sigma) * scale))
        return run_from_init(base_mov[None, :, :] + noise)

    _push(initial['pos'], initial['proxy'], sigma_set)
    no_improve = 0
    hops       = 0
    sig_label  = 'strat' if stratify else None
    while pq and hops < max_hops:
        if improve_quota and no_improve >= improve_quota:
            log(f"[basin-hop] improve-quota {improve_quota} hit after {hops} hops; ending hop loop")
            break
        parent_proxy, _, parent_pos, sigma = heapq.heappop(pq)
        hops += 1
        res = _perturb_and_run(parent_pos, sigma)
        dup = _is_dup(res['pos'], res['proxy'])
        visited.append(np.asarray(res['pos']))
        visited_proxy.append(float(res['proxy']))
        legal = res.get('overlap_area', 1.0) <= 1e-9
        if legal and (best_legal is None or res['proxy'] < best_legal['proxy']):
            best_legal = dict(res)
        # v60: split "any improvement" from "meaningful improvement". Any
        # reduction below incumbent is accepted as the new incumbent (we don't
        # throw away free progress), but only a reduction >= min_improve_frac
        # of the current incumbent resets the quota counter and re-pushes the
        # full sigma_set. Tiny gains count as non-improving for the quota /
        # sigma-fallback logic so the loop still escalates / ends on time.
        improved      = res['proxy'] < incumbent['proxy'] - 1e-6
        rel_drop      = (incumbent['proxy'] - res['proxy']) / max(abs(incumbent['proxy']), 1e-12)
        meaningful    = improved and (rel_drop >= min_improve_frac)
        log_sig  = sig_label if sig_label is not None else f"{sigma:.3f}"
        if improved and not meaningful:
            improved_tag = f" *IMPROVED(+{rel_drop*100:.2f}% <thresh)"
        elif meaningful:
            improved_tag = f" *IMPROVED(+{rel_drop*100:.2f}%)"
        else:
            improved_tag = ""
        log(f"[basin-hop {hops}/{max_hops}] sigma={log_sig} parent={parent_proxy:.4f} "
            f"-> proxy={res['proxy']:.4f} ovlp={res.get('overlap_area', float('nan')):.4g}"
            f"{improved_tag}{' (dup basin)' if dup else ''}")
        if improved:
            # Always accept the new (lower) proxy as the incumbent.
            incumbent = dict(res)
        if meaningful:
            no_improve = 0
            if not dup:
                _push(res['pos'], res['proxy'], sigma_set)
        else:
            no_improve += 1
            if not dup:
                _push(res['pos'], res['proxy'], (min(sigma_set),))
        # In stratify mode, _push collapses to one entry per parent so the PQ
        # holds at most a single seed in steady state. A "not-pushed" outcome
        # (e.g. dup-basin) would drain it and end the loop early — before
        # improve_quota / max_hops actually fired. Refill from incumbent so
        # the loop terminates on the intended budget, not on PQ exhaustion.
        if stratify and not pq:
            _push(incumbent['pos'], incumbent['proxy'], sigma_set)

    # In stratify mode the main loop leaves the PQ at most one entry deep;
    # seed final_explore directly from incumbent so it actually runs.
    if stratify:
        while len(pq) < final_explore_n:
            _push(incumbent['pos'], incumbent['proxy'], sigma_set)

    explored = 0
    while pq and explored < final_explore_n:
        parent_proxy, _, parent_pos, sigma = heapq.heappop(pq)
        explored += 1
        res = _perturb_and_run(parent_pos, sigma)
        legal = res.get('overlap_area', 1.0) <= 1e-9
        if legal and (best_legal is None or res['proxy'] < best_legal['proxy']):
            best_legal = dict(res)
        if res['proxy'] < incumbent['proxy'] - 1e-6:
            incumbent = dict(res)
        log_sig = sig_label if sig_label is not None else f"{sigma:.3f}"
        log(f"[basin-hop final {explored}/{final_explore_n}] sigma={log_sig} "
            f"-> proxy={res['proxy']:.4f} ovlp={res.get('overlap_area', float('nan')):.4g}")

    out = best_legal if best_legal is not None else incumbent
    log(f"[basin-hop] done: {hops} hops + {explored} final-explore;  "
        f"incumbent proxy={incumbent['proxy']:.4f}  "
        f"best_legal={('%.4f' % best_legal['proxy']) if best_legal else 'none'}")
    return out


