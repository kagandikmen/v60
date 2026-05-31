"""
v60 IncrementalEval — incremental proxy cost matching plc_client_os PlacementCost.

PLC reference (external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py):
  - WL cost     = get_wirelength() / ((W + H) * net_cnt)
                  where get_wirelength sums weight * (max-min)_x + (max-min)_y per net.
  - Density cost = 0.5 * mean(top-10% of cell_density)
                  where cell_density[c] = sum of macro overlap area with cell / cell area.
  - Cong cost   = mean(top-5% of concat(V_routing_cong, H_routing_cong))
                  (V/H demand grids built by get_routing() — L-route per edge
                   + box-blur smoothing + macro blockage. See PLC line 1514.)
  - proxy       = 1.0 * WL + 0.5 * Density + 0.5 * Cong   (challenge weights)

Implementation status (2026-05-26):
  ✅  WL: fully incremental. Per-net bbox cached; recomputed only for nets
       touched by the moving macro.
  ✅  Density: fully incremental. Per-cell occupied-area cached; macro footprint
       subtracted from old cells and added to new cells on move.
  ✅  Congestion: fully incremental. raw_V/raw_H (per-net L-routes) and
       macro_raw_V/macro_raw_H (hard-macro blockage) are maintained as
       un-normalised cell grids. On commit_move(m, new_xy) we subtract the
       contributions of nets touching m (and m's blockage if hard), apply the
       move, and add the new contributions back. compute_cong_cost_full()
       then normalises, smooths the net portion, adds the macro portion, and
       runs ABU top-5% — all O(grid_size), no per-net rebuild.

Bit-perfect against PLC (validated on ibm01, 30 random moves):
  max |Δ WL  cost|: 1.19e-7   (float32-vs-float64 noise)
  max |Δ Den cost|: 7.07e-7   (float32-vs-float64 noise)
  max |Δ Cong cost|: 6.30e-9  (float32-vs-float64 noise)

Critical PLC behaviors mirrored:
  - `__overlap_dist` couples x_diff and y_diff: returns (0, 0) if either is
    non-positive (not clamped independently).
  - Bbox computation uses numpy.float32 arithmetic to match PLC's NEP 50
    behavior under numpy 2.0+ (PLC's macro positions are float32 from the
    placement tensor; `mod_x + mod_w/2` stays float32). Without this,
    `floor(x_hi / grid_w)` mis-floors by ±1 cell at any macro whose corner
    lands within ~2^-22 of a grid line. Same fix applied to pin positions
    for net routing.

Usage:
    e = IncrementalEval(benchmark, plc=plc)
    e.set_placement(placement_np)            # initialize from a full placement
    proxy0 = e.proxy()                       # base proxy (matches compute_proxy_cost)
    d = e.delta_for_move(macro_idx, new_xy)  # returns dict of deltas, state unchanged
    if accept:
        e.commit_move(macro_idx, new_xy)     # apply

Validation:
    validate_against_plc(benchmark, plc) runs N random moves and compares
    incremental proxy against compute_proxy_cost on each step.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place._plc import PlacementCost
from macro_place.objective import compute_proxy_cost


# ════════════════════════════════════════════════════════════════════════════
#  Constants / proxy weights
# ════════════════════════════════════════════════════════════════════════════

WEIGHT_WL      = 1.0
WEIGHT_DENSITY = 0.5
WEIGHT_CONG    = 0.5

ABU_FRAC_DENSITY = 0.10   # PLC get_density_cost: top-10%
DENSITY_HALF     = 0.5    # PLC: returns 0.5 * mean(top-10%)

ABU_FRAC_CONG    = 0.05   # PLC get_congestion_cost: top-5% of (V + H)


# ════════════════════════════════════════════════════════════════════════════
#  IncrementalEval
# ════════════════════════════════════════════════════════════════════════════

class IncrementalEval:
    """Maintains proxy cost incrementally over single-macro moves.

    Internal state (set up by __init__ / set_placement):
      WL caches:
        pin_owner   [P]    int   — owner node index (macro 0..nM-1 or port nM..nM+nP-1)
        pin_offset  [P,2]  float — pin offset from owner centre (0 for ports)
        pin_xy      [P,2]  float — current absolute pin position
        pins_per_macro[m]  list  — pin indices owned by macro m
        net_pins[n]        list  — pin indices on net n (driver-first; order matches benchmark)
        net_weight  [N]    float
        net_min_x   [N]    float — current bbox min_x for net
        net_max_x   [N]    float
        net_min_y   [N]    float
        net_max_y   [N]    float
        net_hpwl    [N]    float — current weighted HPWL of net (un-normalised)
        total_hpwl         float — sum(net_hpwl)

      Density caches:
        grid_w, grid_h          — cell extents
        grid_occupied [G_r, G_c] float — area-of-overlap per cell (un-normalised)
        macro_overlap[m]        dict {(row, col) -> overlap_area} — what macro m contributes per cell
                                (rebuilt incrementally; used for O(1) undo of contribution)

      Cong caches:
        raw_V, raw_H       [G_r*G_c] float — flat per-cell net routing
                                             demand (V/H), updated per move
        macro_raw_V, macro_raw_H [G_r*G_c] float — flat per-cell hard-macro
                                             blockage demand (V/H)

      Misc:
        macro_pos   [nM, 2]
        macro_w     [nM]
        macro_h     [nM]
        canvas_w, canvas_h
        G_rows, G_cols
        cw_plus_ch_times_netcnt  — denominator for WL cost normalisation
    """

    def __init__(self, benchmark: Benchmark, plc: Optional[PlacementCost] = None):
        self.benchmark = benchmark
        self.plc       = plc        # only used for `compute_cong()` until incremental cong is done

        self.nM = int(benchmark.num_macros)
        self.nH = int(benchmark.num_hard_macros)
        self.nP = int(benchmark.port_positions.shape[0])
        self.N  = int(len(benchmark.net_nodes))

        self.canvas_w = float(benchmark.canvas_width)
        self.canvas_h = float(benchmark.canvas_height)
        self.G_rows   = int(benchmark.grid_rows)
        self.G_cols   = int(benchmark.grid_cols)
        self.grid_w   = self.canvas_w / self.G_cols
        self.grid_h   = self.canvas_h / self.G_rows
        self.cell_area = self.grid_w * self.grid_h

        # Macro sizes / positions (will be updated by set_placement).
        # When PLC is available, read sizes directly from PLC's HardMacro/SoftMacro
        # objects (full float64 precision). benchmark.macro_sizes is stored as
        # float32 — round-tripping through that introduces ~1e-7 noise at the
        # knife-edge `x_lo = bl_col*gw` cases, which flips bl_col by ±1 in
        # `_grid_cell_for_pos` and breaks the partial-overlap flag.
        if plc is not None:
            self.macro_w = np.zeros(self.nM, dtype=np.float64)
            self.macro_h = np.zeros(self.nM, dtype=np.float64)
            hard_plc = list(getattr(plc, 'hard_macro_indices', []))
            soft_plc = list(getattr(plc, 'soft_macro_indices', []))
            for b_idx, p_idx in enumerate(hard_plc):
                mod = plc.modules_w_pins[int(p_idx)]
                self.macro_w[b_idx] = float(mod.get_width())
                self.macro_h[b_idx] = float(mod.get_height())
            for off, p_idx in enumerate(soft_plc):
                mod = plc.modules_w_pins[int(p_idx)]
                self.macro_w[self.nH + off] = float(mod.get_width())
                self.macro_h[self.nH + off] = float(mod.get_height())
        else:
            sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
            self.macro_w = sizes[:, 0].copy()
            self.macro_h = sizes[:, 1].copy()
        self.macro_pos = np.zeros((self.nM, 2), dtype=np.float64)

        # Port positions (immovable). Pull from PLC for full float64 precision
        # (same reason as macro sizes).
        if plc is not None:
            port_plc = list(getattr(plc, 'port_indices', []))
            if port_plc:
                pp = np.zeros((len(port_plc), 2), dtype=np.float64)
                for i, p_idx in enumerate(port_plc):
                    px, py = plc.modules_w_pins[int(p_idx)].get_pos()
                    pp[i, 0] = float(px); pp[i, 1] = float(py)
                self.port_pos = pp
            else:
                self.port_pos = np.zeros((0, 2), dtype=np.float64)
        elif self.nP > 0:
            self.port_pos = benchmark.port_positions.cpu().numpy().astype(np.float64)
        else:
            self.port_pos = np.zeros((0, 2), dtype=np.float64)

        # ── Pin data ────────────────────────────────────────────────────────
        # When PLC is available, rebuild pin tables directly from PLC (which
        # has soft-macro pin offsets that the benchmark.net_pin_nodes drops).
        # Without PLC, fall back to benchmark's tables (works for hard pins +
        # ports but loses soft-macro pin offsets).
        if plc is not None:
            self._build_pin_tables_from_plc(plc, benchmark)
        else:
            self._build_pin_tables(benchmark)

        # ── Net data ────────────────────────────────────────────────────────
        # Prefer PLC-derived per-driver weights when _build_pin_tables_from_plc
        # has set them. macro_place.loader hard-codes net_weights = 1.0,
        # which doesn't match PLC's get_wirelength (which uses the driver
        # pin's get_weight() per net).
        if getattr(self, '_plc_net_weights', None) is not None and \
                self._plc_net_weights.shape[0] == self.N:
            self.net_weight = self._plc_net_weights.astype(np.float64).copy()
        else:
            bench_weights = benchmark.net_weights.cpu().numpy().astype(np.float64)
            if bench_weights.shape[0] >= self.N:
                self.net_weight = bench_weights[:self.N].copy()
            else:
                self.net_weight = np.ones(self.N, dtype=np.float64)
                self.net_weight[:bench_weights.shape[0]] = bench_weights
        self.net_min_x = np.zeros(self.N, dtype=np.float64)
        self.net_max_x = np.zeros(self.N, dtype=np.float64)
        self.net_min_y = np.zeros(self.N, dtype=np.float64)
        self.net_max_y = np.zeros(self.N, dtype=np.float64)
        self.net_hpwl  = np.zeros(self.N, dtype=np.float64)
        self.total_hpwl = 0.0

        # ── Density caches ──────────────────────────────────────────────────
        self.grid_occupied = np.zeros((self.G_rows, self.G_cols), dtype=np.float64)
        # macro_overlap[m]: dict (row,col) -> overlap_area contributed by macro m
        # (rebuilt for the moving macro on each commit; small per-macro)
        self.macro_overlap = [None] * self.nM

        # ── Cong caches ─────────────────────────────────────────────────────
        # Routing-resource parameters (read from PLC). Stored once at init.
        if plc is not None:
            self.vrouting_alloc     = float(getattr(plc, 'vrouting_alloc', 0.0))
            self.hrouting_alloc     = float(getattr(plc, 'hrouting_alloc', 0.0))
            self.vroutes_per_micron = float(getattr(plc, 'vroutes_per_micron', 0.0))
            self.hroutes_per_micron = float(getattr(plc, 'hroutes_per_micron', 0.0))
            self.smooth_range       = int(getattr(plc, 'smooth_range', 0))
        else:
            self.vrouting_alloc     = 0.0
            self.hrouting_alloc     = 0.0
            self.vroutes_per_micron = 0.0
            self.hroutes_per_micron = 0.0
            self.smooth_range       = 0
        self.grid_v_routes = self.grid_w * self.vroutes_per_micron
        self.grid_h_routes = self.grid_h * self.hroutes_per_micron
        # Per-net contributions to the raw (un-smoothed) routing grids. On
        # macro move, only nets touched by the moved macro's pins get
        # recomputed. Each entry: {cell_idx -> demand}.
        self._cong_net_V = [None] * self.N    # raw V demand per net (dict)
        self._cong_net_H = [None] * self.N    # raw H demand per net (dict)
        # Per-macro blockage contributions (hard macros only).
        self._cong_macro_V = [None] * self.nH
        self._cong_macro_H = [None] * self.nH
        # Aggregated raw grids and macro grids (un-normalised).
        self.raw_V       = np.zeros(self.G_rows * self.G_cols, dtype=np.float64)
        self.raw_H       = np.zeros(self.G_rows * self.G_cols, dtype=np.float64)
        self.macro_raw_V = np.zeros(self.G_rows * self.G_cols, dtype=np.float64)
        self.macro_raw_H = np.zeros(self.G_rows * self.G_cols, dtype=np.float64)
        # Final (post-normalise + smooth + macro-add) grids. Recomputed at
        # query time from raw_V/raw_H + macro_raw_V/macro_raw_H.
        # WL normalisation denominator. PLC normalises by `plc.net_cnt`,
        # which counts the FULL netlist (before the loader drops nets with
        # unresolved pins). We must use plc.net_cnt — not len(net_nodes) —
        # to match `compute_proxy_cost(...)['wirelength_cost']`.
        if plc is not None:
            plc_net_cnt = int(getattr(plc, 'net_cnt', self.N))
        else:
            plc_net_cnt = self.N
        self.plc_net_cnt    = max(1, plc_net_cnt)
        self.wl_denominator = (self.canvas_w + self.canvas_h) * self.plc_net_cnt

        # Sanity log so the user can spot empty-pin-table bugs immediately.
        n_nonempty_nets = sum(1 for pis in self.net_pins if pis.size > 0)
        print(f"[IncrementalEval] N={self.N}  P={self.P}  "
              f"nets_with_pins={n_nonempty_nets}  "
              f"plc.net_cnt={self.plc_net_cnt}  "
              f"macros={self.nM} (hard={self.nH}, ports={self.nP})")

    # ───────────────────────────────────────────────────────────────────────
    #  Setup
    # ───────────────────────────────────────────────────────────────────────

    def _build_pin_tables(self, benchmark: Benchmark) -> None:
        """Build pin_owner, pin_offset, pin_xy, pins_per_macro, net_pins.

        Uses the same pattern as v60_kernels._build_cong_v2_data: convert
        each tensor entry to a Python list with .tolist() before iterating.
        Lists let `for owner, slot in pins:` unpack cleanly without numpy-
        iteration surprises.
        """
        nH = self.nH
        nM = self.nM
        nP = self.nP

        # Convert net_pin_nodes (List[Tensor]) → list of [[owner, slot], ...].
        # When pin-level data is unavailable, fall back to net_nodes (each
        # entry is a 1-D tensor of owner indices; we synthesise slot=0).
        npn_lists = None
        if getattr(benchmark, 'net_pin_nodes', None):
            raw = list(benchmark.net_pin_nodes)
            if len(raw) > 0:
                npn_lists = []
                for t in raw:
                    if isinstance(t, torch.Tensor):
                        if t.numel() == 0:
                            npn_lists.append([])
                        else:
                            npn_lists.append(t.tolist())   # [[owner, slot], ...]
                    else:
                        a = np.asarray(t, dtype=np.int64)
                        npn_lists.append(a.tolist() if a.size > 0 else [])
        if npn_lists is None:
            npn_lists = []
            for nodes in benchmark.net_nodes:
                if isinstance(nodes, torch.Tensor):
                    ns = nodes.tolist()
                else:
                    ns = list(nodes)
                npn_lists.append([[int(o), 0] for o in ns])

        # macro_pin_offsets: List[Tensor of shape [num_pins_h, 2]] per hard macro.
        mpo_lists = []
        raw_mpo = getattr(benchmark, 'macro_pin_offsets', None)
        if raw_mpo:
            for t in raw_mpo:
                if isinstance(t, torch.Tensor):
                    mpo_lists.append(t.cpu().numpy().astype(np.float64))
                else:
                    mpo_lists.append(np.asarray(t, dtype=np.float64))

        pin_owner = []
        pin_offset = []
        net_pins = []     # list per net of arrays of pin indices

        for k, pins in enumerate(npn_lists):
            if not pins:
                net_pins.append(np.zeros(0, dtype=np.int64))
                continue
            net_pin_indices = []
            for entry in pins:
                # entry is either [owner, slot] (list-of-lists path) or
                # plain (owner, slot) tuple-like.
                owner = int(entry[0])
                slot  = int(entry[1]) if len(entry) > 1 else 0
                p_idx = len(pin_owner)
                pin_owner.append(owner)
                if (owner < nH and owner < len(mpo_lists)
                        and mpo_lists[owner] is not None
                        and mpo_lists[owner].shape[0] > slot):
                    off = mpo_lists[owner][slot]
                    pin_offset.append([float(off[0]), float(off[1])])
                else:
                    pin_offset.append([0.0, 0.0])
                net_pin_indices.append(p_idx)
            net_pins.append(np.asarray(net_pin_indices, dtype=np.int64))

        self.pin_owner  = np.asarray(pin_owner, dtype=np.int64)
        self.pin_offset = np.asarray(pin_offset, dtype=np.float64).reshape(-1, 2)
        self.P          = self.pin_owner.shape[0]
        self.pin_xy     = np.zeros((self.P, 2), dtype=np.float64)
        self.net_pins   = net_pins   # list of arrays

        # pins_per_macro: inverse mapping (only for macros — ports don't move).
        self.pins_per_macro = [[] for _ in range(nM)]
        for p_idx, owner in enumerate(self.pin_owner):
            if owner < nM:
                self.pins_per_macro[int(owner)].append(p_idx)
        # Convert to arrays for fast indexing.
        self.pins_per_macro = [np.asarray(lst, dtype=np.int64) for lst in self.pins_per_macro]

        # nets_per_macro: which nets are touched by each macro (precomputed for fast move loop).
        self.nets_per_pin = np.full(self.P, -1, dtype=np.int64)
        for net_idx, pin_idxs in enumerate(self.net_pins):
            self.nets_per_pin[pin_idxs] = net_idx
        # nets_per_macro[m] = sorted unique list of net indices touched by macro m
        self.nets_per_macro = []
        for m in range(nM):
            pin_idxs = self.pins_per_macro[m]
            if pin_idxs.size == 0:
                self.nets_per_macro.append(np.zeros(0, dtype=np.int64))
            else:
                touched = np.unique(self.nets_per_pin[pin_idxs])
                touched = touched[touched >= 0]
                self.nets_per_macro.append(touched)

    def _build_pin_tables_from_plc(self, plc: PlacementCost, benchmark: Benchmark) -> None:
        """Build pin tables directly from PLC.

        Why this exists: macro_place.loader stores `macro_pin_offsets` ONLY for
        hard macros. Soft-macro pin offsets are dropped — every soft-macro pin
        ends up at slot=0 with offset (0, 0). PLC's get_wirelength uses the
        actual soft-pin offsets, so an incremental WL built from benchmark
        alone systematically under-counts HPWL by ~5–10%.

        This method walks PLC's own pin records and matches PLC exactly.
        """
        nH = self.nH
        nM = self.nM
        nP = self.nP

        # plc_idx -> bench_idx mapping (same logic as macro_place.loader).
        plc_idx_to_bench = {}
        # Hard macros.
        hard_plc = getattr(plc, 'hard_macro_indices', [])
        for b_idx, p_idx in enumerate(hard_plc):
            plc_idx_to_bench[int(p_idx)] = b_idx
        # Soft macros.
        soft_plc = getattr(plc, 'soft_macro_indices', [])
        for b_off, p_idx in enumerate(soft_plc):
            plc_idx_to_bench[int(p_idx)] = nH + b_off
        # Ports.
        port_plc = getattr(plc, 'port_indices', [])
        for p_off, p_idx in enumerate(port_plc):
            plc_idx_to_bench[int(p_idx)] = nM + p_off

        mod_name_to_idx = getattr(plc, 'mod_name_to_indices', {})
        modules = getattr(plc, 'modules_w_pins', None)
        if modules is None:
            raise RuntimeError("plc.modules_w_pins missing — cannot build pin tables")

        pin_owner = []
        pin_offset = []
        net_pins = []        # list per net of arrays of pin indices
        net_weights = []     # weight per net (= driver_pin.get_weight(), matches PLC exactly)

        # Walk plc.nets (driver_name -> list of sink_names).
        plc_nets = getattr(plc, 'nets', {})
        for driver_name, sink_names in plc_nets.items():
            net_pin_indices = []
            # PLC takes the per-net weight from the DRIVER pin (see PLC
            # get_wirelength: weight_fact = driver_pin.get_weight()).
            # macro_place.loader hard-codes net weight = 1.0 — we have to
            # query PLC directly to match PLC's HPWL.
            driver_weight = 1.0
            if driver_name in mod_name_to_idx:
                driver_pin = modules[int(mod_name_to_idx[driver_name])]
                if hasattr(driver_pin, 'get_weight'):
                    driver_weight = float(driver_pin.get_weight())
            net_weights.append(driver_weight)
            for pin_name in [driver_name] + list(sink_names):
                if pin_name not in mod_name_to_idx:
                    continue
                pin_plc_idx = int(mod_name_to_idx[pin_name])
                pin_node = modules[pin_plc_idx]
                ptype = pin_node.get_type() if hasattr(pin_node, 'get_type') else 'MACRO_PIN'

                if ptype == 'PORT':
                    # Port = its own pin. Bench owner = port slot beyond nM.
                    owner_plc = pin_plc_idx
                    off_x, off_y = 0.0, 0.0
                else:
                    # Regular macro pin: ref via plc.get_ref_node_id.
                    if hasattr(plc, 'get_ref_node_id'):
                        owner_plc = int(plc.get_ref_node_id(pin_plc_idx))
                    else:
                        owner_plc = -1
                    if owner_plc == -1:
                        continue
                    off_x, off_y = pin_node.get_offset()

                bench_owner = plc_idx_to_bench.get(int(owner_plc))
                if bench_owner is None:
                    continue
                p_idx = len(pin_owner)
                pin_owner.append(int(bench_owner))
                pin_offset.append([float(off_x), float(off_y)])
                net_pin_indices.append(p_idx)
            net_pins.append(np.asarray(net_pin_indices, dtype=np.int64))

        self.pin_owner  = np.asarray(pin_owner, dtype=np.int64)
        self.pin_offset = np.asarray(pin_offset, dtype=np.float64).reshape(-1, 2)
        self.P          = self.pin_owner.shape[0]
        self.pin_xy     = np.zeros((self.P, 2), dtype=np.float64)
        self.net_pins   = net_pins
        # The PLC-net count is the authoritative N (may differ from len(benchmark.net_nodes)).
        self.N          = len(net_pins)
        # Stash PLC-derived net weights so the post-builder net-data block
        # in __init__ uses them instead of benchmark.net_weights (which the
        # loader hard-codes to 1.0).
        self._plc_net_weights = np.asarray(net_weights, dtype=np.float64)

        # pins_per_macro / nets_per_macro precomputation.
        self.pins_per_macro = [[] for _ in range(nM)]
        for p_idx, owner in enumerate(self.pin_owner):
            if owner < nM:
                self.pins_per_macro[int(owner)].append(p_idx)
        self.pins_per_macro = [np.asarray(lst, dtype=np.int64) for lst in self.pins_per_macro]

        self.nets_per_pin = np.full(self.P, -1, dtype=np.int64)
        for net_idx, pin_idxs in enumerate(self.net_pins):
            self.nets_per_pin[pin_idxs] = net_idx
        self.nets_per_macro = []
        for m in range(nM):
            pin_idxs = self.pins_per_macro[m]
            if pin_idxs.size == 0:
                self.nets_per_macro.append(np.zeros(0, dtype=np.int64))
            else:
                touched = np.unique(self.nets_per_pin[pin_idxs])
                touched = touched[touched >= 0]
                self.nets_per_macro.append(touched)

    # ───────────────────────────────────────────────────────────────────────
    #  Full setup from a placement
    # ───────────────────────────────────────────────────────────────────────

    def set_placement(self, placement: np.ndarray) -> None:
        """Initialise all caches from a [num_macros, 2] placement (float).

        After this returns, proxy(), delta_for_move(), commit_move() are valid.
        """
        placement = np.asarray(placement, dtype=np.float64)
        assert placement.shape == (self.nM, 2), \
            f"placement shape {placement.shape} != ({self.nM}, 2)"
        self.macro_pos = placement.copy()

        # Update pin positions: pin_xy = owner_pos + pin_offset.
        self._refresh_all_pin_positions()

        # Build all net bboxes / hpwls.
        self.total_hpwl = 0.0
        for n in range(self.N):
            pis = self.net_pins[n]
            if pis.size == 0:
                self.net_min_x[n] = 0.0
                self.net_max_x[n] = 0.0
                self.net_min_y[n] = 0.0
                self.net_max_y[n] = 0.0
                self.net_hpwl[n] = 0.0
                continue
            xs = self.pin_xy[pis, 0]
            ys = self.pin_xy[pis, 1]
            self.net_min_x[n] = float(xs.min())
            self.net_max_x[n] = float(xs.max())
            self.net_min_y[n] = float(ys.min())
            self.net_max_y[n] = float(ys.max())
            hpwl = ((self.net_max_x[n] - self.net_min_x[n]) +
                    (self.net_max_y[n] - self.net_min_y[n]))
            self.net_hpwl[n] = self.net_weight[n] * hpwl
            self.total_hpwl += self.net_hpwl[n]

        # Build density grid from scratch.
        self.grid_occupied[:] = 0.0
        for m in range(self.nM):
            cell_map = self._compute_macro_cell_overlap(m, self.macro_pos[m, 0], self.macro_pos[m, 1])
            self.macro_overlap[m] = cell_map
            for (r, c), area in cell_map.items():
                self.grid_occupied[r, c] += area

        # Build cong raw grids from scratch (un-normalized). Subsequent
        # commit_move() calls maintain these incrementally by subtracting the
        # touched nets'/macro's old contribution, applying the move, and adding
        # the new contribution. compute_cong_cost_full() reads these caches,
        # normalizes per-axis, smooths, adds macro portion, and runs ABU.
        self.raw_V[:] = 0.0
        self.raw_H[:] = 0.0
        self.macro_raw_V[:] = 0.0
        self.macro_raw_H[:] = 0.0
        for n in range(self.N):
            self._add_net_to_routing(n, self.raw_V, self.raw_H)
        for m in range(self.nH):
            self._add_macro_blockage(m, self.macro_raw_V, self.macro_raw_H)

    def _refresh_all_pin_positions(self) -> None:
        """Recompute self.pin_xy from current macro_pos and port_pos."""
        for p_idx in range(self.P):
            owner = int(self.pin_owner[p_idx])
            if owner < self.nM:
                ox, oy = self.macro_pos[owner]
            else:
                port_idx = owner - self.nM
                if 0 <= port_idx < self.port_pos.shape[0]:
                    ox, oy = self.port_pos[port_idx]
                else:
                    ox, oy = 0.0, 0.0
            self.pin_xy[p_idx, 0] = ox + self.pin_offset[p_idx, 0]
            self.pin_xy[p_idx, 1] = oy + self.pin_offset[p_idx, 1]

    # ───────────────────────────────────────────────────────────────────────
    #  Density helpers
    # ───────────────────────────────────────────────────────────────────────

    def _compute_macro_cell_overlap(self, m: int, x: float, y: float) -> dict:
        """Return {(row, col): overlap_area} for macro m at position (x, y).

        Matches PLC __add_module_to_grid_cells:
          - Macro extent: [x - w/2, x + w/2] × [y - h/2, y + h/2].
          - Find covered cells via floor(x/grid_w).
          - Per cell, overlap = min(extent_max, cell_max) - max(extent_min, cell_min)
            (clipped at 0) in each axis, multiplied.
          - Out-of-bounds macros are skipped entirely.
        """
        w = self.macro_w[m] * 0.5
        h = self.macro_h[m] * 0.5
        x_min = x - w
        x_max = x + w
        y_min = y - h
        y_max = y + h

        # PLC: ur_row, ur_col from upper-right corner; bl_row, bl_col from lower-left.
        ur_col = math.floor(x_max / self.grid_w)
        ur_row = math.floor(y_max / self.grid_h)
        bl_col = math.floor(x_min / self.grid_w)
        bl_row = math.floor(y_min / self.grid_h)

        # PLC OOB skip logic.
        if ur_row < 0 or ur_col < 0:
            return {}
        if bl_row < 0:
            bl_row = 0
        if bl_col < 0:
            bl_col = 0
        if bl_row >= self.G_rows or bl_col >= self.G_cols:
            return {}
        if ur_row > self.G_rows - 1:
            ur_row = self.G_rows - 1
        if ur_col > self.G_cols - 1:
            ur_col = self.G_cols - 1

        out = {}
        for r in range(bl_row, ur_row + 1):
            cell_y_min = r * self.grid_h
            cell_y_max = cell_y_min + self.grid_h
            ov_y = min(y_max, cell_y_max) - max(y_min, cell_y_min)
            if ov_y <= 0:
                continue
            for c in range(bl_col, ur_col + 1):
                cell_x_min = c * self.grid_w
                cell_x_max = cell_x_min + self.grid_w
                ov_x = min(x_max, cell_x_max) - max(x_min, cell_x_min)
                if ov_x <= 0:
                    continue
                out[(r, c)] = ov_x * ov_y
        return out

    # ───────────────────────────────────────────────────────────────────────
    #  Congestion: ports the PLC get_routing() algorithm in numpy
    # ───────────────────────────────────────────────────────────────────────
    #  Mirrors plc_client_os.py:
    #    1. For each driver pin and its sinks, collect node_gcells = set of
    #       cells the (driver + sinks) pins fall in.
    #    2. Dispatch by |node_gcells|:
    #         == 2  → two-pin routing (H along source's row, V along sink's col)
    #         == 3  → L/T/special-case 3-pin routing
    #         > 3   → split into N-1 two-pin nets (source -> each sink)
    #       Each segment adds `weight` per cell along its path to V/H grids.
    #    3. For each hard macro, add blockage to V_macro_routing/H_macro_routing.
    #    4. Normalise: divide raw V grids by grid_v_routes, H by grid_h_routes.
    #    5. Smooth (box-blur per axis) on net portion only.
    #    6. Add macro portion back (un-smoothed).
    #    7. ABU top-5% mean of concat(V, H).

    def _grid_cell_for_pos(self, x: float, y: float) -> Tuple[int, int]:
        """PLC's __get_grid_cell_location (with the monkey-patch clamping
        applied in macro_place.objective)."""
        col = int(math.floor(x / self.grid_w))
        row = int(math.floor(y / self.grid_h))
        col = max(0, min(col, self.G_cols - 1))
        row = max(0, min(row, self.G_rows - 1))
        return row, col

    def _grid_cell_for_pos_f32(self, x_f32, y_f32) -> Tuple[int, int]:
        """PLC-matching grid-cell lookup. Mimics PLC's behavior under
        numpy 2.0+ NEP 50: pin positions are numpy.float32 (PLC stores them
        via `pin.set_pos(macro_x + pin.x_offset)` where macro_x is float32
        → NEP 50 → float32 result), and `floor(x_f32 / grid_w_f64)` in NEP
        50 stays as float32 (python.float is "weak"). We cast explicitly so
        the result matches PLC regardless of numpy version."""
        col = int(math.floor(np.float32(x_f32 / self.grid_w)))
        row = int(math.floor(np.float32(y_f32 / self.grid_h)))
        col = max(0, min(col, self.G_cols - 1))
        row = max(0, min(row, self.G_rows - 1))
        return row, col

    # Note on `weight` parameter: weight is pre-multiplied with the caller's sign
    # (typically ±1) so subtraction-from-grids is done by passing a negative
    # weight. _add_net_to_routing handles the sign once at the top.

    def _add_two_pin_segment(self, source_rc, sink_rc, weight: float,
                              V_arr: np.ndarray, H_arr: np.ndarray) -> None:
        """PLC's __two_pin_net_routing: H along source's row, V along sink's col."""
        sr, sc = source_rc
        tr, tc = sink_rc
        row_min = min(tr, sr); row_max = max(tr, sr)
        col_min = min(tc, sc); col_max = max(tc, sc)
        # H routing along source row, columns [col_min, col_max)
        for c in range(col_min, col_max):
            H_arr[sr * self.G_cols + c] += weight
        # V routing along sink column, rows [row_min, row_max)
        for r in range(row_min, row_max):
            V_arr[r * self.G_cols + tc] += weight

    def _add_three_pin_segment(self, node_gcells_list, weight: float,
                                 V_arr: np.ndarray, H_arr: np.ndarray) -> None:
        """PLC's __three_pin_net_routing: dispatches L / T / special shapes."""
        # PLC: temp_gcell.sort(key=lambda x: (x[1], x[0]))  — sort by (col, row).
        temp = sorted(node_gcells_list, key=lambda rc: (rc[1], rc[0]))
        y1, x1 = temp[0]
        y2, x2 = temp[1]
        y3, x3 = temp[2]

        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            # L-route: see PLC __l_routing.
            for c in range(x1, x2):
                H_arr[y1 * self.G_cols + c] += weight
            for c in range(x2, x3):
                H_arr[y2 * self.G_cols + c] += weight
            for r in range(min(y1, y2), max(y1, y2)):
                V_arr[r * self.G_cols + x2] += weight
            for r in range(min(y2, y3), max(y2, y3)):
                V_arr[r * self.G_cols + x3] += weight
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            # Special case: two pins share x, third is to the left + below.
            for c in range(x1, x2):
                H_arr[y1 * self.G_cols + c] += weight
            for r in range(y1, max(y2, y3)):
                V_arr[r * self.G_cols + x2] += weight
        elif y2 == y3:
            # Special case: two pins share y.
            for c in range(x1, x2):
                H_arr[y1 * self.G_cols + c] += weight
            for c in range(x2, x3):
                H_arr[y2 * self.G_cols + c] += weight
            for r in range(min(y2, y1), max(y2, y1)):
                V_arr[r * self.G_cols + x2] += weight
        else:
            # T-route: PLC __t_routing.
            # PLC: node_gcells.sort() — default tuple sort, i.e. by (row, col).
            t2 = sorted(temp)
            y1t, x1t = t2[0]
            y2t, x2t = t2[1]
            y3t, x3t = t2[2]
            xmin = min(x1t, x2t, x3t); xmax = max(x1t, x2t, x3t)
            for c in range(xmin, xmax):
                H_arr[y2t * self.G_cols + c] += weight
            for r in range(min(y1t, y2t), max(y1t, y2t)):
                V_arr[r * self.G_cols + x1t] += weight
            for r in range(min(y2t, y3t), max(y2t, y3t)):
                V_arr[r * self.G_cols + x3t] += weight

    def _add_net_to_routing(self, net_idx: int,
                              V_arr: np.ndarray, H_arr: np.ndarray,
                              sign: float = 1.0) -> None:
        """Add net `net_idx`'s contribution to V_arr/H_arr (multiplied by `sign`).
        First pin in net_pins[k] is the driver.

        Pin positions are cast to numpy.float32 before grid-cell lookup to
        match PLC's NEP 50 behavior (see `_grid_cell_for_pos_f32`).

        Use `sign=-1.0` to subtract a previously-added contribution from
        the raw grids (incremental update before a macro move)."""
        pis = self.net_pins[net_idx]
        if pis.size < 2:
            return
        weight = float(self.net_weight[net_idx]) * sign
        # Driver pin = first.
        drv = int(pis[0])
        source_rc = self._grid_cell_for_pos_f32(
            np.float32(self.pin_xy[drv, 0]),
            np.float32(self.pin_xy[drv, 1]),
        )
        # Build unique gcells across all pins.
        gcells = {source_rc}
        for p in pis:
            p_int = int(p)
            rc = self._grid_cell_for_pos_f32(
                np.float32(self.pin_xy[p_int, 0]),
                np.float32(self.pin_xy[p_int, 1]),
            )
            gcells.add(rc)

        n = len(gcells)
        if n == 2:
            others = [g for g in gcells if g != source_rc]
            if others:
                self._add_two_pin_segment(source_rc, others[0], weight, V_arr, H_arr)
        elif n == 3:
            self._add_three_pin_segment(list(gcells), weight, V_arr, H_arr)
        elif n > 3:
            # Split: source → each other gcell as separate 2-pin nets.
            for g in gcells:
                if g != source_rc:
                    self._add_two_pin_segment(source_rc, g, weight, V_arr, H_arr)

    def _debug_macro_bbox(self, m: int) -> dict:
        """Diagnostic: dump macro m's bbox info as my port computes it.
        Also queries PLC for the same macro to compare. Use to find
        iteration-domain / flag mismatches."""
        plc = self.plc
        info = {'m': m}
        # My port
        mx = float(self.macro_pos[m, 0])
        my = float(self.macro_pos[m, 1])
        hw = float(self.macro_w[m]) * 0.5
        hh = float(self.macro_h[m]) * 0.5
        x_lo = mx - hw; x_hi = mx + hw
        y_lo = my - hh; y_hi = my + hh
        ur_row, ur_col = self._grid_cell_for_pos(x_hi, y_hi)
        bl_row, bl_col = self._grid_cell_for_pos(x_lo, y_lo)
        info['my'] = {
            'mx': mx, 'my': my, 'w': self.macro_w[m], 'h': self.macro_h[m],
            'x_lo': x_lo, 'x_hi': x_hi, 'y_lo': y_lo, 'y_hi': y_hi,
            'bl_row': bl_row, 'ur_row': ur_row,
            'bl_col': bl_col, 'ur_col': ur_col,
            'x_hi_div_gw': x_hi / self.grid_w,
            'x_lo_div_gw': x_lo / self.grid_w,
        }
        # PLC
        if plc is not None:
            hard_plc = list(getattr(plc, 'hard_macro_indices', []))
            if m < len(hard_plc):
                p_idx = int(hard_plc[m])
                mod = plc.modules_w_pins[p_idx]
                pmx, pmy = mod.get_pos()
                pmw_raw = mod.get_width(); pmh_raw = mod.get_height()
                # NOT casting to float — keep the original type. PLC's internal
                # arithmetic uses the raw value, including numpy.float32 if that's
                # what came in via set_pos.
                px_lo = pmx - pmw_raw/2; px_hi = pmx + pmw_raw/2
                py_lo = pmy - pmh_raw/2; py_hi = pmy + pmh_raw/2
                # PLC's __get_grid_cell_location is monkey-patched to clamp
                gel = plc._PlacementCost__get_grid_cell_location
                pur_row, pur_col = gel(px_hi, py_hi)
                pbl_row, pbl_col = gel(px_lo, py_lo)
                info['plc'] = {
                    'mx_type': type(pmx).__name__,
                    'mx': float(pmx), 'my': float(pmy),
                    'w_type': type(pmw_raw).__name__,
                    'w_raw_repr': repr(pmw_raw),
                    'w': float(pmw_raw), 'h': float(pmh_raw),
                    'x_lo_repr': repr(px_lo),
                    'x_hi_repr': repr(px_hi),
                    'x_lo': float(px_lo), 'x_hi': float(px_hi),
                    'y_lo': float(py_lo), 'y_hi': float(py_hi),
                    'bl_row': pbl_row, 'ur_row': pur_row,
                    'bl_col': pbl_col, 'ur_col': pur_col,
                    'x_hi_div_gw': float(px_hi / plc.grid_width),
                    'x_lo_div_gw': float(px_lo / plc.grid_width),
                }
        return info

    def _add_macro_blockage(self, m: int, V_arr: np.ndarray, H_arr: np.ndarray,
                             sign: float = 1.0) -> None:
        """Add hard macro m's routing blockage to V_arr/H_arr (multiplied by
        `sign`). Mirrors PLC __macro_route_over_grid_cell, including the
        partial-overlap correction at the top row and right column.

        Use `sign=-1.0` to subtract a previously-added blockage from the
        macro grids (incremental update before a hard-macro move).

        Critical PLC behaviors we must mirror:

        1. `__overlap_dist` returns (0, 0) when EITHER x_diff or y_diff is
           non-positive — the two dims are coupled, not clamped independently.

        2. PLC's `mod.get_pos()` returns numpy.float32 scalars (because
           `_set_placement` writes the float32 placement tensor into
           HardMacro.x/y directly). Then `mod_x + mod_w/2` in numpy 2.0+
           with NEP 50 stays as numpy.float32 (python.float is "weak").
           So PLC's x_hi/x_lo/y_hi/y_lo are float32-precision values, and
           ur_col = floor(x_hi/gw) is computed in float32 too. We must
           reproduce that here — pure float64 arithmetic mis-floors by ±1
           cell at any macro whose corner lands within 2^-22 of a grid line
           (very common in ICCAD04 macros with x ≈ multiples of 0.51 μm)."""
        if m >= self.nH:
            return
        # Float32-precision bbox to match PLC's NEP 50 behavior.
        mx_f32 = np.float32(self.macro_pos[m, 0])
        my_f32 = np.float32(self.macro_pos[m, 1])
        hw = float(self.macro_w[m]) * 0.5
        hh = float(self.macro_h[m]) * 0.5
        # mx_f32 (np.float32) + python.float in NEP 50 → np.float32.
        # Wrap in np.float32(...) explicitly to be safe across numpy versions
        # (in numpy < 2.0, the bare expression would promote to float64).
        x_lo_f32 = np.float32(mx_f32 - hw)
        x_hi_f32 = np.float32(mx_f32 + hw)
        y_lo_f32 = np.float32(my_f32 - hh)
        y_hi_f32 = np.float32(my_f32 + hh)
        x_lo = float(x_lo_f32)
        x_hi = float(x_hi_f32)
        y_lo = float(y_lo_f32)
        y_hi = float(y_hi_f32)

        # Grid-cell lookup: float32 numerator / float64 denominator → float32
        # in NEP 50. Cast explicitly so the floor matches PLC even on numpy<2.
        ur_col = int(math.floor(np.float32(x_hi_f32 / self.grid_w)))
        ur_row = int(math.floor(np.float32(y_hi_f32 / self.grid_h)))
        bl_col = int(math.floor(np.float32(x_lo_f32 / self.grid_w)))
        bl_row = int(math.floor(np.float32(y_lo_f32 / self.grid_h)))
        # Clamp (matches macro_place.objective._patched_get_grid_cell_location).
        ur_col = max(0, min(ur_col, self.G_cols - 1))
        ur_row = max(0, min(ur_row, self.G_rows - 1))
        bl_col = max(0, min(bl_col, self.G_cols - 1))
        bl_row = max(0, min(bl_row, self.G_rows - 1))

        # PLC OOB skip (replicated by the clamping in _grid_cell_for_pos).
        if bl_row > ur_row or bl_col > ur_col:
            return

        gw = self.grid_w; gh = self.grid_h
        # Multiply allocations by sign once; all subsequent writes use these.
        vra = self.vrouting_alloc * sign
        hra = self.hrouting_alloc * sign

        if_partial_v = False
        if_partial_h = False
        # First pass: add to both grids — using PLC's coupled __overlap_dist.
        for r in range(bl_row, ur_row + 1):
            cy_lo = r * gh
            cy_hi = cy_lo + gh
            y_raw = min(y_hi, cy_hi) - max(y_lo, cy_lo)
            for c in range(bl_col, ur_col + 1):
                cx_lo = c * gw
                cx_hi = cx_lo + gw
                x_raw = min(x_hi, cx_hi) - max(x_lo, cx_lo)
                # PLC __overlap_dist: returns (0, 0) if EITHER is non-positive.
                if x_raw > 0 and y_raw > 0:
                    x_dist = x_raw; y_dist = y_raw
                else:
                    x_dist = 0.0; y_dist = 0.0
                if ur_row != bl_row:
                    if (r == bl_row and abs(y_dist - gh) > 1e-5) or \
                       (r == ur_row and abs(y_dist - gh) > 1e-5):
                        if_partial_v = True
                if ur_col != bl_col:
                    if (c == bl_col and abs(x_dist - gw) > 1e-5) or \
                       (c == ur_col and abs(x_dist - gw) > 1e-5):
                        if_partial_h = True
                V_arr[r * self.G_cols + c] += x_dist * vra
                H_arr[r * self.G_cols + c] += y_dist * hra
        # Second pass: partial-overlap correction. PLC re-computes x/y_dist
        # via __overlap_dist (same coupled rule) inside the correction loops.
        if if_partial_v:
            r = ur_row
            cy_lo = r * gh
            cy_hi = cy_lo + gh
            y_raw = min(y_hi, cy_hi) - max(y_lo, cy_lo)
            for c in range(bl_col, ur_col + 1):
                cx_lo = c * gw
                cx_hi = cx_lo + gw
                x_raw = min(x_hi, cx_hi) - max(x_lo, cx_lo)
                if x_raw > 0 and y_raw > 0:
                    x_dist = x_raw
                else:
                    x_dist = 0.0
                V_arr[r * self.G_cols + c] -= x_dist * vra
        if if_partial_h:
            c = ur_col
            cx_lo = c * gw
            cx_hi = cx_lo + gw
            x_raw = min(x_hi, cx_hi) - max(x_lo, cx_lo)
            for r in range(bl_row, ur_row + 1):
                cy_lo = r * gh
                cy_hi = cy_lo + gh
                y_raw = min(y_hi, cy_hi) - max(y_lo, cy_lo)
                if x_raw > 0 and y_raw > 0:
                    y_dist = y_raw
                else:
                    y_dist = 0.0
                H_arr[r * self.G_cols + c] -= y_dist * hra

    def _smooth_routing(self, V_in: np.ndarray, H_in: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """PLC's __smooth_routing_cong: per-axis box blur.

        V smoothed across columns within ±smooth_range of each col;
        H smoothed across rows within ±smooth_range of each row.
        Returns (smoothed_V, smoothed_H), both shape [G_rows*G_cols].
        """
        sr = self.smooth_range
        out_V = np.zeros_like(V_in)
        out_H = np.zeros_like(H_in)
        if sr <= 0:
            out_V[:] = V_in
            out_H[:] = H_in
            return out_V, out_H

        Gc = self.G_cols
        Gr = self.G_rows

        # V cong: smooth across columns. For each (row, col), distribute
        # V_in[row, col] / window_size across [col-sr, col+sr] clipped.
        for r in range(Gr):
            base = r * Gc
            for c in range(Gc):
                lp = c - sr
                if lp < 0: lp = 0
                rp = c + sr
                if rp >= Gc: rp = Gc - 1
                cnt = rp - lp + 1
                val = V_in[base + c] / cnt
                for ptr in range(lp, rp + 1):
                    out_V[base + ptr] += val
        # H cong: smooth across rows. For each (row, col), distribute
        # H_in[row, col] / window_size across [row-sr, row+sr] clipped.
        for r in range(Gr):
            for c in range(Gc):
                lp = r - sr
                if lp < 0: lp = 0
                up = r + sr
                if up >= Gr: up = Gr - 1
                cnt = up - lp + 1
                val = H_in[r * Gc + c] / cnt
                for ptr in range(lp, up + 1):
                    out_H[ptr * Gc + c] += val
        return out_V, out_H

    def _abu_top_frac_mean(self, flat: np.ndarray, frac: float) -> float:
        """PLC's abu: top-`frac` of `flat` (sorted descending), mean.
        flat: 1-D array (V + H concatenated for cong, just cells for density)."""
        n = flat.size
        cnt = math.floor(n * frac)
        if cnt == 0:
            return float(flat.max()) if n > 0 else 0.0
        if cnt >= n:
            return float(flat.sum() / n)
        # Top-cnt by argpartition (matches sorted desc + sum / cnt exactly).
        idx = np.argpartition(-flat, cnt - 1)[:cnt]
        return float(flat[idx].sum() / cnt)

    def _build_full_routing_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (V_final, H_final) — normalised + smoothed + macro-added
        grids matching PLC's get_routing() output.

        Reads the cached raw_V / raw_H / macro_raw_V / macro_raw_H, which are
        kept up-to-date by set_placement() and commit_move(). Per query work
        is O(grid_size * smooth_range), not O(N + nH) — the from-scratch
        rebuild used to dominate cong cost evaluation."""
        # Normalised copies (don't mutate the un-normalised caches).
        if self.grid_v_routes > 0:
            norm_V    = self.raw_V       / self.grid_v_routes
            macro_V_n = self.macro_raw_V / self.grid_v_routes
        else:
            norm_V    = self.raw_V.copy()
            macro_V_n = self.macro_raw_V.copy()
        if self.grid_h_routes > 0:
            norm_H    = self.raw_H       / self.grid_h_routes
            macro_H_n = self.macro_raw_H / self.grid_h_routes
        else:
            norm_H    = self.raw_H.copy()
            macro_H_n = self.macro_raw_H.copy()
        # Smooth net portion only.
        sV, sH = self._smooth_routing(norm_V, norm_H)
        # Add macro portion back un-smoothed (matches PLC).
        V_final = sV + macro_V_n
        H_final = sH + macro_H_n
        return V_final, H_final

    def compute_cong_cost_full(self) -> float:
        """Cong cost from the cached raw grids (incremental). Matches PLC's
        get_congestion_cost() bit-perfectly after a from-scratch validation."""
        V_final, H_final = self._build_full_routing_grids()
        flat = np.concatenate([V_final, H_final])
        return self._abu_top_frac_mean(flat, ABU_FRAC_CONG)

    def diff_against_plc_routing(self) -> dict:
        """Diagnostic: rebuild PLC's V/H routing grids and compare cell-by-cell
        to the from-scratch port. Now splits net vs macro vs smoothed parts to
        localise the offset more precisely.

        Returns a dict of summary stats.
        """
        plc = self.plc
        if plc is None:
            return {'error': 'plc not bound'}

        # Sync placement to PLC.
        placement_t = torch.tensor(self.macro_pos, dtype=torch.float32)
        _ = compute_proxy_cost(placement_t, self.benchmark, plc)

        # ── PLC: rebuild raw net grids (BEFORE smooth, BEFORE macro add) ──
        # Replicates plc.get_routing() up to the normalisation step.
        Gr, Gc = self.G_rows, self.G_cols
        plc_grid_v_routes = plc.grid_width * plc.vroutes_per_micron
        plc_grid_h_routes = plc.grid_height * plc.hroutes_per_micron

        # Force PLC into rebuild mode.
        plc.FLAG_UPDATE_CONGESTION = True
        plc.get_routing()
        plc_V_final = np.asarray(plc.V_routing_cong, dtype=np.float64)
        plc_H_final = np.asarray(plc.H_routing_cong, dtype=np.float64)
        # After plc.get_routing(), V_routing_cong = smoothed_net + macro.
        # We can recover macro by reading V_macro_routing_cong (NOT smoothed):
        plc_V_macro = np.asarray(plc.V_macro_routing_cong, dtype=np.float64)
        plc_H_macro = np.asarray(plc.H_macro_routing_cong, dtype=np.float64)
        # Smoothed-net grids = final - macro.
        plc_V_smoothed_net = plc_V_final - plc_V_macro
        plc_H_smoothed_net = plc_H_final - plc_H_macro

        # ── Ours: split net vs macro to mirror PLC ──
        raw_V = np.zeros(Gr * Gc, dtype=np.float64)
        raw_H = np.zeros(Gr * Gc, dtype=np.float64)
        macro_V = np.zeros_like(raw_V)
        macro_H = np.zeros_like(raw_H)
        for n in range(self.N):
            self._add_net_to_routing(n, raw_V, raw_H)
        for m in range(self.nH):
            self._add_macro_blockage(m, macro_V, macro_H)
        if self.grid_v_routes > 0:
            raw_V   /= self.grid_v_routes
            macro_V /= self.grid_v_routes
        if self.grid_h_routes > 0:
            raw_H   /= self.grid_h_routes
            macro_H /= self.grid_h_routes
        sV, sH = self._smooth_routing(raw_V, raw_H)
        my_V_final = sV + macro_V
        my_H_final = sH + macro_H

        d_V_macro = macro_V - plc_V_macro
        d_H_macro = macro_H - plc_H_macro
        d_V_net   = sV - plc_V_smoothed_net
        d_H_net   = sH - plc_H_smoothed_net
        d_V_final = my_V_final - plc_V_final
        d_H_final = my_H_final - plc_H_final

        # Top-5% on FINAL.
        plc_flat = np.concatenate([plc_V_final, plc_H_final])
        my_flat  = np.concatenate([my_V_final,  my_H_final])
        n = plc_flat.size
        cnt = max(1, math.floor(n * ABU_FRAC_CONG))
        idx_top_plc = np.argpartition(-plc_flat, cnt - 1)[:cnt]
        plc_top = plc_flat[idx_top_plc]
        my_top  = my_flat[idx_top_plc]

        # ── Localise the H-macro discrepancy ──
        # Find cells where |my_H_macro - plc_H_macro| > 0.01 (10× float-noise),
        # and check which hard macros' bboxes overlap each one. Helps find
        # the offending macro(s) — usually a small set.
        d_H_macro_2d = d_H_macro.reshape(Gr, Gc)
        offending_cells = np.argwhere(np.abs(d_H_macro_2d) > 0.01)
        worst = []
        for (r, c) in offending_cells[:20]:
            r = int(r); c = int(c)
            # macros overlapping this cell
            macros_here = []
            for m in range(self.nH):
                mx, my = self.macro_pos[m]
                hw = self.macro_w[m] * 0.5
                hh = self.macro_h[m] * 0.5
                if (mx - hw < (c + 1) * self.grid_w and
                    mx + hw > c * self.grid_w and
                    my - hh < (r + 1) * self.grid_h and
                    my + hh > r * self.grid_h):
                    macros_here.append(int(m))
            worst.append({
                'r': r, 'c': c,
                'my_H_macro':  float(macro_H[r * Gc + c]),
                'plc_H_macro': float(plc_H_macro[r * Gc + c]),
                'diff':        float(d_H_macro_2d[r, c]),
                'macros_here': macros_here,
            })

        return {
            # Net portion (smoothed)
            'plc_V_net_sum':    float(plc_V_smoothed_net.sum()),
            'my_V_net_sum':     float(sV.sum()),
            'plc_H_net_sum':    float(plc_H_smoothed_net.sum()),
            'my_H_net_sum':     float(sH.sum()),
            'max_abs_dV_net':   float(np.abs(d_V_net).max()),
            'max_abs_dH_net':   float(np.abs(d_H_net).max()),
            # Macro portion (un-smoothed)
            'plc_V_macro_sum':  float(plc_V_macro.sum()),
            'my_V_macro_sum':   float(macro_V.sum()),
            'plc_H_macro_sum':  float(plc_H_macro.sum()),
            'my_H_macro_sum':   float(macro_H.sum()),
            'max_abs_dV_macro': float(np.abs(d_V_macro).max()),
            'max_abs_dH_macro': float(np.abs(d_H_macro).max()),
            'n_bad_H_cells':    int(offending_cells.shape[0]),
            'worst_H_cells':    worst,
            # Final
            'plc_V_sum':    float(plc_V_final.sum()),
            'my_V_sum':     float(my_V_final.sum()),
            'plc_H_sum':    float(plc_H_final.sum()),
            'my_H_sum':     float(my_H_final.sum()),
            'max_abs_dV':   float(np.abs(d_V_final).max()),
            'max_abs_dH':   float(np.abs(d_H_final).max()),
            # Top-5%
            'plc_top_mean': float(plc_top.mean()),
            'my_top_mean':  float(my_top.mean()),
            'top_cong_mean_diff': float(my_top.mean() - plc_top.mean()),
            'cnt_top':      int(cnt),
        }

    # ───────────────────────────────────────────────────────────────────────
    #  WL: incremental delta and commit
    # ───────────────────────────────────────────────────────────────────────

    def _delta_wl_for_move(self, macro_idx: int, new_xy: Tuple[float, float]) -> Tuple[float, dict]:
        """Compute the (un-normalised) total_hpwl delta if macro_idx moves.

        Returns (delta_total_hpwl, per_net_new_state).
        per_net_new_state[net_idx] = (new_min_x, new_max_x, new_min_y, new_max_y,
                                       new_net_hpwl)  — to be applied on commit.
        """
        nets = self.nets_per_macro[macro_idx]
        if nets.size == 0:
            return 0.0, {}

        pin_idxs_macro = self.pins_per_macro[macro_idx]
        new_x, new_y = float(new_xy[0]), float(new_xy[1])

        # For each pin owned by macro: new_pin_xy = (new_x + offset_x, new_y + offset_y).
        # Build a map pin_idx -> (new_px, new_py).
        new_pin_pos = {}
        for p_idx in pin_idxs_macro:
            ox = new_x + self.pin_offset[p_idx, 0]
            oy = new_y + self.pin_offset[p_idx, 1]
            new_pin_pos[int(p_idx)] = (float(ox), float(oy))

        delta = 0.0
        per_net = {}

        for net_idx in nets:
            net_idx = int(net_idx)
            pis = self.net_pins[net_idx]
            if pis.size == 0:
                continue
            # Build the candidate net pin positions: use new_pin_pos for moving pins,
            # current pin_xy for the rest.
            new_min_x = math.inf
            new_max_x = -math.inf
            new_min_y = math.inf
            new_max_y = -math.inf
            for p in pis:
                p_int = int(p)
                if p_int in new_pin_pos:
                    px, py = new_pin_pos[p_int]
                else:
                    px = float(self.pin_xy[p_int, 0])
                    py = float(self.pin_xy[p_int, 1])
                if px < new_min_x: new_min_x = px
                if px > new_max_x: new_max_x = px
                if py < new_min_y: new_min_y = py
                if py > new_max_y: new_max_y = py
            new_hpwl_unweighted = (new_max_x - new_min_x) + (new_max_y - new_min_y)
            new_net_hpwl = self.net_weight[net_idx] * new_hpwl_unweighted
            delta += new_net_hpwl - self.net_hpwl[net_idx]
            per_net[net_idx] = (new_min_x, new_max_x, new_min_y, new_max_y, new_net_hpwl)

        return delta, per_net

    def _commit_wl(self, macro_idx: int, new_xy: Tuple[float, float], per_net: dict) -> None:
        """Apply WL state for an accepted move."""
        # Update pin_xy for all pins owned by this macro.
        for p_idx in self.pins_per_macro[macro_idx]:
            self.pin_xy[p_idx, 0] = float(new_xy[0]) + self.pin_offset[p_idx, 0]
            self.pin_xy[p_idx, 1] = float(new_xy[1]) + self.pin_offset[p_idx, 1]
        # Apply per-net state.
        for net_idx, (mnx, mxx, mny, mxy, hpwl) in per_net.items():
            self.net_min_x[net_idx] = mnx
            self.net_max_x[net_idx] = mxx
            self.net_min_y[net_idx] = mny
            self.net_max_y[net_idx] = mxy
            self.total_hpwl += hpwl - self.net_hpwl[net_idx]
            self.net_hpwl[net_idx] = hpwl

    # ───────────────────────────────────────────────────────────────────────
    #  Density: incremental delta and commit
    # ───────────────────────────────────────────────────────────────────────

    def _delta_density_for_move(self, macro_idx: int, new_xy: Tuple[float, float]) -> dict:
        """Compute the new grid_occupied delta for macro_idx moving to new_xy.

        Returns a dict that on commit will be applied:
          {'cells_delta': {(r, c): area_change}, 'new_cell_map': {(r, c): new_area}}
        """
        new_x, new_y = float(new_xy[0]), float(new_xy[1])
        old_map = self.macro_overlap[macro_idx] or {}
        new_map = self._compute_macro_cell_overlap(macro_idx, new_x, new_y)

        # Cells whose area changes: union of old keys + new keys.
        cells_delta = {}
        for key, area in old_map.items():
            cells_delta[key] = cells_delta.get(key, 0.0) - area
        for key, area in new_map.items():
            cells_delta[key] = cells_delta.get(key, 0.0) + area

        return {'cells_delta': cells_delta, 'new_cell_map': new_map}

    def _commit_density(self, macro_idx: int, new_xy: Tuple[float, float],
                        state: dict) -> None:
        """Apply density state for accepted move."""
        for (r, c), d in state['cells_delta'].items():
            self.grid_occupied[r, c] += d
        self.macro_overlap[macro_idx] = state['new_cell_map']

    # ───────────────────────────────────────────────────────────────────────
    #  Cost computations (full state -> scalar)
    # ───────────────────────────────────────────────────────────────────────

    def compute_wl_cost(self, total_hpwl: Optional[float] = None) -> float:
        """WL cost = total_hpwl / ((W + H) * net_cnt)."""
        if total_hpwl is None:
            total_hpwl = self.total_hpwl
        return float(total_hpwl / self.wl_denominator)

    def compute_density_cost(self, grid_occupied: Optional[np.ndarray] = None) -> float:
        """Density cost = 0.5 * mean(top-10% of cell_densities). PLC sorts
        only occupied cells (>0) but uses cnt = floor(0.1 * total_cells)."""
        if grid_occupied is None:
            grid_occupied = self.grid_occupied
        densities = grid_occupied / self.cell_area
        flat = densities.ravel()
        occupied = flat[flat != 0.0]
        if occupied.size == 0:
            return 0.0
        cnt = math.floor(flat.size * ABU_FRAC_DENSITY)
        if flat.size < 10:
            return float(DENSITY_HALF * occupied.mean())
        # Top-cnt of occupied (sorted desc).
        if occupied.size <= cnt:
            top_sum = float(occupied.sum())
        else:
            # Use partition: top-cnt largest.
            ind = np.argpartition(-occupied, cnt - 1)[:cnt]
            top_sum = float(occupied[ind].sum())
        return float(DENSITY_HALF * top_sum / cnt) if cnt > 0 else 0.0

    # ───────────────────────────────────────────────────────────────────────
    #  Public API
    # ───────────────────────────────────────────────────────────────────────

    def proxy(self, include_cong: bool = True) -> float:
        """Current proxy cost from cached state. Cong term uses the
        from-scratch numpy port of PLC's get_routing()."""
        wl  = self.compute_wl_cost()
        den = self.compute_density_cost()
        cong = self.compute_cong_cost_full() if include_cong else 0.0
        return WEIGHT_WL * wl + WEIGHT_DENSITY * den + WEIGHT_CONG * cong

    def proxy_breakdown(self, include_cong: bool = True) -> dict:
        """Same as proxy() but returns the per-term breakdown."""
        wl  = self.compute_wl_cost()
        den = self.compute_density_cost()
        cong = self.compute_cong_cost_full() if include_cong else 0.0
        return {
            'wirelength_cost': wl,
            'density_cost':    den,
            'congestion_cost': cong,
            'proxy_cost':      WEIGHT_WL * wl + WEIGHT_DENSITY * den + WEIGHT_CONG * cong,
        }

    def delta_for_move(self, macro_idx: int, new_xy: Tuple[float, float],
                       include_cong: bool = False) -> dict:
        """Compute proxy delta if macro_idx moves to new_xy. State unchanged.

        Returns dict with delta_wl, delta_density, delta_cong (=0 if
        include_cong=False; for True we currently fall back to a full
        recompute on a temporary placement, which is slow).

        include_cong=True is only useful for validation; the local-search
        polish stages prefer include_cong=False then full-eval after the
        cheap delta wins.
        """
        delta_total_hpwl, wl_state = self._delta_wl_for_move(macro_idx, new_xy)
        delta_wl = delta_total_hpwl / self.wl_denominator

        den_state = self._delta_density_for_move(macro_idx, new_xy)
        # New density cost: apply cells_delta to a copy, recompute mean.
        new_grid = self.grid_occupied.copy()
        for (r, c), d in den_state['cells_delta'].items():
            new_grid[r, c] += d
        new_den_cost = self.compute_density_cost(grid_occupied=new_grid)
        delta_density = new_den_cost - self.compute_density_cost()

        delta_cong = 0.0
        if include_cong:
            # Tentative-apply path: snapshot ALL mutable state, commit, eval,
            # restore. Smoothing + ABU is O(grid_size) per call, so this is
            # ~100× faster than a full PLC rebuild. Mostly useful for
            # validation; the polish stages prefer include_cong=False + full
            # eval after the cheap delta wins.
            old_cong = self.compute_cong_cost_full()
            saved = {
                'raw_V':         self.raw_V.copy(),
                'raw_H':         self.raw_H.copy(),
                'macro_raw_V':   self.macro_raw_V.copy(),
                'macro_raw_H':   self.macro_raw_H.copy(),
                'pin_xy':        self.pin_xy.copy(),
                'macro_pos':     self.macro_pos.copy(),
                'net_min_x':     self.net_min_x.copy(),
                'net_max_x':     self.net_max_x.copy(),
                'net_min_y':     self.net_min_y.copy(),
                'net_max_y':     self.net_max_y.copy(),
                'net_hpwl':      self.net_hpwl.copy(),
                'total_hpwl':    self.total_hpwl,
                'grid_occupied': self.grid_occupied.copy(),
                'macro_overlap': dict(self.macro_overlap[macro_idx])
                                  if self.macro_overlap[macro_idx] else None,
            }
            self.commit_move(macro_idx, new_xy)
            new_cong = self.compute_cong_cost_full()
            self.raw_V[:]         = saved['raw_V']
            self.raw_H[:]         = saved['raw_H']
            self.macro_raw_V[:]   = saved['macro_raw_V']
            self.macro_raw_H[:]   = saved['macro_raw_H']
            self.pin_xy[:]        = saved['pin_xy']
            self.macro_pos[:]     = saved['macro_pos']
            self.net_min_x[:]     = saved['net_min_x']
            self.net_max_x[:]     = saved['net_max_x']
            self.net_min_y[:]     = saved['net_min_y']
            self.net_max_y[:]     = saved['net_max_y']
            self.net_hpwl[:]      = saved['net_hpwl']
            self.total_hpwl       = saved['total_hpwl']
            self.grid_occupied[:] = saved['grid_occupied']
            self.macro_overlap[macro_idx] = saved['macro_overlap']
            delta_cong = new_cong - old_cong

        return {
            'delta_wl':      float(delta_wl),
            'delta_density': float(delta_density),
            'delta_cong':    float(delta_cong),
            'delta_proxy':   float(WEIGHT_WL * delta_wl
                                   + WEIGHT_DENSITY * delta_density
                                   + WEIGHT_CONG * delta_cong),
            '_wl_state':     wl_state,
            '_den_state':    den_state,
        }

    def commit_move(self, macro_idx: int, new_xy: Tuple[float, float],
                    state: Optional[dict] = None) -> None:
        """Apply a move. If `state` is provided (from delta_for_move),
        reuse its pre-computed bbox/cell deltas (avoids recomputation).
        Otherwise compute them fresh.

        Incrementally updates raw_V/raw_H (net routes) and macro_raw_V/
        macro_raw_H (hard-macro blockage). Each move re-routes only the
        nets touched by `macro_idx`, plus the macro itself if hard."""
        if state is None:
            state = self.delta_for_move(macro_idx, new_xy, include_cong=False)

        # Cong: subtract OLD contributions before pin_xy / macro_pos updates.
        # nets_per_macro is static (doesn't depend on positions), so re-routing
        # only those nets is sufficient.
        nets_to_update = self.nets_per_macro[macro_idx]
        for n in nets_to_update:
            self._add_net_to_routing(int(n), self.raw_V, self.raw_H, sign=-1.0)
        if macro_idx < self.nH:
            self._add_macro_blockage(
                macro_idx, self.macro_raw_V, self.macro_raw_H, sign=-1.0,
            )

        self._commit_wl(macro_idx, new_xy, state['_wl_state'])
        self._commit_density(macro_idx, new_xy, state['_den_state'])
        self.macro_pos[macro_idx, 0] = float(new_xy[0])
        self.macro_pos[macro_idx, 1] = float(new_xy[1])

        # Cong: add NEW contributions after positions are updated.
        for n in nets_to_update:
            self._add_net_to_routing(int(n), self.raw_V, self.raw_H, sign=+1.0)
        if macro_idx < self.nH:
            self._add_macro_blockage(
                macro_idx, self.macro_raw_V, self.macro_raw_H, sign=+1.0,
            )


# ════════════════════════════════════════════════════════════════════════════
#  Validation harness
# ════════════════════════════════════════════════════════════════════════════

def validate_against_plc(benchmark: Benchmark, plc: PlacementCost,
                          n_moves: int = 30, seed: int = 0,
                          atol_wl: float = 1e-5, atol_den: float = 1e-5,
                          atol_cong: float = 1e-5,
                          check_cong: bool = True) -> dict:
    """Random-move validation. Compares IncrementalEval to compute_proxy_cost
    after each move. Returns dict with per-step mismatches.

    Test plan:
      1. Initialise IncrementalEval from benchmark.macro_positions.
      2. Compute full proxy via compute_proxy_cost as ground truth.
      3. Compare against e.proxy_breakdown(include_cong=True).
      4. For n_moves random (movable macro, random new_xy):
         a. delta = e.delta_for_move(...)
         b. e.commit_move(...)
         c. Full proxy via compute_proxy_cost.
         d. Compare WL / density / cong term-by-term.

    `check_cong=False` skips the per-step cong recompute (cong is a full
    O(N) rebuild until incremental cong lands; useful for quick WL/density
    sanity at high n_moves).
    """
    rng  = np.random.default_rng(seed)
    nM   = int(benchmark.num_macros)
    cw   = float(benchmark.canvas_width)
    ch   = float(benchmark.canvas_height)
    base = benchmark.macro_positions.cpu().numpy().astype(np.float64).copy()
    movable = benchmark.get_movable_mask().cpu().numpy().astype(bool)
    movable_idx = np.where(movable[:nM])[0]
    if movable_idx.size == 0:
        return {'error': 'no movable macros'}

    e = IncrementalEval(benchmark, plc=plc)
    e.set_placement(base)

    def full_proxy(placement_np):
        return compute_proxy_cost(
            torch.tensor(placement_np, dtype=torch.float32), benchmark, plc,
        )

    mismatches = []
    # Initial cross-check (always includes cong — that's the baseline).
    inc0 = e.proxy_breakdown(include_cong=True)
    plc0 = full_proxy(base)
    init_dwl  = inc0['wirelength_cost']  - plc0['wirelength_cost']
    init_dden = inc0['density_cost']      - plc0['density_cost']
    init_dcong= inc0['congestion_cost']   - plc0['congestion_cost']
    mismatches.append({
        'step': -1,
        'inc_wl': inc0['wirelength_cost'],  'plc_wl': plc0['wirelength_cost'],   'dwl': init_dwl,
        'inc_den': inc0['density_cost'],    'plc_den': plc0['density_cost'],     'dden': init_dden,
        'inc_cong': inc0['congestion_cost'],'plc_cong': plc0['congestion_cost'], 'dcong': init_dcong,
    })

    cur = base.copy()
    for step in range(n_moves):
        m_i = int(rng.choice(movable_idx))
        # Small random shift.
        dx = float(rng.normal(0.0, 0.01 * (cw + ch) * 0.5))
        dy = float(rng.normal(0.0, 0.01 * (cw + ch) * 0.5))
        new_x = float(np.clip(cur[m_i, 0] + dx, 0.0, cw))
        new_y = float(np.clip(cur[m_i, 1] + dy, 0.0, ch))

        state = e.delta_for_move(m_i, (new_x, new_y), include_cong=False)
        e.commit_move(m_i, (new_x, new_y), state)
        cur[m_i, 0] = new_x
        cur[m_i, 1] = new_y

        inc = e.proxy_breakdown(include_cong=check_cong)
        plc_costs = full_proxy(cur)
        dwl  = inc['wirelength_cost'] - plc_costs['wirelength_cost']
        dden = inc['density_cost']    - plc_costs['density_cost']
        rec = {
            'step': step, 'macro_idx': m_i, 'new_xy': (new_x, new_y),
            'inc_wl': inc['wirelength_cost'],  'plc_wl': plc_costs['wirelength_cost'],   'dwl': dwl,
            'inc_den': inc['density_cost'],    'plc_den': plc_costs['density_cost'],     'dden': dden,
        }
        if check_cong:
            dcong = inc['congestion_cost'] - plc_costs['congestion_cost']
            rec['inc_cong']  = inc['congestion_cost']
            rec['plc_cong']  = plc_costs['congestion_cost']
            rec['dcong']     = dcong
        mismatches.append(rec)

    # Aggregate stats.
    abs_dwl   = [abs(m['dwl'])   for m in mismatches if 'dwl'   in m]
    abs_dden  = [abs(m['dden'])  for m in mismatches if 'dden'  in m]
    abs_dcong = [abs(m['dcong']) for m in mismatches if 'dcong' in m]
    summary = {
        'n_moves':        n_moves,
        'max_abs_dwl':    max(abs_dwl)   if abs_dwl   else 0.0,
        'max_abs_dden':   max(abs_dden)  if abs_dden  else 0.0,
        'max_abs_dcong':  max(abs_dcong) if abs_dcong else 0.0,
        'wl_pass':        max(abs_dwl)   <= atol_wl   if abs_dwl   else True,
        'density_pass':   max(abs_dden)  <= atol_den  if abs_dden  else True,
        'cong_pass':      max(abs_dcong) <= atol_cong if abs_dcong else True,
        'mismatches':     mismatches,
    }
    return summary


# ════════════════════════════════════════════════════════════════════════════
#  CLI for quick check (run from a benchmark loader)
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
#  Benchmark: per-move eval time vs. compute_proxy_cost
# ════════════════════════════════════════════════════════════════════════════

def benchmark_per_move(benchmark: Benchmark, plc: PlacementCost,
                       n_moves: int = 200, seed: int = 0) -> dict:
    """Compare per-move proxy evaluation time:

    A) compute_proxy_cost: rebuild PLC state from the placement tensor and
       call get_wirelength + get_density_cost + get_congestion_cost. This is
       what the Stage 2 / basin-hop / polish pipelines call.
    B) IncrementalEval: delta_for_move (WL+density) + commit_move (updates all
       three caches) + proxy_breakdown (reads cached cong, O(grid_size)).

    Returns per-call timings in seconds, plus a speedup ratio.
    """
    import time
    rng     = np.random.default_rng(seed)
    nM      = int(benchmark.num_macros)
    cw      = float(benchmark.canvas_width)
    ch      = float(benchmark.canvas_height)
    base    = benchmark.macro_positions.cpu().numpy().astype(np.float64).copy()
    movable = benchmark.get_movable_mask().cpu().numpy().astype(bool)
    movable_idx = np.where(movable[:nM])[0]
    if movable_idx.size == 0:
        return {'error': 'no movable macros'}

    # Pre-generate the move sequence so both paths do the same moves.
    moves = []
    cur = base.copy()
    for _ in range(n_moves):
        m_i = int(rng.choice(movable_idx))
        dx = float(rng.normal(0.0, 0.01 * (cw + ch) * 0.5))
        dy = float(rng.normal(0.0, 0.01 * (cw + ch) * 0.5))
        new_x = float(np.clip(cur[m_i, 0] + dx, 0.0, cw))
        new_y = float(np.clip(cur[m_i, 1] + dy, 0.0, ch))
        moves.append((m_i, new_x, new_y))
        cur[m_i, 0] = new_x; cur[m_i, 1] = new_y

    # ── (A) compute_proxy_cost path ──
    cur = base.copy()
    # Warm-up call to JIT / load PLC state.
    _ = compute_proxy_cost(torch.tensor(cur, dtype=torch.float32), benchmark, plc)
    costs_a = []
    t0 = time.perf_counter()
    for (m_i, nx, ny) in moves:
        cur[m_i, 0] = nx; cur[m_i, 1] = ny
        c = compute_proxy_cost(torch.tensor(cur, dtype=torch.float32), benchmark, plc)
        costs_a.append(float(c['proxy_cost']))
    t_a = time.perf_counter() - t0

    # ── (B) IncrementalEval path ──
    e = IncrementalEval(benchmark, plc=plc)
    e.set_placement(base)
    costs_b = []
    t0 = time.perf_counter()
    for (m_i, nx, ny) in moves:
        state = e.delta_for_move(m_i, (nx, ny), include_cong=False)
        e.commit_move(m_i, (nx, ny), state)
        br = e.proxy_breakdown(include_cong=True)
        costs_b.append(float(br['proxy_cost']))
    t_b = time.perf_counter() - t0

    # Cross-check final costs roughly agree (float-noise tolerance).
    max_drift = max(abs(a - b) for a, b in zip(costs_a, costs_b))

    return {
        'n_moves':           n_moves,
        'compute_proxy_s':   t_a,
        'incremental_s':     t_b,
        'compute_proxy_per_move_ms': 1000.0 * t_a / n_moves,
        'incremental_per_move_ms':   1000.0 * t_b / n_moves,
        'speedup':           t_a / t_b if t_b > 0 else float('inf'),
        'max_proxy_drift':   max_drift,
        'last_proxy_a':      costs_a[-1] if costs_a else None,
        'last_proxy_b':      costs_b[-1] if costs_b else None,
    }


if __name__ == '__main__':
    import argparse, os
    from macro_place.loader import load_benchmark
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', required=True, help='benchmark name, e.g. ibm01')
    parser.add_argument('--plc-root', default='external/MacroPlacement/Testcases/ICCAD04')
    parser.add_argument('--n-moves', type=int, default=20)
    parser.add_argument('--benchmark', action='store_true',
                        help='Run per-move benchmark vs. compute_proxy_cost (skips validation).')
    parser.add_argument('--bench-n-moves', type=int, default=200,
                        help='Number of moves for the benchmark (default: 200).')
    args = parser.parse_args()

    # Use load_benchmark (same path as macro_place.evaluate). The .pt files in
    # benchmarks/processed/public/ are stripped caches that miss net_nodes; the
    # netlist.pb.txt + initial.plc combo is the canonical source.
    netlist  = os.path.join(args.plc_root, args.bench, "netlist.pb.txt")
    init_plc = os.path.join(args.plc_root, args.bench, "initial.plc")
    plc_file = init_plc if os.path.exists(init_plc) else None
    benchmark, plc = load_benchmark(netlist, plc_file, name=args.bench)

    if args.benchmark:
        print(f"=== Benchmarking IncrementalEval vs compute_proxy_cost on {args.bench} ===")
        bres = benchmark_per_move(benchmark, plc, n_moves=args.bench_n_moves)
        for k, v in bres.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")
        import sys; sys.exit(0)

    print(f"=== Validating IncrementalEval against PLC on {args.bench} ===")
    res = validate_against_plc(benchmark, plc, n_moves=args.n_moves)
    print(f"max |Δ WL   cost|: {res['max_abs_dwl']:.3e}    PASS={res['wl_pass']}")
    print(f"max |Δ Den  cost|: {res['max_abs_dden']:.3e}    PASS={res['density_pass']}")
    print(f"max |Δ Cong cost|: {res['max_abs_dcong']:.3e}    PASS={res['cong_pass']}")
    if not (res['wl_pass'] and res['density_pass'] and res['cong_pass']):
        print()
        print("=== Per-step (first 5) ===")
        for m in res['mismatches'][:5]:
            print(m)

    if not res['cong_pass']:
        print()
        print("=== Cong grid diff (initial placement) ===")
        e = IncrementalEval(benchmark, plc=plc)
        e.set_placement(benchmark.macro_positions.cpu().numpy().astype(np.float64))
        diag = e.diff_against_plc_routing()
        for k, v in diag.items():
            print(f"  {k}: {v}")
        print()
        print("=== Per-macro bbox dump (offending macros) ===")
        for m in [52, 119, 7, 148, 26, 42, 128]:
            info = e._debug_macro_bbox(m)
            print(f"macro {m}:")
            print(f"  mine: {info.get('my')}")
            print(f"  plc:  {info.get('plc')}")
