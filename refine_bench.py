#!/usr/bin/env python
"""
refine_bench.py — fast, fixed-input bench for v60's CPU refinement stages.

The problem this solves: on the big designs the GPU engine is 68–78 % of
wall-clock (ibm14 ≈ 3450 s engine vs ≈ 1600 s refinement), so re-running the
whole pipeline to test a refinement tweak costs ~5000 s and you end up only
ever tuning the small designs. This decouples the two: dump the pre-refinement
placement ONCE, then A/B refinement changes against that fixed input in a few
hundred seconds — on the real big design.

Two subcommands:

  dump   Run the full pipeline up to (but not including) the CPU refinement
         stages and save the candidate pool to a cache dir. Pays the engine
         cost a single time. Deterministic by default so the cache is stable.

  run    Load a cached placement and run a chosen refinement stage (cd / swap /
         softswap / all) on it, reporting the proxy breakdown (stage-reported
         AND an independent compute_proxy_cost ground-truth check) and timing.
         Refinement is pure-CPU float64, so this is bit-reproducible with a
         fixed seed at full speed — no GPU determinism tax.

Examples:
  # one-time: cache ibm14's pre-refinement placement (~engine time, once)
  uv run python refine_bench.py dump -b ibm14 --cache-dir .refine_cache

  # iterate: run CD on the best cached candidate, 3 sweeps for a fast signal
  uv run python refine_bench.py run -b ibm14 --cache-dir .refine_cache \
      --stage cd --sweeps 3

  # full chain on candidate 0 (cd -> swap -> softswap)
  uv run python refine_bench.py run -b ibm14 --cache-dir .refine_cache --stage all
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

# Make the v60 submission importable (it self-adds its own dir to sys.path for
# its sibling modules once imported).
_V60_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "submissions", "v60")
if _V60_DIR not in sys.path:
    sys.path.insert(0, _V60_DIR)

from macro_place.loader import load_benchmark_from_dir   # noqa: E402
from macro_place.objective import compute_proxy_cost      # noqa: E402
from v60_placer import v60_Placer                          # noqa: E402

DEFAULT_PLC_ROOT = "external/MacroPlacement/Testcases/ICCAD04"


def _load_bench(name, plc_root):
    benchmark, plc = load_benchmark_from_dir(osp.join(plc_root, name))
    return benchmark, plc


def _breakdown(pos_t, benchmark, plc):
    """Independent ground-truth proxy of a full placement tensor."""
    nM = int(benchmark.num_macros)
    c = compute_proxy_cost(pos_t[:nM].to(torch.float32), benchmark, plc)
    return c


def cmd_dump(args):
    benchmark, plc = _load_bench(args.benchmark, args.plc_root)

    placer = v60_Placer()
    placer.deterministic = not args.no_deterministic   # stable cache by default
    placer.refine_cache_dir = args.cache_dir
    placer.refine_dump_only = True

    t0 = time.time()
    print(f"[bench] dumping pre-refinement cache for {args.benchmark} "
          f"(deterministic={placer.deterministic}) ...")
    placer.place(benchmark)
    print(f"[bench] dump done in {time.time()-t0:.1f}s -> {args.cache_dir}")


def cmd_run(args):
    benchmark, plc = _load_bench(args.benchmark, args.plc_root)

    meta_path = osp.join(args.cache_dir, f"{args.benchmark}_meta.json")
    if not osp.exists(meta_path):
        sys.exit(f"[bench] no cache for {args.benchmark} at {meta_path}; "
                 f"run `dump` first.")
    with open(meta_path) as f:
        meta = json.load(f)

    cands = meta["candidates"]
    if args.cand >= len(cands):
        sys.exit(f"[bench] candidate {args.cand} out of range "
                 f"(cache has {len(cands)})")
    cand = cands[args.cand]
    if args.load_pos:
        pos_np = np.load(args.load_pos)
        print(f"[bench] starting from {args.load_pos} (overrides cached cand)")
    else:
        pos_np = np.load(osp.join(args.cache_dir, cand["file"]))
    pos_t = torch.tensor(pos_np, dtype=torch.float64)

    # Fresh placer with defaults. deterministic=True only seeds the refinement
    # RNG (numpy) for a bit-reproducible A/B — it does NOT touch the GPU here,
    # so it costs nothing.
    placer = v60_Placer()
    placer.deterministic = True
    placer.seed = args.seed
    if args.sweeps is not None:
        placer.cd_polish_sweeps = args.sweeps
        placer.pair_swap_sweeps = args.sweeps
        placer.soft_pair_swap_sweeps = args.sweeps

    start_bd = _breakdown(pos_t, benchmark, plc)
    print(f"[bench] {args.benchmark}  cand{args.cand} (tag={cand['tag']})")
    print(f"[bench] start: proxy={start_bd['proxy_cost']:.6f}  "
          f"wl={start_bd['wirelength_cost']:.3f} "
          f"den={start_bd['density_cost']:.3f} "
          f"cong={start_bd['congestion_cost']:.3f}  "
          f"overlaps={start_bd['overlap_count']}")

    expand = {"all": ["cd", "swap", "softswap"],
              "all+bhop": ["cd", "swap", "softswap", "basinhop"]}
    stages = expand.get(args.stage, [args.stage])
    cur = pos_t
    t_all = time.time()
    for st in stages:
        t0 = time.time()
        if st == "cd":
            out, costs = placer._cd_polish(cur, benchmark, plc,
                                           max_workers=args.workers)
        elif st == "swap":
            out, costs = placer._pair_swap_polish(cur, benchmark, plc)
        elif st == "softswap":
            out, costs = placer._pair_swap_polish_soft(cur, benchmark, plc)
        elif st == "basinhop":
            # Per-hop σ line-search on this single floor (the production path).
            _apply_bhop_args(placer, args)
            out, _pr = placer._refine_basin_hop(cur, benchmark, plc)
            bd = _breakdown(out, benchmark, plc)
            costs = {'proxy_cost': bd['proxy_cost'],
                     'wirelength_cost': bd['wirelength_cost'],
                     'density_cost': bd['density_cost'],
                     'congestion_cost': bd['congestion_cost']}
        else:
            sys.exit(f"[bench] unknown stage {st!r}")
        dt = time.time() - t0
        if costs is None:
            print(f"[bench] {st:>8}: (no-op)  {dt:.1f}s")
            continue
        cur = out
        gt = _breakdown(cur, benchmark, plc)   # independent ground-truth check
        drift = abs(gt['proxy_cost'] - costs['proxy_cost'])
        print(f"[bench] {st:>8}: proxy={costs['proxy_cost']:.6f}  "
              f"wl={costs['wirelength_cost']:.3f} "
              f"den={costs['density_cost']:.3f} "
              f"cong={costs['congestion_cost']:.3f}  "
              f"overlaps={gt['overlap_count']}  {dt:.1f}s"
              f"   [gt|Δproxy|={drift:.2e}]")

    final = _breakdown(cur, benchmark, plc)
    print(f"[bench] FINAL: proxy={final['proxy_cost']:.6f}  "
          f"(start {start_bd['proxy_cost']:.6f}, "
          f"Δ={start_bd['proxy_cost']-final['proxy_cost']:+.6f})  "
          f"total {time.time()-t_all:.1f}s")

    if args.save_out:
        nM = int(benchmark.num_macros)
        np.save(args.save_out, cur[:nM].detach().cpu().numpy().astype(np.float64))
        print(f"[bench] saved final placement -> {args.save_out}")


def cmd_multicand(args):
    """Reproduce the pipeline's best-of-N refined floor: refine ALL cached
    candidates concurrently via _run_cpu_side (one fork per candidate, so wall ~=
    one candidate, NOT N), pick the best FINAL. The best PRE-refinement seed is not
    always the best POST-refinement one, so this is the apples-to-apples
    (production-level) floor a single-cand floor misses. `run --stage basinhop
    --load-pos <saved floor>` then exercises the production basin-hop on it."""
    benchmark, plc = _load_bench(args.benchmark, args.plc_root)
    nM = int(benchmark.num_macros)
    meta_path = osp.join(args.cache_dir, f"{args.benchmark}_meta.json")
    if not osp.exists(meta_path):
        sys.exit(f"[bench] no cache for {args.benchmark} at {meta_path}; "
                 f"run `dump` first.")
    with open(meta_path) as f:
        meta = json.load(f)

    placer = v60_Placer()
    placer.deterministic = True
    placer.seed = args.seed
    placer.cd_polish_parallel_workers = 1   # N candidate forks only, no inner CD pool

    cpu_cands = []
    for c in meta["candidates"]:
        pos_np = np.load(osp.join(args.cache_dir, c["file"]))
        cpu_cands.append({"pos": torch.tensor(pos_np, dtype=torch.float64),
                          "proxy": float(c["proxy"]), "tag": c["tag"]})

    print(f"[bench] {args.benchmark}: refining {len(cpu_cands)} candidates "
          f"concurrently (_run_cpu_side, cd->swap->softswap)")
    t0 = time.time()
    results = placer._run_cpu_side(cpu_cands, benchmark, plc)
    best_i, best = -1, None
    for i, (pos, proxy, tag) in enumerate(results):
        gt = _breakdown(pos, benchmark, plc)
        mark = ""
        if best is None or float(proxy) < best[1]:
            best, best_i = (pos, float(proxy)), i
            mark = "  <-- best"
        print(f"[bench]   cand{i}: floor={float(proxy):.6f}  "
              f"(gt={gt['proxy_cost']:.6f} cong={gt['congestion_cost']:.3f} "
              f"ovlp={gt['overlap_count']}){mark}")
    print(f"[bench] BEST-OF-{len(cpu_cands)} floor = cand{best_i} "
          f"proxy={best[1]:.6f}  (wall {time.time()-t0:.0f}s)")

    cur = best[0]
    if args.save_out:
        np.save(args.save_out, cur[:nM].detach().cpu().numpy().astype(np.float64))
        print(f"[bench] saved best-of-N floor -> {args.save_out}")


def main():
    p = argparse.ArgumentParser(
        prog="refine_bench",
        description="Fixed-input bench for v60 CPU refinement stages.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("dump", help="cache the pre-refinement placement once")
    pd.add_argument("-b", "--benchmark", default="ibm14")
    pd.add_argument("--cache-dir", default=".refine_cache")
    pd.add_argument("--plc-root", default=DEFAULT_PLC_ROOT)
    pd.add_argument("--no-deterministic", action="store_true",
                    help="faster but the cache won't be reproducible")
    pd.set_defaults(func=cmd_dump)

    pr = sub.add_parser("run", help="run a refinement stage on a cached input")
    pr.add_argument("-b", "--benchmark", default="ibm14")
    pr.add_argument("--cache-dir", default=".refine_cache")
    pr.add_argument("--plc-root", default=DEFAULT_PLC_ROOT)
    pr.add_argument("--stage",
                    choices=["cd", "swap", "softswap", "all", "basinhop", "all+bhop"],
                    default="cd",
                    help="'basinhop' = exact-cost refinement basin-hop; "
                         "'all+bhop' = cd,swap,softswap,basinhop")
    pr.add_argument("--cand", type=int, default=0,
                    help="which cached candidate (0 = best)")
    pr.add_argument("--load-pos", default=None,
                    help="start from this .npy placement instead of the cache "
                         "(e.g. a saved post-CD placement to A/B a later stage)")
    pr.add_argument("--save-out", default=None,
                    help="save the final placement to this .npy")
    pr.add_argument("--sweeps", type=int, default=None,
                    help="override sweeps for all stages (fast-signal A/B)")
    pr.add_argument("--workers", type=int, default=1,
                    help="CD candidate-scoring workers")
    pr.add_argument("--seed", type=int, default=0)
    _add_bhop_args(pr)
    pr.set_defaults(func=cmd_run)

    pm = sub.add_parser("multicand",
                        help="refine ALL cached candidates concurrently "
                             "(_run_cpu_side), pick best-of-N floor")
    pm.add_argument("-b", "--benchmark", default="ibm14")
    pm.add_argument("--cache-dir", default=".refine_cache")
    pm.add_argument("--plc-root", default=DEFAULT_PLC_ROOT)
    pm.add_argument("--save-out", default=None, help="save best-of-N floor .npy")
    pm.add_argument("--seed", type=int, default=0)
    pm.set_defaults(func=cmd_multicand)

    args = p.parse_args()
    args.func(args)


def _add_bhop_args(ap):
    """Shared basin-hop knobs (used by `run --stage basinhop` and `multicand`)."""
    ap.add_argument("--hops", type=int, default=10,
                    help="basin-hop ceiling (production default: 10, early-stopped)")
    ap.add_argument("--sigma-set", type=str, default=None,
                    help="comma-separated per-hop σ line-search set, as fractions of "
                         "mean canvas side (default: the placer's refine_basin_hop_sigma_set)")
    ap.add_argument("--cong-frac", type=float, default=0.05,
                    help="congestion tail fraction defining 'hot' cells")
    ap.add_argument("--cap", type=int, default=60,
                    help="max hot soft macros perturbed per hop (0 = all)")
    ap.add_argument("--bhop-sweeps", type=int, default=15,
                    help="CD sweep cap per hop (production default: 15)")
    ap.add_argument("--bhop-min-improve", type=float, default=1e-3,
                    help="per-hop relative-improvement early-stop threshold "
                         "(production default: 1e-3 = 0.1%%; 0 disables)")


def _apply_bhop_args(placer, args):
    """Push the shared basin-hop CLI knobs onto the placer's ctor attributes."""
    placer.refine_basin_hop_hops      = args.hops
    placer.refine_basin_hop_cong_frac = args.cong_frac
    placer.refine_basin_hop_cap       = args.cap
    placer.refine_basin_hop_cd_sweeps = args.bhop_sweeps
    placer.refine_basin_hop_min_improve_frac = args.bhop_min_improve
    if args.sigma_set:
        placer.refine_basin_hop_sigma_set = tuple(
            float(x) for x in args.sigma_set.split(","))


if __name__ == "__main__":
    main()
