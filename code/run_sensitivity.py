"""
run_sensitivity.py
-------------------
One-factor-at-a-time (OFAT) parameter sensitivity sweep for a SINGLE
instance: how total distance (and, where meaningful, fallback/effective
distance) changes as p_bias, departure_h, n_iter, alpha_c or alpha_d vary,
holding the others at a baseline.

Model A/B removed (see design notes): this file used to also sweep beta
for learn_A/B/dynamic_A/B, which no longer exist. What's left:
  - 'standard'/'dynamic' (no ML) sweeps for p_bias, departure_h, n_iter --
    still meaningful without any trained model.
  - 'HSR'/'SRA' (heuristic_c.py / heuristic_d.py, WITH their trained ML
    penalty at baseline alpha/beta) added to the SAME p_bias/departure_h/
    n_iter sweeps whenever --model-c/--model-d resolve to an existing
    bundle -- both build_graph_c/build_graph_d only depend on
    (bundle, locker_cap), never on alpha/beta/departure_h/p_bias, so build
    once per instance and reuse across every sweep value (same reasoning
    as sweep_alpha_c/sweep_alpha_d below).
  - sweep_alpha_c / sweep_alpha_d -- each heuristic's own ablation (alpha at
    beta=0 vs beta=baseline), self-contained.

Usage
-----
    python run_sensitivity.py <instance.txt>
    python run_sensitivity.py <instance.txt> --sweep p_bias,departure_h
    python run_sensitivity.py <instance.txt> --sweep alpha_c,alpha_d \
        --model-c ../data/models_C/saturation_C_best.pkl \
        --model-d ../data/models_D/saturation_D_best.pkl
    # Run on more than one instance without overwriting: tag the output
    python run_sensitivity.py <low_occ_instance.txt>  --tag low  --sweep p_bias,departure_h,n_iter,alpha_c,alpha_d --model-c ... --model-d ...
    python run_sensitivity.py <high_occ_instance.txt> --tag high --sweep p_bias,departure_h,n_iter,alpha_c,alpha_d --model-c ... --model-d ...

Output
------
    data/sensitivity_{p_bias,departure_h,n_iter,alpha_c,alpha_d}[_{tag}].csv
    (only the ones in --sweep are written; the _{tag} suffix is added only
    if --tag is given, so single-instance runs keep the old filenames)
"""

from __future__ import annotations
import os
import sys
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import read_full_instance
from heuristic import INF, N_ITER, P_BIAS, build_graph, br_CWS
from heuristic_dynamic import build_graph_dynamic, br_CWS_dynamic
from heuristic_c import build_graph_c, br_CWS_c, ALPHA_DEFAULT
from heuristic_d import build_graph_d
from heuristic_learn import BETA_DEFAULT, load_model
from simulator import simulate_solution

DEFAULT_P_BIASES     = [0.10, 0.25, 0.40]
DEFAULT_DEPARTURE_HS = [7.0, 8.0, 12.0, 17.0]   # morning peak, baseline, midday, evening peak
DEFAULT_ALPHAS       = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0=pure time, 1=pure distance

_HERE       = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(_HERE, '..', 'data')
MODEL_C_CANDIDATES = [os.path.join(_HERE, '..', 'data', 'models_C', 'saturation_C_best.pkl')]
MODEL_D_CANDIDATES = [os.path.join(_HERE, '..', 'data', 'models_D', 'saturation_D_best.pkl')]


def _resolve_model_path(explicit: str | None, candidates: list[str], label: str) -> str | None:
    """Explicit --model-c/--model-d path if given, else the first existing candidate."""
    if explicit:
        return explicit
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _try_load(path: str | None, label: str):
    if not path:
        print(f"  (no {label} model found -- skipping)")
        return None
    if not os.path.exists(path):
        print(f"  (no {label} model at {path} -- skipping)")
        return None
    return load_model(path)


# =============================================================================
# GENERIC BUILD-ONCE / LOOP-MANY DRIVERS
# =============================================================================

def _grasp_loop(br_fn, active_nodes, savings_list, vehicle_cap, depot, n_iter: int,
                checkpoint_every: int | None = None, **br_kwargs):
    """Run n_iter GRASP iterations of br_fn over an already-built candidate
    list, tracking best-cost-so-far. Returns (best_sol, curve) where curve is
    [(iter, best_cost_km), ...] recorded every checkpoint_every iterations
    (plus the final one); checkpoint_every=None records only the final point."""
    ck = checkpoint_every or n_iter
    best_sol, best_cost = None, INF
    curve = []
    for it in range(1, n_iter + 1):
        sol = br_fn(active_nodes, savings_list, vehicle_cap, depot, **br_kwargs)
        if sol.cost < best_cost:
            best_sol, best_cost = sol, sol.cost
        if it % ck == 0 or it == n_iter:
            curve.append((it, round(best_cost / 1000, 4)))
    return best_sol, curve


def _run_loop(kind: str, active_nodes, savings, vehicle_cap, depot, n_iter: int,
             p_bias: float, dep_h: float, checkpoint_every: int | None = None):
    """kind in {'std', 'dynamic'} -- dispatches to the right br_CWS*."""
    if kind == 'std':
        br_fn, kwargs = br_CWS, {'p': p_bias}
    else:
        br_fn, kwargs = br_CWS_dynamic, {'departure_h': dep_h, 'p': p_bias}
    return _grasp_loop(br_fn, active_nodes, savings, vehicle_cap, depot,
                       n_iter, checkpoint_every, **kwargs)


def _run_loop_penalized(a_nodes, sav, vehicle_cap, depot, n_iter: int, p_bias: float,
                        hourly_probs, d_fb, alpha: float, beta: float, dep_h: float,
                        checkpoint_every: int | None = None):
    """HSR/SRA share the same merge core (heuristic_c.br_CWS_c) -- only the
    hourly_probs table's SOURCE differs (_hourly_saturation_probs vs
    _hourly_release_risk, chosen by which build_graph_* produced it)."""
    return _grasp_loop(br_CWS_c, a_nodes, sav, vehicle_cap, depot, n_iter, checkpoint_every,
                       hourly_probs=hourly_probs, d_fb=d_fb, alpha=alpha, beta=beta,
                       departure_h=dep_h, p=p_bias)


def _fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km: float) -> dict:
    """Locker-overflow fallback KPIs (simulator.simulate_solution, reused as-is).
    Empty dict if no capacity data is available for this instance."""
    if not locker_cap or best_sol is None:
        return {}
    fb_km, n_fb = simulate_solution(best_sol, orders, locker_cap, dist_matrix)
    return {'fallback_km': round(fb_km, 4), 'n_fallbacks': n_fb,
           'effective_km': round(dist_km + fb_km, 4)}


def _build_penalized_methods(nodes, dist_matrix, locker_cap, bundle_c, bundle_d):
    """
    Build HSR (heuristic_c) and/or SRA (heuristic_d) graphs ONCE, if their
    bundles are available -- both only depend on (bundle, locker_cap), never
    on alpha/beta/departure_h/p_bias, so the same build is reused across
    every sweep value (p_bias, departure_h, n_iter, or the alpha ablations).

    Returns {name: (a_nodes, sav, hourly_probs, d_fb, build_s)}.
    """
    methods: dict[str, tuple] = {}
    if bundle_c is not None:
        t0 = time.perf_counter()
        a_nodes, sav, hourly_probs, d_fb = build_graph_c(nodes, dist_matrix, bundle_c, locker_cap)
        methods['HSR'] = (a_nodes, sav, hourly_probs, d_fb, time.perf_counter() - t0)
    if bundle_d is not None:
        t0 = time.perf_counter()
        a_nodes, sav, hourly_probs, d_fb = build_graph_d(nodes, dist_matrix, bundle_d, locker_cap)
        methods['SRA'] = (a_nodes, sav, hourly_probs, d_fb, time.perf_counter() - t0)
    return methods


# =============================================================================
# SWEEPS
# =============================================================================

def sweep_p_bias(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders, dep_h,
                 p_biases: list[float], n_iter: int, bundle_c=None, bundle_d=None,
                 alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                 verbose: bool = False) -> list[dict]:
    """p_bias is loop-level for every method -- build once, reuse across values."""
    rows = []
    builds: dict[str, tuple] = {}   # name -> (kind, active_nodes, savings, build_s)

    t0 = time.perf_counter()
    a_nodes, sav = build_graph(nodes, dist_matrix)
    builds['standard'] = ('std', a_nodes, sav, time.perf_counter() - t0)

    t0 = time.perf_counter()
    a_nodes, sav = build_graph_dynamic(nodes, dist_matrix)   # bundle=None
    builds['dynamic'] = ('dynamic', a_nodes, sav, time.perf_counter() - t0)

    for name, (kind, a_nodes, sav, build_s) in builds.items():
        for pb in p_biases:
            t1 = time.perf_counter()
            best_sol, _ = _run_loop(kind, a_nodes, sav, vehicle_cap, depot, n_iter,
                                    pb, dep_h)
            loop_s = time.perf_counter() - t1
            dist_km = round(best_sol.cost / 1000, 4)
            row = {'method': name, 'p_bias': pb, 'dist_km': dist_km,
                  'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    {name:<10} p_bias={pb:<5} -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")

    penalized = _build_penalized_methods(nodes, dist_matrix, locker_cap, bundle_c, bundle_d)
    for name, (a_nodes, sav, hourly_probs, d_fb, build_s) in penalized.items():
        for pb in p_biases:
            t1 = time.perf_counter()
            best_sol, _ = _run_loop_penalized(a_nodes, sav, vehicle_cap, depot, n_iter, pb,
                                              hourly_probs, d_fb, alpha, beta, dep_h)
            loop_s = time.perf_counter() - t1
            dist_km = round(best_sol.cost / 1000, 4)
            row = {'method': name, 'p_bias': pb, 'dist_km': dist_km,
                  'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    {name:<10} p_bias={pb:<5} -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")
    return rows


def sweep_departure_h(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                      p_bias: float, departure_hs: list[float], n_iter: int,
                      bundle_c=None, bundle_d=None,
                      alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                      verbose: bool = False) -> list[dict]:
    """departure_h only affects 'dynamic'/'HSR'/'SRA' (all three read it at
    loop time, never bake it into the build). 'standard' has no time
    dependency at all, so it's excluded from this sweep."""
    rows = []
    t0 = time.perf_counter()
    dyn_nodes, dyn_sav = build_graph_dynamic(nodes, dist_matrix)   # bundle=None
    dyn_build_s = time.perf_counter() - t0
    for dep_h in departure_hs:
        t1 = time.perf_counter()
        best_sol, _ = _run_loop('dynamic', dyn_nodes, dyn_sav, vehicle_cap, depot,
                                n_iter, p_bias, dep_h)
        loop_s = time.perf_counter() - t1
        dist_km = round(best_sol.cost / 1000, 4)
        row = {'method': 'dynamic', 'departure_h': dep_h,
              'dist_km': dist_km, 'n_routes': len(best_sol.routes),
              'build_s': round(dyn_build_s, 3), 'loop_s': round(loop_s, 3)}
        row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
        rows.append(row)
        if verbose:
            eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
            print(f"    dynamic    dep_h={dep_h:<5} -> {dist_km:.3f} km{eff} "
                  f"(build {dyn_build_s:.2f}s, loop {loop_s:.2f}s)")

    penalized = _build_penalized_methods(nodes, dist_matrix, locker_cap, bundle_c, bundle_d)
    for name, (a_nodes, sav, hourly_probs, d_fb, build_s) in penalized.items():
        for dep_h in departure_hs:
            t1 = time.perf_counter()
            best_sol, _ = _run_loop_penalized(a_nodes, sav, vehicle_cap, depot, n_iter, p_bias,
                                              hourly_probs, d_fb, alpha, beta, dep_h)
            loop_s = time.perf_counter() - t1
            dist_km = round(best_sol.cost / 1000, 4)
            row = {'method': name, 'departure_h': dep_h,
                  'dist_km': dist_km, 'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    {name:<10} dep_h={dep_h:<5} -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")
    return rows


def sweep_n_iter(nodes, dist_matrix, vehicle_cap, depot, locker_cap, dep_h,
                 p_bias: float, n_iter_max: int, checkpoint_every: int,
                 bundle_c=None, bundle_d=None,
                 alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                 verbose: bool = False) -> list[dict]:
    """Single run per method at baseline params, checkpointing best-cost-so-far
    every checkpoint_every iterations -- one pass yields the full convergence
    curve instead of separate reruns per candidate n_iter value.

    Distance-only (no fallback_km/effective_km): the checkpoint mechanism only
    tracks a scalar best-cost-so-far, not a full Solution snapshot at each
    checkpoint, so there's no solution to run simulate_solution() against per
    point without re-architecting the checkpointing. See sweep_p_bias/
    sweep_departure_h for effective-distance (with fallback) comparisons."""
    rows = []
    methods = [('standard', 'std', build_graph(nodes, dist_matrix)),
              ('dynamic', 'dynamic', build_graph_dynamic(nodes, dist_matrix))]
    for name, kind, (a_nodes, sav) in methods:
        _, curve = _run_loop(kind, a_nodes, sav, vehicle_cap, depot, n_iter_max,
                             p_bias, dep_h, checkpoint_every)
        for it, cost_km in curve:
            rows.append({'method': name, 'iter': it, 'dist_km': cost_km})
        if verbose:
            print(f"    {name:<10} final @ iter {n_iter_max}: {curve[-1][1]:.3f} km")

    penalized = _build_penalized_methods(nodes, dist_matrix, locker_cap, bundle_c, bundle_d)
    for name, (a_nodes, sav, hourly_probs, d_fb, _build_s) in penalized.items():
        _, curve = _grasp_loop(br_CWS_c, a_nodes, sav, vehicle_cap, depot, n_iter_max,
                               checkpoint_every, hourly_probs=hourly_probs, d_fb=d_fb,
                               alpha=alpha, beta=beta, departure_h=dep_h, p=p_bias)
        for it, cost_km in curve:
            rows.append({'method': name, 'iter': it, 'dist_km': cost_km})
        if verbose:
            print(f"    {name:<10} final @ iter {n_iter_max}: {curve[-1][1]:.3f} km")
    return rows


def sweep_alpha_c(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders, dep_h,
                  bundle_c, alphas: list[float], beta: float, p_bias: float, n_iter: int,
                  verbose: bool = False) -> list[dict]:
    """
    Ablation sweep for heuristic_c.py (HSR): alpha (distance/time blend) at
    two beta settings -- 0 (no ML penalty, pure distance+time) and the given
    baseline beta (with penalty) -- directly answering "how much better is
    distance+time+penalty vs. just distance+time".

    alpha AND beta are BOTH loop-level for HSR (unlike beta for the old
    learn/dynamic+ML): build_graph_c's hourly P_j table only depends on
    bundle/locker_cap, not alpha/beta (those only enter _fused_saving_c at
    merge-decision time) -- so build once, reuse across the whole grid.
    """
    rows = []
    t0 = time.perf_counter()
    a_nodes, sav, hourly_probs, d_fb = build_graph_c(nodes, dist_matrix, bundle_c, locker_cap)
    build_s = time.perf_counter() - t0

    for beta_val, penalty_label in ((0.0, 'no_penalty'), (beta, 'with_penalty')):
        for alpha in alphas:
            t1 = time.perf_counter()
            best_sol, _ = _grasp_loop(br_CWS_c, a_nodes, sav, vehicle_cap, depot, n_iter,
                                      None, hourly_probs=hourly_probs, d_fb=d_fb,
                                      alpha=alpha, beta=beta_val, departure_h=dep_h, p=p_bias)
            loop_s = time.perf_counter() - t1

            dist_km = round(best_sol.cost / 1000, 4)
            row = {'alpha': alpha, 'beta': beta_val, 'penalty': penalty_label,
                  'dist_km': dist_km, 'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    alpha={alpha:<5} [{penalty_label:<12}] -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")
    return rows


def sweep_alpha_d(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders, dep_h,
                  bundle_d, alphas: list[float], beta: float, p_bias: float, n_iter: int,
                  verbose: bool = False) -> list[dict]:
    """
    Ablation sweep for heuristic_d.py (SRA) -- identical structure to
    sweep_alpha_c, only the build (build_graph_d, hourly RELEASE-RISK table
    instead of overflow-risk) and hence the ML signal being ablated differ.
    Reuses the same merge core (heuristic_c.br_CWS_c) as HSR.
    """
    rows = []
    t0 = time.perf_counter()
    a_nodes, sav, hourly_probs, d_fb = build_graph_d(nodes, dist_matrix, bundle_d, locker_cap)
    build_s = time.perf_counter() - t0

    for beta_val, penalty_label in ((0.0, 'no_penalty'), (beta, 'with_penalty')):
        for alpha in alphas:
            t1 = time.perf_counter()
            best_sol, _ = _grasp_loop(br_CWS_c, a_nodes, sav, vehicle_cap, depot, n_iter,
                                      None, hourly_probs=hourly_probs, d_fb=d_fb,
                                      alpha=alpha, beta=beta_val, departure_h=dep_h, p=p_bias)
            loop_s = time.perf_counter() - t1

            dist_km = round(best_sol.cost / 1000, 4)
            row = {'alpha': alpha, 'beta': beta_val, 'penalty': penalty_label,
                  'dist_km': dist_km, 'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    alpha={alpha:<5} [{penalty_label:<12}] -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")
    return rows


# =============================================================================
# CSV OUTPUT
# =============================================================================

def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print(f"  (no rows -- skipping {path})")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    if 'method' in fieldnames:
        fieldnames = ['method'] + [f for f in fieldnames if f != 'method']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description='OFAT parameter sensitivity sweep for one concrete instance'
    )
    p.add_argument('instance', help='Instance .txt file')
    p.add_argument('--model-c', default=None)
    p.add_argument('--model-d', default=None)
    p.add_argument('--tag', default=None,
                   help='Suffix appended to output CSV filenames (e.g. "low"/"high"), '
                        'so sweeps over multiple instances (different occupancy regimes) '
                        "don't overwrite each other. Omit for the old unsuffixed filenames.")
    p.add_argument('--sweep', default='p_bias,departure_h,n_iter',
                   help='Comma-separated subset of {p_bias,departure_h,n_iter,alpha_c,alpha_d} '
                        '(default: all except alpha_c/alpha_d -- add them explicitly)')
    p.add_argument('--p_biases',     default=','.join(map(str, DEFAULT_P_BIASES)))
    p.add_argument('--departure_hs', default=','.join(map(str, DEFAULT_DEPARTURE_HS)))
    p.add_argument('--alphas',       default=','.join(map(str, DEFAULT_ALPHAS)),
                   help='alpha_c/alpha_d sweep values (0=pure time, 1=pure distance)')
    p.add_argument('--alpha',  type=float, default=ALPHA_DEFAULT,
                   help='Baseline alpha held fixed for HSR/SRA rows in p_bias/departure_h/n_iter sweeps')
    p.add_argument('--beta',    type=float, default=BETA_DEFAULT,
                   help='Baseline beta held fixed for HSR/SRA rows in every sweep')
    p.add_argument('--p_bias',  type=float, default=P_BIAS,
                   help='Baseline p_bias held fixed in non-p_bias sweeps')
    p.add_argument('--n_iter',  type=int, default=25,
                   help='GRASP iterations per cell for p_bias/departure_h/alpha_c/alpha_d sweeps '
                        '(default 25 -- lighter than the usual n_iter=100)')
    p.add_argument('--n_iter_max', type=int, default=100,
                   help='Max iterations for the n_iter convergence-curve sweep')
    p.add_argument('--checkpoint_every', type=int, default=5)
    p.add_argument('--out_dir', default=None)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(args.instance)
    vehicle_cap    = params.capacity
    depot          = nodes[0]
    dep_h_baseline = params.departure
    out_dir        = args.out_dir or RESULTS_DIR
    sweeps         = {s.strip() for s in args.sweep.split(',') if s.strip()}
    suffix         = f'_{args.tag}' if args.tag else ''

    model_c_path = _resolve_model_path(args.model_c, MODEL_C_CANDIDATES, 'C')
    model_d_path = _resolve_model_path(args.model_d, MODEL_D_CANDIDATES, 'D')
    bundle_c = _try_load(model_c_path, 'HSR')
    bundle_d = _try_load(model_d_path, 'SRA')

    print(f"\n{'='*70}")
    print(f"  SENSITIVITY — {params.name}" + (f"  [tag={args.tag}]" if args.tag else ""))
    print(f"  baseline: p_bias={args.p_bias}  departure_h={dep_h_baseline}  "
          f"alpha={args.alpha}  beta={args.beta}")
    print(f"  sweep n_iter (per cell)={args.n_iter}  n_iter_max (curve)={args.n_iter_max}")
    print(f"  HSR model: {'loaded' if bundle_c else 'none'}   "
          f"SRA model: {'loaded' if bundle_d else 'none'}")
    print(f"{'='*70}\n")

    if 'p_bias' in sweeps:
        p_biases = [float(x) for x in args.p_biases.split(',')]
        print(f"[sweep] p_bias = {p_biases}")
        rows = sweep_p_bias(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                            dep_h_baseline, p_biases, args.n_iter, bundle_c, bundle_d,
                            args.alpha, args.beta, args.verbose)
        _write_csv(rows, os.path.join(out_dir, f'sensitivity_p_bias{suffix}.csv'))

    if 'departure_h' in sweeps:
        departure_hs = [float(x) for x in args.departure_hs.split(',')]
        print(f"[sweep] departure_h = {departure_hs}")
        rows = sweep_departure_h(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                                 args.p_bias, departure_hs, args.n_iter, bundle_c, bundle_d,
                                 args.alpha, args.beta, args.verbose)
        _write_csv(rows, os.path.join(out_dir, f'sensitivity_departure_h{suffix}.csv'))

    if 'n_iter' in sweeps:
        print(f"[sweep] n_iter convergence curve, max={args.n_iter_max}, "
              f"checkpoint every {args.checkpoint_every}")
        rows = sweep_n_iter(nodes, dist_matrix, vehicle_cap, depot, locker_cap,
                            dep_h_baseline, args.p_bias,
                            args.n_iter_max, args.checkpoint_every, bundle_c, bundle_d,
                            args.alpha, args.beta, args.verbose)
        _write_csv(rows, os.path.join(out_dir, f'sensitivity_n_iter{suffix}.csv'))

    if 'alpha_c' in sweeps:
        # alpha_c always runs, even without a trained HSR model (beta=0
        # cells then equal the with-penalty cells, which is itself an
        # informative "no ML available yet" baseline).
        alphas = [float(x) for x in args.alphas.split(',')]
        print(f"[sweep] alpha_c = {alphas}  (HSR: distance/time blend, "
              f"{'with' if bundle_c else 'without'} trained ML penalty)")
        rows = sweep_alpha_c(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                             dep_h_baseline, bundle_c, alphas, args.beta, args.p_bias,
                             args.n_iter, args.verbose)
        _write_csv(rows, os.path.join(out_dir, f'sensitivity_alpha_c{suffix}.csv'))

    if 'alpha_d' in sweeps:
        alphas = [float(x) for x in args.alphas.split(',')]
        print(f"[sweep] alpha_d = {alphas}  (SRA: distance/time blend, "
              f"{'with' if bundle_d else 'without'} trained ML penalty)")
        rows = sweep_alpha_d(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                             dep_h_baseline, bundle_d, alphas, args.beta, args.p_bias,
                             args.n_iter, args.verbose)
        _write_csv(rows, os.path.join(out_dir, f'sensitivity_alpha_d{suffix}.csv'))

    print(f"\n{'='*70}")
    print("  Done.")
    print('='*70)


if __name__ == '__main__':
    main()
