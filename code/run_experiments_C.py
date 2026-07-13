"""
run_experiments_C.py  — OPTION C
-----------------------------------
Compares THREE methods per instance, not two -- isolating each component's
contribution instead of a single distance-only standard-vs-Option-C
comparison (which is misleading: Option C deliberately trades some raw
route distance for less overflow, so comparing raw distance alone always
makes the distance-only baseline look "better"):

  1. std     -- heuristic.run_grasp: pure distance CWS baseline.
  2. c_time  -- heuristic_c.run_grasp_c(bundle=None, beta=0): the SAME
               distance/time-blend mechanism as Option C, with the ML
               penalty OFF. Isolates "does using real time help at all",
               independent of ML.
  3. c_full  -- heuristic_c.run_grasp_c(bundle=<trained Option C model>):
               distance+time+hourly-ML penalty, the complete formula.
               Isolates "does adding the ML penalty help further, on top of
               already using time" (c_full vs c_time), as well as the
               overall effect vs the plain baseline (c_full vs std).

Metrics reported per method (not just distance): route distance, MAKESPAN
(Tmax -- the completion time of the LAST delivery in the whole solution;
pickups don't count, matching the multi-criteria definition from the
reference paper), and fallback/effective distance (simulator.py). This is
the "distancia total, tiempo total y rutas/km extra" triplet requested --
raw distance alone doesn't show the trade-off Option C is designed to make.

Run in two phases -- low-occupancy (data/instances/) first, then
high-saturation (data/instances_B/) -- so the expected pattern is directly
visible: on low-occupancy instances, c_time/c_full should barely differ from
std (little/no overflow to avoid, P_j near 0); on high-saturation instances,
c_full should show REAL fallback-km avoided vs both std and c_time.

Train/test split: evaluates on the TEST 20% from
generate_labels_C.split_all_instances() -- the SAME shuffle-split (matching
seed) that generate_labels_C.py used to pick its TRAIN 80% for label
generation, so evaluation instances were never used to train the model.

Requires
--------
    data/models_C/saturation_C_best.pkl  — trained with train_model_C.py
    (itself needs data/datasets_C/labels_C.csv from generate_labels_C.py)

Usage
-----
    python generate_labels_C.py
    python train_model_C.py --compare
    python run_experiments_C.py --alpha 0.5 --beta 0.3
    python run_experiments_C.py --max-low 10 --max-high 10   # quick test
"""

from __future__ import annotations
import os, sys, csv, time as _time, argparse, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import read_full_instance
from heuristic        import run_grasp, N_ITER, P_BIAS
from heuristic_learn  import _compute_arrival_times
from heuristic_c      import run_grasp_c, load_model, ALPHA_DEFAULT, BETA_DEFAULT
from simulator         import simulate_solution
from generate_labels_C import split_all_instances, DATA_DIR_LOW, DATA_DIR_HIGH, TEST_SPLIT, RANDOM_SEED

# =============================================================================
# CONFIGURATION
# =============================================================================
_HERE       = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(_HERE, '..', 'data', 'models_C')
RESULTS_DIR = os.path.join(_HERE, '..', 'data')
ALPHA       = ALPHA_DEFAULT
BETA        = BETA_DEFAULT
N_ITER_EXP  = N_ITER

METHODS = ['std', 'c_time', 'c_full']
METHOD_LABELS = {'std': 'Standard', 'c_time': 'C (dist+time)', 'c_full': 'C (dist+time+ML)'}


# =============================================================================
# MAKESPAN (Tmax) -- second multi-criteria objective, deliveries only
# =============================================================================

def _makespan_h(sol, departure_h: float, dist_matrix) -> float:
    """
    Tmax: completion time (hours from midnight) of the LAST delivery in the
    whole solution. Only nodes with n_deliveries > 0 count -- pickup timing
    doesn't affect delivery-side customer satisfaction (per the reference
    paper's multi-criteria definition).
    """
    arrivals = _compute_arrival_times(sol, departure_h, dist_matrix)
    max_t = departure_h * 3600.0
    for route in sol.routes:
        for node in route.nodes:
            if node.n_deliveries > 0 and node.Id in arrivals:
                completion = arrivals[node.Id] + node.service_time
                if completion > max_t:
                    max_t = completion
    return max_t / 3600.0


# =============================================================================
# SINGLE INSTANCE
# =============================================================================

def run_instance_C(filepath: str, bundle, phase: str, n_iter: int = N_ITER_EXP,
                   alpha: float = ALPHA, beta: float = BETA,
                   verbose: bool = False) -> dict:
    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(filepath)
    n_active = sum(1 for n in nodes[1:] if n.is_active)
    dep_h = params.departure

    print(f"  [{phase:<4}] {params.name:<28}  orders={params.n_orders:>5}  "
          f"active={n_active:>4}  lockers_capped={len(locker_cap)}")

    row: dict = {
        'phase':      phase,
        'instance':   params.name,
        'n_orders':   params.n_orders,
        'n_nodes':    params.n_nodes,
        'n_active':   n_active,
        'capacity':   params.capacity,
    }

    solutions: dict[str, tuple] = {}

    sol_std, t_std = run_grasp(
        nodes, dist_matrix, params.capacity,
        n_iter=n_iter, p_bias=P_BIAS, verbose=verbose)
    solutions['std'] = (sol_std, t_std)

    sol_time, t_time = run_grasp_c(
        nodes, dist_matrix, params.capacity,
        bundle=None, locker_cap=locker_cap,
        n_iter=n_iter, p_bias=P_BIAS, alpha=alpha, beta=0.0,
        departure_h=dep_h, verbose=verbose)
    solutions['c_time'] = (sol_time, t_time)

    sol_full, t_full = run_grasp_c(
        nodes, dist_matrix, params.capacity,
        bundle=bundle, locker_cap=locker_cap,
        n_iter=n_iter, p_bias=P_BIAS, alpha=alpha, beta=beta,
        departure_h=dep_h, verbose=verbose)
    solutions['c_full'] = (sol_full, t_full)

    for name, (sol, t) in solutions.items():
        if sol is None:
            continue
        dist_km = round(sol.cost / 1000, 4)
        row[f'{name}_dist_km']    = dist_km
        row[f'{name}_n_routes']   = len(sol.routes)
        row[f'{name}_elapsed_s']  = round(t, 3)
        row[f'{name}_makespan_h'] = round(_makespan_h(sol, dep_h, dist_matrix), 3)
        if locker_cap:
            fb_km, n_fb = simulate_solution(sol, orders, locker_cap, dist_matrix)
            row[f'{name}_fallback_km']  = round(fb_km, 4)
            row[f'{name}_n_fallbacks']  = n_fb
            row[f'{name}_effective_km'] = round(dist_km + fb_km, 4)

    # --- Isolated comparisons: does TIME help? does ML help ON TOP OF time? ---
    for a, b, label in (('std', 'c_time', 'time_vs_std'),
                        ('c_time', 'c_full', 'ml_vs_time'),
                        ('std', 'c_full', 'full_vs_std')):
        if f'{a}_effective_km' in row and f'{b}_effective_km' in row:
            base = row[f'{a}_effective_km']
            row[f'{label}_effective_improvement_pct'] = (
                round(100 * (base - row[f'{b}_effective_km']) / base, 3) if base > 0 else 0.0)
        if f'{a}_fallback_km' in row and f'{b}_fallback_km' in row:
            row[f'{label}_fallback_km_avoided'] = round(
                row[f'{a}_fallback_km'] - row[f'{b}_fallback_km'], 4)
        if f'{a}_makespan_h' in row and f'{b}_makespan_h' in row:
            row[f'{label}_makespan_delta_h'] = round(
                row[f'{b}_makespan_h'] - row[f'{a}_makespan_h'], 3)

    return row


# =============================================================================
# MAIN
# =============================================================================

def run_experiments_C(data_dir_low=DATA_DIR_LOW, data_dir_high=DATA_DIR_HIGH,
                      model_dir=MODEL_DIR, results_dir=RESULTS_DIR,
                      model_type=None, alpha=ALPHA, beta=BETA, n_iter=N_ITER_EXP,
                      max_low=None, max_high=None, test_split=TEST_SPLIT,
                      seed=RANDOM_SEED, verbose=False) -> list[dict]:

    random.seed(seed); np.random.seed(seed)

    model_path = os.path.join(model_dir,
                              f'saturation_C_{model_type}.pkl' if model_type
                              else 'saturation_C_best.pkl')
    bundle = None
    if os.path.exists(model_path):
        bundle = load_model(model_path)
    else:
        print(f"WARNING: No model at {model_path}. c_full will equal c_time "
              "(P_j=0) -- train_model_C.py first for the full effect.")

    print(f"\n{'='*70}")
    print(f"  OPTION C — std vs. dist+time vs. dist+time+ML  "
          f"(alpha={alpha}, beta={beta}, GRASP iter={n_iter})")
    print(f"{'='*70}\n")

    # Held-out 20% test split -- same seed as generate_labels_C.py's 80%
    # train split, so these instances were never used to generate training
    # labels (see module docstring).
    _train_items, test_items = split_all_instances(data_dir_low, data_dir_high, test_split, seed)
    low_paths  = [p for p, pool in test_items if pool == 'low']
    high_paths = [p for p, pool in test_items if pool == 'high']
    if max_low:
        low_paths = low_paths[:max_low]
    if max_high:
        high_paths = high_paths[:max_high]

    all_rows: list[dict] = []
    t0 = _time.perf_counter()
    for phase, paths in (('low', low_paths), ('high', high_paths)):
        for idx, fpath in enumerate(paths, 1):
            print(f"[{phase} {idx:>3}/{len(paths)}] ", end='', flush=True)
            try:
                all_rows.append(run_instance_C(fpath, bundle, phase, n_iter, alpha, beta, verbose))
            except Exception as exc:
                print(f"  ERROR: {exc}")
    elapsed = _time.perf_counter() - t0

    # --- Summary, per phase ---
    if all_rows:
        print(f"\n{'='*88}")
        print(f"  OPTION C SUMMARY  ({len(all_rows)} instances, {elapsed:.1f}s total)")
        for phase, label in (('low', 'LOW OCCUPANCY'), ('high', 'HIGH SATURATION')):
            rows = [r for r in all_rows if r['phase'] == phase]
            if not rows:
                continue

            def _col(k): return [r[k] for r in rows if k in r]

            print(f"\n  --- {label} ({len(rows)} instances) ---")
            header = f"    {'Metric':<20}" + ''.join(f"{METHOD_LABELS[m]:>18}" for m in METHODS)
            print(header)
            for metric, fmt in (('dist_km', '.3f'), ('makespan_h', '.3f'),
                               ('fallback_km', '.3f'), ('n_fallbacks', '.1f'),
                               ('effective_km', '.3f')):
                vals = [_col(f'{m}_{metric}') for m in METHODS]
                if not any(vals):
                    continue
                cells = ''.join(f"{np.mean(v):>18{fmt}}" if v else f"{'--':>18}" for v in vals)
                print(f"    {metric:<20}{cells}")

            print(f"    {'-'*(len(header)-4)}")
            for label2, key in (('Time alone   (c_time vs std)   ', 'time_vs_std'),
                               ('ML on top    (c_full vs c_time)', 'ml_vs_time'),
                               ('Full (dist+time+ML vs std)     ', 'full_vs_std')):
                eff = _col(f'{key}_effective_improvement_pct')
                fb  = _col(f'{key}_fallback_km_avoided')
                ms  = _col(f'{key}_makespan_delta_h')
                if not eff and not fb:
                    continue
                n_better = sum(1 for i in eff if i > 0) if eff else 0
                eff_str = f"{np.mean(eff):+.3f}% ({n_better}/{len(eff)} better)" if eff else '--'
                fb_str  = f"{np.mean(fb):+.3f} km avoided" if fb else '--'
                ms_str  = f"{np.mean(ms):+.3f} h makespan delta" if ms else '--'
                print(f"    {label2}: effective {eff_str}  |  fallback {fb_str}  |  {ms_str}")
        print(f"\n{'='*88}")

    # --- Export CSV ---
    if all_rows:
        os.makedirs(results_dir, exist_ok=True)
        suffix = model_type or 'best'
        csv_path = os.path.join(results_dir, f'results_C_{suffix}_alpha{alpha}_beta{beta}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()),
                                    extrasaction='ignore')
            writer.writeheader(); writer.writerows(all_rows)
        print(f"\n  Results -> {csv_path}")

    return all_rows


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Option C: std vs. dist+time vs. dist+time+ML')
    p.add_argument('--model',    default=None, help='rf/gbm/lr (default: best)')
    p.add_argument('--alpha',    type=float, default=ALPHA)
    p.add_argument('--beta',     type=float, default=BETA)
    p.add_argument('--n_iter',   type=int,   default=N_ITER_EXP)
    p.add_argument('--max-low',  type=int,   default=None, help='Cap on the low-occupancy test phase')
    p.add_argument('--max-high', type=int,   default=None, help='Cap on the high-saturation test phase')
    p.add_argument('--data-low',  default=DATA_DIR_LOW)
    p.add_argument('--data-high', default=DATA_DIR_HIGH)
    p.add_argument('--test-split', type=float, default=TEST_SPLIT,
                   help='Must match generate_labels_C.py\'s split so test instances stay held-out')
    p.add_argument('--seed', type=int, default=RANDOM_SEED,
                   help='Must match generate_labels_C.py\'s seed so test instances stay held-out')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    run_experiments_C(
        data_dir_low  = args.data_low,
        data_dir_high = args.data_high,
        model_type    = args.model,
        alpha         = args.alpha,
        beta          = args.beta,
        n_iter        = args.n_iter,
        max_low       = args.max_low,
        max_high      = args.max_high,
        test_split    = args.test_split,
        seed          = args.seed,
        verbose       = args.verbose,
    )
