"""
run_experiments_D.py  — OPTION D
-----------------------------------
Mirrors run_experiments_C.py's 3-way isolated comparison, swapping in
Option D's release-aware penalty:

  1. std     -- heuristic.run_grasp: pure distance CWS baseline.
  2. d_time  -- heuristic_d.run_grasp_d(bundle=None, beta=0): the SAME
               distance/time-blend mechanism, ML penalty OFF. Isolates
               "does using real time help at all", independent of ML.
  3. d_full  -- heuristic_d.run_grasp_d(bundle=<trained Option D model>):
               distance+time+hourly release-risk penalty, the complete
               formula. Isolates "does the release-aware ML penalty help
               further, on top of already using time" (d_full vs d_time),
               and the overall effect vs the plain baseline (d_full vs std).

Where Option C's diagnosis (§13, design notes) found its ML component only
made fallbacks CHEAPER (shorter detours), not FEWER (because its label is a
static overflow check with arrival_hour a minor contributor), Option D's
whole premise is a genuinely different mechanism -- a locker that looks
full may have emptied out by the time the vehicle arrives, so postponing
the visit can avoid the fallback entirely. n_fallbacks is therefore
surfaced explicitly and prominently below, not just fallback_km/effective_km.

Metrics reported per method, TWO evaluations side by side (see design notes
for the full discussion of why one metric alone is misleading):

  1. DETERMINISTIC (fallback_km/n_fallbacks/effective_km, simulator.
     simulate_solution): locker capacity is fixed for the whole day, nothing
     is ever released by background self-collection. This is the SAME rigid
     yardstick used for std and Option C -- useful for a like-for-like
     comparison, and to show Option D doesn't hurt even under the pessimistic
     assumption that nobody ever picks up their package. But it CANNOT credit
     any real timing benefit (arrival hour has zero effect on capacity here),
     so any delta under this metric alone reflects route-structure side
     effects, not the mechanism Option D actually targets.
  2. STOCHASTIC (stoch_fail_rate/stoch_n_failed_lockers, simulator.
     simulate_solution_stochastic, averaged over STOCH_N_REPS replicas with
     the SAME release distribution assumed during training): this is the
     metric that actually tests the hypothesis -- "if lockers really do
     empty out over the day, does routing that accounts for this reduce real
     delivery failures". All three methods are evaluated against the
     IDENTICAL sequence of random release draws per instance (common random
     numbers, see _stochastic_kpis) to isolate the route-structure effect
     from evaluation noise.

Plus route distance and MAKESPAN (Tmax, reusing run_experiments_C._makespan_h).

Run in two phases -- low-occupancy (data/instances/) first, then
high-saturation (data/instances_real_highocc/) -- same convention as run_experiments_C.py.

Train/test split: evaluates on the TEST 20% from
generate_labels_C.split_all_instances() -- the SAME shuffle-split (matching
seed) that generate_labels_D.py used to pick its TRAIN 80% for label
generation (identical to Option C's split too, so both are directly
comparable on the same held-out instances).

Requires
--------
    data/models_D/saturation_D_best.pkl  — trained with train_model_D.py
    (itself needs data/datasets_D/labels_D.csv from generate_labels_D.py)

Usage
-----
    python generate_labels_D.py
    python train_model_D.py --compare
    python run_experiments_D.py --alpha 0.5 --beta 0.3
    python run_experiments_D.py --max-low 10 --max-high 10   # quick test
"""

from __future__ import annotations
import os, sys, csv, time as _time, argparse, random, zlib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import read_full_instance
from heuristic        import run_grasp, N_ITER, P_BIAS
from heuristic_d      import run_grasp_d, ALPHA_DEFAULT
from heuristic_learn  import load_model, BETA_DEFAULT
from simulator         import simulate_solution, simulate_solution_stochastic
from generate_labels_C import split_all_instances, DATA_DIR_LOW, DATA_DIR_HIGH, TEST_SPLIT, RANDOM_SEED
from generate_labels_D import (MEAN_HOUR_DEFAULT, STD_HOUR_DEFAULT,
                               START_H_DEFAULT, END_H_DEFAULT)
from run_experiments_C import _makespan_h

# =============================================================================
# CONFIGURATION
# =============================================================================
_HERE       = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(_HERE, '..', 'data', 'models_D')
RESULTS_DIR = os.path.join(_HERE, '..', 'data')
ALPHA       = ALPHA_DEFAULT
BETA        = BETA_DEFAULT
N_ITER_EXP  = N_ITER

# Stochastic (release-model) evaluation: SAME distribution assumed during
# training (generate_labels_D.py's defaults) -- this is what actually tests
# Option D's hypothesis ("does the locker empty out in time"). The plain
# deterministic simulate_solution (below) never releases anything, so it
# CANNOT credit any timing benefit -- any n_fallbacks delta there comes from
# route-structure side effects, not from the mechanism Option D targets. See
# design notes/§15 for the full discussion of why both are reported.
STOCH_N_REPS    = 30
STOCH_MEAN_HOUR = MEAN_HOUR_DEFAULT
STOCH_STD_HOUR  = STD_HOUR_DEFAULT
STOCH_START_H   = START_H_DEFAULT
STOCH_END_H     = END_H_DEFAULT

METHODS = ['std', 'd_time', 'd_full']
METHOD_LABELS = {'std': 'Standard', 'd_time': 'D (dist+time)', 'd_full': 'D (dist+time+ML)'}


# =============================================================================
# STOCHASTIC (RELEASE-MODEL) EVALUATION -- tests the actual hypothesis
# =============================================================================

def _stochastic_kpis(sol, orders, locker_cap, dist_matrix, departure_h: float,
                     base_seed: int, n_reps: int = STOCH_N_REPS,
                     mean_hour: float = STOCH_MEAN_HOUR, std_hour: float = STOCH_STD_HOUR,
                     start_h: int = STOCH_START_H, end_h: int = STOCH_END_H) -> tuple[float, float]:
    """
    Average, over n_reps independent replicas of simulator.simulate_solution_
    stochastic, the fraction of visited lockers where >=1 delivery failed
    (stoch_fail_rate) and the average COUNT of such lockers (stoch_n_failed_
    lockers). NOT the same unit as the deterministic n_fallbacks (which
    counts individual failed ORDERS) -- this counts failed LOCKER VISITS,
    matching simulate_solution_stochastic's own granularity.

    Uses base_seed + rep as the RNG seed for every rep -- callers should pass
    the SAME base_seed for std/d_time/d_full on a given instance, so all
    three methods are evaluated against the IDENTICAL sequence of random
    release draws (common random numbers -- isolates the effect of route
    structure alone from the noise of which draws happened to fire).
    """
    if not locker_cap:
        return 0.0, 0.0
    fail_rates, fail_counts = [], []
    for rep in range(n_reps):
        rng = random.Random(base_seed + rep)
        visit_log = simulate_solution_stochastic(
            sol, orders, locker_cap, dist_matrix, departure_h=departure_h,
            mean_hour=mean_hour, std_hour=std_hour, start_h=start_h, end_h=end_h, rng=rng)
        if not visit_log:
            continue
        n_fail = sum(1 for v in visit_log if not v['delivered_ok'])
        fail_counts.append(n_fail)
        fail_rates.append(n_fail / len(visit_log))
    if not fail_rates:
        return 0.0, 0.0
    return float(np.mean(fail_rates)), float(np.mean(fail_counts))


# =============================================================================
# SINGLE INSTANCE
# =============================================================================

def run_instance_D(filepath: str, bundle, phase: str, n_iter: int = N_ITER_EXP,
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

    # Common random numbers: SAME per-instance seed re-applied (as a FRESH
    # random.Random each time, not one shared/advancing object) before every
    # method's GRASP loop, so std/d_time/d_full all draw the IDENTICAL
    # sequence of biased-random u's for this instance. Without this, all
    # three consumed from one shared global `random` stream advancing
    # across calls, so part of the observed difference between methods
    # could come from which candidates happened to be sampled, not just the
    # algorithmic difference (see the randomness/seeding note in the module docstring).
    stoch_base_seed = zlib.crc32(params.name.encode()) & 0xffffffff

    sol_std, t_std = run_grasp(
        nodes, dist_matrix, params.capacity,
        n_iter=n_iter, p_bias=P_BIAS, verbose=verbose,
        rng=random.Random(stoch_base_seed))
    solutions['std'] = (sol_std, t_std)

    sol_time, t_time = run_grasp_d(
        nodes, dist_matrix, params.capacity,
        bundle=None, locker_cap=locker_cap,
        n_iter=n_iter, p_bias=P_BIAS, alpha=alpha, beta=0.0,
        departure_h=dep_h, verbose=verbose,
        rng=random.Random(stoch_base_seed))
    solutions['d_time'] = (sol_time, t_time)

    sol_full, t_full = run_grasp_d(
        nodes, dist_matrix, params.capacity,
        bundle=bundle, locker_cap=locker_cap,
        n_iter=n_iter, p_bias=P_BIAS, alpha=alpha, beta=beta,
        departure_h=dep_h, verbose=verbose,
        rng=random.Random(stoch_base_seed))
    solutions['d_full'] = (sol_full, t_full)

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

            stoch_rate, stoch_n = _stochastic_kpis(
                sol, orders, locker_cap, dist_matrix, dep_h, stoch_base_seed)
            row[f'{name}_stoch_fail_rate']      = round(stoch_rate, 4)
            row[f'{name}_stoch_n_failed_lockers'] = round(stoch_n, 3)

    # --- Isolated comparisons: does TIME help? does ML help ON TOP OF time? ---
    for a, b, label in (('std', 'd_time', 'time_vs_std'),
                        ('d_time', 'd_full', 'ml_vs_time'),
                        ('std', 'd_full', 'full_vs_std')):
        if f'{a}_effective_km' in row and f'{b}_effective_km' in row:
            base = row[f'{a}_effective_km']
            row[f'{label}_effective_improvement_pct'] = (
                round(100 * (base - row[f'{b}_effective_km']) / base, 3) if base > 0 else 0.0)
        if f'{a}_fallback_km' in row and f'{b}_fallback_km' in row:
            row[f'{label}_fallback_km_avoided'] = round(
                row[f'{a}_fallback_km'] - row[f'{b}_fallback_km'], 4)
        if f'{a}_n_fallbacks' in row and f'{b}_n_fallbacks' in row:
            row[f'{label}_n_fallbacks_avoided'] = row[f'{a}_n_fallbacks'] - row[f'{b}_n_fallbacks']
        if f'{a}_makespan_h' in row and f'{b}_makespan_h' in row:
            row[f'{label}_makespan_delta_h'] = round(
                row[f'{b}_makespan_h'] - row[f'{a}_makespan_h'], 3)
        if f'{a}_stoch_n_failed_lockers' in row and f'{b}_stoch_n_failed_lockers' in row:
            row[f'{label}_stoch_failed_lockers_avoided'] = round(
                row[f'{a}_stoch_n_failed_lockers'] - row[f'{b}_stoch_n_failed_lockers'], 3)

    return row


# =============================================================================
# MAIN
# =============================================================================

def run_experiments_D(data_dir_low=DATA_DIR_LOW, data_dir_high=DATA_DIR_HIGH,
                      model_dir=MODEL_DIR, results_dir=RESULTS_DIR,
                      model_type=None, alpha=ALPHA, beta=BETA, n_iter=N_ITER_EXP,
                      max_low=None, max_high=None, test_split=TEST_SPLIT,
                      seed=RANDOM_SEED, verbose=False) -> list[dict]:

    random.seed(seed); np.random.seed(seed)

    model_path = os.path.join(model_dir,
                              f'saturation_D_{model_type}.pkl' if model_type
                              else 'saturation_D_best.pkl')
    bundle = None
    if os.path.exists(model_path):
        bundle = load_model(model_path)
    else:
        print(f"WARNING: No model at {model_path}. d_full will equal d_time "
              "(risk=0) -- train_model_D.py first for the full effect.")

    print(f"\n{'='*70}")
    print(f"  OPTION D — std vs. dist+time vs. dist+time+release-risk-ML  "
          f"(alpha={alpha}, beta={beta}, GRASP iter={n_iter})")
    print(f"{'='*70}\n")

    # Held-out 20% test split -- SAME split as generate_labels_D.py's (and
    # generate_labels_C.py's) 80% train split, so these instances were never
    # used to generate training labels for either Option C or D.
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
                all_rows.append(run_instance_D(fpath, bundle, phase, n_iter, alpha, beta, verbose))
            except Exception as exc:
                print(f"  ERROR: {exc}")
    elapsed = _time.perf_counter() - t0

    # --- Summary, per phase ---
    if all_rows:
        print(f"\n{'='*88}")
        print(f"  OPTION D SUMMARY  ({len(all_rows)} instances, {elapsed:.1f}s total)")
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
                               ('effective_km', '.3f'),
                               ('stoch_fail_rate', '.4f'), ('stoch_n_failed_lockers', '.2f')):
                vals = [_col(f'{m}_{metric}') for m in METHODS]
                if not any(vals):
                    continue
                cells = ''.join(f"{np.mean(v):>18{fmt}}" if v else f"{'--':>18}" for v in vals)
                tag = '  [DETERMINISTA, sin liberacion]' if metric in ('fallback_km', 'n_fallbacks', 'effective_km') \
                    else ('  [ESTOCASTICO, valida la hipotesis]' if metric.startswith('stoch_') else '')
                print(f"    {metric:<20}{cells}{tag}")

            print(f"    {'-'*(len(header)-4)}")
            for label2, key in (('Time alone   (d_time vs std)   ', 'time_vs_std'),
                               ('ML on top    (d_full vs d_time)', 'ml_vs_time'),
                               ('Full (dist+time+ML vs std)     ', 'full_vs_std')):
                eff = _col(f'{key}_effective_improvement_pct')
                fb  = _col(f'{key}_fallback_km_avoided')
                nfb = _col(f'{key}_n_fallbacks_avoided')
                ms  = _col(f'{key}_makespan_delta_h')
                sfl = _col(f'{key}_stoch_failed_lockers_avoided')
                if not eff and not fb:
                    continue
                n_better = sum(1 for i in eff if i > 0) if eff else 0
                eff_str = f"{np.mean(eff):+.3f}% ({n_better}/{len(eff)} better)" if eff else '--'
                fb_str  = f"{np.mean(fb):+.3f} km avoided" if fb else '--'
                nfb_str = f"{np.mean(nfb):+.2f} fallbacks avoided" if nfb else '--'
                ms_str  = f"{np.mean(ms):+.3f} h makespan delta" if ms else '--'
                sfl_str = f"{np.mean(sfl):+.3f} STOCH failed lockers avoided" if sfl else '--'
                print(f"    {label2}: effective {eff_str}  |  fallback {fb_str}  |  {nfb_str}  |  {ms_str}  |  {sfl_str}")
        print(f"\n{'='*88}")

    # --- Export CSV ---
    if all_rows:
        os.makedirs(results_dir, exist_ok=True)
        suffix = model_type or 'best'
        csv_path = os.path.join(results_dir, f'results_D_{suffix}_alpha{alpha}_beta{beta}.csv')
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
    p = argparse.ArgumentParser(description='Option D: std vs. dist+time vs. dist+time+release-risk-ML')
    p.add_argument('--model',    default=None, help='rf/gbm/lr (default: best)')
    p.add_argument('--alpha',    type=float, default=ALPHA)
    p.add_argument('--beta',     type=float, default=BETA)
    p.add_argument('--n_iter',   type=int,   default=N_ITER_EXP)
    p.add_argument('--max-low',  type=int,   default=None, help='Cap on the low-occupancy test phase')
    p.add_argument('--max-high', type=int,   default=None, help='Cap on the high-saturation test phase')
    p.add_argument('--data-low',  default=DATA_DIR_LOW)
    p.add_argument('--data-high', default=DATA_DIR_HIGH)
    p.add_argument('--test-split', type=float, default=TEST_SPLIT,
                   help='Must match generate_labels_D.py\'s split so test instances stay held-out')
    p.add_argument('--seed', type=int, default=RANDOM_SEED,
                   help='Must match generate_labels_D.py\'s seed so test instances stay held-out')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    run_experiments_D(
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
