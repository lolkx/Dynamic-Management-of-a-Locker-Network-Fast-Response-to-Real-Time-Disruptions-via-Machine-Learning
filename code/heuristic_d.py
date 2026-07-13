"""
heuristic_d.py
---------------
"Option D": a unified time-aware learnheuristic like heuristic_c.py, but
built around a genuinely different mechanism than Option C's static-overflow
penalty. §13's diagnosis (see CLAUDE.md) showed that penalising a locker for
being "at risk" only made the heuristic's own fallbacks cheaper, not fewer --
because the ML signal there predicts overflow purely from static capacity
features, with arrival_hour a minor contributor.

The insight behind Option D: a locker's existing occupants collect their
own packages throughout the day, independent of the delivery route -- so a
locker that looks "full" in the static instance data may well have emptied
out by the time a vehicle *could* reach it later. Option D trains directly
on this:

    P(delivered_ok | initial_occupancy_ratio, arrival_hour)

via simulator.simulate_solution_stochastic's Monte-Carlo replicas (see
generate_labels_D.py) -- not a static overflow check. The savings formula
has the identical structure to Option C, just fed a different penalty
source (a RISK table, i.e. 1 - P(delivered_ok|...,hour), see
_hourly_release_risk below):

    s(i,j) = alpha * s_dist(i,j) + (1-alpha) * s_time(i,j)
             - beta * (1 - P(delivered_ok | j, arrival_hour)) * d_fallback(j)

heuristic_c.py's merge core (_fused_saving_c / br_CWS_c) is REUSED UNCHANGED
here -- both only ever treat hourly_probs as an opaque per-node 24-length
penalty array, with no assumption baked in about what it represents (static
overflow probability for Option C, release-aware failure risk for Option D).
Only the precompute step (_hourly_release_risk vs
heuristic_c._hourly_saturation_probs) and the feature schema
(FEATURE_COLS_D vs FEATURE_COLS_C) differ.

bundle=None (no trained Option-D model yet) makes the risk table identically
0 everywhere -- same ablation convention as Option C.
"""

from __future__ import annotations
import time as _time
import numpy as np

from model import Solution
from heuristic import INF, P_BIAS, N_ITER, build_graph, print_solution
from heuristic_learn import _fallback_distances, load_model, BETA_DEFAULT
from heuristic_c import br_CWS_c, ALPHA_DEFAULT
from instance_reader import MAX_COMPARTMENTS


# =============================================================================
# 1. HOURLY RELEASE RISK  (precomputed once per build, not per merge)
# =============================================================================

def _hourly_release_risk(nodes, bundle: dict | None,
                         locker_cap: dict | None = None) -> dict[int, np.ndarray]:
    """
    risk[h] = 1 - P(delivered_ok | node, hour=h) for hour=0..23, for every
    active node, via ONE batched predict_proba call. bundle=None -> all-zero
    risk tables (no penalty -- pure distance+time ablation).

    Feature schema is instance_reader.FEATURE_COLS_D: the same 13
    FEATURE_COLS_B base columns heuristic_c._hourly_saturation_probs
    reconstructs from Node aggregates, plus the locker's own
    initial_occupancy_ratio (static, computed directly from locker_cap via
    MAX_COMPARTMENTS -- no simulation needed, identical at train and
    inference time) and a varying 14th 'arrival_hour' column.
    """
    active = [n for n in nodes[1:] if n.is_active]
    if not active:
        return {}
    if bundle is None:
        return {n.Id: np.zeros(24) for n in active}

    total_del_w = sum(n.delivery_weight for n in active) or 1.0
    mean_del_w  = total_del_w / len(active)
    total_max   = float(sum(MAX_COMPARTMENTS.values()))

    rows: list[list[float]] = []
    ids: list[int] = []
    for node in active:
        total_ord  = node.n_deliveries + node.n_pickups
        total_comp = (float(sum(locker_cap.get(node.Id, {}).values()))
                     if locker_cap is not None else 0.0)
        occ_ratio  = 1.0 - total_comp / total_max if total_max > 0 else 0.0
        base = [
            float(node.n_deliveries),
            float(node.n_pickups),
            float(total_ord),
            node.delivery_weight,
            node.pickup_weight,
            node.n_deliveries / total_ord if total_ord > 0 else 0.0,
            node.delivery_weight / node.n_deliveries if node.n_deliveries > 0 else 0.0,
            node.delivery_weight / mean_del_w,
            node.delivery_weight / total_del_w,
            total_comp,
            node.n_deliveries / total_comp if total_comp > 0 else 0.0,
            (node.n_large_deliveries / node.n_deliveries
             if node.n_deliveries > 0 else 0.0),
            node.size_weighted_demand,
            occ_ratio,
        ]
        for h in range(24):
            rows.append(base + [float(h)])
        ids.append(node.Id)

    X   = np.array(rows, dtype=float)
    clf = bundle['model']
    probs = (clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba')
             else clf.predict(X).astype(float))
    probs = probs.reshape(len(ids), 24)
    risk  = 1.0 - probs
    return {node_id: risk[i] for i, node_id in enumerate(ids)}


# =============================================================================
# 2. CANDIDATE GRAPH  (distance-only K-NN pruning, reused from heuristic.py)
# =============================================================================

def build_graph_d(nodes, dist_matrix: np.ndarray, bundle: dict | None = None,
                  locker_cap: dict | None = None):
    """Same shape as heuristic_c.build_graph_c, using _hourly_release_risk
    instead of _hourly_saturation_probs as the penalty source."""
    active_nodes, savings_list = build_graph(nodes, dist_matrix)
    hourly_risk = _hourly_release_risk(nodes, bundle, locker_cap)
    d_fb        = _fallback_distances(active_nodes, dist_matrix)
    return active_nodes, savings_list, hourly_risk, d_fb


# =============================================================================
# 3. GRASP OPTION D
# =============================================================================

def run_grasp_d(nodes, dist_matrix: np.ndarray, vehicle_cap: float,
                bundle: dict | None = None, locker_cap: dict | None = None,
                n_iter: int = N_ITER, p_bias: float = P_BIAS,
                alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                departure_h: float = 8.0, verbose: bool = True
                ) -> tuple[Solution, float]:
    """
    GRASP loop: build the candidate graph + hourly release-risk table once
    (build_graph_d), then repeat heuristic_c.br_CWS_c n_iter times -- the
    merge core is reused UNCHANGED (see module docstring).

    Parameters
    ----------
    bundle     : optional Option-D model bundle (heuristic_learn.load_model()).
                 None keeps the risk table == 0 everywhere (pure distance+time
                 ablation).
    locker_cap : required for initial_occupancy_ratio and the ML penalty's
                 other capacity features; see _hourly_release_risk.
    alpha      : distance/time blend weight [0,1]; 1=pure distance, 0=pure time.
    beta       : ML release-risk penalty weight [0,1] (only used if bundle set).
    """
    t0 = _time.perf_counter()

    active_nodes, savings_list, hourly_risk, d_fb = build_graph_d(
        nodes, dist_matrix, bundle, locker_cap)
    depot     = nodes[0]
    best_sol  = None
    best_cost = INF

    for it in range(n_iter):
        sol = br_CWS_c(active_nodes, savings_list, vehicle_cap, depot,
                       hourly_risk, d_fb, alpha, beta, departure_h, p_bias)
        if sol.cost < best_cost:
            best_sol  = sol
            best_cost = sol.cost
            if verbose:
                print(f"    iter {it+1:>4}: new best -> "
                      f"{best_cost/1000:.3f} km | {len(sol.routes)} routes")

    return best_sol, _time.perf_counter() - t0


# =============================================================================
# QUICK SELF-TEST
# =============================================================================

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from instance_reader import read_full_instance

    if len(sys.argv) < 2:
        print("Usage: python heuristic_d.py <instance.txt> [model.pkl] [alpha] [beta] [departure_h]")
        print("       (omit model.pkl for the pure distance+time mode, no ML)")
        sys.exit(1)

    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    alpha_val  = float(sys.argv[3]) if len(sys.argv) > 3 else ALPHA_DEFAULT
    beta_val   = float(sys.argv[4]) if len(sys.argv) > 4 else BETA_DEFAULT
    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(sys.argv[1])
    dep_h = float(sys.argv[5]) if len(sys.argv) > 5 else params.departure

    bundle = load_model(model_path) if model_path else None

    print(f"Instance: {params.name}  "
          f"({params.n_orders} orders, {params.n_nodes} nodes, cap={params.capacity})")

    label = f"D dep={dep_h}h alpha={alpha_val}" + (f" +ML beta={beta_val}" if bundle else "")
    best, elapsed = run_grasp_d(nodes, dist_matrix, params.capacity,
                               bundle=bundle, locker_cap=locker_cap,
                               n_iter=100, alpha=alpha_val, beta=beta_val,
                               departure_h=dep_h, verbose=True)
    print_solution(best, f"{params.name} [{label}]")
    print(f"\nSolved in {elapsed:.2f} s")
