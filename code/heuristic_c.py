"""
heuristic_c.py
---------------
"Option C": a single unified time-aware learnheuristic, kept alongside
heuristic_learn.py (Model A/B) and heuristic_dynamic.py rather than
replacing them.

Savings formula, recalculated live at merge-decision time for every
candidate arc (i -> j):

    s(i,j) = alpha * s_dist(i,j) + (1-alpha) * s_time(i,j)
             - beta * P_j(j, arrival_hour) * d_fallback(j)

Where:
    s_dist(i,j)    -- classic Clarke-Wright distance saving (static, metres)
    s_time(i,j)    -- the SAME saving in time units, converted to a
                      metres-equivalent, computed LIVE using the route's
                      real accumulated arrival time (iRoute.tail_time) --
                      exactly the mechanism heuristic_dynamic.py already
                      uses, NOT the old, circular, pre-route lambda_t blend
                      (heuristic_learn.py's module docstring explains why
                      that one was removed). This is safe to blend with
                      s_dist because it uses genuinely live state.
    P_j(j, hour)   -- probability that locker j overflows GIVEN it is
                      visited at that hour of day, from a model trained on
                      REAL simulated (arrival_hour, overflowed) labels (see
                      generate_labels_C.py) -- not a time-independent
                      per-node probability like Model A/B.
    d_fallback(j)  -- distance to j's nearest alternative locker (reused
                      from heuristic_learn._fallback_distances, unchanged).

Blending s_dist and s_time via alpha (a convex combination, alpha+(1-alpha)
=1) is a standard multi-criteria scalarisation of two normalised, comparable
-unit signals -- it does NOT have the double-counting pathology found and
fixed in heuristic_dynamic.py's ADDITIVE fusion (that was about stacking two
correlated absolute quantities on top of each other; a weighted average of
two bounded scores doesn't compound the same way).

Why P_j needs a PRECOMPUTED per-hour lookup table, not live inference:
a fresh sklearn predict_proba() call per candidate arc (thousands per GRASP
iteration x up to 100 iterations) would be far too slow -- the same
performance concern that keeps heuristic_learn._saturation_probs to exactly
ONE batched call per graph build, not per merge. Fix: _hourly_saturation_
probs computes P(sat_j | hour) for ALL 24 hour buckets x all active nodes in
ONE batched predict_proba call at build time, then br_CWS_c does an O(1)
array lookup at merge-decision time using the estimated arrival hour
(iRoute.tail_time + travel time i->j, using the KNOWN speed at i -- the same
single-step estimate _congestion_delta_m/_merge_routes_dynamic already use
elsewhere, no fixed-point iteration needed since speed-at-i is already known
before j is reached).

Why the accept/reject gate (s_fused <= 0 -> reject) is kept: same
established rationale as heuristic_dynamic.py -- without it (or without an
expensive full re-sort of the candidate list after every merge), the
recalculated score wouldn't influence anything, and this heuristic would
collapse to the plain distance-based K-NN ranking it's built on top of.

bundle=None (no trained Option-C model yet) makes P_j identically 0
everywhere -- this lets alpha/beta ablation runs (distance-only, or
distance+time with no penalty) work for the sensitivity study in
run_sensitivity.py's sweep_alpha_c.
"""

from __future__ import annotations
import math
import random
import time as _time
import numpy as np

from model import Solution
from heuristic import (
    INF, P_BIAS, N_ITER,
    build_graph, _check_merging, _merge_routes,
    print_solution,
)
from heuristic_learn import _fallback_distances, load_model, BETA_DEFAULT
from heuristic_dynamic import _build_dummy_solution_dynamic, _merge_routes_dynamic
from speed_profile import speed_ms, AVG_SPEED_MS

ALPHA_DEFAULT = 0.5


# =============================================================================
# 1. HOURLY SATURATION PROBABILITIES  (precomputed once per build, not per merge)
# =============================================================================

def _hourly_saturation_probs(nodes, bundle: dict | None,
                             locker_cap: dict | None = None) -> dict[int, np.ndarray]:
    """
    P(sat_j | hour) for hour=0..23, for every active node, via ONE batched
    predict_proba call. bundle=None -> all-zero tables (no ML penalty).

    Features mirror heuristic_learn._saturation_probs' 'capacity' branch
    (reconstructed from Node aggregates, since routing time has no raw
    orders list) plus a varying 14th 'arrival_hour' column -- the schema is
    instance_reader.FEATURE_COLS_C.
    """
    active = [n for n in nodes[1:] if n.is_active]
    if not active:
        return {}
    if bundle is None:
        return {n.Id: np.zeros(24) for n in active}

    total_del_w = sum(n.delivery_weight for n in active) or 1.0
    mean_del_w  = total_del_w / len(active)

    rows: list[list[float]] = []
    ids: list[int] = []
    for node in active:
        total_ord  = node.n_deliveries + node.n_pickups
        total_comp = (float(sum(locker_cap.get(node.Id, {}).values()))
                     if locker_cap is not None else 0.0)
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
        ]
        for h in range(24):
            rows.append(base + [float(h)])
        ids.append(node.Id)

    X   = np.array(rows, dtype=float)
    clf = bundle['model']
    probs = (clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba')
             else clf.predict(X).astype(float))
    probs = probs.reshape(len(ids), 24)
    return {node_id: probs[i] for i, node_id in enumerate(ids)}


# =============================================================================
# 2. CANDIDATE GRAPH  (distance-only K-NN pruning, reused from heuristic.py)
# =============================================================================

def build_graph_c(nodes, dist_matrix: np.ndarray, bundle: dict | None = None,
                  locker_cap: dict | None = None):
    """
    Candidate arcs come from heuristic.build_graph, UNCHANGED -- selection is
    purely geometric (K-nearest by distance), unrelated to alpha/beta/time;
    see module docstring for why P_j is precomputed here rather than baked
    into a static ranking.
    """
    active_nodes, savings_list = build_graph(nodes, dist_matrix)
    hourly_probs = _hourly_saturation_probs(nodes, bundle, locker_cap)
    d_fb         = _fallback_distances(active_nodes, dist_matrix)
    return active_nodes, savings_list, hourly_probs, d_fb


# =============================================================================
# 3. FUSED SAVING  (recalculated live at merge-decision time)
# =============================================================================

def _fused_saving_c(inode, jnode, iRoute, edge,
                    hourly_probs: dict[int, np.ndarray], d_fb: dict[int, float],
                    alpha: float, beta: float, dep_speed_ms: float) -> float:
    """
    s(i,j) = alpha*s_dist(i,j) + (1-alpha)*s_time(i,j) - beta*P_j(j,hour)*d_fallback(j)

    s_dist is the static Clarke-Wright saving; s_time is the same saving in
    time units (converted to metres-equivalent), evaluated LIVE using
    iRoute's real accumulated tail_time -- see module docstring.
    """
    speed_i   = speed_ms(iRoute.tail_time)
    t_i_depot = inode.ndEdge.cost / speed_i
    t_depot_j = jnode.dnEdge.cost / dep_speed_ms
    t_i_j     = edge.cost / speed_i
    s_time_m  = (t_i_depot + t_depot_j - t_i_j) * AVG_SPEED_MS

    s_dist_m = inode.ndEdge.cost + jnode.dnEdge.cost - edge.cost

    # Estimated arrival hour at j: depart i (after service) at the speed
    # known at i's current tail_time -- single-step estimate, same pattern
    # as heuristic_dynamic._merge_routes_dynamic uses for the same purpose.
    t_est_j     = iRoute.tail_time + inode.service_time + edge.cost / speed_i
    hour_bucket = int(t_est_j / 3600.0) % 24
    p_j = hourly_probs.get(jnode.Id, np.zeros(24))[hour_bucket]

    return alpha * s_dist_m + (1.0 - alpha) * s_time_m - beta * p_j * d_fb.get(jnode.Id, 0.0)


# =============================================================================
# 4. BR-CWS OPTION C  (one GRASP iteration)
# =============================================================================

def br_CWS_c(active_nodes, savings_list, vehicle_cap: float, depot,
            hourly_probs: dict[int, np.ndarray], d_fb: dict[int, float],
            alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
            departure_h: float = 8.0, p: float = P_BIAS) -> Solution:
    """
    Candidates sampled in the same distance-ranked, biased-random order as
    heuristic.br_CWS, but accepted only if the fused, recalculated saving
    (distance + live time + hourly ML penalty) is still positive.
    """
    sol          = _build_dummy_solution_dynamic(active_nodes, depot, departure_h)
    local_sav    = list(savings_list)
    log_p        = math.log(p)
    dep_speed_ms = speed_ms(departure_h * 3600.0)

    while local_sav:
        u   = random.random()
        idx = min(int(math.floor(math.log(max(u, 1e-300)) / log_p)),
                  len(local_sav) - 1)

        edge = local_sav[idx]
        del local_sav[idx]

        inode  = edge.origin
        jnode  = edge.end
        iRoute = inode.inRoute
        jRoute = jnode.inRoute

        if iRoute is None or jRoute is None:
            continue
        if not _check_merging(inode, jnode, iRoute, jRoute, vehicle_cap):
            continue

        s_fused = _fused_saving_c(inode, jnode, iRoute, edge, hourly_probs, d_fb,
                                  alpha, beta, dep_speed_ms)
        if s_fused <= 0.0:
            continue

        _merge_routes_dynamic(inode, jnode, iRoute, jRoute, edge, sol)

    return sol


# =============================================================================
# 5. GRASP OPTION C
# =============================================================================

def run_grasp_c(nodes, dist_matrix: np.ndarray, vehicle_cap: float,
                bundle: dict | None = None, locker_cap: dict | None = None,
                n_iter: int = N_ITER, p_bias: float = P_BIAS,
                alpha: float = ALPHA_DEFAULT, beta: float = BETA_DEFAULT,
                departure_h: float = 8.0, verbose: bool = True
                ) -> tuple[Solution, float]:
    """
    GRASP loop: build the candidate graph + hourly P_j table once
    (build_graph_c), then repeat br_CWS_c n_iter times.

    Parameters
    ----------
    bundle     : optional Option-C model bundle (heuristic_learn.load_model()).
                 None keeps P_j == 0 everywhere (pure distance+time ablation).
    locker_cap : required for the ML penalty's capacity features; see
                 _hourly_saturation_probs.
    alpha      : distance/time blend weight [0,1]; 1=pure distance, 0=pure time.
    beta       : ML saturation penalty weight [0,1] (only used if bundle set).
    """
    t0 = _time.perf_counter()

    active_nodes, savings_list, hourly_probs, d_fb = build_graph_c(
        nodes, dist_matrix, bundle, locker_cap)
    depot     = nodes[0]
    best_sol  = None
    best_cost = INF

    for it in range(n_iter):
        sol = br_CWS_c(active_nodes, savings_list, vehicle_cap, depot,
                       hourly_probs, d_fb, alpha, beta, departure_h, p_bias)
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
        print("Usage: python heuristic_c.py <instance.txt> [model.pkl] [alpha] [beta] [departure_h]")
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

    label = f"C dep={dep_h}h alpha={alpha_val}" + (f" +ML beta={beta_val}" if bundle else "")
    best, elapsed = run_grasp_c(nodes, dist_matrix, params.capacity,
                               bundle=bundle, locker_cap=locker_cap,
                               n_iter=100, alpha=alpha_val, beta=beta_val,
                               departure_h=dep_h, verbose=True)
    print_solution(best, f"{params.name} [{label}]")
    print(f"\nSolved in {elapsed:.2f} s")
