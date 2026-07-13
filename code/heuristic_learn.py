"""
heuristic_learn.py
------------------
Learnheuristic: BR-CWS + GRASP with ML-guided savings penalisation.

Modified savings formula:
    s_learn(i->j) = s_std(i->j) - beta * P_eff(saturation_j)

Where:
    s_std(i->j)     -- normalised Clarke-Wright saving (distance-only, same
                       formula as heuristic.py)
    P(saturation_j) -- probability [0,1] that node j is saturated,
                       predicted by a trained sklearn / XGBoost model
    P_eff(sat_j)    -- P(saturation_j) amplified by how late in the day node
                       j is estimated to be reached (see _saturation_probs)
    beta            -- penalty weight (configurable, default 0.3)

Intuition: high-risk nodes (P close to 1) are penalised in the savings
ranking, causing the heuristic to serve them via shorter, more direct routes.
Vehicles arrive earlier -> lower probability that lockers are already full.

NOTE: an earlier version of this module also blended a time-dependent
savings term (s_time, estimated via a naive direct-from-depot arrival guess)
into s_raw. That blend was removed: because CWS savings are computed once,
before any route exists, the arrival-time estimate it relied on assumed the
very direct-visit pattern CWS is trying to avoid, and its two depot-distance
terms were just s_dist's components rescaled by a constant -- so it added
estimation noise rather than independent signal. Savings ranking here is now
distance-only, exactly like heuristic.py, plus the ML saturation penalty
below. Genuine time-of-day-aware routing (using each route's REAL
accumulated arrival time, not a pre-route guess) now lives in
heuristic_dynamic.py, which recomputes savings at merge time.
"""

from __future__ import annotations
import math
import os
import pickle
import random
import time as _time
import numpy as np

from model import Node, Edge, Route, Solution
from heuristic import (
    INF, P_BIAS, N_ITER, K_NEAREST,
    _build_dummy_solution, _check_merging, _merge_routes,
    print_solution,
)
from speed_profile import SPEED_PROFILE_KMH, speed_ms as _speed_ms

BETA_DEFAULT = 0.3


# =============================================================================
# MODEL I/O
# =============================================================================

def load_model(model_path: str) -> dict:
    """Load a model bundle saved by train_model.py."""
    with open(model_path, 'rb') as fh:
        bundle = pickle.load(fh)
    m = bundle.get('metrics', {})
    auc_str = f"  AUC={m.get('roc_auc_mean', 0):.3f}" if m else ''
    print(f"  Loaded {bundle['model_type'].upper()} model "
          f"(trained on {bundle['n_train']} samples{auc_str})")
    return bundle


# =============================================================================
# SATURATION PROBABILITY + FALLBACK DISTANCE
# =============================================================================

def _fallback_distances(active_nodes: list[Node],
                        dist_matrix: np.ndarray) -> dict[int, float]:
    """
    For each active node j, compute d_fallback(j) = distance (m) to its
    nearest active neighbour.

    This is the minimum extra travel when locker j is full and the vehicle
    must redirect to the closest alternative locker.
    """
    active_ids = np.array([n.Id for n in active_nodes], dtype=int)
    d_fb: dict[int, float] = {}
    for a_idx, node in enumerate(active_nodes):
        i     = node.Id
        dists = dist_matrix[i, active_ids].copy().astype(float)
        dists[a_idx] = np.inf          # exclude self
        d_fb[i] = float(np.min(dists)) if dists.size > 1 else 0.0
    return d_fb


def _saturation_probs(nodes: list[Node], bundle: dict,
                      dist_matrix=None, locker_cap: dict | None = None,
                      departure_h: float = 8.0) -> dict[int, float]:
    """
    Predict P(saturation) for every active node using the trained model.
    Features are computed from aggregated demand — same columns as training.

    Bundles trained with only the 9 base demand features get no extra
    columns. Bundles with 13 features branch on bundle['feature_set']:
      - 'time'     (Model A) -- 4 distance/congestion features, needs
                    dist_matrix. Falls back to zeros if not provided.
      - 'capacity' (Model B) -- 4 capacity-aware features computed from
                    locker_cap (total_compartments, utilization_ratio,
                    large_size_share, size_weighted_demand). Falls back to
                    zeros if locker_cap not provided.
    Older bundles saved before 'feature_set' existed default to 'time' for
    backward compatibility.
    """
    active = [n for n in nodes[1:] if n.is_active]
    if not active:
        return {}

    total_del_w = sum(n.delivery_weight for n in active) or 1.0
    mean_del_w  = total_del_w / len(active)
    n_features  = len(bundle.get('features', []))
    feature_set = bundle.get('feature_set', 'time')
    dep_t       = departure_h * 3600.0

    rows, ids = [], []
    for node in active:
        total_ord = node.n_deliveries + node.n_pickups
        row = [
            float(node.n_deliveries),
            float(node.n_pickups),
            float(total_ord),
            node.delivery_weight,
            node.pickup_weight,
            node.n_deliveries / total_ord if total_ord > 0 else 0.0,
            node.delivery_weight / node.n_deliveries if node.n_deliveries > 0 else 0.0,
            node.delivery_weight / mean_del_w,
            node.delivery_weight / total_del_w,
        ]
        if n_features > 9 and feature_set == 'capacity':
            total_comp = (float(sum(locker_cap.get(node.Id, {}).values()))
                          if locker_cap is not None else 0.0)
            row.extend([
                total_comp,                                                # total_compartments
                node.n_deliveries / total_comp if total_comp > 0 else 0.0, # utilization_ratio
                (node.n_large_deliveries / node.n_deliveries
                 if node.n_deliveries > 0 else 0.0),                       # large_size_share
                node.size_weighted_demand,                                 # size_weighted_demand
            ])
        elif n_features > 9:
            if dist_matrix is not None:
                v_max      = max(SPEED_PROFILE_KMH) / 3.6  # m/s
                speed_dep  = _speed_ms(dep_t)
                d0j        = dist_matrix[0, node.Id]
                t_arr_s    = dep_t + d0j / speed_dep
                v_arr      = _speed_ms(t_arr_s)
                cong       = (v_max - v_arr) / v_max
                row.extend([
                    d0j / 1000.0,                    # dist_to_depot_km
                    t_arr_s / 3600.0,                # estimated_arrival_h
                    cong,                             # congestion_factor
                    node.n_deliveries * cong,         # time_pressure
                ])
            else:
                row.extend([0.0, departure_h, 0.0, 0.0])
        rows.append(row)
        ids.append(node.Id)

    X   = np.array(rows, dtype=float)
    clf = bundle['model']
    probs = (clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba')
             else clf.predict(X).astype(float))
    return dict(zip(ids, probs.tolist()))


# =============================================================================
# LEARNHEURISTIC GRAPH BUILDER
# =============================================================================

def build_graph_learn(nodes:       list[Node],
                      dist_matrix: np.ndarray,
                      bundle:      dict,
                      beta:        float = BETA_DEFAULT,
                      departure_h: float = 8.0,
                      working_hours: float = 8.0,
                      locker_cap:  dict | None = None,
                      ) -> tuple[list[Node], list[Edge]]:
    """
    Build savings list with ML saturation penalty.

    s_learn(i->j) = s_dist(i->j) - beta * P_eff(saturation_j) * d_fallback(j)

    Savings ranking is distance-only (same formula as heuristic.build_graph);
    see module docstring for why the earlier time-blended s_raw was removed.

    locker_cap : required for 'capacity' feature-set bundles (Model B); see
                 _saturation_probs. Ignored by 'time' feature-set bundles.
    """
    depot        = nodes[0]
    active_nodes = [n for n in nodes[1:] if n.is_active]

    # Depot arcs
    for node in active_nodes:
        i           = node.Id
        node.dnEdge = Edge(depot, node, dist_matrix[0][i])
        node.ndEdge = Edge(node,  depot, dist_matrix[i][0])

    # Saturation probabilities and fallback distances
    sat_probs = _saturation_probs(nodes, bundle, dist_matrix=dist_matrix,
                                  locker_cap=locker_cap, departure_h=departure_h)
    d_fb      = _fallback_distances(active_nodes, dist_matrix)

    # Time factor: speed at departure hour, used to estimate how late each
    # node is visited (farther from depot → later arrival → higher risk).
    # This only scales the ML penalty below (p_eff_j), not the savings
    # ranking itself.
    speed_dep_ms = _speed_ms(departure_h * 3600.0)
    t_span_s     = working_hours * 3600.0 or 1.0

    # K-nearest neighbours per node (same logic as build_graph in heuristic.py)
    n_active   = len(active_nodes)
    k          = min(K_NEAREST, n_active - 1)
    active_ids = np.array([n.Id for n in active_nodes], dtype=int)

    edges: list[Edge] = []
    for a_idx, inode in enumerate(active_nodes):
        i     = inode.Id
        dists = dist_matrix[i, active_ids].copy()
        dists[a_idx] = np.inf                          # exclude self

        if k > 0:
            nn_pos = (np.argpartition(dists, k)[:k]
                      if n_active - 1 >= k
                      else np.where(dists < np.inf)[0])
        else:
            nn_pos = np.array([], dtype=int)

        for pos in nn_pos:
            jnode = active_nodes[int(pos)]
            if jnode is inode:
                continue
            j    = jnode.Id
            d_ij = float(dist_matrix[i][j])
            if d_ij >= INF:
                continue
            edge = Edge(inode, jnode, d_ij)

            # Distance-only Clarke-Wright saving (same as heuristic.build_graph)
            s_raw = inode.ndEdge.cost + jnode.dnEdge.cost - d_ij  # metres

            # s_learn = s_raw − β × P_eff(sat_j) × d_fallback(j)
            #
            # P_eff incorporates time: nodes far from the depot are reached
            # later in the day, when lockers are more likely to be full
            # (cumulative deliveries from other routes). We amplify P(sat_j)
            # by a time factor derived from the estimated direct-drive time
            # depot→j at the departure-hour speed.
            #
            # t_factor_j ∈ [0,1]: 0 = departs immediately, 1 = end of shift
            # P_eff_j    = min(1, P(sat_j) × (1 + t_factor_j))
            #
            # This makes the penalty both dimensionally consistent (metres)
            # and time-aware: isolated, far, high-risk lockers are ranked last.
            p_sat_j    = sat_probs.get(j, 0.0)
            t_to_j_s   = dist_matrix[0, j] / speed_dep_ms   # depot→j travel time
            t_factor_j = min(1.0, t_to_j_s / t_span_s)
            p_eff_j    = min(1.0, p_sat_j * (1.0 + t_factor_j))
            edge.s_raw = s_raw - beta * p_eff_j * d_fb.get(j, 0.0)
            edges.append(edge)

    # Normalise adjusted raw savings to [0, 1] for CWS ordering
    if edges:
        raw_vals     = [e.s_raw for e in edges]
        min_s, max_s = min(raw_vals), max(raw_vals)
        span         = max_s - min_s or 1.0
        for e in edges:
            e.savings = (e.s_raw - min_s) / span

    edges.sort(key=lambda e: e.savings, reverse=True)

    n_penalised = sum(1 for e in edges if sat_probs.get(e.end.Id, 0.0) > 0.5)
    print(f"  LearnGraph: {n_active} nodes | {len(edges)} arcs (K={k}) | "
          f"{n_penalised} penalised (P>0.5) | β={beta}")

    return active_nodes, edges


# =============================================================================
# BR-CWS LEARN  (one iteration)
# =============================================================================

def br_CWS_learn(active_nodes: list[Node],
                 savings_list:  list[Edge],
                 vehicle_cap:   float,
                 depot:         Node,
                 p:             float = P_BIAS) -> Solution:
    """One BR-CWS iteration using the learnheuristic savings list."""
    sol       = _build_dummy_solution(active_nodes, depot)
    local_sav = list(savings_list)
    log_p     = math.log(p)

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
        if _check_merging(inode, jnode, iRoute, jRoute, vehicle_cap):
            _merge_routes(inode, jnode, iRoute, jRoute, edge, sol)

    return sol


# =============================================================================
# GRASP LEARN
# =============================================================================

def run_grasp_learn(nodes:        list[Node],
                    dist_matrix:  np.ndarray,
                    vehicle_cap:  float,
                    bundle:       dict,
                    n_iter:       int   = N_ITER,
                    p_bias:       float = P_BIAS,
                    beta:         float = BETA_DEFAULT,
                    departure_h:  float = 8.0,
                    working_hours: float = 8.0,
                    locker_cap:   dict | None = None,
                    verbose:      bool  = True
                    ) -> tuple[Solution, float]:
    """
    GRASP with learnheuristic savings.

    Parameters
    ----------
    bundle     : model bundle loaded with load_model()
    beta       : saturation penalty weight [0, 1]
    locker_cap : required to correctly score 'capacity' feature-set (Model B)
                 bundles; see build_graph_learn / _saturation_probs.
    """
    t0 = _time.perf_counter()

    active_nodes, savings_list = build_graph_learn(
        nodes, dist_matrix, bundle, beta, departure_h, working_hours, locker_cap)
    depot     = nodes[0]
    best_sol  = None
    best_cost = INF

    for it in range(n_iter):
        sol = br_CWS_learn(active_nodes, savings_list, vehicle_cap, depot, p_bias)
        if sol.cost < best_cost:
            best_sol  = sol
            best_cost = sol.cost
            if verbose:
                print(f"    iter {it+1:>4}: new best -> "
                      f"{best_cost/1000:.3f} km | {len(sol.routes)} routes")

    return best_sol, _time.perf_counter() - t0


# =============================================================================
# ARRIVAL TIME COMPUTATION
# =============================================================================

def _compute_arrival_times(sol: Solution,
                            departure_h: float,
                            dist_matrix: np.ndarray) -> dict[int, float]:
    """
    Estimate arrival time (seconds from midnight) at each node in the solution.

    For route [depot → n1 → n2 → ... → depot]:
        t(n1) = departure + dist(depot, n1) / speed
        t(n2) = t(n1) + service_time(n1) + dist(n1, n2) / speed
        ...

    Travel time  = edge.cost / speed_ms(t)  (time-dependent, metres / (m/s) = seconds)
    Service time = node.service_time (parking + S × n_orders, set in instance_reader)
    """
    t0 = departure_h * 3600.0
    arrivals: dict[int, float] = {}
    for route in sol.routes:
        t = t0
        for edge in route.edges:
            dest = edge.end
            if dest.Id == 0:          # returned to depot — stop
                break
            t += edge.cost / _speed_ms(t)   # travel seconds (time-dependent speed)
            arrivals[dest.Id] = t
            t += dest.service_time          # service + parking seconds
    return arrivals


# =============================================================================
# EXPECTED FALLBACK COST  (KPI for evaluation)
# =============================================================================

def expected_fallback_km(sol: Solution,
                          nodes: list[Node],
                          dist_matrix: np.ndarray,
                          bundle: dict,
                          departure_h: float = 8.0,
                          working_hours: float = 8.0,
                          locker_cap: dict | None = None) -> float:
    """
    Total expected fallback detour distance (km), weighted by arrival time.

    Base formula (same for every solution, since all nodes are visited once):
        contribution(j) = P(sat_j) × d_fallback(j)

    Time-weighted formula (solution-dependent):
        t_factor_j   = (t_arrival_j − t_departure) / (working_hours × 3600)
        P_eff(sat_j) = min(1, P(sat_j) × (1 + t_factor_j))
        contribution = P_eff(sat_j) × d_fallback(j)

    Rationale: a vehicle arriving at locker j early in the day is less likely
    to find it full than one arriving late.  The learnheuristic tends to route
    high-risk nodes into shorter routes → earlier arrival → lower P_eff.

    Parameters
    ----------
    departure_h   : vehicle start time in decimal hours (from instance params.departure)
    working_hours : assumed working day length for time normalisation
    locker_cap    : required to correctly score 'capacity' feature-set
                    (Model B) bundles; see _saturation_probs. Bundles were
                    previously scored here with no dist_matrix/locker_cap at
                    all, which raised a feature-count mismatch for any real
                    13-feature model -- fixed by passing both through.
    """
    active_nodes = [n for n in nodes[1:] if n.is_active]
    sat_probs    = _saturation_probs(nodes, bundle, dist_matrix=dist_matrix,
                                     locker_cap=locker_cap, departure_h=departure_h)
    d_fb         = _fallback_distances(active_nodes, dist_matrix)
    arrivals     = _compute_arrival_times(sol, departure_h, dist_matrix)

    t_start = departure_h * 3600.0
    t_span  = working_hours * 3600.0 or 1.0   # normalisation denominator

    total = 0.0
    for route in sol.routes:
        for node in route.nodes:
            p_sat  = sat_probs.get(node.Id, 0.0)
            d_fall = d_fb.get(node.Id, 0.0)
            t_j    = arrivals.get(node.Id, t_start)
            # t_factor ∈ [0, 1]: 0 = depart time, 1 = end of working day
            t_fac  = max(0.0, min(1.0, (t_j - t_start) / t_span))
            # Later arrival amplifies effective saturation risk
            p_eff  = min(1.0, p_sat * (1.0 + t_fac))
            total += p_eff * d_fall
    return total / 1000.0          # metres → km


# =============================================================================
# QUICK SELF-TEST
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from instance_reader import read_full_instance

    if len(sys.argv) < 3:
        print("Usage: python heuristic_learn.py <instance.txt> <model.pkl> [beta]")
        sys.exit(1)

    beta_val = float(sys.argv[3]) if len(sys.argv) > 3 else BETA_DEFAULT
    params, nodes, dist_matrix, *_ = read_full_instance(sys.argv[1])
    bundle = load_model(sys.argv[2])

    best, elapsed = run_grasp_learn(
        nodes, dist_matrix, params.capacity,
        bundle=bundle, n_iter=100, beta=beta_val, verbose=True
    )
    print_solution(best, f"{params.name} [learn beta={beta_val}]")
    print(f"\nSolved in {elapsed:.2f} s")
