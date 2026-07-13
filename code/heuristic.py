"""
heuristic.py
------------
Standard Biased Randomised Clarke-Wright Savings (BR-CWS) + GRASP
for the weight-based VRPSPD on Rudy (2025) benchmark instances.

Savings formula (normalised to [0, 1]):
    s_raw(i->j) = d(i->depot) + d(depot->j) - d(i->j)   [metres]
    s(i->j)     = (s_raw - min_s) / (max_s - min_s)

VRPSPD capacity constraint:
    delivery_weight + pickup_weight  <=  vehicle_capacity
"""

from __future__ import annotations
import math
import random
import time as _time
import numpy as np

from model import Node, Edge, Route, Solution

# =============================================================================
# CONSTANTS
# =============================================================================
INF          = float('inf')
P_BIAS       = 0.25    # geometric bias for biased-random selection
N_ITER       = 100     # default GRASP iterations
AVG_SPEED_MS = 8.33    # m/s (~30 km/h), used for travel-time estimates
K_NEAREST    = 20      # max neighbours per node in savings list
               # Limits list to n×K arcs instead of n², making large
               # instances tractable. Distant arcs rarely improve CWS.


# =============================================================================
# 1. BUILD GRAPH & SAVINGS LIST
# =============================================================================

def build_graph(nodes: list[Node],
                dist_matrix: np.ndarray
                ) -> tuple[list[Node], list[Edge]]:
    """
    Create depot-arcs and compute the normalised savings list.

    Only keeps arcs to the K_NEAREST neighbours per node (by distance).
    This reduces the savings list from O(n²) to O(n×K), making BR-CWS
    tractable on large instances (m ~ 300-900 nodes).

    Parameters
    ----------
    nodes       : list of Node objects, index 0 = depot
    dist_matrix : (m+1) x (m+1) distance matrix in metres

    Returns
    -------
    active_nodes : nodes with positive demand
    savings_list : list of Edge sorted by savings descending
    """
    depot        = nodes[0]
    active_nodes = [n for n in nodes[1:] if n.is_active]
    n_active     = len(active_nodes)

    # Depot arcs for every active node
    for node in active_nodes:
        i           = node.Id
        node.dnEdge = Edge(depot, node, dist_matrix[0][i])
        node.ndEdge = Edge(node,  depot, dist_matrix[i][0])

    # K-nearest neighbours per node  (both directions i->j and j->i)
    k            = min(K_NEAREST, n_active - 1)
    active_ids   = np.array([n.Id for n in active_nodes], dtype=int)

    savings_list: list[Edge] = []
    for a_idx, inode in enumerate(active_nodes):
        i = inode.Id
        # Distance from i to all active nodes
        dists = dist_matrix[i, active_ids].copy()
        dists[a_idx] = np.inf                          # exclude self

        # Indices of k nearest (unsorted; we only need the k smallest)
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
            edge       = Edge(inode, jnode, d_ij)
            edge.s_raw = inode.ndEdge.cost + jnode.dnEdge.cost - d_ij
            savings_list.append(edge)

    # Normalise raw savings to [0, 1]
    if savings_list:
        raw_vals      = [e.s_raw for e in savings_list]
        min_s, max_s  = min(raw_vals), max(raw_vals)
        span          = max_s - min_s or 1.0
        for e in savings_list:
            e.savings = (e.s_raw - min_s) / span

    savings_list.sort(key=lambda e: e.savings, reverse=True)

    print(f"  Graph: {n_active} active nodes | "
          f"{len(savings_list)} directed arcs (K={k})")
    return active_nodes, savings_list


# =============================================================================
# 2. DUMMY SOLUTION  (one route per active node)
# =============================================================================

def _build_dummy_solution(active_nodes: list[Node], depot: Node) -> Solution:
    """Initialise solution with one trivial depot->node->depot route each."""
    sol = Solution()
    for node in active_nodes:
        node.inRoute = None

    for node in active_nodes:
        route                 = Route()
        route.edges           = [node.dnEdge, node.ndEdge]
        route.cost            = node.dnEdge.cost + node.ndEdge.cost
        route.delivery_weight = node.delivery_weight
        route.pickup_weight   = node.pickup_weight
        route.service_time    = node.service_time
        route.nodes           = [node]
        node.inRoute          = route
        sol.routes.append(route)
        sol.cost             += route.cost
    return sol


# =============================================================================
# 3. CHECK MERGING
# =============================================================================

def _check_merging(inode: Node, jnode: Node,
                   iRoute: Route, jRoute: Route,
                   vehicle_cap: float) -> bool:
    """
    Check whether iRoute (tail=inode) can be merged with jRoute (head=jnode).

    Conditions:
      A) Different route objects.
      B) inode is the TAIL of iRoute (asymmetric: no reversal allowed).
      C) jnode is the HEAD of jRoute.
      D) VRPSPD weight capacity: combined weight <= vehicle_cap.
    """
    if iRoute is jRoute:
        return False
    if iRoute.edges[-1].origin is not inode:
        return False
    if jRoute.edges[0].end is not jnode:
        return False
    if (iRoute.delivery_weight + jRoute.delivery_weight +
            iRoute.pickup_weight  + jRoute.pickup_weight) > vehicle_cap:
        return False
    return True


# =============================================================================
# 4. MERGE ROUTES
# =============================================================================

def _merge_routes(inode: Node, jnode: Node,
                  iRoute: Route, jRoute: Route,
                  edge: Edge, sol: Solution) -> None:
    """Merge jRoute into iRoute via arc inode->jnode."""
    tail_edge    = iRoute.edges.pop()
    iRoute.cost -= tail_edge.cost

    head_edge    = jRoute.edges.pop(0)
    jRoute.cost -= head_edge.cost

    iRoute.edges.append(edge)
    iRoute.cost += edge.cost

    for e in jRoute.edges:
        iRoute.edges.append(e)
        iRoute.cost += e.cost
    for n in jRoute.nodes:
        n.inRoute = iRoute

    iRoute.delivery_weight += jRoute.delivery_weight
    iRoute.pickup_weight   += jRoute.pickup_weight
    iRoute.service_time    += jRoute.service_time
    iRoute.nodes.extend(jRoute.nodes)

    sol.cost -= (tail_edge.cost + head_edge.cost - edge.cost)
    sol.routes.remove(jRoute)


# =============================================================================
# 5. BR-CWS  (one iteration)
# =============================================================================

def br_CWS(active_nodes: list[Node],
           savings_list:  list[Edge],
           vehicle_cap:   float,
           depot:         Node,
           p:             float = P_BIAS) -> Solution:
    """
    One pass of Clarke-Wright Savings with biased-random arc selection.

    Index sampling via inverse-CDF of the geometric distribution: O(1).
    Removal by index (del list[i]): O(n) C-level shift, ~100x faster
    than the O(n) Python-loop approach used with random.choices + list.remove.
    """
    sol       = _build_dummy_solution(active_nodes, depot)
    local_sav = list(savings_list)
    log_p     = math.log(p)

    while local_sav:
        # Sample position from geometric distribution in O(1):
        # P(i) ∝ p^i  =>  i = floor(log(U) / log(p))
        u   = random.random()
        idx = min(int(math.floor(math.log(max(u, 1e-300)) / log_p)),
                  len(local_sav) - 1)

        edge = local_sav[idx]
        del local_sav[idx]          # C-level shift; no Python equality scan

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
# 6. GRASP
# =============================================================================

def run_grasp(nodes:       list[Node],
              dist_matrix: np.ndarray,
              vehicle_cap: float,
              n_iter:      int   = N_ITER,
              p_bias:      float = P_BIAS,
              verbose:     bool  = True
              ) -> tuple[Solution, float]:
    """
    GRASP loop: build graph once, repeat br_CWS n_iter times.

    Returns
    -------
    best_sol  : best Solution found
    elapsed_s : wall-clock time in seconds
    """
    t0 = _time.perf_counter()

    active_nodes, savings_list = build_graph(nodes, dist_matrix)
    depot    = nodes[0]
    best_sol  = None
    best_cost = INF

    for it in range(n_iter):
        sol = br_CWS(active_nodes, savings_list, vehicle_cap, depot, p_bias)
        if sol.cost < best_cost:
            best_sol  = sol
            best_cost = sol.cost
            if verbose:
                print(f"    iter {it+1:>4}: new best -> "
                      f"{best_cost/1000:.3f} km | {len(sol.routes)} routes")

    return best_sol, _time.perf_counter() - t0


# =============================================================================
# 7. PRINT SOLUTION
# =============================================================================

def print_solution(sol: Solution, instance_name: str = '') -> None:
    title = f"SOLUTION — {instance_name}" if instance_name else "SOLUTION"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Total distance : {sol.cost/1000:.3f} km")
    print(f"  Routes used    : {len(sol.routes)}")
    print(f"{'='*60}")
    for i, route in enumerate(sol.routes):
        nodes_str = " -> ".join(str(n.Id) for n in route.nodes)
        print(f"  Route {i+1:>3}: [{nodes_str}]  |  "
              f"dist={route.cost/1000:.2f}km  "
              f"del={route.delivery_weight:.0f}kg  "
              f"pck={route.pickup_weight:.0f}kg")
    print('='*60)


# =============================================================================
# QUICK SELF-TEST
# =============================================================================

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from instance_reader import read_full_instance

    if len(sys.argv) < 2:
        print("Usage: python heuristic.py <instance.txt>")
        sys.exit(1)

    params, nodes, dist_matrix, *_ = read_full_instance(sys.argv[1])
    print(f"Instance: {params.name}  "
          f"({params.n_orders} orders, {params.n_nodes} nodes, cap={params.capacity})")

    best, elapsed = run_grasp(nodes, dist_matrix, params.capacity,
                              n_iter=100, verbose=True)
    print_solution(best, params.name)
    print(f"\nSolved in {elapsed:.2f} s")
