"""
model.py
--------
Data structures for the Rudy (2025) PLBDP benchmark instances.

Key differences from LockerNYC model:
  - Capacity is weight-based (problem parameter C), not VU-based.
  - No physical compartments — locker saturation is handled by the ML model.
  - No GPS coordinates — distances come directly from the embedded matrix.
"""


# =============================================================================
# NODE  — a parcel locker station (or the depot, ID=0)
# =============================================================================
class Node:
    """
    Represents a parcel locker location visited by a vehicle.

    Demand fields (populated by instance_reader):
        delivery_weight  -- total weight of parcels to deliver here
        pickup_weight    -- total weight of parcels to collect here
        n_deliveries     -- number of delivery orders
        n_pickups        -- number of pickup orders
        service_time     -- total vehicle stop time in seconds
                           (= parking + service_per_order * n_orders)

    Routing state (set during heuristic execution):
        inRoute  -- Route this node currently belongs to (None = unassigned)
        dnEdge   -- Edge: depot -> this node
        ndEdge   -- Edge: this node -> depot
    """
    def __init__(self, node_id: int):
        self.Id = node_id

        # Demand aggregates (reset for each instance)
        self.delivery_weight: float = 0.0
        self.pickup_weight:   float = 0.0
        self.n_deliveries:    int   = 0
        self.n_pickups:       int   = 0
        self.service_time:    float = 0.0

        # Order-size aggregates (used by Model B's capacity-aware features --
        # see instance_reader.extract_features_capacity / FEATURE_COLS_B)
        self.size_weighted_demand: float = 0.0   # sum of delivery order sizes
        self.n_large_deliveries:   int   = 0     # deliveries with size >= 3

        # CWS routing pointers
        self.inRoute = None
        self.dnEdge  = None   # Edge: depot -> this node
        self.ndEdge  = None   # Edge: this node -> depot

    @property
    def total_weight(self) -> float:
        return self.delivery_weight + self.pickup_weight

    @property
    def is_active(self) -> bool:
        """Node has demand and should be included in routing."""
        return self.total_weight > 0.0

    def __repr__(self):
        return (f"Node(id={self.Id}, "
                f"del={self.delivery_weight:.0f}kg, "
                f"pck={self.pickup_weight:.0f}kg)")


# =============================================================================
# EDGE  — directed arc between two nodes
# =============================================================================
class Edge:
    """
    Directed arc from `origin` to `end`.

    cost    -- travel distance in metres
    savings -- Clarke-Wright saving value (normalised, set by build_graph)
    s_raw   -- raw saving in metres (before normalisation)
    """
    def __init__(self, origin: 'Node', end: 'Node', cost: float):
        self.origin  = origin
        self.end     = end
        self.cost    = cost    # metres
        self.savings = 0.0     # normalised saving in [0, 1]
        self.s_raw   = 0.0     # raw saving in metres


# =============================================================================
# ROUTE  — ordered sequence of nodes served by one vehicle
# =============================================================================
class Route:
    """
    A single vehicle route: depot -> node_1 -> ... -> node_k -> depot.

    VRPSPD capacity constraint (weight-based):
        delivery_weight + pickup_weight  <=  vehicle_capacity
    """
    def __init__(self):
        self.edges:           list  = []
        self.cost:            float = 0.0   # total travel distance (m)
        self.delivery_weight: float = 0.0   # sum of delivery weights on route
        self.pickup_weight:   float = 0.0   # sum of pickup weights on route
        self.service_time:    float = 0.0   # sum of service times at all stops
        self.nodes:           list  = []    # visited Node objects (ordered)

        # Seconds-from-midnight arrival time at the route's current tail node.
        # Only set/used by heuristic_dynamic.py (time-of-day-aware BR-CWS);
        # left None (unused) by heuristic.py and heuristic_learn.py.
        self.tail_time: float | None = None

    @property
    def total_weight(self) -> float:
        """Total weight the vehicle must handle on this route."""
        return self.delivery_weight + self.pickup_weight

    def __repr__(self):
        nodes_str = " -> ".join(str(n.Id) for n in self.nodes)
        return (f"Route([{nodes_str}] | "
                f"dist={self.cost/1000:.2f}km | "
                f"del={self.delivery_weight:.0f}kg pck={self.pickup_weight:.0f}kg)")


def route_peak_load(route: 'Route') -> float:
    """
    DIAGNOSTIC ONLY -- not used by _check_merging's default feasibility test
    (see design notes point 3). Simulates the vehicle's actual load node by
    node -- starts with every delivery on board, subtracts each node's
    delivery_weight on arrival, adds its pickup_weight -- and returns the
    PEAK load observed along the route.

    route.total_weight (delivery_weight + pickup_weight summed over the
    whole route) is the conservative upper bound _check_merging actually
    enforces: it assumes the worst case where every delivery and every
    pickup are on board simultaneously. route_peak_load is the tighter,
    real bound; comparing the two on a solved solution shows how much
    slack the conservative check leaves on the table (it can only ever
    reject merges the real load profile would have allowed, never the
    other way around).
    """
    load = sum(n.delivery_weight for n in route.nodes)
    peak = load
    for n in route.nodes:
        load += n.pickup_weight - n.delivery_weight
        peak = max(peak, load)
    return peak


def audit_route_capacity(sol: 'Solution', vehicle_cap: float) -> list[dict]:
    """
    DIAGNOSTIC ONLY (see route_peak_load). For every route in a solved
    Solution, compares the conservative static bound (route.total_weight,
    what _check_merging actually enforces) against the real peak load
    (route_peak_load) -- both as absolute kg and as a percentage of
    vehicle_cap. Use this to audit how conservative the current capacity
    check is on a given solution, without changing routing behaviour.
    """
    rows = []
    for i, route in enumerate(sol.routes):
        static_bound = route.total_weight
        peak         = route_peak_load(route)
        rows.append({
            'route_idx':              i,
            'n_stops':                len(route.nodes),
            'static_bound_kg':        round(static_bound, 1),
            'actual_peak_kg':         round(peak, 1),
            'slack_kg':               round(static_bound - peak, 1),
            'static_utilization_pct': round(100 * static_bound / vehicle_cap, 1) if vehicle_cap else 0.0,
            'actual_utilization_pct': round(100 * peak / vehicle_cap, 1) if vehicle_cap else 0.0,
        })
    return rows


# =============================================================================
# SOLUTION  — complete assignment of nodes to vehicle routes
# =============================================================================
class Solution:
    """A complete set of routes covering all active nodes."""
    _count: int = 0

    def __init__(self):
        Solution._count += 1
        self.ID:     int   = Solution._count
        self.routes: list  = []
        self.cost:   float = 0.0   # total travel distance (m)

    def __repr__(self):
        return (f"Solution(ID={self.ID} | "
                f"dist={self.cost/1000:.2f}km | "
                f"routes={len(self.routes)})")
