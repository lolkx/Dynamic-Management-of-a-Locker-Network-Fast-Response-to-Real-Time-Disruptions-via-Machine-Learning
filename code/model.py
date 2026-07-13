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
