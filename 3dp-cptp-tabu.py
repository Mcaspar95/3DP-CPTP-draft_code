# Tabu Search for the 3DP-CPTP
#
# Inspired by: Gendreau, Iori, Laporte & Martello (2006)
#   "A Tabu Search Algorithm for a Routing and Container Loading Problem"
#
# Adapted from the 3L-CVRP tabu search to the 3DP-CPTP where:
#   - Not all customers must be served (selective purchasing)
#   - Objective: maximize total revenue – total travel cost
#   - 3D packing feasibility is checked via a Gurobi sub-model
#   - Neighborhood moves include insert/remove/swap customers between routes
#     and intra-route 2-opt reordering, plus serve/unserve decisions

# Short description of the Tabu Search algorithm:

# Initial solution via a greedy savings-inspired heuristic (customers ranked by attractiveness = revenue / travel distance, inserted cheapest-first)
# 5 neighborhood move types:
# Relocate — move a customer between routes
# Swap — exchange two customers between routes
# Insert — serve an unserved customer (3DP-CPTP selective serving)
# Remove — unserve an unprofitable customer
# 2-opt — reverse a segment within a route to improve routing
# Tabu list with configurable tenure (default 10 iterations) — records reverse-move attributes to prevent cycling
# Aspiration criterion — a tabu move is accepted if it yields a new global best
# 3D packing feasibility checked via Gurobi sub-model with caching (frozenset of item IDs)
# Diversification after 30 stagnating iterations — removes ~30% of served customers and re-inserts unserved ones
#Function check_packing_feasible() has a time_limit, but this can enhance exploration as feasible solutions are still rejected

# Optimize tabu_tenure variable, small value - revisit recent solutions, high value - better global exploration
import math
import os
import random
import time
import copy
from itertools import permutations

from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from instances import cvrp01, cvrp02, cvrp03, cvrp04, cvrp05, cvrp06, cvrp07, cvrp08, cvrp09, cvrp10, cvrp11, cvrp12, cvrp13, cvrp14, cvrp15, cvrp16, cvrp17, cvrp18, cvrp19, cvrp20, cvrp21, cvrp22, cvrp23, cvrp24, cvrp25, cvrp26, cvrp27
from instances_cutted import ccvrp01, ccvrp02, ccvrp03, ccvrp04, ccvrp05
# -------------------------
# Parameters
# -------------------------
TABU_TENURE = 15          # number of iterations a move stays tabu
MAX_ITERATIONS = 500000      # maximum tabu search iterations
TIME_LIMIT = 1800          # total wall-clock seconds
PACKING_TIME_LIMIT = 5    # time limit per packing feasibility check (seconds)
GAMMA = 1.0               # travel cost factor
P_NEAREST = 1           # neighborhood size: p nearest customers (Gendreau 2006 §3.1)
diversify_threshold = 30  # diversify after this many non-improving iterations

# -------------------------
# Instance generation (shared with other solvers)
# -------------------------
def generate_instance(
    instance,
    gamma=1.0, # travel cost factor
    scenario="P3",
):
    # derive a short name for the instance (module basename if a module was passed)
    if hasattr(instance, "__name__"):
        name = instance.__name__.split(".")[-1]
    else:
        name = str(instance)
    m = instance.m
    coords = instance.coords
    n = instance.n
    C = instance.C
    Q = instance.Q
    items = instance.items
    weights = instance.weights

    if scenario == "P1":
        revenues = instance.revenues
    elif scenario == "P2":
        revenues = {i: 1 + ((7141 * i + 73) % 100) for i in range(1, n + 1)}
    elif scenario == "P3":
        max_dist = max(math.dist(coords[0], coords[j]) for j in range(1, n + 1))
        revenues = {i: 1 + math.ceil(99 * math.dist(coords[0], coords[i]) / max_dist) for i in range(1, n + 1)}
    elif scenario == "P4":
        revenues = {i: round(math.dist(coords[0], coords[i])) for i in range(1, n + 1)}
    else:
        raise ValueError(f"Unknown scenario {scenario}")

    # Vehicles (homogeneous fleet)
    vehicles = {k: C for k in range(1, m + 1)}

    # Travel costs
    c = {}
    N0 = list(range(0, n + 1))
    for i in N0:
        for j in N0:
            if i == j:
                continue
            c[(i, j)] = gamma * math.dist(coords[i], coords[j])

    return {
        "n": n,
        "m": m,
        "coords": coords,
        "vehicles": vehicles,
        "Q": Q,
        "items": items,
        "revenues": revenues,
        "weights": weights,
        "costs": c,
        "tightness": None,
        "name": name,
        "scenario": scenario,
        "p_nearest": _precompute_p_nearest(n, c, P_NEAREST),
    }

# -------------------------
# Helper: precompute p-nearest neighbors (Gendreau 2006 §3.1)
# -------------------------
def _precompute_p_nearest(n, costs, p):
    """
    For each customer i, return the set of p nearest customers by travel cost.
    """
    N = list(range(1, n + 1))
    p_nearest = {}
    for i in N:
        # sort other customers by cost from i
        others = [(costs.get((i, j), float('inf')), j) for j in N if j != i]
        others.sort()
        p_nearest[i] = set(j for _, j in others[:p])
    return p_nearest

# -------------------------
# Helper: unique orientations
# -------------------------
def unique_orientations(l, w, h):
    return list(dict.fromkeys(permutations([l, w, h], 3)))

# -------------------------
# 3D Packing feasibility check via Gurobi
# -------------------------
def check_packing_feasible(item_ids, instance, time_limit=None, deadline=None):
    """
    Check whether items in 'item_ids' can be feasibly packed into
    one container of dimensions C = (L, W, H).
    Returns True if feasible, False otherwise.
    """
    # resolve at call time so main_slurm can override the module global
    if time_limit is None:
        time_limit = PACKING_TIME_LIMIT
    if not item_ids:
        return True

    # Respect global deadline
    if deadline is not None:
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        time_limit = min(time_limit, remaining)

    items = instance["items"]
    C = list(instance["vehicles"].values())[0]  # homogeneous fleet
    L, W, H = C

    N = list(item_ids)
    R_i = {i: unique_orientations(*items[i]) for i in N}

    model = gp.Model("packing_check")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    # Position variables
    p_x = model.addVars(N, lb=0.0, name="px")
    p_y = model.addVars(N, lb=0.0, name="py")
    p_z = model.addVars(N, lb=0.0, name="pz")

    # Orientation binaries
    o = {}
    for i in N:
        for ridx in range(len(R_i[i])):
            o[(i, ridx)] = model.addVar(vtype=GRB.BINARY, name=f"o[{i},{ridx}]")
    o = gp.tupledict(o)

    # Non-overlap precedence
    pairs_ordered = [(i, j) for i in N for j in N if i != j]
    a_x = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="ax")
    a_y = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="ay")
    a_z = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="az")

    def size_expr(i, axis):
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        return gp.quicksum(R_i[i][ridx][idx] * o[(i, ridx)] for ridx in range(len(R_i[i])))

    # Feasibility only — no real objective
    model.setObjective(0, GRB.MINIMIZE)

    # Exactly one orientation per item
    for i in N:
        model.addConstr(gp.quicksum(o[(i, ridx)] for ridx in range(len(R_i[i]))) == 1)

    # Container bounds
    for i in N:
        model.addConstr(p_x[i] + size_expr(i, "x") <= L)
        model.addConstr(p_y[i] + size_expr(i, "y") <= W)
        model.addConstr(p_z[i] + size_expr(i, "z") <= H)

    # Non-overlap
    pairs = [(i, j) for i in N for j in N if i < j]
    for i, j in pairs:
        model.addConstr(
            a_x[(i, j)] + a_x[(j, i)]
            + a_y[(i, j)] + a_y[(j, i)]
            + a_z[(i, j)] + a_z[(j, i)]
            >= 1
        )

    for i, j in pairs_ordered:
        model.addConstr(p_x[i] + size_expr(i, "x") <= p_x[j] + L * (1 - a_x[(i, j)]))
        model.addConstr(p_y[i] + size_expr(i, "y") <= p_y[j] + W * (1 - a_y[(i, j)]))
        model.addConstr(p_z[i] + size_expr(i, "z") <= p_z[j] + H * (1 - a_z[(i, j)]))

    model.optimize()

    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL): #if Gurobi no feasible solution in time_limit - packing is infeasible
        return True
    return False

# -------------------------
# 3D Packing solver — returns item positions & oriented dimensions
# -------------------------
def solve_packing(item_ids, instance, time_limit=None, deadline=None):
    """
    Solve the packing sub-problem and return a list of packed-item dicts:
      [{"i": customer_id, "pos": (x,y,z), "dims": (dx,dy,dz), "container": (L,W,H)}, ...]
    Returns an empty list if infeasible or no items.
    """
    # resolve at call time so main_slurm can override the module global
    if time_limit is None:
        time_limit = PACKING_TIME_LIMIT
    if not item_ids:
        return []

    # Respect global deadline
    if deadline is not None:
        remaining = deadline - time.time()
        if remaining <= 0:
            return []
        time_limit = min(time_limit, remaining)

    items = instance["items"]
    C = list(instance["vehicles"].values())[0]  # homogeneous fleet
    L, W, H = C

    N = list(item_ids)
    R_i = {i: unique_orientations(*items[i]) for i in N}

    model = gp.Model("packing_solve")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit

    # Position variables
    p_x = model.addVars(N, lb=0.0, name="px")
    p_y = model.addVars(N, lb=0.0, name="py")
    p_z = model.addVars(N, lb=0.0, name="pz")

    # Orientation binaries
    o = {}
    for i in N:
        for ridx in range(len(R_i[i])):
            o[(i, ridx)] = model.addVar(vtype=GRB.BINARY, name=f"o[{i},{ridx}]")
    o = gp.tupledict(o)

    # Non-overlap precedence
    pairs_ordered = [(i, j) for i in N for j in N if i != j]
    a_x = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="ax")
    a_y = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="ay")
    a_z = model.addVars(pairs_ordered, vtype=GRB.BINARY, name="az")

    def size_expr(i, axis):
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        return gp.quicksum(R_i[i][ridx][idx] * o[(i, ridx)] for ridx in range(len(R_i[i])))

    model.setObjective(0, GRB.MINIMIZE)

    # Exactly one orientation per item
    for i in N:
        model.addConstr(gp.quicksum(o[(i, ridx)] for ridx in range(len(R_i[i]))) == 1)

    # Container bounds
    for i in N:
        model.addConstr(p_x[i] + size_expr(i, "x") <= L)
        model.addConstr(p_y[i] + size_expr(i, "y") <= W)
        model.addConstr(p_z[i] + size_expr(i, "z") <= H)

    # Non-overlap
    pairs = [(i, j) for i in N for j in N if i < j]
    for i, j in pairs:
        model.addConstr(
            a_x[(i, j)] + a_x[(j, i)]
            + a_y[(i, j)] + a_y[(j, i)]
            + a_z[(i, j)] + a_z[(j, i)]
            >= 1
        )

    for i, j in pairs_ordered:
        model.addConstr(p_x[i] + size_expr(i, "x") <= p_x[j] + L * (1 - a_x[(i, j)]))
        model.addConstr(p_y[i] + size_expr(i, "y") <= p_y[j] + W * (1 - a_y[(i, j)]))
        model.addConstr(p_z[i] + size_expr(i, "z") <= p_z[j] + H * (1 - a_z[(i, j)]))

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return []

    packed = []
    for i in N:
        chosen = None
        for ridx in range(len(R_i[i])):
            if o[(i, ridx)].X > 0.5:
                chosen = R_i[i][ridx]
                break
        if chosen is None:
            chosen = items[i]
        packed.append({
            "i": i,
            "pos": (p_x[i].X, p_y[i].X, p_z[i].X),
            "dims": chosen,
            "container": (L, W, H),
        })
    return packed

# Packing feasibility cache to avoid re-checking identical item sets
_packing_cache = {}

def check_packing_cached(item_ids, instance, deadline=None):
    """Cached version of packing feasibility check."""
    key = frozenset(item_ids)
    if key not in _packing_cache:
        _packing_cache[key] = check_packing_feasible(item_ids, instance, deadline=deadline)
    return _packing_cache[key]

# -------------------------
# Solution representation
# -------------------------
# A solution is a dict:
#   {
#     "routes": {k: [0, i1, i2, ..., 0]},  # route per vehicle (depot-to-depot)
#     "served": set of served customer ids,
#     "unserved": set of unserved customer ids,
#   }

def route_cost(route, costs):
    """Total travel cost of a route [0, i1, i2, ..., 0]."""
    total = 0.0
    for idx in range(len(route) - 1):
        total += costs.get((route[idx], route[idx + 1]), 0.0)
    return total

def route_weight(route, weights):
    """Total weight of items on a route (excluding depot)."""
    return sum(weights.get(i, 0) for i in route if i != 0)

def route_items(route):
    """List of customer/item ids on a route (excluding depot)."""
    return [i for i in route if i != 0]

def solution_objective(sol, instance):
    """Compute objective: total revenue – total travel cost."""
    revenues = instance["revenues"]
    costs = instance["costs"]
    total_rev = sum(revenues.get(i, 0) for i in sol["served"])
    total_cost = 0.0
    for k, route in sol["routes"].items():
        if len(route) > 2:  # not empty route
            total_cost += route_cost(route, costs)
    return total_rev - total_cost

def copy_solution(sol):
    """Deep-copy a solution."""
    return {
        "routes": {k: list(route) for k, route in sol["routes"].items()},
        "served": set(sol["served"]),
        "unserved": set(sol["unserved"]),
    }

# -------------------------
# Initial solution: greedy savings heuristic
# -------------------------
def build_initial_solution(instance, deadline=None):
    """
    Greedy construction heuristic inspired by Clarke-Wright savings.
    1. Rank customers by revenue / (weight + travel_to_depot) ratio
    2. Insert customers greedily into routes maintaining feasibility
    """
    n = instance["n"]
    m = instance["m"]
    Q = instance["Q"]
    costs = instance["costs"]
    revenues = instance["revenues"]
    weights = instance["weights"]
    coords = instance["coords"]
    K = list(range(1, m + 1))
    N = list(range(1, n + 1))

    # Rank customers by attractiveness (revenue / distance-to-depot / weight)
    attractiveness = []
    for i in N:
        dist_depot = costs.get((0, i), 1.0) + costs.get((i, 0), 1.0)
        ratio = revenues[i] / max(dist_depot, 0.01)
        attractiveness.append((ratio, i))
    attractiveness.sort(reverse=True)  # most attractive first

    # Initialise empty routes
    routes = {k: [0, 0] for k in K}
    served = set()
    unserved = set(N)

    for _, cust in attractiveness:
        best_k = None
        best_pos = None
        best_increase = float("inf")

        for k in K:
            # check weight capacity
            curr_weight = route_weight(routes[k], weights)
            if curr_weight + weights[cust] > Q:
                continue

            # try each insertion position
            r = routes[k]
            for pos in range(1, len(r)):  # between existing nodes
                # cost increase of inserting cust between r[pos-1] and r[pos]
                c_insert = (costs.get((r[pos - 1], cust), 0)
                            + costs.get((cust, r[pos]), 0)
                            - costs.get((r[pos - 1], r[pos]), 0))
                if c_insert < best_increase:
                    # check packing feasibility with cust added
                    trial_items = route_items(r) + [cust]
                    if check_packing_cached(trial_items, instance, deadline=deadline):
                        best_increase = c_insert
                        best_k = k
                        best_pos = pos

        # Respect global deadline
        if deadline is not None and time.time() >= deadline:
            break

        # Insert if profitable (revenue covers insertion cost)
        if best_k is not None:
            net = revenues[cust] - best_increase
            if net > 0:
                routes[best_k].insert(best_pos, cust)
                served.add(cust)
                unserved.discard(cust)

    sol = {
        "routes": routes,
        "served": served,
        "unserved": unserved,
    }
    return sol

# -------------------------
# Cheapest insertion position for a customer into a route
# -------------------------
def cheapest_insertion(cust, route, costs):
    """
    Find the position in 'route' that results in minimal travel cost increase
    when inserting 'cust'. Returns (position, cost_increase).
    """
    best_pos = None
    best_delta = float("inf")
    for pos in range(1, len(route)):
        delta = (costs.get((route[pos - 1], cust), 0)
                 + costs.get((cust, route[pos]), 0)
                 - costs.get((route[pos - 1], route[pos]), 0))
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
    return best_pos, best_delta

# -------------------------
# Remove customer from its route, returning cost savings
# -------------------------
def remove_customer(cust, routes, costs):
    """
    Remove 'cust' from whatever route it's in.
    Returns (vehicle_id, old_position, cost_savings).
    """
    for k, route in routes.items():
        if cust in route:
            idx = route.index(cust)
            prev_node = route[idx - 1]
            next_node = route[idx + 1]
            savings = (costs.get((prev_node, cust), 0)
                       + costs.get((cust, next_node), 0)
                       - costs.get((prev_node, next_node), 0))
            route.remove(cust)
            return k, idx, savings
    return None, None, 0.0

# -------------------------
# Neighborhood moves (Gendreau 2006 style): 5 types following line 348-538
# -------------------------

def generate_relocate_moves(sol, instance):
    """
    RELOCATE: Move a served customer from its current route to a different
    route that contains at least one of its p-nearest candidates (Gendreau 2006 §3.1).
    """
    Q = instance["Q"]
    costs = instance["costs"]
    weights = instance["weights"]
    p_nearest = instance["p_nearest"]
    K = list(sol["routes"].keys())
    moves = []

    # Build a map: customer -> vehicle it belongs to
    cust_to_route = {}
    for k, route in sol["routes"].items():
        for c in route_items(route):
            cust_to_route[c] = k

    for k_from in K:
        route_from = sol["routes"][k_from]
        customers_from = route_items(route_from)
        for cust in customers_from:
            # Identify candidate target routes: those containing a p-nearest neighbor
            candidate_routes = set()
            for neighbor in p_nearest.get(cust, set()):
                if neighbor in cust_to_route and cust_to_route[neighbor] != k_from:
                    candidate_routes.add(cust_to_route[neighbor])

            for k_to in candidate_routes:
                route_to = sol["routes"][k_to]
                # weight check
                if route_weight(route_to, weights) + weights[cust] > Q:
                    continue
                # cheapest insertion
                pos, insert_cost = cheapest_insertion(cust, route_to, costs)
                if pos is None:
                    continue
                # removal savings
                idx = route_from.index(cust)
                prev_n = route_from[idx - 1]
                next_n = route_from[idx + 1]
                remove_saving = (costs.get((prev_n, cust), 0)
                                 + costs.get((cust, next_n), 0)
                                 - costs.get((prev_n, next_n), 0))
                delta = insert_cost - remove_saving  # negative = improvement (cost decrease)
                moves.append({
                    "type": "relocate",
                    "cust": cust,
                    "from_k": k_from,
                    "to_k": k_to,
                    "to_pos": pos,
                    "delta_cost": delta,  # change in travel cost (negative = saving)
                    "delta_obj": -delta,  # change in objective (positive = improvement)
                })
    return moves

def generate_swap_moves(sol, instance):
    """
    SWAP: Exchange two customers between two different routes.
    Only considers pairs where c2 is among c1's p-nearest candidates (Gendreau 2006 §3.1).
    """
    Q = instance["Q"]
    costs = instance["costs"]
    weights = instance["weights"]
    p_nearest = instance["p_nearest"]
    K = list(sol["routes"].keys())
    moves = []

    # Build a map: customer -> vehicle
    cust_to_route = {}
    for k, route in sol["routes"].items():
        for c in route_items(route):
            cust_to_route[c] = k

    for k1 in K:
        custs1 = route_items(sol["routes"][k1])
        for c1 in custs1:
            # Only consider swapping with p-nearest neighbors on other routes
            for c2 in p_nearest.get(c1, set()):
                if c2 not in cust_to_route:
                    continue
                k2 = cust_to_route[c2]
                if k2 <= k1:
                    continue  # avoid duplicate pairs
                # weight feasibility after swap
                w1_new = route_weight(sol["routes"][k1], weights) - weights[c1] + weights[c2]
                w2_new = route_weight(sol["routes"][k2], weights) - weights[c2] + weights[c1]
                if w1_new > Q or w2_new > Q:
                    continue
                # cost delta: remove both, insert both in opposite routes
                r1 = sol["routes"][k1]
                r2 = sol["routes"][k2]
                idx1 = r1.index(c1)
                idx2 = r2.index(c2)
                # removal costs
                rem1 = (costs.get((r1[idx1 - 1], c1), 0) + costs.get((c1, r1[idx1 + 1]), 0)
                        - costs.get((r1[idx1 - 1], r1[idx1 + 1]), 0))
                rem2 = (costs.get((r2[idx2 - 1], c2), 0) + costs.get((c2, r2[idx2 + 1]), 0)
                        - costs.get((r2[idx2 - 1], r2[idx2 + 1]), 0))
                # insertion costs (insert at same position after removal)
                # for simplicity, use cheapest insertion
                # idx1/idx2 are already known and each customer occurs exactly once,
                # so copy-and-delete-by-index is faster than filtering the whole route.
                r1_temp = list(r1)
                del r1_temp[idx1]
                r2_temp = list(r2)
                del r2_temp[idx2]
                pos2_in_r1, ins_c2_r1 = cheapest_insertion(c2, r1_temp, costs)
                pos1_in_r2, ins_c1_r2 = cheapest_insertion(c1, r2_temp, costs)
                if pos2_in_r1 is None or pos1_in_r2 is None:
                    continue
                delta = ins_c2_r1 + ins_c1_r2 - rem1 - rem2
                moves.append({
                    "type": "swap",
                    "cust1": c1,
                    "cust2": c2,
                    "k1": k1,
                    "k2": k2,
                    "delta_cost": delta,
                    "delta_obj": -delta,
                })
    return moves

def generate_insert_unserved_moves(sol, instance):
    """
    INSERT: Try inserting an unserved customer into routes containing a
    p-nearest candidate, or any route if no candidate is currently served (Gendreau 2006 §3.1).
    """
    Q = instance["Q"]
    costs = instance["costs"]
    weights = instance["weights"]
    revenues = instance["revenues"]
    p_nearest = instance["p_nearest"]
    K = list(sol["routes"].keys())
    moves = []

    # Build a map: customer -> vehicle
    cust_to_route = {}
    for k, route in sol["routes"].items():
        for c in route_items(route):
            cust_to_route[c] = k

    for cust in sol["unserved"]:
        # Candidate routes: those containing a p-nearest neighbor
        candidate_routes = set()
        for neighbor in p_nearest.get(cust, set()):
            if neighbor in cust_to_route:
                candidate_routes.add(cust_to_route[neighbor])
        # If no served neighbor, fall back to all routes
        if not candidate_routes:
            candidate_routes = set(K)

        for k in candidate_routes:
            route = sol["routes"][k]
            if route_weight(route, weights) + weights[cust] > Q:
                continue
            pos, insert_cost = cheapest_insertion(cust, route, costs)
            if pos is None:
                continue
            # objective change: gain revenue, pay insertion cost
            delta_obj = revenues[cust] - insert_cost
            moves.append({
                "type": "insert",
                "cust": cust,
                "to_k": k,
                "to_pos": pos,
                "delta_cost": insert_cost,
                "delta_obj": delta_obj,
            })
    return moves

def generate_remove_served_moves(sol, instance):
    """
    REMOVE: Remove a served customer from its route (if unprofitable on net).
    """
    costs = instance["costs"]
    revenues = instance["revenues"]
    K = list(sol["routes"].keys())
    moves = []

    for k in K:
        route = sol["routes"][k]
        for cust in route_items(route):
            idx = route.index(cust)
            prev_n = route[idx - 1]
            next_n = route[idx + 1]
            remove_saving = (costs.get((prev_n, cust), 0)
                             + costs.get((cust, next_n), 0)
                             - costs.get((prev_n, next_n), 0))
            # objective change: lose revenue, save travel
            delta_obj = -revenues[cust] + remove_saving
            moves.append({
                "type": "remove",
                "cust": cust,
                "from_k": k,
                "delta_cost": -remove_saving,
                "delta_obj": delta_obj,
            })
    return moves

def generate_2opt_moves(sol, instance):
    """
    INTRA-ROUTE 2-OPT: Reverse a segment within a single route to reduce travel cost.
    """
    costs = instance["costs"]
    K = list(sol["routes"].keys())
    moves = []

    for k in K:
        route = sol["routes"][k]
        customers = route_items(route)
        if len(customers) < 2:
            continue
        # try reversing segment route[i..j] (1-indexed within route, excluding depots)
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                # current cost of edges (i-1,i) and (j,j+1)
                old_cost = (costs.get((route[i - 1], route[i]), 0)
                            + costs.get((route[j], route[j + 1]), 0))
                # new cost after reversal: edges (i-1,j) and (i,j+1)
                new_cost = (costs.get((route[i - 1], route[j]), 0)
                            + costs.get((route[i], route[j + 1]), 0))
                delta = new_cost - old_cost
                if delta < -1e-6:  # only improving moves for 2-opt
                    moves.append({
                        "type": "2opt",
                        "k": k,
                        "i": i,
                        "j": j,
                        "delta_cost": delta,
                        "delta_obj": -delta,
                    })
    return moves

# -------------------------
# Apply a move to a solution
# -------------------------
def apply_move(sol, move, instance):
    """Apply a neighborhood move to the solution (in-place). Returns the tabu attribute."""
    costs = instance["costs"]
    weights = instance["weights"]

    if move["type"] == "relocate":
        cust = move["cust"]
        k_from = move["from_k"]
        k_to = move["to_k"]
        sol["routes"][k_from].remove(cust)
        # re-compute cheapest insertion in target (positions may have shifted)
        pos, _ = cheapest_insertion(cust, sol["routes"][k_to], costs)
        sol["routes"][k_to].insert(pos, cust)
        return ("relocate", cust, k_to, k_from)  # reverse move as tabu key

    elif move["type"] == "swap":
        c1, c2 = move["cust1"], move["cust2"]
        k1, k2 = move["k1"], move["k2"]
        sol["routes"][k1].remove(c1)
        sol["routes"][k2].remove(c2)
        # insert c2 into k1, c1 into k2
        pos2, _ = cheapest_insertion(c2, sol["routes"][k1], costs)
        sol["routes"][k1].insert(pos2, c2)
        pos1, _ = cheapest_insertion(c1, sol["routes"][k2], costs)
        sol["routes"][k2].insert(pos1, c1)
        return ("swap", c2, c1, k2, k1)  # reverse

    elif move["type"] == "insert":
        cust = move["cust"]
        k_to = move["to_k"]
        pos, _ = cheapest_insertion(cust, sol["routes"][k_to], costs)
        sol["routes"][k_to].insert(pos, cust)
        sol["served"].add(cust)
        sol["unserved"].discard(cust)
        return ("remove", cust)  # reverse is removing

    elif move["type"] == "remove":
        cust = move["cust"]
        k_from = move["from_k"]
        sol["routes"][k_from].remove(cust)
        sol["served"].discard(cust)
        sol["unserved"].add(cust)
        return ("insert", cust)  # reverse is inserting

    elif move["type"] == "2opt":
        k = move["k"]
        i, j = move["i"], move["j"]
        route = sol["routes"][k]
        route[i:j + 1] = route[i:j + 1][::-1]
        return ("2opt", k, i, j)  # same reversal undoes itself

    return None

# -------------------------
# Check packing feasibility for all routes in a solution
# -------------------------
def check_solution_packing(sol, instance):
    """Check packing feasibility for all non-empty routes."""
    for k, route in sol["routes"].items():
        items_on_route = route_items(route)
        if items_on_route:
            if not check_packing_cached(items_on_route, instance):
                return False
    return True

def check_route_packing(route, instance, deadline=None):
    """Check packing feasibility for a single route."""
    items_on_route = route_items(route)
    if not items_on_route:
        return True
    return check_packing_cached(items_on_route, instance, deadline=deadline)

# -------------------------
# Tabu Search main loop
# -------------------------
def tabu_search(instance, max_iterations=MAX_ITERATIONS, tabu_tenure=TABU_TENURE,
                time_limit=TIME_LIMIT, diversify_threshold=diversify_threshold,
                verbose=True):
    """
    Tabu search for the 3DP-CPTP following Gendreau et al. (2006).

    Neighborhood moves:
      1. Relocate — move a customer between routes
      2. Swap    — exchange two customers between routes
      3. Insert  — serve an unserved customer
      4. Remove  — unserve a served customer (if unprofitable)
      5. 2-opt   — reverse a segment within a route

    Tabu mechanism:
      - Each move records a "reverse attribute" in the tabu list
      - A move is tabu if its attribute is in the tabu list
      - Aspiration: tabu move accepted if it yields a new global best

    Packing feasibility:
      - Checked via Gurobi sub-model (cached by item set)
    """
    t_start = time.time()
    deadline = t_start + time_limit

    # ---- Build initial solution ----
    if verbose:
        print("Building initial solution...")
    current = build_initial_solution(instance, deadline=deadline)
    current_obj = solution_objective(current, instance)

    best = copy_solution(current)
    best_obj = current_obj

    if verbose:
        n_served = len(current["served"])
        print(f"Initial solution: served={n_served}, obj={current_obj:.2f}")

    # ---- Tabu list: maps tabu_attribute -> expiry iteration ----
    tabu_list = {}

    # ---- Iteration counters ----
    no_improve_count = 0

    for iteration in range(1, max_iterations + 1):
        elapsed = time.time() - t_start
        if elapsed > time_limit:
            if verbose:
                print(f"Time limit reached at iteration {iteration}.")
            break

        # ---- Generate candidate moves ----
        all_moves = []
        all_moves.extend(generate_relocate_moves(current, instance))
        all_moves.extend(generate_swap_moves(current, instance))
        all_moves.extend(generate_insert_unserved_moves(current, instance))
        all_moves.extend(generate_remove_served_moves(current, instance))
        all_moves.extend(generate_2opt_moves(current, instance))

        if not all_moves:
            if verbose:
                print(f"[iter {iteration}] No moves available. Stopping.")
            break

        # ---- Sort moves by delta_obj (best improvement first) ----
        all_moves.sort(key=lambda mv: mv["delta_obj"], reverse=True)

        # ---- Select best non-tabu move (or tabu with aspiration) ----
        chosen_move = None
        for move in all_moves:
            # Check time limit inside the move evaluation loop
            if time.time() >= deadline:
                break

            # Determine tabu attribute for this move
            tabu_attr = _move_tabu_attribute(move)
            is_tabu = tabu_attr in tabu_list and tabu_list[tabu_attr] > iteration

            new_obj = current_obj + move["delta_obj"]

            if is_tabu:
                # Aspiration: accept if it improves global best
                if new_obj > best_obj + 1e-6:
                    pass  # accept despite tabu
                else:
                    continue  # skip tabu move

            # Check packing feasibility for affected routes
            trial = copy_solution(current)
            apply_move(trial, move, instance)

            # Determine which routes need packing check
            affected_routes = _affected_routes(move)
            feasible = True
            for k in affected_routes:
                if k in trial["routes"]:
                    if not check_route_packing(trial["routes"][k], instance, deadline=deadline):
                        feasible = False
                        break

            if feasible:
                chosen_move = move
                break

        # Break outer loop if time is up
        if time.time() >= deadline:
            if verbose:
                print(f"Time limit reached at iteration {iteration}.")
            break

        if chosen_move is None:
            no_improve_count += 1
            if verbose and iteration % 20 == 0:
                print(f"[iter {iteration}] No feasible move found this iteration.")
            if no_improve_count >= diversify_threshold:
                _diversify(current, instance, deadline=deadline)
                current_obj = solution_objective(current, instance)
                no_improve_count = 0
                if verbose:
                    print(f"[iter {iteration}] No feasible move found this iteration.")
            continue

        # ---- Apply chosen move ----
        reverse_attr = apply_move(current, chosen_move, instance)
        current_obj = solution_objective(current, instance)

        # Record tabu
        if reverse_attr is not None:
            tabu_list[reverse_attr] = iteration + tabu_tenure

        # Update best
        if current_obj > best_obj + 1e-6:
            best = copy_solution(current)
            best_obj = current_obj
            no_improve_count = 0
            if verbose:
                print(f"[iter {iteration}] *** NEW BEST: obj={best_obj:.2f}, "
                      f"served={len(best['served'])}, "
                      f"elapsed={time.time()-t_start:.1f}s")
        else:
            no_improve_count += 1

        # Diversification
        if no_improve_count >= diversify_threshold:
            _diversify(current, instance, deadline=deadline)
            current_obj = solution_objective(current, instance)
            no_improve_count = 0
            if verbose:
                print(f"[iter {iteration}] Diversification applied. obj={current_obj:.2f}")

        # Periodic status
        if verbose and iteration % 25 == 0:
            print(f"[iter {iteration}] current_obj={current_obj:.2f}, "
                  f"best_obj={best_obj:.2f}, served={len(current['served'])}, "
                  f"elapsed={time.time()-t_start:.1f}s")

        # Clean expired tabu entries periodically
        if iteration % 50 == 0:
            tabu_list = {attr: exp for attr, exp in tabu_list.items() if exp > iteration}

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nTabu search finished in {elapsed:.2f}s after {min(iteration, max_iterations)} iterations.")
        print(f"Best objective: {best_obj:.2f}")
        print(f"Served customers: {sorted(best['served'])}")
        print(f"Un-Served customers: {sorted(best['unserved'])}")
        for k, route in best["routes"].items():
            if len(route) > 2:
                print(f"  Route {k}: {route}")

    return best, best_obj

# -------------------------
# Helper: tabu attribute of a move
# -------------------------
def _move_tabu_attribute(move):
    """Return the attribute that would be made tabu if this move is executed."""
    if move["type"] == "relocate":
        return ("relocate", move["cust"], move["from_k"], move["to_k"])
    elif move["type"] == "swap":
        return ("swap", move["cust1"], move["cust2"], move["k1"], move["k2"])
    elif move["type"] == "insert":
        return ("insert", move["cust"])
    elif move["type"] == "remove":
        return ("remove", move["cust"])
    elif move["type"] == "2opt":
        return ("2opt", move["k"], move["i"], move["j"])
    return None

# -------------------------
# Helper: affected routes for packing check
# -------------------------
def _affected_routes(move):
    """Return set of vehicle ids whose routes are modified by the move."""
    if move["type"] == "relocate":
        return {move["from_k"], move["to_k"]}
    elif move["type"] == "swap":
        return {move["k1"], move["k2"]}
    elif move["type"] == "insert":
        return {move["to_k"]}
    elif move["type"] == "remove":
        return {move["from_k"]}
    elif move["type"] == "2opt":
        return {move["k"]}
    return set()

# -------------------------
# Diversification: randomly perturb solution
# -------------------------
def _diversify(sol, instance, deadline=None):
    """
    Diversification (Gendreau 2006 §3.4): randomly remove several customers
    and re-insert a subset of unserved ones.
    """
    costs = instance["costs"]
    weights = instance["weights"]
    revenues = instance["revenues"]
    Q = instance["Q"]
    K = list(sol["routes"].keys())

    served_list = list(sol["served"])
    if not served_list:
        return

    # Remove ~30% of served customers randomly
    n_remove = max(1, len(served_list) // 3)
    to_remove = random.sample(served_list, min(n_remove, len(served_list)))

    for cust in to_remove:
        for k in K:
            if cust in sol["routes"][k]:
                sol["routes"][k].remove(cust)
                break
        sol["served"].discard(cust)
        sol["unserved"].add(cust)

    # Try to insert some unserved customers
    unserved_list = list(sol["unserved"])
    random.shuffle(unserved_list)
    for cust in unserved_list[:n_remove * 2]:
        if deadline is not None and time.time() >= deadline:
            break
        best_k = None
        best_pos = None
        best_delta = float("inf")
        for k in K:
            route = sol["routes"][k]
            if route_weight(route, weights) + weights[cust] > Q:
                continue
            pos, delta = cheapest_insertion(cust, route, costs)
            if pos is not None and delta < best_delta:
                trial_items = route_items(route) + [cust]
                if check_packing_cached(trial_items, instance, deadline=deadline):
                    best_delta = delta
                    best_k = k
                    best_pos = pos

        if best_k is not None and revenues[cust] - best_delta > 0:
            sol["routes"][best_k].insert(best_pos, cust)
            sol["served"].add(cust)
            sol["unserved"].discard(cust)


# -------------------------
# Plotting utilities (shared)
# -------------------------
def _box_faces(x, y, z, dx, dy, dz):
    p000 = (x,     y,     z)
    p100 = (x+dx,  y,     z)
    p010 = (x,     y+dy,  z)
    p110 = (x+dx,  y+dy,  z)
    p001 = (x,     y,     z+dz)
    p101 = (x+dx,  y,     z+dz)
    p011 = (x,     y+dy,  z+dz)
    p111 = (x+dx,  y+dy,  z+dz)
    return [
        [p000, p100, p110, p010],
        [p001, p101, p111, p011],
        [p000, p100, p101, p001],
        [p010, p110, p111, p011],
        [p000, p010, p011, p001],
        [p100, p110, p111, p101],
    ]

def _item_color(i, cmap_name="tab20"):
    cmap = plt.get_cmap(cmap_name)
    return cmap((i - 1) % cmap.N)

def plot_all_tours_from_solution(instance, sol, out_path=None):
    """Plot tours from the tabu solution representation."""
    coords = instance["coords"]
    plt.figure()
    xs = [coords[i][0] for i in coords]
    ys = [coords[i][1] for i in coords]
    plt.scatter(xs, ys, c="gray", s=20, zorder=1)

    # highlight served
    for i in sol["served"]:
        plt.scatter(coords[i][0], coords[i][1], c="green", s=50, zorder=2)
    # highlight depot
    plt.scatter(coords[0][0], coords[0][1], c="red", s=100, marker="s", zorder=3)

    for i, (cx, cy) in coords.items():
        plt.text(cx, cy, str(i), fontsize=7)

    for k, route in sol["routes"].items():
        if len(route) > 2:
            rx = [coords[i][0] for i in route]
            ry = [coords[i][1] for i in route]
            plt.plot(rx, ry, label=f"truck {k}")

    plt.legend()
    plt.title("Tabu Search — Selected tours")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

def plot_all_tours(instance, routes_by_k, out_path=None):
    coords = instance["coords"]
    plt.figure()
    xs = [coords[i][0] for i in coords]
    ys = [coords[i][1] for i in coords]
    plt.scatter(xs, ys)
    for i, (cx, cy) in coords.items():
        plt.text(cx, cy, str(i), fontsize=9)
    for k, route in routes_by_k.items():
        if route and len(route) >= 2:
            rx = [coords[i][0] for i in route]
            ry = [coords[i][1] for i in route]
            plt.plot(rx, ry, label=f"truck {k}")
    if any(routes_by_k[k] for k in routes_by_k):
        plt.legend()
    plt.title("Selected tours (multi-truck)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    if out_path:
        dir_name = os.path.dirname(out_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

def plot_packing_for_vehicle(k, packed_items, container_dims, out_path=None):
    L, W, H = container_dims
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    edges = [
        ((0,0,0),(L,0,0)), ((L,0,0),(L,W,0)), ((L,W,0),(0,W,0)), ((0,W,0),(0,0,0)),
        ((0,0,H),(L,0,H)), ((L,0,H),(L,W,H)), ((L,W,H),(0,W,H)), ((0,W,H),(0,0,H)),
        ((0,0,0),(0,0,H)), ((L,0,0),(L,0,H)), ((L,W,0),(L,W,H)), ((0,W,0),(0,W,H))
    ]
    for (x1,y1,z1),(x2,y2,z2) in edges:
        ax.plot([x1,x2],[y1,y2],[z1,z2])
    for item in packed_items:
        i = item["i"]
        (px, py, pz) = item["pos"]
        (dx, dy, dz) = item["dims"]
        faces = _box_faces(px, py, pz, dx, dy, dz)
        color = _item_color(i)
        poly = Poly3DCollection(
            faces, facecolors=[color],
            edgecolors=[mcolors.to_rgba("black", 0.25)],
            linewidths=0.5, alpha=0.35
        )
        ax.add_collection3d(poly)
        cx, cy, cz = px + dx/2, py + dy/2, pz + dz/2
        ax.text(cx, cy, cz, str(i), size=9)
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    ax.set_box_aspect((L, W, H))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"3D packing - truck {k}")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

def extract_packing_solution_by_vehicle(instance, vars_dict):
    n = instance["n"]
    m = instance["m"]
    vehicles = instance["vehicles"]
    items = instance["items"]
    N = list(range(1, n + 1))
    K = list(range(1, m + 1))
    y = vars_dict["y"]
    p_x = vars_dict["p_x"]
    p_y = vars_dict["p_y"]
    p_z = vars_dict["p_z"]
    o = vars_dict["o"]
    R_i = vars_dict["R_i"]
    packed_by_k = {k: [] for k in K}
    for i in N:
        for k in K:
            if y[(i, k)].X > 0.5:
                chosen = None
                for ridx in range(len(R_i[i])):
                    if o[(i, k, ridx)].X > 0.5:
                        chosen = R_i[i][ridx]
                        break
                if chosen is None:
                    chosen = items[i]
                packed_by_k[k].append({
                    "i": i,
                    "pos": (p_x[(i, k)].X, p_y[(i, k)].X, p_z[(i, k)].X),
                    "dims": chosen,
                    "container": vehicles[k]
                })
    return packed_by_k

def extract_routes_by_vehicle(instance, vars_dict):
    n = instance["n"]
    m = instance["m"]
    N0 = list(range(0, n + 1))
    K = list(range(1, m + 1))
    x = vars_dict["x"]
    v = vars_dict["v"]
    routes = {k: [] for k in K}
    for k in K:
        if v[k].X < 0.5:
            continue
        arcs = [(i, j) for i in N0 for j in N0 if i != j and x[(i, j, k)].X > 0.5]
        if not arcs:
            continue
        succ = {}
        for i, j in arcs:
            succ[i] = j
        route = [0]
        cur = 0
        seen = set([0])
        while cur in succ:
            nxt = succ[cur]
            route.append(nxt)
            cur = nxt
            if cur in seen:
                break
            seen.add(cur)
            if cur == 0:
                break
        routes[k] = route
    return routes

# -------------------------
# main
# -------------------------
def main(instance_module=None, scenario="P3", timelimit=None, diversify_threshold=None):
    if timelimit is None:
        timelimit = TIME_LIMIT
    if diversify_threshold is None:
        diversify_threshold = globals()["diversify_threshold"]
    if instance_module is None:
        instance_module = cvrp01
    t_start = time.time()

    instance = generate_instance(
        instance=instance_module,
        gamma=GAMMA,
        scenario=scenario,
    )

    print("=" * 60)
    print("3DP-CPTP Tabu Search (Gendreau et al. 2006 style)")
    print(f"Instance: n={instance['n']}, m={instance['m']}, Q={instance['Q']}")
    print(f"Container: {list(instance['vehicles'].values())[0]}")
    print("=" * 60)

    # ---- Phase 1: Tabu Search ----
    best_sol, best_obj = tabu_search(
        instance,
        max_iterations=MAX_ITERATIONS,
        tabu_tenure=TABU_TENURE,
        time_limit=timelimit,
        diversify_threshold=diversify_threshold,
        verbose=True
    )

    elapsed = time.time() - t_start

    print(f"\nTotal running time: {elapsed:.2f} seconds")
    print(f"Final objective: {best_obj:.2f}")

    # Print route details
    used_trucks = [k for k, route in best_sol["routes"].items() if len(route) > 2]
    print(f"Used trucks: {used_trucks}")
    print(f"Served customers ({len(best_sol['served'])}): {sorted(best_sol['served'])}")

    # Solve packing for each used truck to get positions & oriented dims
    packed_by_k = {}
    for k in used_trucks:
        items_on_route = route_items(best_sol["routes"][k])
        packed_by_k[k] = solve_packing(items_on_route, instance)

    for k in used_trucks:
        route = best_sol["routes"][k]
        print(f"Route truck {k}: {route}")
        total_weight = sum(instance["weights"][i] for i in route_items(route))
        print(f"Truck {k} weight utilization: {total_weight} / {instance['Q']}")

        packed = packed_by_k.get(k, [])
        print(f"\nPacked items for vehicle {k} (i, pos, dims):")
        for it in packed:
            print(it["i"], it["pos"], it["dims"])
        if packed:
            Lk, Wk, Hk = instance["vehicles"][k]
            cap = Lk * Wk * Hk
            vol = sum(it["dims"][0] * it["dims"][1] * it["dims"][2] for it in packed)
            print(f"Oriented-volume utilization: {round(vol / cap, 3)}")

    # Plot tabu solution routes
    #plot_all_tours_from_solution(instance, best_sol)

    results_dir = Path(__file__).parent / "results"
    os.chdir(results_dir)
    with open('T-' + str(instance["name"]) + '-' + str(instance["scenario"]) + '-' + str(timelimit) + 's-' + 'v' + str(TABU_TENURE) + '-p' + str(P_NEAREST) + '-' + 'd' + str(diversify_threshold) + '-' + 't' + str(PACKING_TIME_LIMIT) + '-' + '.lsg', 'w') as f:
        f.write(f"Runtime: {elapsed}\n")
        f.write(f"Objective: {best_obj}\n")
        f.write(f"Used vehicles: {used_trucks}\n")
        f.write(f"Served customers ({len(best_sol['served'])}): {sorted(best_sol['served'])}\n")

        used_trucks = [k for k, route in best_sol["routes"].items() if len(route) > 2]
        Qk = instance["Q"]
        for k in used_trucks:
            route = best_sol["routes"][k]
            f.write(f"Route vehicle {k}: {route}\n")
            total_weight = sum(instance["weights"][i] for i in route_items(route))
            f.write(f"Vehicle {k} weight utilization: {total_weight} / {Qk}\n")
            packed = packed_by_k.get(k, [])
            f.write(f"Packed items for vehicle {k} (i, pos, dims):\n")
            for it in packed:
                f.write(f"{it['i']} {it['pos']} {it['dims']}\n")
            if packed:
                Lk, Wk, Hk = instance["vehicles"][k]
                cap = Lk * Wk * Hk
                vol = sum(it["dims"][0] * it["dims"][1] * it["dims"][2] for it in packed)
                f.write(f"Oriented-volume utilization: {round(vol / cap, 3)}\n")

if __name__ == "__main__":
    main()
