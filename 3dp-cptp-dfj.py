# 3DP-CPTP (OF: maximize value in one container) so it is a knapsack problem (= packing problem in literature)

# Here we extend the model that a customer can have multiple items as in Gendreau et al. 2006 instances.
# Every item is specified and can have shared locations (same customer)

# HOW to set the revenues? Current: uniform 100

# Instance in line 600 defined

import math
import os
from itertools import permutations

from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from instances import cvrp01, cvrp02, cvrp03, cvrp04, cvrp05, cvrp06, cvrp07, cvrp08, cvrp09, cvrp10, cvrp11, cvrp12, cvrp13, cvrp14, cvrp15, cvrp16, cvrp17, cvrp18, cvrp19, cvrp20, cvrp21, cvrp22, cvrp23, cvrp24, cvrp25, cvrp26, cvrp27
from instances_cutted import ccvrp01, ccvrp02, ccvrp03, ccvrp04, ccvrp05
from instances2D import cvrp0101, cvrp1001, cvrp3605
timelimit = 30 #seconds

# -------------------------
# Based on instance: 3l_cvrp01.txt from Gendreau et al. 2006
# -------------------------
def generate_instance(
    instance,
    gamma=1.0, # travel cost factor
    scenario="P1",
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
    }

# -------------------------
# DFJ subtour elimination callback + subtour finder
# -------------------------
def subtour(edges, reallist, include_depot=False):

    # Make a mutable copy
    unvisited = list(reallist)
    # initial cycle large
    cycle = list(range(len(reallist) + 1))
    while unvisited:
        thiscycle = []
        # neighbors = unvisited copy
        neighbors = list(unvisited)
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            # remove from unvisited
            if current in unvisited:
                unvisited.remove(current)
            # find outgoing neighbors in edges among the remaining unvisited nodes
            neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
        # decide if thiscycle is the new shortest
        if include_depot:
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        else:
            # only update if thiscycle doesn't contain depot (0)
            if len(cycle) > len(thiscycle) and (0 not in thiscycle):
                cycle = thiscycle
    # If cycle still default-length, return empty
    if len(cycle) == len(reallist) + 1:
        return []
    return cycle

def subtourelim(model, where):

    if where == GRB.Callback.MIPSOL:
        # get solution values for all x variables (in same order as model._x_vars)
        vals = model.cbGetSolution(model._x_vars)  # returns a list aligned with _x_vars

        # build mapping key -> value for quick lookup
        valdict = {}
        for idx, key in enumerate(model._x_keys):
            valdict[key] = vals[idx]

        K = list(model._K)  # vehicle indices
        # Build selected edges per vehicle (as gp.tuplelist)
        selected = {k: gp.tuplelist() for k in K}
        for (i, j, k) in model._x_keys:
            if valdict.get((i, j, k), 0.0) > 0.5:
                selected[k].append((i, j))

        # For each vehicle, detect subtours and add lazy constraints
        for k in K:
            # collect visited nodes by this vehicle (both origins and destinations)
            used_nodes = list({v for e in selected[k] for v in e})
            if len(used_nodes) == 0:
                continue

            # find shortest subtour in selected edges for this vehicle
            tour = subtour(selected[k], used_nodes, include_depot=False)

            # If subtour is found and it's a true subtour (size < number of visited) add lazy cut
            if tour and (len(tour) < len(used_nodes) or 0 not in tour):
                # Build lazy constraint: sum_{i in S, j in S, i!=j} x_{i,j,k} <= |S|-1
                expr = gp.quicksum(
                    model._x_dict[(i, j, k)]
                    for i in tour for j in tour if i != j
                )
                model.cbLazy(expr <= len(tour) - 1)
                # Note: we add one cut per subtour per vehicle. If multiple subtours exist,
                # the callback will be invoked again for the next solution and eliminate them.

# -------------------------
# helper: unique orientations
# -------------------------
def unique_orientations(l, w, h):
    """Unique permutations of (l,w,h). Thus there are 6 possible orientations. Each orientation corresponds to choosing a different side as the base and a different dimension as the height.
    But Gendreau 2006: We assume that the boxes can be rotated by 90 degrees on the horizontal plane,
    while upside-down rotations are not allowed. Thus they only allow 2 possible orientations""" 
    return list(dict.fromkeys(permutations([l, w, h], 3)))


# -------------------------
# Multi-truck model builder (WITHOUT MTZ)
# -------------------------
def build_multi_truck_model(instance, verbose=True, symmetry_breaking=True):
    n = instance["n"]
    m = instance["m"]
    coords = instance["coords"]
    vehicles = instance["vehicles"]
    items = instance["items"]
    r = instance["revenues"]
    weights = instance["weights"]
    c = instance["costs"]

    N = list(range(1, n + 1)) #items/customers
    N0 = [0] + N #including depot
    K = list(range(1, m + 1)) #vehicles
    Q = instance["Q"] #weight capacity per vehicle

    # orientations per item
    R_i = {i: unique_orientations(*items[i]) for i in N}

    # big-M per axis
    Mx = max(vehicles[k][0] for k in K)
    My = max(vehicles[k][1] for k in K)
    Mz = max(vehicles[k][2] for k in K)

    model = gp.Model("3DP-CPTP")
    if not verbose:
        model.Params.OutputFlag = 0

    # -------------------------
    # Variables
    # -------------------------
    x = model.addVars(
        [(i, j, k) for i in N0 for j in N0 if i != j for k in K],
        vtype=GRB.BINARY, name="x") # arc (i,j) used by vehicle k

    y = model.addVars([(i, k) for i in N for k in K], vtype=GRB.BINARY, name="y") # item i packed in vehicle k
    z = model.addVars(N, vtype=GRB.BINARY, name="z") # customer served or not

    v = model.addVars(K, vtype=GRB.BINARY, name="v") # vehicle used or not

    # NOTE: MTZ vars removed (u)

    p_x = model.addVars([(i, k) for i in N for k in K], lb=0.0, name="p_x") # position of items in x-coordinate
    p_y = model.addVars([(i, k) for i in N for k in K], lb=0.0, name="p_y")
    p_z = model.addVars([(i, k) for i in N for k in K], lb=0.0, name="p_z")

    # orientation binaries
    o = {}
    for i in N:
        for k in K:
            for ridx in range(len(R_i[i])):
                o[(i, k, ridx)] = model.addVar(vtype=GRB.BINARY, name=f"o[{i},{k},{ridx}]")
    o = gp.tupledict(o)

    # non-overlap precedence
    a_x = model.addVars([(i, j, k) for i in N for j in N if i != j for k in K],
                        vtype=GRB.BINARY, name="a_x")
    a_y = model.addVars([(i, j, k) for i in N for j in N if i != j for k in K],
                        vtype=GRB.BINARY, name="a_y")
    a_z = model.addVars([(i, j, k) for i in N for j in N if i != j for k in K],
                        vtype=GRB.BINARY, name="a_z")

    # -------------------------
    # Helper expressions
    # -------------------------
    def size_expr(i, k, axis):
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        return gp.quicksum(R_i[i][ridx][idx] * o[(i, k, ridx)] for ridx in range(len(R_i[i])))

    # -------------------------
    # Objective
    # -------------------------
    model.setObjective(
        gp.quicksum(r[i] * z[i] for i in N)
        - gp.quicksum(c[(i, j)] * x[(i, j, k)] for i in N0 for j in N0 if i != j for k in K),
        GRB.MAXIMIZE
    )

    # -------------------------
    # 1) Customer selection
    # -------------------------
    for i in N:
        model.addConstr(gp.quicksum(y[(i, k)] for k in K) == z[i], name=f"select[{i}]")

    # -------------------------
    # 2) Vehicle routing
    # -------------------------
    for i in N:
        for k in K:
            model.addConstr(
                gp.quicksum(x[(i, j, k)] for j in N0 if j != i) == y[(i, k)],
                name=f"outflow[{i},{k}]"
            )
            model.addConstr(
                gp.quicksum(x[(j, i, k)] for j in N0 if j != i) == y[(i, k)],
                name=f"inflow[{i},{k}]"
            )

            # only assign if vehicle active
            model.addConstr(y[(i, k)] <= v[k], name=f"assign_active[{i},{k}]")

    for k in K:
        model.addConstr(gp.quicksum(x[(0, j, k)] for j in N) == v[k], name=f"depart[{k}]")
        model.addConstr(gp.quicksum(x[(i, 0, k)] for i in N) == v[k], name=f"return[{k}]")

    # Arc–customer linking
    for k in K:
        for i in N0:
            for j in N0:
                if i == j:
                    continue
                if i != 0:
                    model.addConstr(x[(i, j, k)] <= y[(i, k)], name=f"link_i[{i},{j},{k}]")
                if j != 0:
                    model.addConstr(x[(i, j, k)] <= y[(j, k)], name=f"link_j[{i},{j},{k}]")

    # -------------------------
    # 3.5) Weight Capacity
    # -------------------------
    for k in K:
        model.addConstr(
            gp.quicksum(weights[i] * y[(i, k)] for i in N) <= Q,
            name=f"weight_cap[{k}]"
        )

    # -------------------------
    # 3.6) Volume Capacity (valid inequality)
    # -------------------------
    for k in K:
        Lk, Wk, Hk = vehicles[k]
        container_volume = Lk * Wk * Hk
        model.addConstr(
            gp.quicksum(items[i][0] * items[i][1] * items[i][2] * y[(i, k)] for i in N) <= container_volume,
            name=f"vol_cap[{k}]"
        )

    # -------------------------
    # 4) 3D Orientation
    # -------------------------
    for i in N:
        for k in K:
            model.addConstr(
                gp.quicksum(o[(i, k, ridx)] for ridx in range(len(R_i[i]))) == y[(i, k)],
                name=f"orient[{i},{k}]"
            )

    # -------------------------
    # 5) Container bounds
    # -------------------------
    for i in N:
        for k in K:
            Lk, Wk, Hk = vehicles[k]
            model.addConstr(p_x[(i, k)] + size_expr(i, k, "x") <= Lk, name=f"bound_x[{i},{k}]")
            model.addConstr(p_y[(i, k)] + size_expr(i, k, "y") <= Wk, name=f"bound_y[{i},{k}]")
            model.addConstr(p_z[(i, k)] + size_expr(i, k, "z") <= Hk, name=f"bound_z[{i},{k}]")

    # -------------------------
    # 6) Tight position upper bounds
    # -------------------------
    for i in N:
        for k in K:
            Lk, Wk, Hk = vehicles[k]
            model.addConstr(p_x[(i, k)] <= Lk - size_expr(i, k, "x"), name=f"pos_ub_x[{i},{k}]")
            model.addConstr(p_y[(i, k)] <= Wk - size_expr(i, k, "y"), name=f"pos_ub_y[{i},{k}]")
            model.addConstr(p_z[(i, k)] <= Hk - size_expr(i, k, "z"), name=f"pos_ub_z[{i},{k}]")

    # -------------------------
    # 7) Non-overlap constraints
    # -------------------------
    # Activation
    for k in K:
        for i in N:
            for j in N:
                if i == j:
                    continue
                model.addConstr(a_x[(i, j, k)] <= y[(i, k)], name=f"act_ax_i[{i},{j},{k}]")
                model.addConstr(a_x[(i, j, k)] <= y[(j, k)], name=f"act_ax_j[{i},{j},{k}]")
                model.addConstr(a_y[(i, j, k)] <= y[(i, k)], name=f"act_ay_i[{i},{j},{k}]")
                model.addConstr(a_y[(i, j, k)] <= y[(j, k)], name=f"act_ay_j[{i},{j},{k}]")
                model.addConstr(a_z[(i, j, k)] <= y[(i, k)], name=f"act_az_i[{i},{j},{k}]")
                model.addConstr(a_z[(i, j, k)] <= y[(j, k)], name=f"act_az_j[{i},{j},{k}]")

    pairs = [(i, j) for i in N for j in N if i < j]

    # Direction exclusivity and separating axis
    for k in K:
        for i, j in pairs:
            model.addConstr(a_x[(i, j, k)] + a_x[(j, i, k)] <= 1, name=f"excl_x[{i},{j},{k}]")
            model.addConstr(a_y[(i, j, k)] + a_y[(j, i, k)] <= 1, name=f"excl_y[{i},{j},{k}]")
            model.addConstr(a_z[(i, j, k)] + a_z[(j, i, k)] <= 1, name=f"excl_z[{i},{j},{k}]")

            model.addConstr(
                a_x[(i, j, k)] + a_x[(j, i, k)]
                + a_y[(i, j, k)] + a_y[(j, i, k)]
                + a_z[(i, j, k)] + a_z[(j, i, k)]
                >= y[(i, k)] + y[(j, k)] - 1,
                name=f"sep_axis[{i},{j},{k}]"
            )

    # Axis-wise non-overlap (ordered)
    for k in K:
        for i in N:
            for j in N:
                if i == j:
                    continue
                model.addConstr(
                    p_x[(i, k)] + size_expr(i, k, "x")
                    <= p_x[(j, k)] + Mx * (1 - a_x[(i, j, k)]),
                    name=f"nox[{i},{j},{k}]"
                )
                model.addConstr(
                    p_y[(i, k)] + size_expr(i, k, "y")
                    <= p_y[(j, k)] + My * (1 - a_y[(i, j, k)]),
                    name=f"noy[{i},{j},{k}]"
                )
                model.addConstr(
                    p_z[(i, k)] + size_expr(i, k, "z")
                    <= p_z[(j, k)] + Mz * (1 - a_z[(i, j, k)]),
                    name=f"noz[{i},{j},{k}]"
                )

    # -------------------------
    # 8) Optional symmetry breaking on vehicles
    # -------------------------
    if symmetry_breaking and m > 1:
        for k_idx in range(1, m):
            model.addConstr(v[k_idx] >= v[k_idx + 1], name=f"sym[{k_idx}]")

    model.update()

    # Attach helpful handles to model for the callback
    # _x_vars: list of Var objects (in stable order)
    # _x_keys: list of keys (i,j,k) aligned with _x_vars
    # _x_dict: direct mapping (i,j,k) -> Var (for building lazy expressions)
    model._x_keys = list(x.keys())
    model._x_vars = [x[key] for key in model._x_keys]
    model._x_dict = x
    model._K = K

    vars_dict = {
        "x": x, "y": y, "z": z, "v": v,
        "p_x": p_x, "p_y": p_y, "p_z": p_z,
        "o": o, "R_i": R_i
    }
    return model, vars_dict

# -------------------------
# Solution extraction
# -------------------------
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

        # reconstruct from depot
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
# Plotting utilities
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

def plot_all_tours(instance, routes_by_k, out_path=None):
    coords = instance["coords"]

    plt.figure()
    xs = [coords[i][0] for i in coords]
    ys = [coords[i][1] for i in coords]
    plt.scatter(xs, ys)

    labels_by_coord = {}
    for item_id, coord in coords.items():
        labels_by_coord.setdefault(coord, []).append(item_id)

    for (x, y), item_ids in labels_by_coord.items():
        label = ",".join(str(item_id) for item_id in sorted(item_ids))
        plt.text(x, y, label, fontsize=9)

    # plot each vehicle route
    for k, route in routes_by_k.items():
        if route and len(route) >= 2:
            rx = [coords[i][0] for i in route]
            ry = [coords[i][1] for i in route]
            plt.plot(rx, ry, label=f"vehicle {k}")

    if any(routes_by_k[k] for k in routes_by_k):
        plt.legend()

    plt.title("Selected tours (multi-vehicle)")
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

# -------------------------
# UPDATED: Color helper for Matplotlib
# -------------------------
def _item_color(i, cmap_name="tab20"):
    """
    Stable color per item id using a matplotlib categorical colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    return cmap((i - 1) % cmap.N)

# -------------------------
# UPDATED: Matplotlib 3D packing with different colors per box
# -------------------------
def plot_packing_for_vehicle(k, packed_items, container_dims, out_path=None):
    L, W, H = container_dims

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Container edges
    edges = [
        ((0,0,0),(L,0,0)), ((L,0,0),(L,W,0)), ((L,W,0),(0,W,0)), ((0,W,0),(0,0,0)),
        ((0,0,H),(L,0,H)), ((L,0,H),(L,W,H)), ((L,W,H),(0,W,H)), ((0,W,H),(0,0,H)),
        ((0,0,0),(0,0,H)), ((L,0,0),(L,0,H)), ((L,W,0),(L,W,H)), ((0,W,0),(0,W,H))
    ]
    for (x1,y1,z1),(x2,y2,z2) in edges:
        ax.plot([x1,x2],[y1,y2],[z1,z2])

    # Items with distinct colors
    for item in packed_items:
        i = item["i"]
        (x, y, z) = item["pos"]
        (dx, dy, dz) = item["dims"]

        faces = _box_faces(x, y, z, dx, dy, dz)

        color = _item_color(i)
        poly = Poly3DCollection(
            faces,
            facecolors=[color],
            edgecolors=[mcolors.to_rgba("black", 0.25)],
            linewidths=0.5,
            alpha=0.35
        )
        ax.add_collection3d(poly)

        cx, cy, cz = x + dx/2, y + dy/2, z + dz/2
        ax.text(cx, cy, cz, str(i), size=9)

    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    ax.set_box_aspect((L, W, H))  # preserve real container proportions
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"3D packing - vehicle {k}")

    if out_path:
        dir_name = os.path.dirname(out_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def volume_utilization_for_vehicle(instance, packed_items):
    if not packed_items:
        return 0.0
    L, W, H = packed_items[0]["container"]
    cap = L * W * H
    vol = sum(it["dims"][0] * it["dims"][1] * it["dims"][2] for it in packed_items)
    return vol / cap

# -------------------------
# main
# -------------------------
def main(instance_module=None, scenario="P3", timelimit=None):
    if timelimit is None:
        timelimit = globals().get('timelimit', 30)
    if instance_module is None:
        instance_module = cvrp1001
    instance = generate_instance(
        instance=instance_module,
        gamma=1.0, # travel cost factor
        scenario=scenario,
    )

    model, vars_dict = build_multi_truck_model(instance, verbose=True, symmetry_breaking=True)

    # Tell Gurobi we'll be using lazy constraints in the callback
    model.Params.LazyConstraints = 1

    model.Params.TimeLimit = timelimit
    #model.Params.MIPGap = 0.02

    # Optimize with the DFJ callback
    model.optimize(subtourelim)

    status = model.Status
    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        print("\nObjective:", model.ObjVal)

        used_trucks = [k for k in range(1, instance["m"] + 1) if vars_dict["v"][k].X > 0.5]
        print("Used vehicles:", used_trucks)

        served = [i for i in range(1, instance["n"] + 1) if vars_dict["z"][i].X > 0.5]
        print("Served customers:", served)

        # Extended output: non-served customers with extra details
        non_served = [i for i in range(1, instance["n"] + 1) if vars_dict["z"][i].X <= 0.5]
        print("Non-Served customers:", non_served)
        # if non_served:
        #     print("\nNon-served customers (extended info):")
        #     for i in non_served:
        #         wt = instance["weights"].get(i, None)
        #         dims = instance["items"].get(i, None)
        #         rev = instance["revenues"].get(i, None)
        #         coord = instance["coords"].get(i, None)
        #         depot_coord = instance["coords"].get(0, None)
        #         dist_to_depot = None
        #         if coord is not None and depot_coord is not None:
        #             try:
        #                 dist_to_depot = round(math.dist(depot_coord, coord), 3)
        #             except Exception:
        #                 dist_to_depot = None

        #         print(f"Customer {i}: weight={wt}, dims={dims}, revenue={rev}, coords={coord}, dist_to_depot={dist_to_depot}")
        # else:
        #     print("\nAll customers served.")
        # Routes
        routes_by_k = extract_routes_by_vehicle(instance, vars_dict)
        for k in range(1, instance["m"] + 1):
            if routes_by_k.get(k):
                print(f"Route vehicle {k}:", routes_by_k[k])

        plot_all_tours(instance, routes_by_k)

        # Weight utilization per vehicle
        for k in used_trucks:
            total_weight = sum(
                instance["weights"][i]
                for i in range(1, instance["n"] + 1)
                if vars_dict["y"][(i, k)].X > 0.5
            )
            Qk = instance["Q"]
            print(f"Vehicle {k} weight utilization: {total_weight} / {Qk}")

        # Packing per vehicle
        packed_by_k = extract_packing_solution_by_vehicle(instance, vars_dict)

        for k in used_trucks:
            packed = packed_by_k.get(k, [])
            print(f"\nPacked items for vehicle {k} (i, pos, dims):")
            for it in packed:
                print(it["i"], it["pos"], it["dims"])

            if packed:
                Lk, Wk, Hk = instance["vehicles"][k]
                cap = Lk * Wk * Hk
                vol = sum(it["dims"][0] * it["dims"][1] * it["dims"][2] for it in packed)
                print("Oriented-volume utilization:", round(vol / cap, 3))

            # Matplotlib colored plot
            plot_packing_for_vehicle(k, packed, instance["vehicles"][k])

    else:
        print("No feasible/usable solution. Status code:", status)

    results_dir = Path(__file__).parent / "results"
    os.chdir(results_dir)
    with open('D-' + str(instance["name"]) + '-' + str(instance["scenario"]) + '-' + str(timelimit) + 's-' + '.lsg', 'w') as f:
        f.write(f"GAP: {model.MIPGap}\n")
        f.write(f"Runtime: {model.Runtime}\n")
        f.write(f"Objective: {model.ObjVal}\n")
        f.write(f"Used vehicles: {used_trucks}\n")
        f.write(f"Served customers: {served}\n")
        f.write(f"Non-Served customers: {non_served}\n")
        for k in range(1, instance["m"] + 1):
            if routes_by_k.get(k):
                f.write(f"Route vehicle {k}: {routes_by_k[k]}\n")
        Qk = instance["Q"]
        for k in used_trucks:
            total_weight = sum(
                instance["weights"][i]
                for i in range(1, instance["n"] + 1)
                if vars_dict["y"][(i, k)].X > 0.5
            )
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


