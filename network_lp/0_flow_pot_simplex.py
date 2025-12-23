import re
from math import inf

def parse_graph(file_path):
    edges = {}
    all_nodes = set()
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('[source'):
                    continue

                match = re.match(r'(\d+)-(\d+):\s*\(\s*(\d+),\s*(\d+)\s*\)', line)
                if match:
                    u, v, cost, capacity = map(int, match.groups())
                    edges[(u, v)] = {'cost': cost, 'capacity': capacity}
                    all_nodes.add(u)
                    all_nodes.add(v)
    except FileNotFoundError:
        return None, None
    
    return edges, sorted(list(all_nodes))

def parse_node_costs(file_path):
    node_costs = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or 'node costs' in line.lower():
                    continue
                
                match = re.match(r'(\d+):\s*(-?\d+)', line)
                if match:
                    node, cost = map(int, match.groups())
                    node_costs[node] = cost
    except FileNotFoundError:
        return None
    
    return node_costs

def parse_partition(file_path):
    T = set()
    U = set()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            t_match = re.search(r'T\s*=\s*\{([^}]+)\}', content)
            if t_match:
                t_content = t_match.group(1)
                for edge in re.findall(r'\((\d+),\s*(\d+)\)', t_content):
                    T.add((int(edge[0]), int(edge[1])))
            u_match = re.search(r'U\s*=\s*\{([^}]+)\}', content)
            if u_match:
                u_content = u_match.group(1)
                for edge in re.findall(r'\((\d+),\s*(\d+)\)', u_content):
                    U.add((int(edge[0]), int(edge[1])))
    except FileNotFoundError:
        return None, None
    
    return T, U

def calculate_flow_tree(edges, nodes, node_costs, T, U):
    """Calcola il flusso iniziale risolvendo il sistema lineare sull'albero."""
    flow = {}
    
    for edge in edges:
        if edge in U:
            flow[edge] = edges[edge]['capacity']
        elif edge in T:
            flow[edge] = None
        else:
            flow[edge] = 0
    
    T_list = list(T)
    n_vars = len(T_list)
    edge_to_idx = {edge: i for i, edge in enumerate(T_list)}
    
    A = []
    b_vec = []
    
    for node in nodes[:-1]:
        equation = [0] * n_vars
        rhs = node_costs.get(node, 0)
        
        for edge in edges:
            u, v = edge
            f = flow[edge]
            
            if edge in T:
                idx = edge_to_idx[edge]
                if v == node:
                    equation[idx] = 1
                elif u == node:
                    equation[idx] = -1
            else:
                if f is not None:
                    if v == node:
                        rhs -= f
                    elif u == node:
                        rhs += f
        
        A.append(equation)
        b_vec.append(rhs)
    
    for i in range(len(A)):
        A[i].append(b_vec[i])
    
    for col in range(min(len(A), n_vars)):
        max_row = col
        for row in range(col + 1, len(A)):
            if abs(A[row][col]) > abs(A[max_row][col]):
                max_row = row
        A[col], A[max_row] = A[max_row], A[col]
        if abs(A[col][col]) < 1e-10:
            continue
        for row in range(col + 1, len(A)):
            if abs(A[row][col]) > 1e-10:
                factor = A[row][col] / A[col][col]
                for j in range(n_vars + 1):
                    A[row][j] -= factor * A[col][j]
    
    x = [0] * n_vars
    for i in range(min(len(A), n_vars) - 1, -1, -1):
        if abs(A[i][i]) > 1e-10:
            x[i] = A[i][n_vars]
            for j in range(i + 1, n_vars):
                x[i] -= A[i][j] * x[j]
            x[i] /= A[i][i]
    
    for i, edge in enumerate(T_list):
        flow[edge] = round(x[i])
    
    return flow

def calculate_potential(edges, nodes, T):
    potential = {node: None for node in nodes}
    potential[min(nodes)] = 0
    
    max_iterations = 100
    for _ in range(max_iterations):
        updated = False
        for edge in T:
            i, j = edge
            cost = edges[edge]['cost']
            
            if potential[i] is not None and potential[j] is None:
                potential[j] = potential[i] + cost
                updated = True
            elif potential[j] is not None and potential[i] is None:
                potential[i] = potential[j] - cost
                updated = True
        
        if not updated:
            break
    
    return potential

def check_flow_feasibility(flow, node_costs, edges, nodes):
    violations = []
    
    for node in nodes:
        inflow = 0
        outflow = 0
        
        for edge in edges:
            u, v = edge
            if v == node:
                inflow += flow.get(edge, 0)
            if u == node:
                outflow += flow.get(edge, 0)
        
        balance = inflow - outflow
        expected_balance = node_costs.get(node, 0)
        
        if abs(balance - expected_balance) > 0.001:
            violations.append(f"Node {node}: balance = {balance:.2f}, expected = {expected_balance}")
    
    return len(violations) == 0, violations

def check_flow_optimality(edges, flow, potential, T, U):
    violations = []
    
    for edge in edges:
        i, j = edge
        cost = edges[edge]['cost']
        pi_i = potential.get(i)
        pi_j = potential.get(j)
        
        if pi_i is None or pi_j is None:
            continue
        
        reduced_cost = cost - (pi_j - pi_i)
        
        if edge in T:
            if abs(reduced_cost) > 0.001:
                violations.append(f"Edge {edge} in T: reduced cost = {reduced_cost:.2f} (should be 0)")
        elif edge in U:
            if reduced_cost > 0.001:
                violations.append(f"Edge {edge} in U: reduced cost = {reduced_cost:.2f} (should be ≤ 0)")
        else:
            if reduced_cost < -0.001:
                violations.append(f"Edge {edge} in L: reduced cost = {reduced_cost:.2f} (should be ≥ 0)")
    
    return len(violations) == 0, violations

def check_degeneracy(flow, potential, T, U, edges):
    flow_degenerate = False
    potential_degenerate = False
    flow_deg_arcs = []
    potential_deg_arcs = []
    
    for edge in T:
        if flow.get(edge, 0) == 0 or flow.get(edge, 0) == edges[edge]['capacity']:
            flow_degenerate = True
            flow_deg_arcs.append(edge)
    
    L = set(edges.keys()) - T - U
    for edge in L:
        i, j = edge
        cost = edges[edge]['cost']
        pi_i = potential.get(i)
        pi_j = potential.get(j)
        
        if pi_i is not None and pi_j is not None:
            reduced_cost = cost - (pi_j - pi_i)
            if abs(reduced_cost) < 0.001:
                potential_degenerate = True
                potential_deg_arcs.append(edge)
    
    return flow_degenerate, flow_deg_arcs, potential_degenerate, potential_deg_arcs

def simplex_iteration(edges, nodes, node_costs, T, U, flow, potential):
    L = set(edges.keys()) - T - U
    entering_edge = None
    
    # choose entering edge
    sorted_edges = sorted(edges.keys())
    for edge in sorted_edges:
        if edge in T:
            continue
            
        u, v = edge
        cost = edges[edge]['cost']
        pi_u = potential[u]
        pi_v = potential[v]
        reduced_cost = cost - (pi_v - pi_u)
        
        if edge in U:
            if reduced_cost > 0.001:
                entering_edge = edge
                break
        else: # edge in L
            if reduced_cost < -0.001:
                entering_edge = edge
                break
    
    if entering_edge is None:
        return None, None, "Optimal solution reached"
    
    print(f"\n>>> ENTERING EDGE: {entering_edge}")
    i, j = entering_edge
    cost = edges[entering_edge]['cost']
    reduced_cost = cost - (potential[j] - potential[i])
    print(f"    Cost: {cost}, Reduced cost: {reduced_cost:.2f}")
    print(f"    Edge is in {'L' if entering_edge in L else 'U'}")
    
    # cycle
    tree_adj = {node: [] for node in nodes}
    for u, v in T:
        tree_adj[u].append(v)
        tree_adj[v].append(u)
    
    start, end = entering_edge
    
    def find_path_with_parent(current, target, visited, path, parent_map):
        if current == target:
            return True
        visited.add(current)
        for neighbor in tree_adj[current]:
            if neighbor not in visited:
                path.append(neighbor)
                parent_map[neighbor] = current
                if find_path_with_parent(neighbor, target, visited, path, parent_map):
                    return True
                path.pop()
                del parent_map[neighbor]
        return False
    
    path = [start]
    parent_map = {start: None}
    find_path_with_parent(start, end, set(), path, parent_map)
    
    print(f"    Path in tree from {start} to {end}: {' -> '.join(map(str, path))}")
    
    cycle_edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if (u, v) in T:
            cycle_edges.append(((u, v), 'backward'))
        elif (v, u) in T:
            cycle_edges.append(((v, u), 'forward'))
    
    cycle_edges.reverse()
    
    print(f"    Cycle formed by adding entering edge {entering_edge}:")
    print(f"      Entering edge {entering_edge}: increase flow by θ")
    for edge, direction in cycle_edges:
        if direction == 'backward':
            print(f"      Edge {edge}: decrease flow by θ (backward)")
        else:
            print(f"      Edge {edge}: increase flow by θ (forward)")
    
    print(f"\n>>> CALCULATE θ+ AND θ-")
    
    # θ+: min(resid entering edge, resid forward edges)
    theta_plus = edges[entering_edge]['capacity'] - flow.get(entering_edge, 0)
    limiting_edge_plus = entering_edge
    
    print(f"    Entering edge {entering_edge}: residual capacity = {theta_plus}")

    for edge, direction in cycle_edges:
        if direction == 'forward':
            residual = edges[edge]['capacity'] - flow.get(edge, 0)
            print(f"    Edge {edge} (forward): residual capacity = {residual}")
            if residual < theta_plus:
                theta_plus = residual
                limiting_edge_plus = edge
                
    print(f"    θ+ = {theta_plus} (edge: {limiting_edge_plus})")
    
    # theta_minus: min(flusso archi backward)
    theta_minus = float('inf')
    limiting_edge_minus = None
    
    for edge, direction in cycle_edges:
        if direction == 'backward':
            limit = flow.get(edge, 0)
            print(f"    Edge {edge} (backward): current flow = {limit}")
            if limit < theta_minus:
                theta_minus = limit
                limiting_edge_minus = edge
    
    if theta_minus == float('inf'):
        print(f"    θ- = ∞ (no backward edge in cycle)")
    else:
        print(f"    θ- = {theta_minus} (edge: {limiting_edge_minus})")
    
    theta = min(theta_plus, theta_minus)
    print(f"\n    θ = min(θ+, θ-) = {theta}")
    
    # update flow
    new_flow = flow.copy()
    
    # update entering edge flow (flow + theta)
    new_flow[entering_edge] += theta
    
    # update cycle flow (+ / - theta)
    for edge, direction in cycle_edges:
        if direction == 'forward':
            new_flow[edge] += theta
        else: # backward
            new_flow[edge] -= theta

    # precision errors fixes            
    for edge in new_flow:
        if abs(new_flow[edge]) < 1e-9: new_flow[edge] = 0
        if abs(new_flow[edge] - edges[edge]['capacity']) < 1e-9: new_flow[edge] = edges[edge]['capacity']

    # update T,L,U
    # θ = θ-: edge becomes empty -> goes in L
    # θ = θ+: edge becomes full -> goes in U

    if theta == theta_minus:
        leaving_edge = limiting_edge_minus
        target_set = 'L'
    else:
        leaving_edge = limiting_edge_plus
        target_set = 'U'

    print(f"\n>>> LEAVING EDGE: {leaving_edge}")
    print(f"    θ: {theta}")
    
    new_T = T.copy()
    new_U = U.copy()
    new_L = L.copy()
    
    if leaving_edge == entering_edge:
        # edge doesn't go in T
        if entering_edge in L:
            new_L.remove(entering_edge)
            new_U.add(entering_edge)
        else:
            new_U.remove(entering_edge)
            new_L.add(entering_edge)
    else:
        # swap leaving and entering edge
        new_T.remove(leaving_edge)
        new_T.add(entering_edge)
        
        # remove entering from old partition
        if entering_edge in L: new_L.remove(entering_edge)
        elif entering_edge in U: new_U.remove(entering_edge)
        
        # remove entering from old partition based on its final flow
        if new_flow[leaving_edge] <= 0.001: # approx
            new_L.add(leaving_edge)
        else:
            new_U.add(leaving_edge)
    
    print(f"\n>>> NEW PARTITION:")
    print(f"    T: {sorted(new_T)}")
    print(f"    U: {sorted(new_U)}")
    print(f"    L: {sorted(new_L)}")
    
    return (new_T, new_U), new_flow, "Iteration completed"

if __name__ == "__main__":
    GRAPH_FILE = 'graph.txt'
    
    edges, nodes = parse_graph(GRAPH_FILE)
    if edges is None:
        print(f"Error: File {GRAPH_FILE} not found.")
        exit(1)
    
    node_costs = parse_node_costs(GRAPH_FILE)
    if node_costs is None:
        print("Error: Unable to read node costs.")
        exit(1)
    
    T, U = parse_partition(GRAPH_FILE)
    if T is None or U is None:
        print("Error: Unable to read partition T and U.")
        exit(1)
    
    L = set(edges.keys()) - T - U
    
    print(f"\nGraph: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Partition:")
    print(f"  T (basic edges): {sorted(T)}")
    print(f"  U (saturated edges):  {sorted(U)}")
    print(f"  L (free edges):  {sorted(L)}")
    
    # initial flow
    flow = calculate_flow_tree(edges, nodes, node_costs, T, U)
    print("\nFLOW (Initial)")
    
    edges_list = sorted(edges.keys())
    flow_vector = []
    
    for edge in edges_list:
        f = flow.get(edge, 0)
        cap = edges[edge]['capacity']
        partition = "T" if edge in T else ("U" if edge in U else "L")
        flow_vector.append(f)
        print(f"Edge {edge}: flow = {f:>3}, capacity = {cap:>2} [{partition}]")
    
    print(f"\nFlow: x = ({', '.join(map(str, flow_vector))})\n")
    
    is_feasible, feasibility_violations = check_flow_feasibility(flow, node_costs, edges, nodes)
    if not is_feasible:
        print("\nWARNING: Calculated flow does NOT satisfy conservation!")
        for v in feasibility_violations:
            print(f"  - {v}")
    
    # initial potential
    potential = calculate_potential(edges, nodes, T)
    
    print("POTENTIAL (Initial)")
    pi_vector = []
    for node in sorted(nodes):
        pi = potential.get(node)
        pi_vector.append(str(pi))
        print(f"Node {node}: π = {pi}")
        
    print(f"\nPotential: π = ({', '.join(pi_vector)})\n")
    
    is_optimal, violations = check_flow_optimality(edges, flow, potential, T, U)
    
    if is_optimal:
        print("\nFlow is OPTIMAL (all optimality conditions satisfied)")
    else:
        print("\nFlow is NOT optimal. Violations:")
        for v in violations:
            print(f"  - {v}")
    
    flow_deg, flow_deg_arcs, pot_deg, pot_deg_arcs = check_degeneracy(flow, potential, T, U, edges)
    
    if flow_deg:
        print(f"Flow is DEGENERATE")
        print(f"    Edges in T with flow = 0 or = capacity: {flow_deg_arcs}")
    else:
        print("Flow is NOT degenerate")
    
    if pot_deg:
        print(f" Potential is DEGENERATE")
        print(f"  Edges in L with reduced cost = 0: {pot_deg_arcs}")
    else:
        print("\nPotential is NOT degenerate")
    
    if not is_optimal:
        print("\n--- SIMPLEX ITERATION ---")
        
        result_partitions, new_flow, msg = simplex_iteration(edges, nodes, node_costs, T, U, flow, potential)
        
        if result_partitions is not None:
            new_T, new_U = result_partitions
            print("\nSimplex iteration completed!")
            print("\nNew partition:")
            print(f"T = {{{', '.join([f'({u},{v})' for u, v in sorted(new_T)])}}}")
            print(f"U = {{{', '.join([f'({u},{v})' for u, v in sorted(new_U)])}}}")
            
            new_potential = calculate_potential(edges, nodes, new_T)
            
            print("\nNew flow:")
            new_flow_vector = []
            for edge in sorted(edges.keys()):
                val = new_flow.get(edge, 0)
                new_flow_vector.append(str(val))
                print(f"  {edge}: {val}")
            
            print(f"Flow: x = ({', '.join(new_flow_vector)})")
            
            print("\nNew potential:")
            new_pot_vector = []
            for node in sorted(nodes):
                val = new_potential.get(node)
                new_pot_vector.append(str(val))
                print(f"  Node {node}: π = {val}")
            
            print(f"Potential: π = ({', '.join(new_pot_vector)})")
        else:
            print("\n" + msg)