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
    
    # Imposta la radice (nodo 1) a 0
    root = min(nodes)
    potential[root] = 0
    print(f"  Node {root}: 0 (Root fixed)")
    
    max_iterations = 100
    for _ in range(max_iterations):
        updated = False
        sorted_T = sorted(list(T)) 
        
        for edge in sorted_T:
            i, j = edge
            cost = edges[edge]['cost']
            
            # entering edge (+)
            # π_j - π_i = cost  =>  π_j = π_i + cost
            if potential[i] is not None and potential[j] is None:
                val = potential[i] + cost
                potential[j] = val
                print(f"  Node {j}: {potential[i]} + {cost} = {val} \t(via arc {i}->{j})")
                updated = True
            
            # leaving edge (-)
            # π_j - π_i = cost  =>  π_i = π_j - cost
            elif potential[j] is not None and potential[i] is None:
                val = potential[j] - cost
                potential[i] = val
                print(f"  Node {i}: {potential[j]} - {cost} = {val} \t(via arc {i}->{j})")
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
    print("\nChecking optimality details (Reduced Costs):")
    violations = []
    
    sorted_edges = sorted(edges.keys())
    
    for edge in sorted_edges:
        i, j = edge
        cost = edges[edge]['cost']
        pi_i = potential.get(i)
        pi_j = potential.get(j)
        
        if pi_i is None or pi_j is None:
            continue
        
        # red_cost: c_ij - (pi_j - pi_i)
        pi_diff = pi_j - pi_i
        reduced_cost = cost - pi_diff
        
        calc_msg = f"  Edge {edge}: cost {cost} - (π{j}:{pi_j} - π{i}:{pi_i}) = {reduced_cost:.2f}"
        
        if edge in T:
            # must have red_cost = 0
            if abs(reduced_cost) > 0.001:
                print(f"{calc_msg} [T] -> VIOLATION (should be 0)")
                violations.append(f"Edge {edge} in T: reduced cost = {reduced_cost:.2f} (should be 0)")
            else:
                print(f"{calc_msg} [T] OK")
                
        elif edge in U:
            # must have red_cost <= 0
            if reduced_cost > 0.001:
                print(f"{calc_msg} [U] -> VIOLATION (should be ≤ 0)")
                violations.append(f"Edge {edge} in U: reduced cost = {reduced_cost:.2f} (should be ≤ 0)")
            else:
                print(f"{calc_msg} [U] OK")
                
        else: # L
            # must have red_cost >= 0
            if reduced_cost < -0.001:
                print(f"{calc_msg} [L] -> VIOLATION (should be >= 0)")
                violations.append(f"Edge {edge} in L: reduced cost = {reduced_cost:.2f} (should be ≥ 0)")
            else:
                print(f"{calc_msg} [L] OK")
    
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
    entering_from = None  # 'L' / 'U'
    
    # 1. entering edge
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
                entering_from = 'U'
                break
        else: # edge in L
            if reduced_cost < -0.001:
                entering_edge = edge
                entering_from = 'L'
                break
    
    if entering_edge is None:
        return None, None, "Optimal solution reached"
    
    print(f"\n>>> ENTERING EDGE: {entering_edge} (from {entering_from})")
    i, j = entering_edge
    cost = edges[entering_edge]['cost']
    reduced_cost = cost - (potential[j] - potential[i])
    print(f"    Cost: {cost}, Reduced cost: {reduced_cost:.2f}")
    
    tree_adj = {node: [] for node in nodes}
    for u, v in T:
        tree_adj[u].append(v)
        tree_adj[v].append(u)
    
    start, end = entering_edge
    
    def find_path_with_parent(current, target, visited, path):
        if current == target:
            return True
        visited.add(current)
        for neighbor in tree_adj[current]:
            if neighbor not in visited:
                path.append(neighbor)
                if find_path_with_parent(neighbor, target, visited, path):
                    return True
                path.pop()
        return False
    
    path_nodes = [start]
    find_path_with_parent(start, end, set(), path_nodes)
    print(f"    Path in tree: {' -> '.join(map(str, path_nodes))}")

    # 3. Find cycle, find theta
   
    cycle_arcs = []
    
    cap_ent = edges[entering_edge]['capacity']
    flow_ent = flow.get(entering_edge, 0)
    
    if entering_from == 'L':
        cycle_arcs.append((entering_edge, 1, cap_ent, flow_ent))
    else:
        cycle_arcs.append((entering_edge, -1, cap_ent, flow_ent))

    for k in range(len(path_nodes) - 1):
        u_curr, v_curr = path_nodes[k], path_nodes[k+1]
        
        if (u_curr, v_curr) in T:
            tree_edge = (u_curr, v_curr)
            is_forward_traversal = True 
        else:
            tree_edge = (v_curr, u_curr)
            is_forward_traversal = False 
            
        
        if entering_from == 'L':
            if is_forward_traversal:
                change = -1
            else:
                change = 1
        else: # entering_from == 'U'
            if is_forward_traversal:
                change = 1
            else:
                change = -1
        
        cycle_arcs.append((tree_edge, change, edges[tree_edge]['capacity'], flow.get(tree_edge, 0)))

    # theta
    theta = float('inf')
    leaving_edge = entering_edge
    theta_p = []
    theta_m = []
    print("    Cycle (θ = min flow x):")
    for edge, change, cap, f in cycle_arcs:
        if change == 1: # flow increases
            resid = cap - f
            print(f"      {edge}: increases (resid cap: {resid})")
            if edge != entering_edge:
                theta_p.append(resid)
                if resid < theta:
                    theta = resid
                    leaving_edge = edge

        else: # flow decreases
            resid = f
            print(f"      {edge}: decreases (resid flow: {resid})")
            if edge != entering_edge:
                theta_m.append(resid)
                if resid < theta:
                    theta = resid
                    leaving_edge = edge

    if(len(theta_p) != 0):
        print(f"\n    θ+ = {min(theta_p)}")
    if(len(theta_m) != 0):
        print(f"    θ- = {min(theta_m)}")
    print(f"    θ = {theta} (leaving edge: {leaving_edge})")
    if leaving_edge == entering_edge:
        print(">>> DEG CASE: leaving edge == entering edge!")

    # 4. Flow update
    new_flow = flow.copy()
    for edge, change, _, _ in cycle_arcs:
        if change == 1:
            new_flow[edge] += theta
        else:
            new_flow[edge] -= theta
            
    for edge in new_flow:
        if abs(new_flow[edge]) < 1e-9: new_flow[edge] = 0
        cap = edges[edge]['capacity']
        if abs(new_flow[edge] - cap) < 1e-9: new_flow[edge] = cap

    # 5. Update partitions
    leaving_edge = leaving_edge
    print(f">>> LEAVING EDGE: {leaving_edge}")
    
    new_T = T.copy()
    new_U = U.copy()
    new_L = L.copy()
    
    if leaving_edge == entering_edge:
        # L <-> U
        if entering_from == 'L':
            new_L.remove(entering_edge)
            new_U.add(entering_edge) # saturated
        else:
            new_U.remove(entering_edge)
            new_L.add(entering_edge) # emptied
    else:
        # swap edges out and in B
        new_T.add(entering_edge)
        new_T.remove(leaving_edge)
        
        if entering_from == 'L': new_L.remove(entering_edge)
        else: new_U.remove(entering_edge)
        
        f_leave = new_flow[leaving_edge]
        cap_leave = edges[leaving_edge]['capacity']
        
        if f_leave <= 1e-9:
            new_L.add(leaving_edge)
        elif f_leave >= cap_leave - 1e-9:
            new_U.add(leaving_edge) # saturated -> U
        else: # possible logic errors
            raise ValueError(f"ERROR: edge {leaving_edge} with flow {f_leave} / {cap_leave}. Impossible, must be empty (in L) or saturated (in U).")

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
    print("Initial potential")
    potential = calculate_potential(edges, nodes, T)
    
    pi_vector = []
    for node in sorted(nodes):
        pi = potential.get(node)
        pi_vector.append(str(pi))
        
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
            
            
            
            print("\nNew flow (increase/decrease by θ the cycle edges + re-calculate with the new partition):")
            new_flow_vector = []
            for edge in sorted(edges.keys()):
                val = new_flow.get(edge, 0)
                new_flow_vector.append(str(val))
                print(f"  {edge}: {val}")
            
            print(f"Flow: x = ({', '.join(new_flow_vector)})")
            
            print("\nNew potential:")
            new_potential = calculate_potential(edges, nodes, new_T)
            new_pot_vector = []
            for node in sorted(nodes):
                val = new_potential.get(node)
                new_pot_vector.append(str(val))
            
            print(f"Potential: π = ({', '.join(new_pot_vector)})")
        else:
            print("\n" + msg)
