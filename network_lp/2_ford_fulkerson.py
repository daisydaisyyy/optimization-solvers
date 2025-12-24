import re
from collections import defaultdict, deque
from math import inf

def parse_graph_for_max_flow(file_path):
    capacity_graph = defaultdict(dict)
    all_nodes = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                match = re.match(r'(\d+)-(\d+):\s*\(\s*(-?\d+),\s*(\d+)\s*\)', line)
                if match:
                    u, v, cost, capacity = map(int, match.groups())
                    capacity_graph[u][v] = capacity
                    all_nodes.add(u)
                    all_nodes.add(v)
    except FileNotFoundError:
        return None, None
    return capacity_graph, sorted(list(all_nodes))

def bfs_find_path(graph, s, t, parent):
    all_nodes = set(graph.keys())
    for u in graph:
        all_nodes.update(graph[u].keys())
    
    visited = {node: False for node in all_nodes}
    queue = deque([s])
    visited[s] = True
    parent.clear()
    parent[s] = None

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if not visited.get(v, False) and graph[u][v] > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == t:
                    return True
    return False

def max_flow_min_cut(capacity_graph, s, t, nodes):
    max_flow = 0
    iteration_details = []
    flow_values = defaultdict(lambda: defaultdict(int))
    residual_graph = defaultdict(lambda: defaultdict(int))
    
    # Sort edges for consistent vector output
    all_edges = []
    for u in sorted(capacity_graph.keys()):
        for v in sorted(capacity_graph[u].keys()):
            all_edges.append((u, v))

    for u in nodes:
        for v in capacity_graph.get(u, {}):
            residual_graph[u][v] = capacity_graph[u][v]
            residual_graph[v][u] = 0
            
    parent = {}
    iteration = 1

    while bfs_find_path(residual_graph, s, t, parent):
        path = []
        path_flow = inf
        v = t
        while v != s:
            u = parent[v]
            path.append((u, v))
            path_flow = min(path_flow, residual_graph[u][v])
            v = u
        path.reverse()
        path_str = " -> ".join([str(s)] + [str(v) for u, v in path])
        
        # Capture values for calculation string BEFORE update
        previous_flow = max_flow
        max_flow += path_flow 
        calculation_str = f"{previous_flow} + {path_flow} = {max_flow}"
        
        changes = []

        v = t
        while v != s:
            u = parent[v]
            if v in capacity_graph.get(u, {}):
                flow_values[u][v] += path_flow
            elif u in capacity_graph.get(v, {}):
                flow_values[v][u] -= path_flow

            old_res_uv = residual_graph[u][v]
            old_res_vu = residual_graph[v][u]

            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow

            new_res_uv = residual_graph[u][v]
            new_res_vu = residual_graph[v][u]

            changes.append({'edge': (u, v), 'old': old_res_uv, 'new': new_res_uv})
            changes.append({'edge': (v, u), 'old': old_res_vu, 'new': new_res_vu})
            v = u
        
        current_flow_snapshot = {}
        for u_node in nodes:
            for v_node in capacity_graph.get(u_node, {}):
                current_flow_snapshot[(u_node, v_node)] = flow_values[u_node][v_node]
        
        # Create vector string x=(...)
        vector_values = []
        for u_edge, v_edge in all_edges:
            val = flow_values[u_edge][v_edge]
            vector_values.append(str(val))
        vector_str = "x = ( " + "  ".join(vector_values) + " )"

        iteration_details.append({
            'iteration': iteration,
            'path': path_str,
            'path_flow': path_flow,
            'current_max_flow': max_flow,
            'calculation_str': calculation_str, # Added specific calculation string
            'changes': changes,
            'parent_vector': parent.copy(),
            'flow_snapshot': current_flow_snapshot,
            'vector_str': vector_str,
            'all_edges': all_edges
        })
        iteration += 1

    set_A = set()
    queue = deque([s])
    visited_cut = {node: False for node in nodes}
    visited_cut[s] = True
    while queue:
        u = queue.popleft()
        set_A.add(u)
        for v in nodes:
            if not visited_cut.get(v) and residual_graph[u][v] > 0:
                visited_cut[v] = True
                queue.append(v)
    set_B = set(nodes) - set_A
    min_cut_arcs = []
    for u in set_A:
        for v in set_B:
            if v in capacity_graph.get(u, {}):
                min_cut_arcs.append(((u, v), capacity_graph[u][v]))
                
    return max_flow, set_A, set_B, min_cut_arcs, iteration_details

def run_max_flow_analysis(file_path):
    capacity_graph, nodes = parse_graph_for_max_flow(file_path)
    if capacity_graph is None or not nodes:
        return "Error: File not found or graph is empty."
        
    print(f"Detected nodes: {nodes}")
    
    while True:
        try:
            s_input = input(f"Enter FIRST node (s): ")
            s = int(s_input)
            if s in nodes: break
            print(f"Node {s} not found. Try again.")
        except ValueError:
            print("Invalid input.")
            
    while True:
        try:
            t_input = input(f"Enter SECOND node (t): ")
            t = int(t_input)
            if t in nodes: break
            print(f"Node {t} not found. Try again.")
        except ValueError:
            print("Invalid input.")
            
    if s == t:
         return "Error: Source and sink cannot be the same."

    max_flow, set_A, set_B, min_cut_arcs, iteration_details = max_flow_min_cut(capacity_graph, s, t, nodes)
    
    output = ""
    
    if not iteration_details:
        output += "\nNo augmenting path found initially. Max Flow = 0."
    else:
        for detail in iteration_details:
            output += f"\n\n{'='*60}"
            output += f"\nITERATION {detail['iteration']}"
            output += f"\n{'='*60}"
            
            output += f"\n\n>>> Augmenting path: {detail['path']}"
            output += f"\n>>> Delta (min capacity among path nodes): {detail['path_flow']}"
            output += f"\n>>> Max Flow: {detail['calculation_str']}"
            
            edge_headers = []
            for u, v in detail['all_edges']:
                edge_headers.append(f"{u}{v}") 
            
            header_str = "      " + " ".join(edge_headers)
            output += f"\n{header_str}"
            output += f"\n{detail['vector_str']}"

    output += f"\n\n2. FINAL RESULTS"
    output += f"\nMax Flow = {max_flow}"
    
    cut_capacity = sum(cap for edge, cap in min_cut_arcs)
    output += f"\n--- Min Cut (Max-Flow Min-Cut Theorem): ---"
    output += f"\nMin Cut Capacity = {cut_capacity}"
    output += f"\nNode partition (A|B): A={sorted(list(set_A))} | B={sorted(list(set_B))}"
    
    output += "\nEdges in Min Cut (from A to B):"
    if not min_cut_arcs:
        output += "\n- None"
    else:
        for (u,v), cap in sorted(min_cut_arcs):
            output += f"\n- Edge ({u}, {v}): Capacity {cap}"
    
    return output

if __name__ == "__main__":
    print(run_max_flow_analysis('graph.txt'))