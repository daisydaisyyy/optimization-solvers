import re
import heapq
import copy
import sys
import numpy as np

def parse_graph(file_path):
    edges = {}
    nodes = set()
    balances = {}
    parsing_nodes = False
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if "node costs" in line.lower():
                    parsing_nodes = True
                    continue
                
                if not parsing_nodes:
                    clean_line = re.sub(r'\[.*?\]', '', line).strip()
                    # parsing
                    match = re.search(r'(\d+)-(\d+):\s*\(\s*(\d+),\s*(\d+)\s*\)', clean_line) 
                    if match:
                        u, v, cost, cap = map(int, match.groups())
                        edges[(u, v)] = {'cost': cost, 'capacity': cap}
                        nodes.add(u)
                        nodes.add(v)
                else:
                    parts = line.split(':')
                    if len(parts) == 2:
                        try:
                            n = int(parts[0].strip())
                            b = int(parts[1].strip())
                            balances[n] = b
                            nodes.add(n)
                        except ValueError: pass
                            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None, None
        
    for n in nodes:
        if n not in balances: balances[n] = 0
            
    return edges, sorted(list(nodes)), balances

def dijkstra_with_history(edges, nodes, start):
    adj = {n: [] for n in nodes}
    for (u, v), data in edges.items():
        adj[u].append((v, data['cost']))

    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[start] = 0
    prev[start] = start
    
    history = []
    history.append((None, copy.deepcopy(dist), copy.deepcopy(prev)))
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if u in visited: continue
        visited.add(u)
        
        for v, cost in adj[u]:
            if dist[u] + cost < dist[v]:
                dist[v] = dist[u] + cost
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
                
        history.append((u, copy.deepcopy(dist), copy.deepcopy(prev)))
        
    return history, dist, prev

def calculate_flow_matrix(edges, nodes, balances, T):
    flow = {e: 0 for e in edges}
    T_list = sorted(list(T))
    
    if not T_list: return flow
    
    n_vars = len(T_list)
    A = []
    b_vec = []
    
    solver_nodes = nodes[:-1]
    
    for node in solver_nodes:
        row = [0] * n_vars
        rhs = balances[node]
        
        for idx, (u, v) in enumerate(T_list):
            if v == node: # in-flow
                row[idx] = 1
            elif u == node: # out-flow
                row[idx] = -1
        
        A.append(row)
        b_vec.append(rhs)
        
    try:
        x = np.linalg.solve(A, b_vec)
        for i, edge in enumerate(T_list):
            val = x[i]
            # approx
            if abs(val - round(val)) < 1e-9:
                val = int(round(val))
            flow[edge] = val
    except np.linalg.LinAlgError:
        print("error: singular matrix")
        
    return flow

def print_matrix_table(nodes, history):
    col_width = 10
    header_top = " " * 6 + "|"
    for i in range(len(history)):
        header_top += f"{f'i={i+1}':^{col_width}}|"
        
    print("\n" + "-" * len(header_top))
    print(header_top)
    
    header_sub = " " * 6 + "|"
    for _ in history:
        header_sub += f"{'pi   p':^{col_width}}|"
        
    print(header_sub)
    print("-" * len(header_sub))
    
    for node in nodes:
        row_str = f" {node:<4} |"
        for _, step_dist, step_prev in history:
            d = step_dist.get(node, float('inf'))
            p = step_prev.get(node, None)
            
            d_str = "+inf" if d == float('inf') else str(d)
            p_str = "*" if p is None else str(p)
            
            row_str += f"{f'{d_str}  {p_str}':^{col_width}}|"
        print(row_str)
    print("-" * len(header_top))

if __name__ == "__main__":
    FILE_NAME = "graph.txt"
    START_NODE = int(input("Enter start node: "))
    
    edges, nodes, balances = parse_graph(FILE_NAME)

    demand = 1
    total_supply = -(len(nodes) - 1) * demand
    for n in nodes:
        if n == START_NODE:
            balances[n] = total_supply # set source node weight = nodes number - 1 (itself)
        else:
            balances[n] = demand # other nodes with weight = 1
    
    if edges:
        history, final_dist, final_prev = dijkstra_with_history(edges, nodes, START_NODE)
        print_matrix_table(nodes, history)
        
        T = set()
        for v in nodes:
            u = final_prev[v]
            if u is not None and u != v:
                if (u, v) in edges:
                    T.add((u, v))
                else:
                    pass

        print("\nShortest path tree T:")
        t_list_str = [f"({u}, {v})" for u, v in sorted(list(T))]
        print(f"{{ {', '.join(t_list_str)} }}")
        
        # flow
        edge_flows = calculate_flow_matrix(edges, nodes, balances, T)
        
        all_edges = sorted(edges.keys())
        vector_values = []
        for edge in all_edges:
            vector_values.append(edge_flows.get(edge, 0)) # 0 if not in shortest path tree
            
        print(f"\nx = ({', '.join(map(str, vector_values))}) -> flow on shortest path tree")