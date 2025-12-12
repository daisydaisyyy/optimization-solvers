import re
import heapq
import copy

def parse_graph(file_path):
    graph = {}
    nodes = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'(\d+)-(\d+):\s*\(\s*(\d+),\s*(\d+)\s*\)', line)
                if match:
                    u, v, cost, cap = map(int, match.groups())
                    if u not in graph: graph[u] = []
                    if v not in graph: graph[v] = []
                    graph[u].append((v, cost))
                    graph[v].append((u, cost))
                    nodes.add(u)
                    nodes.add(v)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    return graph, sorted(list(nodes))

def dijkstra_with_history(graph, nodes, start):
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
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
        history.append((u, copy.deepcopy(dist), copy.deepcopy(prev)))
    return history, dist, prev

def print_matrix_table(nodes, history):
    col_width = 10
    header_top = " " * 6 + "|"
    for i in range(len(history)):
        iter_label = f" i = {i + 1}"
        header_top += f"{iter_label:^{col_width}}|"
    print("\n" + "-" * len(header_top))
    print(header_top)
    header_sub = " " * 6 + "|"
    for _ in history:
        sub_label = " π    p "
        header_sub += f"{sub_label:^{col_width}}|"
    print(header_sub)
    print("-" * len(header_sub))
    for node in nodes:
        row_str = f" {node:<4} |"
        for _, step_dist, step_prev in history:
            d = step_dist.get(node, float('inf'))
            p = step_prev.get(node, None)
            d_str = "+\u221E" if d == float('inf') else str(d)
            p_str = "*" if p is None else str(p)
            cell_content = f"{d_str}  {p_str}"
            row_str += f"{cell_content:^{col_width}}|"
        print(row_str)
    print("-" * len(header_top))

def print_tree_set(nodes, start_node, final_prev):
    edges = []
    for v in sorted(nodes):
        if v == start_node:
            continue
        u = final_prev[v]
        if u is not None and u != v:
            edges.append(f"({u}, {v})")
    result_str = ", ".join(edges)
    print("\nShortest path tree is")
    print(f"{{{result_str}}}")

if __name__ == "__main__":
    FILE_NAME = "graph.txt"
    START_NODE = 1
    graph, nodes = parse_graph(FILE_NAME)
    if graph:
        history, final_dist, final_prev = dijkstra_with_history(graph, nodes, START_NODE)
        print_matrix_table(nodes, history)
        print_tree_set(nodes, START_NODE, final_prev)
