import sys

class TSPSolver:
    def __init__(self, filename):
        self.matrix = {}
        self.nodes = []
        self.num_nodes = 0
        self.branching_vars = []  
        self.load_data(filename)
        self.global_ub = float('inf')
        self.best_solution_edges = [] 
        self.hub_node = 1 

    def load_data(self, filename):
        try:
            with open(filename, 'r') as f:
                # cleaning strange things
                lines = [line.replace('\xa0', ' ').strip() for line in f if line.strip()]

            # 1. last line is variables to instantiate in branch & bound
            last_line = lines[-1]
            if last_line.lower().startswith('x'):
                raw_vars = last_line.replace(',', ' ').split()
                for var in raw_vars:
                    clean_var = var.lower().replace('x', '').strip()
                    if len(clean_var) >= 2:
                        u, v = int(clean_var[0]), int(clean_var[1])
                        self.branching_vars.append((u, v))
                lines = lines[:-1] 

            # 2. header
            header = lines[0].split()
            col_nodes = [int(x) for x in header]
            first_row_node = int(lines[1].split()[0])
            self.nodes = sorted(list(set([first_row_node] + col_nodes)))
            self.num_nodes = len(self.nodes)

            # initialize matrix
            for n in self.nodes:
                self.matrix[n] = {}

            # 3. parsing matrix rows
            for line in lines[1:]:
                parts = line.split()
                if not parts[0].isdigit(): continue
                
                row_node = int(parts[0])
                costs = [int(x) for x in parts[1:]]
                
                offset = len(col_nodes) - len(costs)
                
                for i, cost in enumerate(costs):
                    col_index = i + offset
                    if col_index < len(col_nodes):
                        col_node = col_nodes[col_index]
                        self.matrix[row_node][col_node] = cost
                        self.matrix[col_node][row_node] = cost

            # fix for malformed files (e.g. missing 4-5 cost)
            # if 4 in self.matrix and 5 in self.matrix:
            #     if 5 not in self.matrix[4]:
            #          if row_node == 4 and len(costs) == 1:
            #              cost = costs[0]
            #              self.matrix[4][5] = cost
            #              self.matrix[5][4] = cost

        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit()

    def get_edges_for_subset(self, subset_nodes):
        edges = []
        seen = set()
        for u in subset_nodes:
            if u not in self.matrix: continue
            for v in self.matrix[u]:
                if v in subset_nodes:
                    if (u, v) not in seen and (v, u) not in seen:
                        edges.append((u, v, self.matrix[u][v]))
                        seen.add((u, v))
        return sorted(edges, key=lambda x: x[2])

    def get_edges_connected_to(self, node):
        edges = []
        if node in self.matrix:
            for v, cost in self.matrix[node].items():
                edges.append((node, v, cost))
        return sorted(edges, key=lambda x: x[2])

    def calculate_n_tree_hub(self, included, excluded, hub_node):
        subset_nodes = [n for n in self.nodes if n != hub_node]
        subset_edges = self.get_edges_for_subset(subset_nodes)
        
        parent = {n: n for n in subset_nodes}
        def find(n):
            if parent[n] != n: parent[n] = find(parent[n])
            return parent[n]
        def union(n1, n2):
            root1, root2 = find(n1), find(n2)
            if root1 != root2:
                parent[root1] = root2
                return True
            return False

        mst_edges = []
        mst_cost = 0
        
        # 1. included node
        for u, v in included:
            if u != hub_node and v != hub_node:
                mst_edges.append((u, v))
                mst_cost += self.matrix[u][v]
                union(u, v)

        # 2. Kruskal
        for u, v, w in subset_edges:
            if (u, v) in excluded or (v, u) in excluded: continue
            if (u, v) in included or (v, u) in included: continue
            if len(mst_edges) < len(subset_nodes) - 1:
                if union(u, v):
                    mst_edges.append((u, v))
                    mst_cost += w
            else:
                break
        
        if len(mst_edges) < len(subset_nodes) - 1:
            return float('inf'), []

        # 3. reconnect node k
        hub_candidates = self.get_edges_connected_to(hub_node)
        hub_edges_selected = []
        
        forced_hub_edges = 0
        for u, v in included:
            if u == hub_node or v == hub_node:
                hub_edges_selected.append((u, v))
                mst_cost += self.matrix[u][v]
                forced_hub_edges += 1

        if forced_hub_edges > 2: return float('inf'), []

        needed = 2 - forced_hub_edges
        for u, v, w in hub_candidates:
            if needed == 0: break
            if (u, v) in excluded or (v, u) in excluded: continue
            
            is_included = False
            for iu, iv in hub_edges_selected:
                if (u==iu and v==iv) or (u==iv and v==iu): is_included = True; break
            if is_included: continue

            hub_edges_selected.append((u, v))
            mst_cost += w
            needed -= 1
            
        if len(hub_edges_selected) < 2: return float('inf'), []

        final_edges = mst_edges + hub_edges_selected
        return mst_cost, final_edges

    def is_tour(self, edges):
        if len(edges) != self.num_nodes: return False
        deg = {n: 0 for n in self.nodes}
        adj = {n: [] for n in self.nodes}
        for u, v in edges:
            deg[u]+=1; deg[v]+=1
            adj[u].append(v); adj[v].append(u)
        if any(d != 2 for d in deg.values()): return False
        visit = set()
        def dfs(x):
            visit.add(x)
            for y in adj[x]:
                if y not in visit: dfs(y)
        dfs(self.nodes[0])
        return len(visit) == self.num_nodes

    def nearest_neighbor(self, start_node):
        path = [start_node]
        curr = start_node
        cost = 0
        
        while len(path) < self.num_nodes:
            nxt, m_dist = None, float('inf')
            for n, d in self.matrix[curr].items():
                if n not in path and d < m_dist:
                    m_dist = d; nxt = n
            
            if nxt is None: return float('inf'), []
            path.append(nxt)
            cost += m_dist
            curr = nxt
            
        if start_node in self.matrix[curr]:
            cost += self.matrix[curr][start_node]
            path.append(start_node)
            return cost, path
        else:
            return float('inf'), []

    def analyze_constraints(self, edges):
        deg = {n: 0 for n in self.nodes}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        violation_found = False
        for n in self.nodes:
            if deg[n] != 2:
                violation_found = True
                print(f"   - Constraint violated for node {n}: Sum(x_{n}j) = {deg[n]} (should be 2)")
        
        if not violation_found:
            print("   - No constraints violated (the 1-tree is a cycle).")

        print("   - Question: Is minimum cost assignment a better evaluation?")
        print("     NO. The 1-tree is usually a better (tighter) bound for the symmetric TSP.")

    def bb_recursive(self, inc, exc, var_index, name):
        lb, edges = self.calculate_n_tree_hub(inc, exc, self.hub_node)
        edges_str = " ".join([f"({u},{v})" for u, v in sorted(edges)])

        print(f"\n[{name}]")
        print(f"   {self.hub_node}-tree: {edges_str}")
        print(f"   LB = v_I = {lb}")

        if lb >= self.global_ub:
            print(f"   >>> PRUNED (LB = v_I = {lb} >= UB = v_S = {self.global_ub})")
            return

        if self.is_tour(edges):
            print(f"   >>> FEASIBLE SOLUTION! Cost: {lb}")
            if lb < self.global_ub:
                self.global_ub = lb
                self.best_solution_edges = edges 
                print(f"   *** NEW OPTIMUM (UB) = {self.global_ub} ***")
            return

        if var_index < len(self.branching_vars):
            u, v = self.branching_vars[var_index]
            print(f"   -> Branch on variable x_{u}{v}")
            self.bb_recursive(inc, exc + [(u, v)], var_index + 1, f"{name} -> x{u}{v}=0")
            self.bb_recursive(inc + [(u, v)], exc, var_index + 1, f"{name} -> x{u}{v}=1")

    def parse_edge_input(self, text):
        """reads '1 2', 'x12', '1,2' etc."""
        try:
            # cleans everything except numbers
            text = text.lower().replace('x', '').replace('(', '').replace(')', '').replace(',', ' ')
            clean = text.split()
            
            if len(clean) == 1 and len(clean[0]) == 2:
                clean = [clean[0][0], clean[0][1]]
                
            if len(clean) >= 2:
                u, v = int(clean[0]), int(clean[1])
                return sorted((u, v)) 
        except:
            pass
        return None

    def check_edge_in_optimal(self, u, v):
        for e1, e2 in self.best_solution_edges:
            if (e1 == u and e2 == v) or (e1 == v and e2 == u):
                return True
        return False

    def reconstruct_cycle(self, edges):
        """Reconstructs the sequence a-b-c-... from edges"""
        if not edges: return "None"
        
        # adj map
        adj = {n: [] for n in self.nodes}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        try:
            path = [self.nodes[0]]
            visited = {self.nodes[0]}
            curr = self.nodes[0]
            
            while len(path) < self.num_nodes:
                found_next = False
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        curr = neighbor
                        found_next = True
                        break
                if not found_next: break
            
            # adds return edge to start node
            path.append(self.nodes[0])
            return "-".join(map(str, path))
        except:
            return "Path Error"

    def solve(self):
        
        try:
            raw = input(f"a) v_I -> Enter the k node to calculate the k-tree: ")
            self.hub_node = int(raw)
            if self.hub_node not in self.nodes:
                self.hub_node = self.nodes[-1]
        except:
            self.hub_node = 5

        try:
            raw = input(f"b) v_S -> Enter the starting node for the heuristic cycle (nearest neighbor algorithm): ")
            start_cycle = int(raw)
            if start_cycle not in self.nodes: start_cycle = 1
        except:
            start_cycle = 1

        # a) k-tree: find lower bound (v_I)
        lb_root, edges_root = self.calculate_n_tree_hub([], [], self.hub_node)
        edges_str = " ".join([f"( {u} , {v} )" for u, v in sorted(edges_root)])
        print(f"\na) {self.hub_node}-tree: {edges_str}")
        print(f"   v_I(P) = {lb_root}")
        self.analyze_constraints(edges_root)

        # b) heuristic cycle: find upper bound (v_S)
        ub, path = self.nearest_neighbor(start_cycle)
        self.global_ub = ub
        if ub != float('inf'):
            path_edges = []
            for i in range(len(path)-1):
                u, v = sorted((path[i], path[i+1]))
                path_edges.append((u, v))
            self.best_solution_edges = path_edges
            path_str = " - ".join(map(str, path[:-1])) 
            print(f"\nb) cycle: {path_str}   v_S(P) = {ub}")
        else:
            print(f"\nb) cycle: IMPOSSIBILE")
        
        
        print("\n--- BRANCH AND BOUND ---")

        self.bb_recursive([], [], 0, "P:")

        # --- OTHER QUESTIONS ---
        print("\n" + "="*60)
        
        cycle_str = self.reconstruct_cycle(self.best_solution_edges)
        print(f"1. optimal cycle: {cycle_str} (v_S: {self.global_ub}).")
    
        print("\n2. Edge cost changes question:")
        raw_edge = input("   Enter edge whose cost changes (e.g. x12): ")
        edge = self.parse_edge_input(raw_edge)
        
        if edge:
            u, v = edge
            if self.check_edge_in_optimal(u, v):
                print(f"   -> Is edge ({u}, {v}) in the optimal solution? YES.")
                print(f"   Since ({u}, {v}) belongs to the optimal cycle, if its cost increased, the value of the optimal solution would increase; if it decreased, the value would decrease.")
            else:
                print(f"   -> Is edge ({u}, {v}) in the optimal solution? NO.")
                print(f"   Since ({u}, {v}) does not belong to the optimal cycle, if its cost increased, the optimal solution would not change; if it decreased, it might change.")
        else:
            print("     Invalid input! Use format '1 2' or 'x12'.")

        print("\n3. Edge constraint Question:")
        raw_edge_force = input("   Enter edge to force (e.g. 3 4): ")
        edge_force = self.parse_edge_input(raw_edge_force)
        
        if edge_force:
            u, v = edge_force
            if self.check_edge_in_optimal(u, v):
                print(f"   -> Is edge ({u}, {v}) in the optimal solution? YES.")
                print(f"   Since ({u}, {v}) already belongs to the optimal solution, forcing it would not change anything.")
            else:
                print(f"   -> Is edge ({u}, {v}) in the optimal solution? NO.")
                print(f"   Since({u}, {v}) does not belong to the optimal solution, forcing it would result in a worsening (or at best equality) of the solution cost.")
        else:
            print("     Invalid input! Use format '3 4'.")

if __name__ == "__main__":
    s = TSPSolver("tsp.txt")
    s.solve()