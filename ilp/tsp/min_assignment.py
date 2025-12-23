import sys
import numpy as np
from scipy.optimize import linear_sum_assignment


class AssignmentSolver:
    def __init__(self, filename):
        self.matrix = {}
        self.nodes = []
        self.num_nodes = 0
        self.load_data(filename)

    def load_data(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = [line.replace('\xa0', ' ').strip() for line in f if line.strip()]
            if lines[-1].lower().startswith('x'): lines = lines[:-1]
            header = lines[0].split()
            col_nodes = [int(x) for x in header if x.isdigit()]
            first_row_node = int(lines[1].split()[0])
            self.nodes = sorted(list(set([first_row_node] + col_nodes)))
            self.num_nodes = len(self.nodes)
            for n in self.nodes: self.matrix[n] = {}
            for line in lines[1:]:
                parts = line.split()
                if not parts[0].isdigit(): continue
                row = int(parts[0])
                costs = [int(x) for x in parts[1:]]
                offset = len(col_nodes) - len(costs)
                for i, cost in enumerate(costs):
                    col = col_nodes[i + offset]
                    self.matrix[row][col] = cost
                    self.matrix[col][row] = cost
            for n in self.nodes: self.matrix[n][n] = float('inf')
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit()

    def print_matrix(self, mat, title):
        print(f"\n--- {title} ---")
        print("      " + "".join([f"{n:^6}" for n in self.nodes]))
        print("    " + "-" * (6 * self.num_nodes + 2))
        
        for i in range(self.num_nodes):
            row_label = self.nodes[i]
            row_str = f"{row_label:^3} |"
            for j in range(self.num_nodes):
                val = mat[i][j]
                if val == float('inf'):
                    row_str += "  inf "
                else:
                    row_str += f"{int(val):^6}"
            print(row_str)

    def print_ascii_cycle(self, edges):
        # build undirected adj map
        adj = {u: [] for u in self.nodes}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # traversing
        start = self.nodes[0]
        path = [start]
        curr = start
        visited = {start}
        
        # traverse until all nodes are covered
        while len(path) < self.num_nodes:
            found = False
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    curr = neighbor
                    found = True
                    break
            if not found: break
                
        # print graph
        out_str = ""
        total_check = 0
        
        for i in range(len(path)):
            u = path[i]
            v = path[(i + 1) % len(path)] # loop back to start
            w = self.matrix[u][v]
            total_check += w
            out_str += f"({u}) =={int(w)}==> "
        
        out_str += f"({path[0]})"
        
        print(out_str)
   

    def solve(self):
        # print initial data
        curr_matrix = np.zeros((self.num_nodes, self.num_nodes))
        nodes_map = {i: n for i, n in enumerate(self.nodes)}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                curr_matrix[i][j] = self.matrix[nodes_map[i]][nodes_map[j]]

        self.print_matrix(curr_matrix, "INITIAL PROBLEM")

        # subtract from rows
        for i in range(self.num_nodes):
            row_vals = [x for x in curr_matrix[i] if x != float('inf')]
            min_val = min(row_vals) if row_vals else 0
            # print(f"   Row {nodes_map[i]} min: {int(min_val)}")
            for j in range(self.num_nodes):
                if curr_matrix[i][j] != float('inf'):
                    curr_matrix[i][j] -= min_val
        
        self.print_matrix(curr_matrix, "subtract min from each row")

        # subtract from columns
        for j in range(self.num_nodes):
            col_vals = [curr_matrix[i][j] for i in range(self.num_nodes) if curr_matrix[i][j] != float('inf')]
            min_val = min(col_vals) if col_vals else 0
            # if min_val > 0: print(f"   Col {nodes_map[j]} min: {int(min_val)}")
            for i in range(self.num_nodes):
                if curr_matrix[i][j] != float('inf'):
                    curr_matrix[i][j] -= min_val

        self.print_matrix(curr_matrix, "FULLY REDUCED (row + column subtraction)")

        # solve assignment using scipy
        original_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                original_matrix[i][j] = self.matrix[nodes_map[i]][nodes_map[j]]

        row_ind, col_ind = linear_sum_assignment(original_matrix)
        ap_cost = original_matrix[row_ind, col_ind].sum()
        
        adj = {}
        eq_parts = []
        print("\n>> OPT ASSIGNMENT (take one 0 for each row, col)")
        for r, c in zip(row_ind, col_ind):
            u, v = nodes_map[r], nodes_map[c]
            adj[u] = v
            w = self.matrix[u][v]
            print(f"   - Node {u} -> {v} (Cost: {int(w)})")
            eq_parts.append(str(int(w)))

        print(f"\n{' + '.join(eq_parts)} = {int(ap_cost)}")
        print(f">>> v_I (Lower Bound) = {int(ap_cost)}")

        # find subtours
        visited = set()
        cycles = []
        for n in self.nodes:
            if n not in visited:
                curr = n
                cycle = []
                while curr not in visited:
                    visited.add(curr)
                    cycle.append(curr)
                    curr = adj[curr]
                if cycle: cycles.append(cycle)
        
        print("\nSubtours found:")
        for i, c in enumerate(cycles):
            print(f"   Cycle {i+1}: {'-'.join(map(str, c + [c[0]]))}")

       
        if len(cycles) > 1:
            # patch algorithm
            print("\nPATCH ALGORITHM")
            c1 = cycles[0]
            c2 = cycles[1]
            best_delta = float('inf')
            best_swap_info = None
            
            print(f"{'REMOVE      '}| {'ADD         '}| {'Delta'}")

            for i in range(len(c1)):
                u1, v1 = c1[i], c1[(i + 1) % len(c1)]
                for j in range(len(c2)):
                    u2, v2 = c2[j], c2[(j + 1) % len(c2)]
                    
                    cost_rem = self.matrix[u1][v1] + self.matrix[u2][v2]
                    
                    # swap cross: (u1->v2, u2->v1)
                    cost_add_a = self.matrix[u1][v2] + self.matrix[u2][v1]
                    delta_a = cost_add_a - cost_rem
                    
                    # swap parallel (u1->u2, v1->v2)
                    cost_add_b = self.matrix[u1][u2] + self.matrix[v1][v2]
                    delta_b = cost_add_b - cost_rem
                    
                    if delta_a < delta_b:
                        delta = delta_a
                        add_str = f"({u1},{v2})&({u2},{v1})"
                        calc_str = f"{int(cost_add_a)} - {int(cost_rem)} = {int(delta)}"
                        current_add = [(u1, v2), (u2, v1)]
                    else:
                        delta = delta_b
                        add_str = f"({u1},{u2})&({v1},{v2})"
                        calc_str = f"{int(cost_add_b)} - {int(cost_rem)} = {int(delta)}"
                        current_add = [(u1, u2), (v1, v2)]
                    
                    rem_str = f"({u1},{v1})&({u2},{v2})"
                    print(f"{rem_str} | {add_str} | {calc_str}")
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_swap_info = {
                            'rem': [(u1, v1), (u2, v2)],
                            'add': current_add
                        }
            final_cost = ap_cost + best_delta
            print(f"\nBest Delta: {int(best_delta)}")
            print(f"{int(ap_cost)} (v_I) + {int(best_delta)} (patch) = {int(final_cost)}")

            # print final graph
            if best_swap_info:
                final_edges = []

                for i in range(len(c1)):
                    u, v = c1[i], c1[(i + 1) % len(c1)]
                    # check if edge is removed
                    is_removed = False
                    for r_u, r_v in best_swap_info['rem']:
                        if (u == r_u and v == r_v) or (u == r_v and v == r_u):
                            is_removed = True
                    if not is_removed: final_edges.append((u, v))
                
                # check edges from C2
                for i in range(len(c2)):
                    u, v = c2[i], c2[(i + 1) % len(c2)]
                    is_removed = False
                    for r_u, r_v in best_swap_info['rem']:
                        if (u == r_u and v == r_v) or (u == r_v and v == r_u):
                            is_removed = True
                    if not is_removed: final_edges.append((u, v))
                
                # add new edges
                for u, v in best_swap_info['add']:
                    final_edges.append((u, v))
                
                self.print_ascii_cycle(final_edges)

        else:
            print("\nOnly 1 cycle found. Solution is already optimal for TSP.")
            c = cycles[0]
            edges = [(c[i], c[(i+1)%len(c)]) for i in range(len(c))]
            self.print_ascii_cycle(edges)

if __name__ == "__main__":
    solver = AssignmentSolver("tsp.txt")
    solver.solve()