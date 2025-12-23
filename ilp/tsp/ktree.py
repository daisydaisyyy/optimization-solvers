import sys

class KTreeSolver:
    def __init__(self, filename):
        self.matrix = {}
        self.nodes = []
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
        except Exception as e:
            print(f"Error: {e}")
            sys.exit()

    def solve(self):
        try:
            raw = input(f"Enter node k [Available: {self.nodes}]: ")
            k = int(raw)
            if k not in self.nodes:
                print(f"Node {k} not found. Defaulting to {self.nodes[-1]}.")
                k = self.nodes[-1]
        except:
            print(f"Invalid input. Defaulting to {self.nodes[-1]}.")
            k = self.nodes[-1]

        others = [n for n in self.nodes if n != k]

        # kruskal
        edges = []
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                u, v = others[i], others[j]
                edges.append((u, v, self.matrix[u][v]))
        edges.sort(key=lambda x: x[2])

        parent = {n: n for n in others}
        def find(n):
            if parent[n] != n: parent[n] = find(parent[n])
            return parent[n]
        def union(n1, n2):
            r1, r2 = find(n1), find(n2)
            if r1 != r2: parent[r1] = r2; return True
            return False

        mst_cost = 0
        eq_parts = []
        
        print("Kruskal on remaining nodes:")
        for u, v, w in edges:
            if union(u, v):
                print(f"   - Selected ({u},{v}) Cost: {w}")
                mst_cost += w
                eq_parts.append(str(w))

        # connect k
        print(f"Connecting k = {k}:")
        k_edges = []
        for v in others:
            k_edges.append((v, self.matrix[k][v]))
        k_edges.sort(key=lambda x: x[1])
        
        best = k_edges[:2]
        for v, w in best:
            print(f"   - Connecting ({k},{v}) Cost: {w}")
            mst_cost += w
            eq_parts.append(str(w))
            
        print(f"\nEquation: {' + '.join(eq_parts)} = {mst_cost}")
        print(f"v_I (Lower Bound) = {mst_cost}")

if __name__ == "__main__":
    s = KTreeSolver("tsp.txt")
    s.solve()