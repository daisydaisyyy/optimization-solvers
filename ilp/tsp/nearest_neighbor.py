import sys

class NNSolver:
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
            print(f"Error: {e}")
            sys.exit()

    def solve(self):
        try:
            raw = input(f"Enter Start Node [Available: {self.nodes}]: ")
            start = int(raw)
            if start not in self.nodes:
                print(f"Node {start} invalid. Defaulting to {self.nodes[0]}.")
                start = self.nodes[0]
        except:
            print(f"Invalid input. Defaulting to {self.nodes[0]}.")
            start = self.nodes[0]

        path = [start]
        curr = start
        visited = {start}
        total_cost = 0
        eq_parts = []

        while len(path) < self.num_nodes:
            # find nearest not visited
            candidates = []
            for n in self.nodes:
                if n not in visited:
                    dist = self.matrix[curr][n]
                    candidates.append((n, dist))
            
            # sort by distance, then by node index
            candidates.sort(key=lambda x: (x[1], x[0]))
            
            if not candidates: break
            
            best_n, best_d = candidates[0]
            
            print(f"   {curr} -> {best_n} (Cost {best_d})")
            
            visited.add(best_n)
            path.append(best_n)
            total_cost += best_d
            eq_parts.append(str(best_d))
            curr = best_n
        
        # Close cycle
        close_cost = self.matrix[curr][start]
        total_cost += close_cost
        eq_parts.append(str(close_cost))
        path.append(start)
        
        print(f"   Closing cycle: {curr} -> {start} (Cost {close_cost})")
        print(f"\nFinal Cycle: {'-'.join(map(str, path))}")
        print(f"v_S (Upper bound) = {' + '.join(eq_parts)} = {total_cost}")

if __name__ == "__main__":
    s = NNSolver("tsp.txt")
    s.solve()