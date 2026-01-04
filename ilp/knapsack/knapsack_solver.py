import sys
import math
import numpy as np 
from scipy.optimize import milp, LinearConstraint, Bounds 
from fractions import Fraction

class Node:
    def __init__(self, level, taken, path_desc):
        self.level = level                  
        self.taken = taken                  # Lista completa delle decisioni prese [(id, val), ...]
        self.path_desc = path_desc          # Descrizione ultima mossa (non usato per la stampa full)
        
        # Dati del nodo
        self.sol_vector = {}
        self.curr_weight = 0.0
        self.curr_value = 0.0
        self.is_integer = False
        self.split_item = None
        self.calc_str = ""

class TreeVisualizer:
    def __init__(self, capacity, items):
        self.capacity = capacity
        for item in items:
            item['r'] = item['v'] / item['p']
        
        self.items_sorted = sorted(items, key=lambda x: x['r'], reverse=True)
        self.items_by_id = sorted(items, key=lambda x: x['id'])
        self.best_value = 0
        self.node_counter = 0 

    def precalc_node(self, taken):
        sol_vector = {}
        current_w = 0
        current_v = 0
        calc_parts = []
        
        fixed_ids = {}
        for item_id, val in taken:
            sol_vector[item_id] = val
            fixed_ids[item_id] = val
            if val == 1:
                item = next(i for i in self.items_sorted if i['id'] == item_id)
                current_w += item['p']
                current_v += item['v']
                calc_parts.append(f"{int(item['v'])}")

        if current_w > self.capacity:
            return None # not feasible

        remaining_cap = self.capacity - current_w
        split_item = None
        
        for item in self.items_sorted:
            if item['id'] in fixed_ids: continue
            
            if item['p'] <= remaining_cap:
                sol_vector[item['id']] = 1
                remaining_cap -= item['p']
                current_v += item['v']
                current_w += item['p']
                calc_parts.append(f"{int(item['v'])}")
            else:
                if remaining_cap > 0:
                    frac = remaining_cap / item['p']
                    sol_vector[item['id']] = frac
                    added_val = item['v'] * frac
                    current_v += added_val
                    current_w += remaining_cap
                    split_item = item['id']
                    calc_parts.append(f"({frac:.2f}*{int(item['v'])})")
                    remaining_cap = 0
                else:
                    sol_vector[item['id']] = 0
        
        for item in self.items_sorted:
            if item['id'] not in sol_vector: sol_vector[item['id']] = 0
        
        calc_str = " + ".join(calc_parts) if calc_parts else "0"
        is_integer = (split_item is None)
        
        return {
            "sol_vector": sol_vector, "curr_weight": current_w, "curr_value": current_v,
            "split_item": split_item, "is_integer": is_integer, "calc_str": calc_str
        }

    def solve(self, initial_best_value=0):
        self.best_value = initial_best_value
        
        root_data = self.precalc_node([])
        root = Node(0, [], "Root")
        if root_data:
            for k, v in root_data.items(): setattr(root, k, v)
        
        stack = [root]
        
        
        while stack:
            curr_node = stack.pop()
            self.node_counter += 1
            node_label = f"N{self.node_counter}"
            indent = "    " * curr_node.level
            
            new_optimum_found = False
            if curr_node.is_integer:
                if curr_node.curr_value > self.best_value:
                    self.best_value = curr_node.curr_value
                    new_optimum_found = True
            
            if not curr_node.taken:
                full_path_str = "Root"
            else:
                full_path_str = ", ".join([f"x{i}={v}" for i, v in curr_node.taken])

            print(f"\n{indent}|-- [{node_label}] {{ {full_path_str} }}")
            
            vec_str_parts = []
            for item in self.items_by_id:
                val = curr_node.sol_vector.get(item['id'], 0)
                if val == 0 or val == 1: vec_str_parts.append(f"x{item['id']}={int(val)}")
                else: vec_str_parts.append(f"x{item['id']}={val:.2f}")
            
            status_str = "FEASIBLE (Integer Solution)" if curr_node.is_integer else f"INFEASIBLE (Fractional on x{curr_node.split_item})"
            
            vec_str = ", ".join(vec_str_parts) + " -> " + f"{status_str}"

            print(f"{indent}    -> [ {vec_str} ]")
            print(f"{indent}    -> Capacity : {curr_node.curr_weight:.2f} / {self.capacity}")            
            print(f"{indent}    -> v_S = {curr_node.calc_str} = {curr_node.curr_value:.2f}")
            print(f"{indent}    -> current best v_I = {int(self.best_value)}")
            
            bound_floor = math.floor(curr_node.curr_value)
            print(f"{indent}    -> [{int(self.best_value)}, {bound_floor}]")

            if new_optimum_found:
                print(f"{indent}    *** NEW OPTIMUM FOUND: v_I updated to {int(self.best_value)} ***")

            # --- PRUNING ---
            if bound_floor < self.best_value:
                 print(f"{indent}    -> ACTION: PRUNING: {bound_floor} < {int(self.best_value)}")
                 continue
            
            if bound_floor == self.best_value and not curr_node.is_integer:
                 print(f"{indent}    -> ACTION: PRUNING: {bound_floor} <= {int(self.best_value)} (Cannot improve)")
                 continue

            if curr_node.is_integer:
                continue 

            # --- BRANCHING ---
            print(f"{indent}    -> ACTION: BRANCHING on x{curr_node.split_item}")
            
            children = []
            
            child_1_data = self.precalc_node(curr_node.taken + [(curr_node.split_item, 1)])
            if child_1_data:
                node_1 = Node(curr_node.level + 1, curr_node.taken + [(curr_node.split_item, 1)], f"x{curr_node.split_item}=1")
                for k, v in child_1_data.items(): setattr(node_1, k, v)
                children.append(node_1)

            child_0_data = self.precalc_node(curr_node.taken + [(curr_node.split_item, 0)])
            if child_0_data:
                node_0 = Node(curr_node.level + 1, curr_node.taken + [(curr_node.split_item, 0)], f"x{curr_node.split_item}=0")
                for k, v in child_0_data.items(): setattr(node_0, k, v)
                children.append(node_0)
            
            # LIFO stack to pop best value first
            children.sort(key=lambda n: (n.is_integer, n.curr_value), reverse=False)
            
            for child in children:
                stack.append(child)

        return self.best_value

# --- CLASSI SUPPORTO ---
class KnapsackProblem:
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = items
        for item in self.items: item['r'] = item['v'] / item['p']
        self.items_sorted = sorted(self.items, key=lambda x: x['r'], reverse=True)
        self.items_original_order = sorted(self.items, key=lambda x: x['id'])
    
    def _format_vector(self, solution_dict):
        vector = []
        for item in self.items_original_order:
            value = solution_dict.get(item['id'], 0)
            if isinstance(value, Fraction):
                val = float(value)
                vector.append("1" if val==1.0 else "0" if val==0.0 else f"{value.numerator}/{value.denominator}")
            else:
                vector.append(str(int(value)) if value == int(value) else f"{value:.2f}")
        return f"( {', '.join(vector)} )"

    def solve_bin_rel(self):
        rem_w = self.capacity; total_value = 0; solution = {}
        for item in self.items_sorted:
            if item['p'] <= rem_w: rem_w -= item['p']; total_value += item['v']; solution[item['id']] = 1
            else: fraction = Fraction(int(rem_w), int(item['p'])); total_value += float(fraction) * item['v']; solution[item['id']] = fraction; rem_w = 0; break
        return total_value, math.floor(total_value), solution

    def solve_binary_greedy(self):
        remaining_w = self.capacity; total_value = 0; solution = {}
        for item in self.items_sorted:
            if item['p'] <= remaining_w: remaining_w -= item['p']; total_value += item['v']; solution[item['id']] = 1
            else: solution[item['id']] = 0
        return total_value, solution

def load_data(filename):
    items = []; capacity = 0
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            counter = 1
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == "CAPACITY": capacity = float(parts[1])
                elif parts[0] == "ITEM": items.append({'id': counter, 'v': float(parts[1]), 'p': float(parts[2])}); counter += 1
                elif len(parts) == 2 and parts[0].replace('.', '', 1).isdigit(): items.append({'id': counter, 'v': float(parts[0]), 'p': float(parts[1])}); counter += 1
    except FileNotFoundError: print(f"File {filename} not found"); sys.exit(1)
    return capacity, items

def main():
    filename = 'knapsack.txt'
    if len(sys.argv) > 1: filename = sys.argv[1]
    capacity, items = load_data(filename)
    kp = KnapsackProblem(capacity, items)
    
    print(">>> Item Details (v, p, ratio):")
    for item in kp.items_sorted:
        print(f"    Item {item['id']}: v={int(item['v'])}, p={int(item['p'])}, r={item['r']:.2f}")

    print("\n=== 1. PRE-PROCESSING ===")
    lower_value_bin, greedy_sol_bin = kp.solve_binary_greedy()
    print(f"Lower Bound (Greedy) v_I = {int(lower_value_bin)}")
    upper_exact, upper_floor, sol_rel = kp.solve_bin_rel()
    print(f"Upper Bound (Relax) v_S = {upper_exact:.2f}")

    print("\n=== 2. BRANCH AND BOUND ===")
    solver = TreeVisualizer(capacity, items)
    solver.solve(initial_best_value=lower_value_bin)
    
    print("\n=== OPTIMAL RESULT ===")
    print(f"   Value = {int(solver.best_value)}")

if __name__ == "__main__":
    main()