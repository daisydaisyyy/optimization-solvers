import sys
import math
import numpy as np 
from fractions import Fraction
import os

class Node:
    def __init__(self, level, taken, path_desc):
        self.level = level                  
        self.taken = taken               
        self.path_desc = path_desc 
        
        self.sol_vector = {}
        self.curr_weight = 0.0
        self.curr_value = 0.0
        self.is_integer = False
        self.split_item = None
        self.calc_str = ""

class TreeVisualizer:
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = [i.copy() for i in items]
        for item in self.items:
            item['r'] = item['v'] / item['p']
        
        self.items_sorted = sorted(self.items, key=lambda x: x['r'], reverse=True)
        self.items_by_id = sorted(self.items, key=lambda x: x['id'])
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
        print(f"\n--- STARTING BRANCH & BOUND (Capacity {self.capacity}, Binary Case) ---")
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

            if bound_floor < self.best_value:
                 print(f"{indent}    -> ACTION: PRUNING: {bound_floor} < {int(self.best_value)}")
                 continue
            
            if bound_floor == self.best_value and not curr_node.is_integer:
                 print(f"{indent}    -> ACTION: PRUNING: {bound_floor} <= {int(self.best_value)} (Cannot improve)")
                 continue

            if curr_node.is_integer:
                continue 

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
            
            if node_1: stack.append(node_1)
            if node_0: stack.append(node_0)

        return self.best_value

class KnapsackProblem:
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = items
        for item in self.items: item['r'] = item['v'] / item['p']
        self.items_sorted = sorted(self.items, key=lambda x: x['r'], reverse=True)
        self.items_original_order = sorted(self.items, key=lambda x: x['id'])
    
    def solve_binary_greedy(self):
        remaining_w = self.capacity; total_value = 0; solution = {}
        for item in self.items_sorted:
            if item['p'] <= remaining_w: remaining_w -= item['p']; total_value += item['v']; solution[item['id']] = 1
            else: solution[item['id']] = 0
        return total_value, solution
    
    def solve_bin_rel(self):
        rem_w = self.capacity; total_value = 0; solution = {}
        for item in self.items_sorted:
            if item['p'] <= rem_w: rem_w -= item['p']; total_value += item['v']; solution[item['id']] = 1
            else: fraction = Fraction(int(rem_w), int(item['p'])); total_value += float(fraction) * item['v']; solution[item['id']] = fraction; rem_w = 0; break
        return total_value, math.floor(total_value), solution

class GomoryKnapsackInteger:
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = [i.copy() for i in items]
        for item in self.items:
            item['r'] = item['v'] / item['p']
        
        self.items_sorted = sorted(self.items, key=lambda x: x['r'], reverse=True)
        self.items_by_id = sorted(self.items, key=lambda x: x['id'])

    def solve_and_generate_cut(self):
        best_item = self.items_sorted[0]
        base_id = best_item['id']
        base_w = best_item['p']
        base_v = best_item['v']
        
        print(f"Most efficient item (Basis): Item {base_id} (v={base_v}, p={base_w}, r={best_item['r']:.4f})")
        
        val_x_base = self.capacity / base_w
        opt_relaxed_val = val_x_base * base_v
        
        print(f"\n--- RELAXED SOLUTION ---")
        print(f"x{base_id} = {self.capacity} / {base_w} = {val_x_base:.2f}")
        for item in self.items_by_id:
            if item['id'] != base_id:
                print(f"x{item['id']} = 0")
        
        print(f"v_S(P) (Relaxed Value) = {opt_relaxed_val:.2f}")
        
        qty_int = math.floor(val_x_base)
        opt_int_val = qty_int * base_v
        rem_cap = self.capacity - (qty_int * base_w)
        
        print(f"\n--- FEASIBLE SOLUTION (Lower Bound) ---")
        print(f"x{base_id} = {qty_int} (integers)")
        print(f"Remaining capacity: {rem_cap}")
        print(f"v_I(P) (Integer Basis Value) = {opt_int_val}")
        
        print(f"\n--- GOMORY CUT GENERATION ---")
        print(f"Optimal row equation (Basis x{base_id}):")
        
        rhs_frac = Fraction(int(self.capacity), int(base_w))
        f0 = rhs_frac - math.floor(rhs_frac)
        
        print(f"RHS = {self.capacity}/{base_w} = {float(rhs_frac):.2f} -> f0 = {f0}")
        
        cut_terms = []
        
        print("Calculating fractional coefficients (f_j):")
        for item in self.items_by_id:
            if item['id'] == base_id:
                continue
            
            coeff = Fraction(int(item['p']), int(base_w))
            fj = coeff - math.floor(coeff)
            
            print(f"  x{item['id']}: w={item['p']} -> coeff={coeff} -> f_{item['id']} = {fj}")
            
            if fj > 0:
                cut_terms.append((fj, f"x{item['id']}"))
                
        coeff_s = Fraction(1, int(base_w))
        fs = coeff_s - math.floor(coeff_s)
        slack_name = f"x{len(self.items)+1}" 
        print(f"  {slack_name} (slack): coeff={coeff_s} -> f_s = {fs}")
        
        if fs > 0:
            cut_terms.append((fs, slack_name))
            
        cut_str_frac = " + ".join([f"{val} {name}" for val, name in cut_terms])
        print(f"\nFractional Cut:\n{cut_str_frac} >= {f0}")
        
        print(f"\nInteger Cut (multiplied by {int(base_w)}):")
        
        cut_int_terms = []
        for val, name in cut_terms:
            int_val = int(val * base_w) 
            cut_int_terms.append(f"{int_val}{name}")
            
        rhs_int = int(f0 * base_w)
        
        cut_str_int = " + ".join(cut_int_terms)
        print(f"{cut_str_int} >= {rhs_int}")

def load_data(filename):
    items = []
    capacity = 0
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            counter = 1
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == "CAPACITY": capacity = float(parts[1])
                elif parts[0] == "ITEM":
                    items.append({'id': counter, 'v': float(parts[1]), 'p': float(parts[2])})
                    counter += 1
                elif len(parts) == 2 and parts[0].replace('.', '', 1).isdigit():
                    items.append({'id': counter, 'v': float(parts[0]), 'p': float(parts[1])})
                    counter += 1
    except FileNotFoundError:
        print(f"File {filename} not found")
        sys.exit(1)
    return capacity, items

def check_coincidence(items):
    print("EXAMPLE WHERE BINARY OPTIMUM == INTEGER OPTIMUM?")
    
    best_item = max(items, key=lambda x: x['v']/x['p'])
    test_cap = int(best_item['p'])

    print(f"Using capacity = weight of most efficient Item {best_item['id']}: {test_cap}")
    
    print(f"\n1) Binary Case (Capacity {test_cap}):")
    kp_bin = KnapsackProblem(test_cap, items)
    val_greedy, _ = kp_bin.solve_binary_greedy()
    solver_bin = TreeVisualizer(test_cap, items)
    solver_bin.solve(initial_best_value=val_greedy)
    bin_opt = solver_bin.best_value
    print(f">>> Binary opt: {int(bin_opt)}")
    
    print(f"\n2) Integer Case (Capacity {test_cap}):")
    int_opt = best_item['v']
    print(f"Item {best_item['id']} has max efficiency.")
    print(f"With capacity {test_cap}, we can fit exactly 1 unit.")
    print(f">>> Integer Optimum: {int(int_opt)}")
    
    if bin_opt == int_opt:
        print("\nOK, Solutions coincide.")
    else:
        print("\nNot equal.")

def main():
    filename = 'knapsack.txt'
    if len(sys.argv) > 1: filename = sys.argv[1]
    
    capacity, items = load_data(filename)
    
    kp_bin = KnapsackProblem(capacity, items)
    
    print(">>> Item Details (v, p, ratio):")
    for item in kp_bin.items_sorted:
        v_int = int(item['v'])
        p_int = int(item['p'])
        ratio_float = item['r']
        print(f"    Item {item['id']}: v={v_int}, p={p_int}, r={ratio_float:.2f}")

    print("\n=== 1. PRE-PROCESSING ===")
    
    lower_value_bin, greedy_sol_bin = kp_bin.solve_binary_greedy()
    print(f"Lower Bound (Greedy) v_I = {int(lower_value_bin)}")
    
    upper_exact_bin, upper_floor_bin, rel_sol_bin = kp_bin.solve_bin_rel()
    print(f"Upper Bound (Relax) v_S = {upper_exact_bin:.2f}")

    print("\n" + "="*50)
    print(f"BINARY CASE (Capacity {capacity})")
    print("="*50)
    
    solver_bin = TreeVisualizer(capacity, items)
    solver_bin.solve(initial_best_value=lower_value_bin)
    print(f"\n>>> Binary Optimum found: {int(solver_bin.best_value)}")
    
    print("\n" + "="*50)
    print("INTEGER PROBLEM, GOMORY CUT")
    print("="*50)
    
    try:
        user_cap_str = input("Enter capacity for integer case: ").strip()
        capacity_int = int(user_cap_str) if user_cap_str else int(capacity)
    except ValueError:
        print(f"Invalid input, using {capacity}.")
        capacity_int = int(capacity)

    gomory_solver = GomoryKnapsackInteger(capacity_int, items)
    gomory_solver.solve_and_generate_cut()

    check_coincidence(items)

if __name__ == "__main__":
    main()