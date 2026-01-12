import sys
import math
import numpy as np 
from fractions import Fraction
from collections import deque, defaultdict
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
        
        # Ordinamento per efficienza
        self.items_sorted = sorted(self.items, key=lambda x: x['r'], reverse=True)
        self.items_by_id = sorted(self.items, key=lambda x: x['id'])
        self.best_value = 0

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
            return None # Not feasible

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
                    
                    # Calcolo esplicito della frazione
                    f_obj = Fraction(frac).limit_denominator()
                    calc_parts.append(f"({f_obj.numerator}/{f_obj.denominator} * {int(item['v'])})")
                    
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
        print(f"\n--- BRANCH & BOUND ---")
        self.best_value = initial_best_value
        
        root_data = self.precalc_node([])
        root = Node(0, [], "Root")
        if root_data:
            for k, v in root_data.items(): setattr(root, k, v)
        
        # USARE DEQUE PER BFS (CODA)
        queue = deque([root])
        
        # Contatori per etichette Pi,j
        level_counters = defaultdict(int)
        
        while queue:
            curr_node = queue.popleft() # PRENDI IL PRIMO (FIFO)
            
            # Etichettatura P_{i,j}
            if curr_node.level == 0:
                node_label = "P"
                level_counters[0] = 1
            else:
                level_counters[curr_node.level] += 1
                idx = level_counters[curr_node.level]
                node_label = f"P_{{{curr_node.level},{idx}}}"

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
            
            status_str = "FEASIBLE (Integer)" if curr_node.is_integer else f"INFEASIBLE (Frac x{curr_node.split_item})"
            
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
            
            # Generazione figli
            # PER ORDINE BFS E GRAFICO STANDARD: Genero prima x=0 (sinistra), poi x=1 (destra)
            
            # Branch x_i = 0
            child_0_data = self.precalc_node(curr_node.taken + [(curr_node.split_item, 0)])
            node_0 = None
            if child_0_data:
                node_0 = Node(curr_node.level + 1, curr_node.taken + [(curr_node.split_item, 0)], f"x{curr_node.split_item}=0")
                for k, v in child_0_data.items(): setattr(node_0, k, v)

            # Branch x_i = 1
            child_1_data = self.precalc_node(curr_node.taken + [(curr_node.split_item, 1)])
            node_1 = None
            if child_1_data:
                node_1 = Node(curr_node.level + 1, curr_node.taken + [(curr_node.split_item, 1)], f"x{curr_node.split_item}=1")
                for k, v in child_1_data.items(): setattr(node_1, k, v)
            
            # AGGIUNTA ALLA CODA (FIFO)
            # Aggiungo prima 0 poi 1 così verranno visitati in quest'ordine nel prossimo livello
            if node_0: queue.append(node_0)
            if node_1: queue.append(node_1)

        return self.best_value

class KnapsackProblem:
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = [i.copy() for i in items]
        for item in self.items: 
            item['r'] = item['v'] / item['p']
        self.items_sorted = sorted(self.items, key=lambda x: x['r'], reverse=True)
        self.items_by_id = sorted(self.items, key=lambda x: x['id'])
    
    def solve_binary_detailed(self):
        print(f"\n" + "="*60)
        print(f"INITIAL V_S, V_I: BINARY CASE (0/1)")
        print(f"="*60)
        
        print(f"\n>>> 1. CALCULATING LOWER BOUND BINARY (GREEDY) v_I")
        rem_w = self.capacity
        val_greedy = 0
        print(f"   Available Capacity: {rem_w}")
        
        for item in self.items_sorted:
            print(f"   -> Item {item['id']} (v={item['v']}, p={item['p']}, r={item['r']:.2f}): ", end="")
            
            if item['p'] <= rem_w: 
                rem_w -= item['p']
                val_greedy += item['v']
                print(f"TAKEN. (v_tot={val_greedy}, Rem. Cap={rem_w})")
            else: 
                print(f"DISCARDED. (Weight {item['p']} > Rem. Cap {rem_w})")
        
        print(f"   [RESULT] v_I (Binary) = {val_greedy}")

        print(f"\n>>> 2. CALCULATING UPPER BOUND BINARY (RELAXED) v_S")
        rem_w = self.capacity
        val_relax = 0
        
        for item in self.items_sorted:
            print(f"   -> Item {item['id']} (v={item['v']}, p={item['p']}): ", end="")
            
            if item['p'] <= rem_w: 
                rem_w -= item['p']
                val_relax += item['v']
                print(f"TAKEN 1.0. (v_tot: {val_relax})")
            else: 
                fraction = rem_w / item['p']
                added = fraction * item['v']
                val_relax += added
                print(f"TAKEN FRACTION: {rem_w} / {item['p']} = {fraction:.4f}")
                print(f"      Calculation: {fraction:.4f} * {item['v']} = {added:.4f}")
                print(f"      Capacity exhausted.")
                break
        
        print(f"   [RESULT] v_S (Binary) = {val_relax:.2f}")
        return val_greedy, val_relax

    def solve_integer_detailed(self):
        print(f"\n" + "="*60)
        print(f"INITIAL V_S, V_I: INTEGER")
        print(f"="*60)
        print(f"we use the most efficient item.")
        
        best_item = self.items_sorted[0]
        print(f"\n   Most Efficient Item (Basis): Item {best_item['id']} (v={best_item['v']}, p={best_item['p']}, r={best_item['r']:.4f})")
        
        print(f"\n>>> 1. CALCULATING UPPER BOUND INTEGER (RELAXED) v_S")
        print(f"   Hypothesis: Fill knapsack entirely with Item {best_item['id']} (fractional).")
        
        count_frac = self.capacity / best_item['p']
        val_relax = count_frac * best_item['v']
        
        print(f"   Copies calculation: Capacity {self.capacity} / Weight {best_item['p']} = {count_frac:.4f}")
        print(f"   Value calculation: {count_frac:.4f} * {best_item['v']} = {val_relax:.4f}")
        print(f"   [RESULT] v_S (Integer) = {val_relax:.2f}")

        print(f"\n>>> 2. CALCULATING LOWER BOUND INTEGER (GREEDY) v_I")
        
        count_int = math.floor(count_frac)
        val_int = count_int * best_item['v']
        rem_cap = self.capacity - (count_int * best_item['p'])
        
        print(f"   Take only integer copies of Item {best_item['id']}: floor({count_frac:.4f}) = {count_int}")
        print(f"   Value calculation: {count_int} * {best_item['v']} = {val_int}")
        print(f"   Remaining capacity: {rem_cap}")
        for item in self.items:
            if item['p'] < rem_cap:
                print(f"  Can also take item {item["id"]} (cap = {item['p']}): {val_int}  + {item['v']} = {val_int  + item['v']}")

        
        return val_int, val_relax

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
        
        val_x_base = self.capacity / base_w
        
        print(f"\n--- GOMORY CUT GENERATION (Basis x{base_id}) ---")
        
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
    print(f"\n" + "="*60)
    print("CHECK COINCIDENCE (Binary Optimum vs Integer Optimum)")
    print(f"="*60)
    
    best_item = max(items, key=lambda x: x['v']/x['p'])
    test_cap = int(best_item['p'])

    print(f"Testing with Capacity = weight of Item {best_item['id']} ({test_cap})")
    
    kp_bin = KnapsackProblem(test_cap, items)
    print("\n--- Binary Case ---")
    val_greedy, _ = kp_bin.solve_binary_detailed()
    
    solver_bin = TreeVisualizer(test_cap, items)
    sys.stdout = open(os.devnull, 'w')
    solver_bin.solve(initial_best_value=val_greedy)
    sys.stdout = sys.__stdout__
    
    bin_opt = solver_bin.best_value
    print(f"   >>> Binary Optimum: {int(bin_opt)}")
    
    int_opt = best_item['v']
    print(f"\n--- Integer Case ---")
    print(f"   With capacity {test_cap}, we can take exactly 1 unit of Item {best_item['id']}.")
    print(f"   >>> Integer Optimum: {int(int_opt)}")
    
    if bin_opt == int_opt:
        print("\n[OK] Solutions coincide.")
    else:
        print("\n[DIFF] Solutions do not coincide.")

def main():
    filename = 'knapsack.txt'
    if len(sys.argv) > 1: filename = sys.argv[1]
    
    capacity, items = load_data(filename)
    
    kp = KnapsackProblem(capacity, items)
    
    print(">>> Item Details (v, p, ratio):")
    for item in kp.items_sorted:
        print(f"    Item {item['id']}: v={int(item['v'])}, p={int(item['p'])}, r={item['r']:.2f}")

    bin_lower, bin_upper = kp.solve_binary_detailed()
    int_lower, int_upper = kp.solve_integer_detailed()

    solver_bin = TreeVisualizer(capacity, items)
    solver_bin.solve(initial_best_value=bin_lower)
    print(f"\n>>> Binary Optimum found: {int(solver_bin.best_value)}")
    
    try:
        user_cap_str = input("\nDo you want to change capacity for the Integer case? (Press Enter to keep " + str(int(capacity)) + "): ").strip()
        capacity_int = int(user_cap_str) if user_cap_str else int(capacity)
    except ValueError:
        capacity_int = int(capacity)

    gomory_solver = GomoryKnapsackInteger(capacity_int, items)
    gomory_solver.solve_and_generate_cut()

    check_coincidence(items)

if __name__ == "__main__":
    main()