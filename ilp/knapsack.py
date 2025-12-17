import sys
import math
import numpy as np # Aggiunto per scipy
from scipy.optimize import milp, LinearConstraint, Bounds # Aggiunto per scipy
from fractions import Fraction

class Node:
    def __init__(self, level, current_weight, current_value, taken, path_desc):
        self.level = level                  # idx in the sorted item array
        self.curr_weight = current_weight
        self.current_value = current_value
        self.taken = taken                  # list of tuples (item_id, 0/1)
        self.path_desc = path_desc          # path description string
        self.bound = 0.0

class TreeVisualizer:
    def __init__(self, capacity, items):
        self.capacity = capacity
        for item in items:
            item['r'] = item['v'] / item['p']
        
        # sort by decreasing profit ratio
        self.items_sorted = sorted(items, key=lambda x: x['r'], reverse=True)
        self.n = len(items)
        self.best_value = 0
        self.best_solution = []
        self.node_counter = 0 

    def calculate_bound(self, node):
        if node.curr_weight > self.capacity:
            return 0, "NOT FEASIBLE"

        bound_value = node.current_value
        remaining_weight = self.capacity - node.curr_weight
        detail_str = f"Base({node.current_value})"

        for i in range(node.level, self.n):
            item = self.items_sorted[i]
            if item['p'] <= remaining_weight:
                remaining_weight -= item['p']
                bound_value += item['v']
                detail_str += f"+B{item['id']}(1)"
            else:
                fraction = remaining_weight / item['p']
                bound_value += item['v'] * fraction
                detail_str += f"+B{item['id']}({fraction:.2f})"
                remaining_weight = 0
                break
        
        return bound_value, detail_str

    def solve(self, initial_best_value=0):
        self.best_value = initial_best_value
        
        root = Node(0, 0, 0, [], "Root")
        bound_value, _ = self.calculate_bound(root)
        root.bound = bound_value
        
        print(f"Variable order (by ratio): {[i['id'] for i in self.items_sorted]}")
        
        stack = [root]

        while stack:
            curr_node = stack.pop()
            self.node_counter += 1
            node_name = f"N{self.node_counter}"
            
            indent = "    " * len(curr_node.taken)
            print(f"{indent}|-- [{node_name}] {curr_node.path_desc}")

            if curr_node.curr_weight > self.capacity:
                print(f"{indent}    -> PRUNED (Weight {curr_node.curr_weight} > Capacity {self.capacity})")
                continue

            if curr_node.level == self.n:
                print(f"{indent}    -> LEAF. Value: {curr_node.current_value}")
                if curr_node.current_value > self.best_value:
                    self.best_value = curr_node.current_value
                    self.best_solution = list(curr_node.taken)
                    print(f"{indent}    *** NEW OPT: {self.best_value} ***")
                continue

            bound_value, detail = self.calculate_bound(curr_node)
            bound_floor = math.floor(bound_value)
            
            print(f"{indent}    Bound: {detail} = {bound_value:.2f} (Floor: {bound_floor}) | Current Best: {self.best_value}")

            if bound_floor <= self.best_value:
                print(f"{indent}    -> PRUNED (Bound {bound_floor} <= Best {self.best_value})")
                continue

            next_item = self.items_sorted[curr_node.level]
            
            # DFS: push x=0 first, then x=1 so x=1 is explored first
            right_child = Node(
                curr_node.level + 1,
                curr_node.curr_weight,
                curr_node.current_value,
                curr_node.taken + [(next_item['id'], 0)],
                curr_node.path_desc + f", x{next_item['id']}=0"
                if curr_node.path_desc != "Root" else f"x{next_item['id']}=0"
            )
            stack.append(right_child)

            left_child = Node(
                curr_node.level + 1,
                curr_node.curr_weight + next_item['p'],
                curr_node.current_value + next_item['v'],
                curr_node.taken + [(next_item['id'], 1)],
                curr_node.path_desc + f", x{next_item['id']}=1"
                if curr_node.path_desc != "Root" else f"x{next_item['id']}=1"
            )
            stack.append(left_child)
        
        return self.best_value, self.best_solution

class KnapsackProblem:
    def __init__(self, capacity, items):
        self.capacity = capacity
        self.items = items
        for item in self.items:
            item['r'] = item['v'] / item['p']
        
        self.items_sorted = sorted(self.items, key=lambda x: x['r'], reverse=True)
        self.items_original_order = sorted(self.items, key=lambda x: x['id'])
        self.n = len(items)

    def _format_vector(self, solution_dict):
        vector = []
        for item in self.items_original_order:
            value = solution_dict.get(item['id'], 0)
            if isinstance(value, Fraction):
                decimal_value = float(value)
                if decimal_value == 1.0: vector.append("1")
                elif decimal_value == 0.0: vector.append("0")
                else: vector.append(f"{value.numerator}/{value.denominator}")
            else:
                if value == int(value): vector.append(str(int(value)))
                else: vector.append(f"{value:.2f}")
        return f"( {', '.join(vector)} )"

    # --- BINARY CASE (0/1) ---
    def solve_bin_rel(self):
        rem_w = self.capacity
        total_value = 0
        solution = {}
        basic_vars = []
        
        for item in self.items_sorted:
            if item['p'] <= rem_w:
                rem_w -= item['p']
                total_value += item['v']
                solution[item['id']] = 1
                basic_vars.append(item['id'])
            else:
                fraction = Fraction(int(rem_w), int(item['p']))
                total_value += float(fraction) * item['v']
                solution[item['id']] = fraction
                basic_vars.append(f"{item['id']} (fractional)")
                rem_w = 0
                break
        
        floor_value = math.floor(total_value)
        return total_value, floor_value, solution, basic_vars

    def solve_binary_greedy(self):
        remaining_w = self.capacity
        total_value = 0
        solution = {}
        for item in self.items_sorted:
            if item['p'] <= remaining_w:
                remaining_w -= item['p']
                total_value += item['v']
                solution[item['id']] = 1
            else:
                solution[item['id']] = 0
        return total_value, solution

    # --- INTEGER CASE ---
    def solve_integer_greedy(self):
        """integer greedy: fill as much as possible with the best item, then move on."""
        remaining_w = self.capacity
        tot_value = 0
        solution = {}
        for item in self.items_sorted:
            if remaining_w <= 0:
                break
            quantity = int(remaining_w // item['p'])
            if quantity > 0:
                tot_value += quantity * item['v']
                remaining_w -= quantity * item['p']
                solution[item['id']] = quantity
        return tot_value, solution

    def solve_integer_relaxation(self):
        """integer rel: fill the knapsack entirely with the best-ratio item."""
        best_item = self.items_sorted[0]
        value = self.capacity * best_item['r']
        return value, best_item

def load_data(filename):
    items = []
    capacity = 0
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            counter = 1
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "CAPACITY":
                    capacity = float(parts[1])
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

def check_conflict(solution_vector, forbidden_ids):
    """checks whether the specified items are selected simultaneously."""
    conflict_count = 0
    names = []
    for fid in forbidden_ids:
        if solution_vector.get(fid, 0) == 1:
            conflict_count += 1
        names.append(str(fid))
    
    conflict_str = ", ".join(names)
    if conflict_count == len(forbidden_ids):
        print(f" -> WARNING: The solution includes ALL forbidden items together ({conflict_str}).")
        print("    The optimal solution WOULD CHANGE if this constraint were active.")
    else:
        print(f" -> OK: The solution does NOT include the items together ({conflict_str}).")
        print("    The optimal solution would remain valid.")

def main():
    filename = 'knapsack.txt'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    capacity, items = load_data(filename)
    kp = KnapsackProblem(capacity, items)
    print(">>> Ratios details (v/p):")
    for item in kp.items_sorted:
        v_int = int(item['v'])
        p_int = int(item['p'])
        ratio_float = item['r']
        print(f"    Item {item['id']}: {v_int}/{p_int} = {ratio_float:.4f}")
    
    print(f">>> Sorted items: {[i['id'] for i in kp.items_sorted]}")

    # 1. BINARY CASE
    print("\n=== BINARY CASE (0/1) ===")
    
    lower_value_bin, greedy_sol_bin = kp.solve_binary_greedy()
    print("1) v_I = Lower Bound (Greedy):")
    print(f"   Feasible solution = {kp._format_vector(greedy_sol_bin)}")
    print(f"   v_I(P) = {int(lower_value_bin)}")

    upper_exact_bin, upper_floor_bin, rel_sol_bin, basis_vars = kp.solve_bin_rel()
    print("2) v_S = Upper Bound (cont rel):")
    print(f"   Opt relaxed solve = {kp._format_vector(rel_sol_bin)}")
    print(f"   v_S(P) = {upper_floor_bin} (exact: {upper_exact_bin:.2f})")
    print(f"   Base: {basis_vars}")

    # INTEGER CASE
    print("\n=== INTEGER CASE ===")
    
    lower_value_int, greedy_solution_int = kp.solve_integer_greedy()
    print("1) v_I = Lower Bound (Greedy):")
    print(f"   Feasible solution = {kp._format_vector(greedy_solution_int)}")
    print(f"   v_I(P) = {int(lower_value_int)}")
    
    upper_exact_int, best_item = kp.solve_integer_relaxation()
    print("2) v_S = Upper Bound (cont rel):")

    print(f"   Fill entirely with item {best_item['id']} (ratio {best_item['r']:.4f})")
    # optimal relaxed solution is all 0 except the best item which is capacity/item weight
    print("   opt rel sol = (", end="")
    
    vector_elems = []
    sorted_items = sorted(kp.items, key=lambda x: x['id'])
    
    for item in sorted_items:
        if item['id'] == best_item['id']:
            frac = Fraction(int(kp.capacity), int(item['p']))
            vector_elems.append(f"{frac.numerator}/{frac.denominator}")
        else:
            vector_elems.append("0")
    
    print(", ".join(vector_elems) + ")")
    print(f"   v_S(P) = {int(upper_exact_int)} (Exact: {upper_exact_int:.2f})")

    # matlab-like intlinprog to find integer optimal solution
    items_by_id = sorted(items, key=lambda x: x['id'])
    v_arr = np.array([i['v'] for i in items_by_id])
    p_arr = np.array([i['p'] for i in items_by_id])
    
    # c = -v (min problem)
    c = -v_arr
    # constraint: p*x <= capacity
    A = p_arr.reshape(1, -1)
    constraints = LinearConstraint(A, lb=[-np.inf], ub=[capacity])
    int_constraint = np.ones(len(items))
    bounds = Bounds(lb=0, ub=np.inf) # set lower and upper bounds
    
    res = milp(c=c, constraints=constraints, integrality=int_constraint, bounds=bounds)
    
    if res.success:
        opt_val_int = -res.fun
        sol_opt_int_dict = {items_by_id[i]['id']: int(round(x)) for i, x in enumerate(res.x)}
        
        print("3) opt integer result:")
        print(f"   opt sol = {kp._format_vector(sol_opt_int_dict)}")
        print(f"   v(P) = {int(opt_val_int)}")
    else:
        print("3) opt result: solver failed")

    # BRANCH AND BOUND (BINARY)
    print("\n=== BRANCH AND BOUND (binary) ===")
    solver = TreeVisualizer(capacity, items)
    opt_value, opt_sol_list = solver.solve(initial_best_value=lower_value_bin)
    
    opt_sol_dict = {item_id: value for (item_id, value) in opt_sol_list}
    for item in items:
        if item['id'] not in opt_sol_dict:
            opt_sol_dict[item['id']] = 0

    print("\n3) FINAL OPT RESULT (binary)")
    print(f"    x = {kp._format_vector(opt_sol_dict)}")
    print(f"   OPT value = {int(opt_value)}")


    # ADDITIONAL CONSTRAINT (i, j, k)
    skip_items = [1, 3, 6]
    
    print("\n--- ADDITIONAL CONSTRAINT CHECK ---")
    print(f"Question: If items {skip_items} could not be loaded together, would the solution change?")
    check_conflict(opt_sol_dict, skip_items)

if __name__ == "__main__":
    main()