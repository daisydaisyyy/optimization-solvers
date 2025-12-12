import numpy as np
from numpy.linalg import matrix_rank
import sys
import re

def file_write(optimal_basis, filename="data.txt"):
    try:
        with open(filename, "a") as f:
            f.write("\n")
            f.write(f"B: {optimal_basis}\n")
        print(f"B saved in {filename}")
    except IOError as e:
        print(f"Error writing file: {e}")

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f.read().splitlines() if l.strip() and not l.startswith("#")]
    except FileNotFoundError:
        print(f"ERROR: File '{filename}' does not exist.")
        sys.exit(1)

    point = []
    constraints = []
    mode = None

    for line in lines:
        if line == "[PUNTO]":
            mode = "POINT"
            continue
        elif line == "[VINCOLI]":
            mode = "CONSTR"
            continue

        if mode == "POINT":
            if "target" in line or line.startswith("c:"):
                continue
            try:
                point = [float(x.strip()) for x in line.split(',')]
            except ValueError:
                continue
        
        elif mode == "CONSTR":
            try:
                if ":" not in line: continue

                parts = line.split(':')
                id_str = parts[0].strip()
                if not id_str.isdigit(): continue
                
                con_id = int(id_str)
                expr_rest = parts[1].strip()
                
                op = ""
                if "<=" in expr_rest: op = "<="
                elif ">=" in expr_rest: op = ">="
                elif "=" in expr_rest: op = "="
                if not op: continue

                lhs_str, rhs_str = expr_rest.split(op)
                rhs = float(rhs_str)
                
                terms = []
                raw_terms = lhs_str.replace("-", "+-").split('+')
                
                for t in raw_terms:
                    t = t.strip()
                    if not t: continue
                    coeff = 1.0
                    var_name = t
                    
                    if '*' in t:
                        c, v = t.split('*')
                        coeff = float(c.strip())
                        var_name = v.strip()
                    elif t.startswith("-") and '*' not in t:
                        if t == "-": continue 
                        if len(t) > 1:
                            coeff = -1.0
                            var_name = t[1:]
                    
                    terms.append((coeff, var_name))
                
                constraints.append((con_id, terms, op, rhs))
            except Exception as e:
                print(f"Error reading constraint line: {line} -> {e}")

    return point, constraints

def solve():
    values, raw_constraints = read_file("data.txt")
    
    if not values:
        print("ERROR: No point found in the file.")
        return

    num_vars = len(values)
    var_names = [f"x{i+1}" for i in range(num_vars)]
    var_map = {n: i for i, n in enumerate(var_names)}

    print(f"{'='*80}")
    print("INITIAL POINT CHECK")
    print(f"{'='*80}")
    
    print(f"Point to analyze: {values}")
    print(f"Detected variables: {var_names}")

    active_indices = []
    active_gradients = []
    feasible = True

    for con_id, terms, op, rhs in raw_constraints:
        print(f"\nChecking Constraint {con_id}:")
        
        eq_parts = []
        subs_parts = []
        total = 0.0
        detail_parts = []
        gradient_row = [0.0] * num_vars

        for coeff, v_name in terms:
            v_name = v_name.strip()
            idx = -1
            if v_name in var_map:
                idx = var_map[v_name]
            else:
                match = re.search(r'x(\d+)', v_name)
                if match:
                    idx = int(match.group(1)) - 1
            
            if idx < 0 or idx >= num_vars:
                continue
            
            eq_parts.append(f"{coeff}*{v_name}")
            subs_parts.append(f"{coeff}({values[idx]})")
            
            partial = coeff * values[idx]
            total += partial
            detail_parts.append(f"{partial:g}")
            
            gradient_row[idx] = coeff

        eq_str = " + ".join(eq_parts)
        subs_str = " + ".join(subs_parts)
        calc_str = " + ".join(detail_parts)

        print(f"  Eq: {eq_str} {op} {rhs}")
        print(f"  Substitution: {subs_str}")

        diff = abs(total - rhs)
        is_active = diff < 1e-9
        
        status_str = ""
        violated = False
        
        if op == "<=" and total > rhs + 1e-9: 
            violated = True
        elif op == ">=" and total < rhs - 1e-9: 
            violated = True
        elif op == "=" and diff > 1e-9:
            violated = True

        if violated:
            status_str = "!!! VIOLATED !!!"
            feasible = False
        elif is_active:
            status_str = ">>> ACTIVE (Basis)"
            active_indices.append(con_id)
            active_gradients.append(gradient_row)
        else:
            status_str = "Not active (ok)"

        print(f"  Calculation:   {calc_str} = {total:g}")
        print(f"  Comparison:    {total:g} {op} {rhs} -> {status_str}")

    print("\nFINAL POINT ANALYSIS")

    if feasible:
        print("FEASIBILITY: YES (Point satisfies all constraints).")
    else:
        print("FEASIBILITY: NO (Point violates some constraints).")

    num_active = len(active_indices)
    print(f"Active constraints found: {num_active} (Indices: {active_indices})")
    
    is_vertex = False
    if num_active >= num_vars:
        if active_gradients:
            active_matrix = np.array(active_gradients)
            rank = matrix_rank(active_matrix)
            print(f"Active constraints matrix rank: {rank} (Variables: {num_vars})")
            
            if rank == num_vars:
                is_vertex = True
                print("VERTEX: YES (Rank = number of variables).")
            else:
                print(f"VERTEX: NO (Rank {rank} < {num_vars} -> dependent constraints).")
        else:
            print("VERTEX: NO (No active gradients calculated).")
    else:
        print(f"VERTEX: NO (Only {num_active} active constraints, at least {num_vars} needed).")

    print("\n" + "="*40)
    if feasible and is_vertex:
        print("CONCLUSION: The point is a valid feasible basic solution.")
        active_indices.sort()
        print(f"Initial Basis B = {active_indices}")
        file_write(active_indices)
    else:
        print("CONCLUSION: The point is NOT valid as a starting point for simplex.")
    print("="*40)

if __name__ == "__main__":
    solve()
