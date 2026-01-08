import re
import numpy as np
import os
import copy
from fractions import Fraction

INPUT_FILE = "data.txt"
OUTPUT_FILE = "dual.txt"

def format_fraction(val):
    if abs(val) < 1e-9: return "0"
    if abs(val - round(val)) < 1e-4: return str(int(round(val)))
    f = Fraction(val).limit_denominator(1000)
    if f.denominator == 1: return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"

def parse_primal(filename):
    if not os.path.exists(filename): raise FileNotFoundError("File not found")
    with open(filename, 'r') as f: lines = f.readlines()

    c = None
    constrs = []
    basis_ids = []
    is_min = True
    
    mode = None 

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"): continue
        
        if "target" in line:
            if "max" in line.lower(): is_min = False
            elif "min" in line.lower(): is_min = True
            continue
        
        if line.startswith("c:"):
            content = line.split(":")[1].strip(" []")
            c = np.array([float(x) for x in content.split(",")])
            continue
            
        if line.startswith("B:"):
            content = line.split(":")[1].strip(" []")
            basis_ids = [int(x) for x in content.split(",")]
            continue

        if "[POINT]" in line: mode = "POINT"; continue
        elif "[CONSTRAINTS]" in line or "[VINCOLI]" in line: mode = "CONSTR"; continue

        if mode == "CONSTR" and ":" in line:
            parts = line.split(":")
            try:
                cid = int(parts[0].strip())
                rest = parts[1]
                
                sign = "="
                if "<=" in rest: sign = "<="; lhs, rhs = rest.split("<=")
                elif ">=" in rest: sign = ">="; lhs, rhs = rest.split(">=")
                else: lhs, rhs = rest.split("=")
                
                row = np.zeros(len(c)) if c is not None else np.zeros(2)
                
                lhs = lhs.replace("- ", "+-").replace("-x", "+-1*x")
                terms = lhs.split("+")
                for t in terms:
                    t = t.strip()
                    if not t: continue
                    coeff = 1.0
                    if "*" in t: 
                        co, var = t.split("*")
                        coeff = float(co.replace(" ", ""))
                        var_clean = var
                    elif "x" in t: var_clean = t
                    else: continue
                    
                    match = re.search(r'x(\d+)', var_clean)
                    if match:
                        idx = int(match.group(1)) - 1
                        if idx < len(row): row[idx] = coeff
                
                constrs.append({
                    'id': cid, 'A': row, 'sign': sign, 'b': float(rhs)
                })
            except Exception: pass

    calc_constrs = []
    if is_min and c is not None:
        c_calc = -c
        for con in constrs:
            new_con = copy.deepcopy(con)
            if new_con['sign'] == ">=":
                new_con['A'] = -new_con['A']
                new_con['b'] = -new_con['b']
            calc_constrs.append(new_con)
    else:
        c_calc = c
        calc_constrs = constrs

    return c, c_calc, constrs, calc_constrs, basis_ids, is_min

def write_dual_file(c, constrs, is_min, filename):
    n = len(c)
    with open(filename, "w") as f:
        target = "max" if is_min else "min"
        
        obj = []
        for con in constrs:
            if abs(con['b']) > 1e-9: 
                val = con['b']
                if abs(val - round(val)) < 1e-4: val = int(round(val))
                obj.append(f"{val}*y{con['id']}")
        
        if not obj: f.write(f"target: {target} 0\n")
        else: f.write(f"target: {target} {' + '.join(obj)}\n")

        f.write("\n[CONSTRAINTS]\n")
        
        rel = "<=" if is_min else ">="
        for j in range(n): 
            terms = []
            for con in constrs:
                coeff = con['A'][j]
                if abs(coeff) > 1e-9:
                    c_val = coeff
                    if abs(c_val - round(c_val)) < 1e-4: c_val = int(round(c_val))
                    terms.append(f"{c_val}*y{con['id']}")
            
            c_rhs = c[j]
            if abs(c_rhs - round(c_rhs)) < 1e-4: c_rhs = int(round(c_rhs))
            f.write(f"{' + '.join(terms)} {rel} {c_rhs}\n")
            
    print(f"Dual text saved to {filename}")

def solve_dual_values(c_calc, calc_constrs, basis_ids):
    if not basis_ids:
        print("No basis found. Cannot solve dual.")
        return

    con_map = {x['id']: x for x in calc_constrs}
    rows = []
    
    try:
        for bid in basis_ids:
            rows.append(con_map[bid]['A'])
    except KeyError:
        print("Error: Basis ID not found in constraints.")
        return

    A_B = np.array(rows)
    
    try:
        y_vals = np.linalg.solve(A_B.T, c_calc)
    except np.linalg.LinAlgError:
        print("Singular matrix. Cannot solve dual.")
        return

    max_id = max(c['id'] for c in calc_constrs)
    full_y = ["0"] * (max_id + 1)
    
    for i, bid in enumerate(basis_ids):
        val = y_vals[i]
        full_y[bid] = format_fraction(val)

    final_str = "(" + ", ".join(full_y[1:]) + ")"
    
    print(f"solution using the starting point: y = {final_str}")
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"\nsolution using the starting point: y = {final_str}\n")

if __name__ == "__main__":
    try:
        c, c_calc, constrs, calc_constrs, basis, is_min = parse_primal(INPUT_FILE)
        write_dual_file(c, constrs, is_min, OUTPUT_FILE)
        print(c,c_calc)
        solve_dual_values(c, calc_constrs, basis)
    except Exception as e:
        print(f"Error: {e}")