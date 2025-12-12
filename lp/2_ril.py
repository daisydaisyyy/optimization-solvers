import numpy as np
import re
import os
from scipy.optimize import linprog
INPUT_FILE = "data.txt"
direction = 0
def parse_input(filename):
    if not os.path.exists(filename):
        print(f"{filename} not found.")
        return None, None, None

    with open(filename, 'r') as f:
        lines = f.readlines()

    c_vec = None
    A_rows = []
    b_vec = []
    num_vars = 0
    
    for line in lines:
        line = line.strip()
        if 'max' in line.lower():
            direction = 'max'
        if line.startswith("c:"):
            content = line.split(":")[1].replace('[','').replace(']','').strip()
            c_vec = np.array([float(x) for x in content.split(',')])
            num_vars = len(c_vec)
            break

    print(direction)
    if num_vars == 0:
        print("vector c not found")
        return None, None, None

    direction = 'max' if direction != 0 else 'min'

    count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("B:") or "obiettivo" in line or "[PUNTO]" in line: 
            continue
        
        if ":" in line and ("<=" in line or ">=" in line or "=" in line):
            if not line.split(':')[0].strip().isdigit(): continue

            if "<=" in line: sep="<="; mult = 1
            elif ">=" in line: sep=">="; mult = -1 
            else: sep="="; mult = 1 
            
            parts = line.split(sep)
            lhs = parts[0].split(':')[1]
            rhs = float(parts[1])
            
            row = np.zeros(num_vars)
            terms = lhs.replace('-', '+-').split('+')
            for term in terms:
                term = term.strip()
                if not term: continue
                coeff = 1.0
                if '*' in term:
                    c_s, v_s = term.split('*')
                    coeff = float(c_s.replace(" ", ""))
                else:
                    v_s = term
                    if v_s.startswith("-"): coeff = -1.0; v_s = v_s[1:]
                
                match = re.search(r'x(\d+)', v_s)
                if match:
                    idx = int(match.group(1)) - 1
                    if idx < num_vars: row[idx] = coeff

            A_rows.append(row * mult)
            b_vec.append(rhs * mult)
            count += 1

    return np.array(A_rows), np.array(b_vec), c_vec

def solve_relaxed(A, b, c):
    print("\nsolve continuous relaxed")
    
    res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)]*len(c), method='highs')
    z_val = -res.fun if direction == 'max' else res.fun
    if not res.success:
        print(f"error: {res.message}")
        return

    print(f"\nZ (f): {-res.fun:,.2f}")
    for i, val in enumerate(res.x):
        print(f"  x{i+1}: {val:.4f}")
        
    formatted_vals = []
    for val in res.x:
        if abs(val - round(val)) < 1e-3:
            formatted_vals.append(f"{int(round(val))}")
        else:
            formatted_vals.append(f"{val:.1f}")
            
    vec_str = ", ".join(formatted_vals)
    print(f"\nx=({vec_str})")
    print("\nvalue:",z_val)
    output_file = "optim_sol.txt";
    try:
        with open(output_file, 'w') as f:
            f.write(vec_str)
        print(f"\nsolution saved: see '{output_file}'.")
    except Exception as e:
        print(f"\nwrite error")
# --- MAIN ---
if __name__ == "__main__":
    A, b, c = parse_input(INPUT_FILE)
    if A is not None:
        solve_relaxed(A, b, c)