import numpy as np
import re
import os
from scipy.optimize import linprog
from fractions import Fraction

INPUT_FILE = "data.txt"

problem_direction = 'max' 

def parse_input(filename):
    global problem_direction
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
        if 'min' in line.lower():
            problem_direction = 'min'
        elif 'max' in line.lower():
            problem_direction = 'max'
            
        if line.startswith("c:"):
            content = line.split(":")[1].replace('[','').replace(']','').strip()
            parts = [x for x in content.split(',') if x.strip()]
            c_vec = np.array([float(x) for x in parts])
            num_vars = len(c_vec)
            break

    print(f"Problem type: {problem_direction}")
    if num_vars == 0:
        print("c not found")
        return None, None, None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("B:") or "target" in line or "c:" in line or "[PUNTO]" in line: 
            continue
        
        if ":" in line and ("<=" in line or ">=" in line or "=" in line):
            if not line.split(':')[0].strip().isdigit(): continue

            mult = 1
            sep = "="
            if "<=" in line: 
                sep="<="
                mult = 1
            elif ">=" in line: 
                sep=">="
                mult = -1
            
            parts = line.split(sep)
            lhs = parts[0].split(':')[1]
            try:
                rhs = float(parts[1])
            except ValueError:
                print(f"Skipping malformed RHS in line: {line}")
                continue
            
            row = np.zeros(num_vars)
            terms = lhs.replace('-', '+-').split('+')
            
            for term in terms:
                term = term.strip()
                if not term: continue
                
                coeff = 1.0
                if '*' in term:
                    c_s, v_s = term.split('*')
                    try:
                        coeff = float(c_s.replace(" ", ""))
                    except ValueError:
                        continue
                else:
                    v_s = term
                    if v_s.startswith("-"): 
                        coeff = -1.0; v_s = v_s[1:]
                
                match = re.search(r'x(\d+)', v_s)
                if match:
                    idx = int(match.group(1)) - 1
                    if idx < num_vars: row[idx] = coeff # Somma coeff se necessario

            A_rows.append(row * mult)
            b_vec.append(rhs * mult)

    return np.array(A_rows), np.array(b_vec), c_vec

def solve_relaxed(A, b, c):
    print("\n--- Continuous relaxed solution ---")
    
    c_prog = -c if problem_direction == 'max' else c
    
    res = linprog(c_prog, A_ub=A, b_ub=b, bounds=[(0, None)]*len(c), method='highs')
    
    if not res.success:
        print(f"Solver error: {res.message}")
        return

    z_val = -res.fun if problem_direction == 'max' else res.fun
    
    print(f"Opt value: {z_val:,.2f}")
    
    print("Variables:")
    for i, val in enumerate(res.x):
        print(f"  x{i+1}: {val:.4f}")
        
    formatted_vals = []
    frac_vals = []
    
    for val in res.x:
        if abs(val - round(val)) < 1e-3:
            formatted_vals.append(f"{int(round(val))}")
        else:
            formatted_vals.append(f"{val:.2f}")
        
        frac = Fraction(val).limit_denominator(1000)
        frac_vals.append(str(frac))
            
    vec_str = ", ".join(formatted_vals)
    vec_frac_str = ", ".join(frac_vals)
    
    print(f"\nx = ({vec_frac_str}) = ({vec_str})")
    print("value:", z_val)
    
    output_file = "optim_sol.txt"
    try:
        with open(output_file, 'w') as f:
            f.write(vec_str)
        print(f"\nOutput SAVED to '{output_file}'")
    except Exception as e:
        print(f"\nWRITE ERROR: {e}")

if __name__ == "__main__":
    A, b, c = parse_input(INPUT_FILE)
    if A is not None and len(A) > 0:
        solve_relaxed(A, b, c)
    else:
        print("Parsing error")