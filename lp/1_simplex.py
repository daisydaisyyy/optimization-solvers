import numpy as np
import re
import os
import sys

INPUT_FILE = "data.txt"
OUTPUT_FILE = "sim_result.txt"

def log(msg):
    print(msg)
    with open(OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")

def format_vec_flat(v):
    vals = []
    for x in v:
        if abs(x - round(x)) < 1e-4: vals.append(f"{int(round(x))}")
        else: vals.append(f"{x:.2f}")
    return "(" + ", ".join(vals) + ")"

def format_matrix(M, name="Matrix"):
    rows, cols = M.shape
    out = f"{name} =\n"
    for i in range(rows):
        out += "  ( "
        for j in range(cols):
            val = M[i, j]
            if abs(val) < 1e-9: val = 0.0
            out += f"{val:6.2f} "
        out += ")\n"
    return out

def parse_input(filename):
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return None, None, None, None, None

    with open(filename, 'r') as f:
        lines = f.readlines()

    c_vec = None
    constraints = [] 
    x_start = None
    basis_ids = []
    
    mode = None
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith("c:"):
            content = line.split(":")[1].replace('[','').replace(']','').strip()
            c_vec = np.array([float(x) for x in content.split(',')])
            continue
        elif line.startswith("B:"):
            content = line.split(":")[1].replace('[','').replace(']','').strip()
            basis_ids = [int(x) for x in content.split(',')]
            continue
            
        if "[PUNTO]" in line: mode = "POINT"; continue
        elif "[VINCOLI]" in line: mode = "CONSTR"; continue
        
        if mode == "POINT":
            if "obiettivo" in line or "c:" in line or "source" in line: continue
            try:
                parts = line.split(',')
                vals = [float(x) for x in parts if x.strip().lstrip('-').replace('.','',1).isdigit()]
                if vals: x_start = np.array(vals)
            except: continue
            
        elif mode == "CONSTR":
            if ":" in line and ("<=" in line or ">=" in line or "=" in line):
                parts = line.split(':')
                try:
                    con_id = int(parts[0].strip())
                    rest = parts[1]
                    if "<=" in rest: sep="<="; mult=1
                    elif ">=" in rest: sep=">="; mult=-1
                    else: sep="="; mult=1
                    lhs, rhs = rest.split(sep)
                    
                    row = np.zeros(len(c_vec) if c_vec is not None else 2)
                    terms = lhs.replace("-", "+-").split('+')
                    for term in terms:
                        term = term.strip()
                        if not term: continue
                        coeff = 1.0
                        if '*' in term:
                            c_s, v_s = term.split('*')
                            coeff = float(c_s)
                        elif term.startswith("-") and "x" in term:
                             coeff = -1.0
                             term = term[1:]
                        
                        if "x" in term:
                            idx = int(re.search(r'x(\d+)', term).group(1)) - 1
                            row[idx] = coeff
                    
                    constraints.append({
                        'id': con_id,
                        'A': row * mult,
                        'b': float(rhs) * mult,
                        'orig_A': row,
                        'orig_b': float(rhs)
                    })
                except: pass

    return c_vec, x_start, basis_ids, constraints

def calculate_y(c, basis_ids, constraints):
    rows = []
    for bid in basis_ids:
        for con in constraints:
            if con['id'] == bid:
                rows.append(con['A'])
                break
    
    if not rows: return None, None
    
    A_B = np.array(rows)
    A_B_T = A_B.T
    
    try:
        y_vals = np.linalg.solve(A_B_T, c)
    except:
        return None, None
        
    y_full = []
    for con in constraints:
        if con['id'] in basis_ids:
            idx = basis_ids.index(con['id'])
            y_full.append(y_vals[idx])
        else:
            y_full.append(0.0)
            
    return y_vals, np.array(y_full)

def solve_step_formulas(c, x_curr, basis_ids, constraints, step_num):
    log(f"=== STEP {step_num} ===")
    
    log(f"Starting Point x = {format_vec_flat(x_curr)}")
    log(f"Starting Basis B = {basis_ids}")
    
    rows = []
    for bid in basis_ids:
        for con in constraints:
            if con['id'] == bid:
                rows.append(con['A'])
                break
    
    if not rows:
        log("Error: Empty basis or constraints not found.")
        return None, None, True

    A_B = np.array(rows)
    A_B_T = A_B.T
    
    log("\n" + format_matrix(A_B_T, "A_B^T"))
    
    y_vals, y_full_vec = calculate_y(c, basis_ids, constraints)
    
    if y_vals is None:
        log("Error: Singular matrix.")
        return None, None, True

    y_display = [f"{val:.0f}" for val in y_full_vec]
    log(f"y = ({', '.join(y_display)})")
    
    negative_indices_local = [i for i, val in enumerate(y_vals) if val < -1e-7]
    
    if not negative_indices_local:
        log("\n>>> OPTIMAL REACHED (All lambda >= 0).")
        log(f"\n--- FINAL RESULT ---")
        log(f"Final x = {format_vec_flat(x_curr)}")
        log(f"Final y = ({', '.join(y_display)})")
        log(f"Final Basis = {basis_ids}")
        return None, None, True
    
    idx_target = min(negative_indices_local, key=lambda i: basis_ids[i])
    leaving_id = basis_ids[idx_target]
    val_leaving = y_vals[idx_target]
    idx_min = idx_target 
    
    log(f"Leaving Index = {leaving_id} (y_{leaving_id} = {val_leaving:.0f})")
    
    try:
        inv_ABT = np.linalg.inv(A_B_T)
        W_matrix = -inv_ABT
    except: return None, None, True
    
    log("\n" + format_matrix(W_matrix, "W = -(A_B^T)^-1"))
    
    w_vec = W_matrix[:, idx_min]
    
    col_str = "".join([f"  ( {val:6.2f} )\n" for val in w_vec])
    log(f"Column w^{leaving_id}:\n{col_str}")
    
    best_r = float('inf')
    entering_id = None
    
    log("--- Calculating Ratios (r) ---")
    
    for con in constraints:
        if con['id'] in basis_ids: continue
        
        A_i = con['A']
        b_i = con['b']
        den = np.dot(A_i, w_vec)
        slack = b_i - np.dot(A_i, x_curr)
        
        if den > 1e-9:
            r = slack / den
            log(f"r_{con['id']} = (b_{con['id']} - A_{con['id']} x) / {den:.2f} = {r:.2f}")
            if r < best_r:
                best_r = r
                entering_id = con['id']
            elif abs(r - best_r) < 1e-9:
                if entering_id is None or con['id'] < entering_id:
                    entering_id = con['id']
            
    if entering_id is not None:
        log(f">>> Entering Index: {entering_id} (r = {best_r:.2f})")
        x_new = x_curr + best_r * w_vec
        new_basis = sorted([b for b in basis_ids if b != leaving_id] + [entering_id])
        
        log(f"\n--- END OF STEP {step_num}: NEW VALUES ---")
        log(f"New x = {format_vec_flat(x_new)}")
        log(f"New Basis B = {new_basis}")
        
        _, y_new_full = calculate_y(c, new_basis, constraints)
        
        if y_new_full is not None:
            y_new_display = [f"{val:.0f}" for val in y_new_full]
            log(f"New y = ({', '.join(y_new_display)})")
        else:
            log("New y = (Error: Singular basis)")

        return x_new, new_basis, False
    else:
        log("Unbounded problem.")
        return None, None, True

if __name__ == "__main__":
    with open(OUTPUT_FILE, "w") as f: f.write("--- REPORT --- \n")
    c, x_curr, basis_ids, constraints = parse_input(INPUT_FILE)
    
    if c is not None:
        try:
            val = input("How many steps do you want to execute? ")
            num_steps = int(val)
        except: num_steps = 1
        
        for i in range(1, num_steps + 1):
            x_next, basis_next, finished = solve_step_formulas(c, x_curr, basis_ids, constraints, i)
            if finished: break
            x_curr = x_next
            basis_ids = basis_next
        print(f"Done. Check: {OUTPUT_FILE}")
