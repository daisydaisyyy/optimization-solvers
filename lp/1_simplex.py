import numpy as np
import re
import os
from fractions import Fraction

INPUT_FILE = "data.txt"
OUTPUT_FILE = "sim_result.txt"

def log(msg, to_file=True):
    print(msg)
    if to_file:
        with open(OUTPUT_FILE, "a") as f:
            f.write(msg + "\n")

def format_fraction(val):
    if abs(val) < 1e-9: return "0"
    if abs(val - round(val)) < 1e-4: return str(int(round(val)))
    f = Fraction(val).limit_denominator(1000)
    if f.denominator == 1: return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"

def format_vec_flat(v):
    vals = []
    for x in v:
        if abs(x - round(x)) < 1e-4: vals.append(f"{int(round(x))}")
        else: vals.append(f"{x:.2f}")
    return "(" + ", ".join(vals) + ")"

def format_vec_fraction(v):
    vals = []
    for x in v:
        f = Fraction(x).limit_denominator(1000)
        if f.denominator == 1: 
            vals.append(str(f.numerator))
        else: 
            vals.append(f"{f.numerator}/{f.denominator}")
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
        print(f"Error: {filename} not found.")
        return None, None, None, None

    with open(filename, 'r') as f:
        lines = f.readlines()

    c_vec = None
    constraints = [] 
    x_start = None
    basis_ids = []
    is_min = False
    
    mode = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"): continue
        
        if "target" in line and "min" in line.lower():
            is_min = True

        if line.startswith("c:"):
            content = line.split(":")[1].replace('[','').replace(']','').strip()
            c_vec = np.array([float(x) for x in content.split(',')])
            continue
        elif line.startswith("B:"):
            content = line.split(":")[1].replace('[','').replace(']','').strip()
            basis_ids = [int(x) for x in content.split(',')]
            continue
            
        if "[POINT]" in line or "[PUNTO]" in line: mode = "POINT"; continue
        elif "[CONSTRAINTS]" in line or "[VINCOLI]" in line: mode = "CONSTR"; continue
        
        if mode == "POINT":
            if "target" in line or "c:" in line: continue
            try:
                # fraction values handling
                parts = line.split(',')
                vals = []
                for p in parts:
                    p = p.strip()
                    if '/' in p:
                        n, d = p.split('/')
                        vals.append(float(n)/float(d))
                    elif p:
                         clean_p = ''.join(c for c in p if c.isdigit() or c in '.-')
                         if clean_p: vals.append(float(clean_p))
                
                if vals: x_start = np.array(vals)
            except Exception as e: 
                print(f"Error parsing point: {e}")
                continue
            
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
                    row = np.zeros(len(c_vec) if c_vec is not None else 3)
                    terms = lhs.replace("-", "+-").split('+')
                    for term in terms:
                        term = term.strip()
                        if not term: continue
                        coeff = 1.0
                        if '*' in term:
                            c_s, v_s = term.split('*')
                            coeff = float(c_s.replace(" ", ""))
                        elif term.startswith("-") and "x" in term:
                             coeff = -1.0
                             term = term[1:]
                        if "x" in term:
                            match = re.search(r'x(\d+)', term)
                            if match:
                                idx = int(match.group(1)) - 1
                                if idx < len(row): row[idx] = coeff
                    
                    constraints.append({
                        'id': con_id, 'A': row * mult, 'b': float(rhs) * mult
                    })
                except Exception as e: 
                    print(f"Parsing error on row {con_id}: {e}")

    if is_min and c_vec is not None:
        print(">>> Minimization detected: Flipping c vector sign.")
        c_vec = -c_vec

    return c_vec, x_start, basis_ids, constraints

def solve_step(c, x_curr, basis_ids, constraints, step_num=1):
    log(f"\n=== STEP {step_num} ===")
    log(f"Starting point x = {format_vec_fraction(x_curr)} = {format_vec_flat(x_curr)}")
    log(f"Starting B = {basis_ids}")

    # 1. A_B matrix
    rows = []
    con_map = {c['id']: c for c in constraints}
    for bid in basis_ids:
        rows.append(con_map[bid]['A'])
    A_B = np.array(rows)
    
    log("\n" + format_matrix(A_B.T, "A_B^T"))

    # 2. y (dual solution)
    try:
        y_vals = np.linalg.solve(A_B.T, c)
    except:
        log("Error: Singular matrix.")
        return None, None, True


    log("\n--- Find h (leaving index = min of y_i (from dual solution) ) ---")
    for i, bid in enumerate(basis_ids):
        val = y_vals[i]
        status = " -> FOUND" if val < -1e-7 else " (OK)"
        log(f"  {bid}: {format_fraction(val):>6} {status}")
    
    y_full_map = {}
    for idx, bid in enumerate(basis_ids):
        y_full_map[bid] = y_vals[idx]
    

    # 3. choose h
    neg_indices = [i for i, y in enumerate(y_vals) if y < -1e-7]
    if not neg_indices:
        log("\n>>> OPTIMAL REACHED (All y >= 0).")
        return None, None, True

    idx_h = neg_indices[0] 

    h_id = basis_ids[idx_h] 
    val_leaving = y_vals[idx_h]

    try:
        W_full = -1 * np.linalg.inv(A_B)
        log("\n" + format_matrix(W_full, "W (A_B^-1)"))
    except np.linalg.LinAlgError:
        log("\nW is a singular matrix.")
    
    log(f"h = {h_id} (y_{h_id} = {format_fraction(val_leaving)}) -> Leaving Index")

    # 4. W
    rhs = np.zeros(len(basis_ids))
    rhs[idx_h] = -1.0 
    w_vec = np.linalg.solve(A_B, rhs)
    
    col_str = "".join([f"  ( {x:6.2f} )\n" for x in w_vec])
    log(f"\nW^{h_id}:\n{col_str}")

    # 5. r
    log("--- Calculating Ratios (r) ---")
    ratios_valid = []
    best_r = float('inf')
    k = None
    
    non_basic_con = sorted([c for c in constraints if c['id'] not in basis_ids], key=lambda x: x['id'])
    
    for con in non_basic_con:
        Ax_val = np.dot(con['A'], x_curr)
        slack = con['b'] - Ax_val
        den = np.dot(con['A'], w_vec)
        
        # consider only den > 0 (we are moving towards the constraint)
        if den <= 1e-9:
            continue
        
        r_val = slack / den
        theta_str = format_fraction(r_val)
        
        cid = con['id']
        # (b_i - A_i x) / (A_i w_h)
        b_str = f"{con['b']:.4f}".rstrip('0').rstrip('.')
        Ax_str = f"{Ax_val:.4f}"
        den_str = f"{den:.4f}"
        
        expanded_calc = f"({b_str} - {Ax_str}) / {den_str}"
        
        log(f"  theta_{cid} = {expanded_calc} = {theta_str}")
        
        if r_val < -1e-9: continue 

        ratios_valid.append(theta_str)
        
        if r_val < best_r:
            best_r = r_val
            k = con['id']
        elif abs(r_val - best_r) < 1e-9:
            if k is None or con['id'] < k: k = con['id']
    
    if k is None:
        log("Unbounded or Optimal inside.")
        return None, None, True

    log(f"\nk = {k} (r = {format_fraction(best_r)}) -> Entering Index")

    x_new = x_curr + best_r * w_vec
    new_basis = sorted([b for b in basis_ids if b != h_id] + [k])
    
    log(f"Next Vertex x = {format_vec_fraction(x_new)} = {format_vec_flat(x_new)}")
    log(f"Next Basis B = {new_basis}")
    
    return x_new, new_basis, False

if __name__ == "__main__":
    with open(OUTPUT_FILE, "w") as f: f.write("--- REPORT --- \n")
    c, x_curr, basis_ids, constraints = parse_input(INPUT_FILE)
    
    if c is not None:
        try:
            try:
                input_str = input("How many simplex steps? ")
                max_steps = int(input_str)
            except ValueError:
                print("Not valid. Executing 1 step by default")
                max_steps = 1

            for i in range(1, max_steps + 1):
                x_next, basis_next, finished = solve_step(c, x_curr, basis_ids, constraints, i)
                if finished: break
                x_curr = x_next
                basis_ids = basis_next
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"\nOutput SAVED to {OUTPUT_FILE}")