import numpy as np
import re
import os
import math
import fractions
from gurobipy import Model, GRB, quicksum

INPUT_FILE = "data.txt"

def get_fractional_part(val):
    tol = 1e-9
    # if it's integer (within tolerance), fractional part = 0
    if abs(val - round(val)) < tol: 
        return 0.0
    return val - math.floor(val + tol)

def format_frac(val):
    f = fractions.Fraction(val).limit_denominator(1000)
    if f.denominator == 1: return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"

def parse_input(data_filename):
    if not os.path.exists(data_filename):
        print("data.txt missing")
        return None, None, None, None

    with open(data_filename, 'r') as f:
        content = f.read()

    # Parsing c vector
    c_match = re.search(r'c:\s*\[(.*?)\]', content)
    c = np.array([float(x.strip()) for x in c_match.group(1).split(',')])
    n_vars = len(c)

    A_rows, b_vec, signs = [], [], []
    lines = content.splitlines()
    parsing = False
    
    for line in lines:
        line = line.strip()
        if '[CONSTRAINTS]' in line or '[VINCOLI]' in line: 
            parsing = True; continue
        if 'B:' in line: 
            parsing = False; break
        if not parsing or not line or line.startswith('#'): continue
        if re.search(r'1\*\s*x\d+\s*[<>]=\s*0', line): continue 
            
        if ':' in line and ('<=' in line or '>=' in line or '=' in line):
            if '<=' in line: sign, parts = '<=', line.split('<=')
            elif '>=' in line: sign, parts = '>=', line.split('>=')
            else: sign, parts = '=', line.split('=')
            
            lhs = parts[0].split(':', 1)[1].strip() if ':' in parts[0] else parts[0].strip()
            rhs = float(parts[1].strip())
            row = np.zeros(n_vars)
            
            terms = lhs.replace('-', '+-').split('+')
            for term in terms:
                term = term.strip()
                if not term: continue
                if '*' in term:
                    c_str, v_s = term.split('*')
                    coeff = float(c_str.replace(' ', ''))
                    var_idx = int(re.search(r'x(\d+)', v_s).group(1)) - 1
                    row[var_idx] += coeff
            
            A_rows.append(row)
            b_vec.append(rhs)
            signs.append(sign)

    print(f"Parsed {n_vars} variables, {len(b_vec)} constraints.")
    return np.array(A_rows), np.array(b_vec), signs, c

def solve_gomory(A, b, signs, c):
    m, n = A.shape
    
    # A MATRIX
    print("\n--- A ---")
    for i in range(m):
        row_str = "  ".join([f"{x:6.2f}" for x in A[i]])
        print(f"Row {i+1}: [ {row_str} ]  {signs[i]} {b[i]}")

    # PRINT SYSTEM + SLACK VARS
    print("\n--- SYSTEM + SLACK VARS ---")
    for i in range(m):
        terms_str = []
        for j in range(n):
            coeff = A[i,j]
            if abs(coeff) > 1e-9:
                sign = "+" if coeff >= 0 else "-"
                abs_val = abs(coeff)
                val_str = f"{abs_val:g}"
                term = f"{sign} {val_str}x{j+1}"
                terms_str.append(term)
        
        # slack vars
        s_name = f"s{i+1}"
        if signs[i] == '<=':
            terms_str.append(f"+ {s_name}")
        elif signs[i] == '>=':
            terms_str.append(f"- {s_name}")
        
        full_line = " ".join(terms_str).strip()
        if full_line.startswith("+ "): full_line = full_line[2:]
            
        print(f"{full_line} = {b[i]:g}")
    
    # solve relaxed problem (LP OPTIMAL SOLUTION)
    model = Model("lp")
    model.Params.OutputFlag = 0
    x = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
    s = model.addVars(m, lb=0, vtype=GRB.CONTINUOUS, name="s") 
    
    model.setObjective(quicksum(c[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
    for i in range(m):
        expr = quicksum(A[i, j] * x[j] for j in range(n))
        if signs[i] == '<=':
            model.addConstr(expr + s[i] == b[i], name=f"c{i}")
        else: 
            model.addConstr(expr - s[i] == b[i], name=f"c{i}")

    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        print("Optimal solution not found.")
        return

    print("\n--- OPT BASIS ---")
    
    names_list = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)]
    vars_gurobi = [x[i] for i in range(n)] + [s[i] for i in range(m)]
    
    # print only non-zero variables
    for i, var in enumerate(vars_gurobi):
        val = var.X
        if abs(val) > 1e-4:
            print(f"  {names_list[i]:<4}: {val:8.4f}")

    # find B
    basis_indices = []
    for idx, var in enumerate(vars_gurobi):
        if var.VBasis == 0: # 0 = Basic
            basis_indices.append(idx)
            
    basis_names = [names_list[i] for i in basis_indices]
    print(f"\nB: {basis_indices} (vars: {basis_names})")

    # A_total = [A | S]
    S_mat = np.zeros((m, m))
    for i, sign in enumerate(signs):
        S_mat[i, i] = 1.0 if sign == '<=' else -1.0
        
    A_total = np.hstack([A, S_mat])
    
    # A_B^-1
    try:
        B_mat = A_total[:, basis_indices] 
        if B_mat.shape[1] != m:
            print(f"Base dimension error.")
            return
        B_inv = np.linalg.inv(B_mat)
    except Exception as e:
        print("Base matrix inversion error:", e)
        return

    print("\n--- A_B^-1 ---")
    for i in range(m):
        row_vals = [f"{val:8.4f}" for val in B_inv[i, :]]
        print(f"  Row {i+1}: " + "  ".join(row_vals))

    x_B_val = B_inv @ b
    
    # --- find fractional parts that will generate a cut plane ---
    print("\n--- FRACTIONAL ROWS ---")
    fractional_rows = []
    
    for i in range(len(basis_indices)):
        global_idx = basis_indices[i]
        var_name = names_list[global_idx]
        val = x_B_val[i]
        f0 = get_fractional_part(val)
        
        print(f"Row {i+1} ({var_name}): Val={val:.4f}, f₀={f0:.4f}")
        
        if f0 > 1e-4 and f0 < (1 - 1e-4):
            fractional_rows.append((i, f0, val))

    print("\n--- GOMORY CUTS ---")
    
    if not fractional_rows:
        print("No fractional rows found.")
    else:
        for row_idx, f0, val in fractional_rows:
            global_idx = basis_indices[row_idx]
            var_name = names_list[global_idx]
            current_row_num = row_idx + 1
            
            print(f"\n>>> ROW {current_row_num} ({var_name})")
            
            # B_inv row (= pi)
            row_B_inv = B_inv[row_idx, :]
            
            print(f"    [Step 1] row {current_row_num} of B^-1:")
            pi_str = ", ".join([f"{x:.3f}" for x in row_B_inv])
            print(f"      pi = [{pi_str}]")
            
            print(f"    [Step 2] (pi * Column_j):")
            
            row_tableau = np.zeros(len(names_list))
            for j in range(A_total.shape[1]):
                col_vec = A_total[:, j]
                coeff = np.dot(row_B_inv, col_vec)
                row_tableau[j] = coeff
                
                is_basic = (j == global_idx)
                if abs(coeff) > 1e-4 and (j not in basis_indices or is_basic):
                    product_terms = []
                    for p_val, a_val in zip(row_B_inv, col_vec):
                        product_terms.append(f"({p_val:.3f}*{a_val:.3g})")
                    expansion_str = " + ".join(product_terms)
                    print(f"      part_coeff({names_list[j]}) = {expansion_str} = {coeff:.4f} ({format_frac(coeff)})")
            
            eq_terms = []
            for j in range(len(row_tableau)):
                coeff = row_tableau[j]
                if abs(coeff) > 1e-5 and j != global_idx:
                     eq_terms.append(f"({format_frac(coeff)}){names_list[j]}")
            
            print(f"\n      {var_name} + {' + '.join(eq_terms)} = {format_frac(val)}")

            cut_lhs_terms = []
            print(f"    [Step 3] find cut equation COEFF = part_coeff - floor(part_coeff):")
            
            slacks_of_interest = range(len(row_tableau)) 
            for col_idx in slacks_of_interest:
                coeff = row_tableau[col_idx]
                if col_idx == global_idx: continue
                if abs(coeff) < 1e-5: continue

                fj = get_fractional_part(coeff)
                int_part = math.floor(coeff + 1e-9)
                
                # --- MODIFICA QUI: Formato esplicito f = val - floor(val) ---
                val_str = format_frac(coeff)
                fj_str = format_frac(fj)
                
                print(f"      {names_list[col_idx]}: coeff = {val_str} - floor({val_str}) = {val_str} - ({int_part}) = {fj_str}")

                if fj > 1e-5:
                    cut_lhs_terms.append((fj, names_list[col_idx]))

            if not cut_lhs_terms:
                print("    No cut generated (integer coefficients).")
                continue

            # 3. Final Cut
            frac_cut_str = " + ".join([f"{{{format_frac(c)}}}{v}" for c, v in cut_lhs_terms])
            print(f"\n    => CUT:")
            print(f"    {frac_cut_str} >= {format_frac(f0)}") 
            
            # integer cut (x8 heuristic)
            common_mult = 8
            int_terms = []
            for fj, vname in cut_lhs_terms:
                val_int = int(round(fj * common_mult))
                int_terms.append(f"{val_int}{vname}")
            rhs_int = int(round(f0 * common_mult))
            

    print("-" * 40)
    
    # --- INTEGER SOLUTION ---
    model_int = Model("gomory_int")
    model_int.Params.OutputFlag = 0
    xi = model_int.addVars(n, lb=0, vtype=GRB.INTEGER, name="x")
    
    model_int.setObjective(quicksum(c[i] * xi[i] for i in range(n)), GRB.MAXIMIZE)
    
    for i in range(m):
        expr = quicksum(A[i, j] * xi[j] for j in range(n))
        if signs[i] == '<=':
            model_int.addConstr(expr <= b[i])
        else: 
            model_int.addConstr(expr >= b[i])

    model_int.optimize()
    
    if model_int.status == GRB.OPTIMAL:
        sol_int = np.array([xi[i].X for i in range(n)])
        obj_int = model_int.ObjVal
        
        sol_int_rounded = tuple(np.round(sol_int).astype(int).tolist())
        
        print(f"optimal value = {obj_int:.1f}")
        print(f"optimal integer solution: {sol_int_rounded}")
    else:
        print("integer solution not found.")

if __name__ == "__main__":
    A, b, signs, c = parse_input(INPUT_FILE)
    if A is not None:
        solve_gomory(A, b, signs, c)