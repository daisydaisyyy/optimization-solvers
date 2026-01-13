import numpy as np
import re
import os
import math
import fractions
from gurobipy import Model, GRB, quicksum

INPUT_FILE = "data.txt"

def format_frac(val):
    try:
        f = fractions.Fraction(val).limit_denominator(1000)
        if f.denominator == 1: return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    except:
        return f"{val:.4f}"

def get_fractional_part(val):
    tol = 1e-9
    if abs(val - round(val)) < tol: return 0.0
    return val - math.floor(val + tol)

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def parse_input(data_filename):
    if not os.path.exists(data_filename):
        print(f"File {data_filename} not found.")
        return None, None, None, None, None

    with open(data_filename, 'r') as f:
        content = f.read()

    sense = GRB.MAXIMIZE 
    if re.search(r'target:.*min', content, re.IGNORECASE): sense = GRB.MINIMIZE
    elif re.search(r'target:.*max', content, re.IGNORECASE): sense = GRB.MAXIMIZE

    c_match = re.search(r'c:\s*\[(.*?)\]', content)
    c = np.array([float(x.strip()) for x in c_match.group(1).split(',')]) if c_match else np.array([])
    n_vars = len(c)

    A_rows, b_vec, signs = [], [], []
    lines = content.splitlines()
    parsing = False
    
    for line in lines:
        line = line.strip()
        if '[CONSTRAINTS]' in line or '[VINCOLI]' in line: parsing = True; continue
        if 'B:' in line: parsing = False; break
        if "Non neg" in line or "non neg" in line.lower(): parsing = False; continue
        if not parsing or not line or line.startswith('#'): continue

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
                    try: coeff = float(c_str.replace(' ', ''))
                    except: coeff = 1.0
                    match = re.search(r'x(\d+)', v_s)
                    if match: row[int(match.group(1)) - 1] += coeff
                elif 'x' in term:
                    coeff = -1.0 if term.startswith('-') else 1.0
                    match = re.search(r'x(\d+)', term)
                    if match: row[int(match.group(1)) - 1] += coeff
            
            A_rows.append(row)
            b_vec.append(rhs)
            signs.append(sign)

    return np.array(A_rows), np.array(b_vec), signs, c, sense

def solve_gomory(A, b, signs, c, sense):
    m, n = A.shape
    
    model = Model("lp")
    model.Params.OutputFlag = 0
    x = model.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="x")
    s = model.addVars(m, lb=0, vtype=GRB.CONTINUOUS, name="s") 
    
    model.setObjective(quicksum(c[i] * x[i] for i in range(n)), sense) 
    
    for i in range(m):
        expr = quicksum(A[i, j] * x[j] for j in range(n))
        if signs[i] == '<=': model.addConstr(expr + s[i] == b[i], name=f"c{i}")
        elif signs[i] == '>=': model.addConstr(expr - s[i] == b[i], name=f"c{i}")
        else: model.addConstr(expr == b[i], name=f"c{i}")

    model.optimize()
    if model.status != GRB.OPTIMAL:
        print("Optimal solution not found.")
        return

    names_list = [f"x_{i+1}" for i in range(n)] + [f"x_{n+(i+1)}" for i in range(m)]
    vars_gurobi = [x[i] for i in range(n)] + [s[i] for i in range(m)]
    
    basis_idxs = []
    non_basis_idxs = []
    
    for idx, var in enumerate(vars_gurobi):
        if var.VBasis == 0:
            basis_idxs.append(idx)
        else:
            non_basis_idxs.append(idx)
            
    basis_idx_print = [i + 1 for i in basis_idxs]
    
    print(f"\nOPT BASIS FOR GOMORY = {{{', '.join(map(str, basis_idx_print))}}}")
    
    S_mat = np.zeros((m, m))
    for i, sign in enumerate(signs):
        if sign == '<=': S_mat[i, i] = 1.0
        elif sign == '>=': S_mat[i, i] = -1.0
        
    A_total = np.hstack([A, S_mat])
    
    try:
        A_B = A_total[:, basis_idxs]
        A_N = A_total[:, non_basis_idxs]
        
        if A_B.shape[1] != m:
            print("Basis dimension error.")
            return
        A_B_inv = np.linalg.inv(A_B)
    except Exception as e:
        print("Matrix Error:", e)
        return
    
    print("\n[1] A_B^-1:")
    for row in A_B_inv:
        print("  " + "  ".join([f"{format_frac(v):>8}" for v in row]))
        
    print("\n[2] A_N:")
    non_basic_names = [names_list[i] for i in non_basis_idxs]
    header_N = "".join([f"{n:>8} " for n in non_basic_names])
    print(f"     {header_N}")
    for row in A_N:
        print("  " + "  ".join([f"{format_frac(v):>8}" for v in row]))

    print("\n[3] A~ = A_B^-1 * A_N")
    
    A_tilde_N = A_B_inv @ A_N
    
    header_tilde = "".join([f"{n:>10} " for n in non_basic_names])
    print(f"     {header_tilde}")
    print("-" * (6 + len(header_tilde)))
    for i in range(m):
        row_str = "  "
        for val in A_tilde_N[i]:
            row_str += f"{format_frac(val):>10} "
        print(row_str)

    print("\n[4] b~ = A_B^-1 * b")
    b_tilde = A_B_inv @ b
    
    for val in b_tilde:
        print(f"     {format_frac(val):>10}")

    print("\n" + "="*60)
    print("      GOMORY CUT GENERATION")
    print("="*60)

    generated_cuts = []
    
    for i in range(m):
        basic_var_idx = basis_idxs[i]
        basic_var_name = names_list[basic_var_idx]
        
        val_b = b_tilde[i]
        f0 = get_fractional_part(val_b)
        
        if f0 > 1e-4 and f0 < (1 - 1e-4):
            print(f"\n>>> ROW {i+1} (Basic Variable: {basic_var_name}) is fractional")
            print(f"    b~ = {format_frac(val_b)} -> Fractional Part f0 = {format_frac(f0)}")
            
            row_coeffs = A_tilde_N[i]
            cut_lhs = []
            denominators = []
            
            print(f"    Row coefficients in A~_N:")
            
            rhs_frac_obj = fractions.Fraction(f0).limit_denominator(1000)
            if rhs_frac_obj.denominator > 1:
                denominators.append(rhs_frac_obj.denominator)

            for j, coeff in enumerate(row_coeffs):
                var_name = non_basic_names[j]
                fj = get_fractional_part(coeff)
                
                if abs(coeff) > 1e-9:
                    print(f"      {var_name}: coeff {format_frac(coeff):>6} -> frac {format_frac(fj):>6}")
                
                if fj > 1e-5:
                    cut_lhs.append((fj, var_name))
                    f_obj = fractions.Fraction(fj).limit_denominator(1000)
                    if f_obj.denominator > 1:
                        denominators.append(f_obj.denominator)
            
            if cut_lhs:
                generated_cuts.append({
                    'id': i+1, 'var': basic_var_name,
                    'lhs': cut_lhs, 'rhs': f0
                })
                
                cut_str = " + ".join([f"{format_frac(c)} {v}" for c, v in cut_lhs])
                print(f"    => GENERATED CUT (Fractional):")
                print(f"       {cut_str} >= {format_frac(f0)}")

                if denominators:
                    current_lcm = 1
                    for d in denominators:
                        current_lcm = lcm(current_lcm, d)
                    
                    int_lhs_parts = []
                    for c_val, v_name in cut_lhs:
                        int_coeff = int(round(c_val * current_lcm))
                        int_lhs_parts.append(f"{int_coeff} {v_name}")
                    
                    int_rhs = int(round(f0 * current_lcm))
                    int_cut_str = " + ".join(int_lhs_parts)
                    
                    print(f"       {int_cut_str} >= {int_rhs}")

            else:
                print("    No cut generated (all coefficients are integers).")

    print("\n" + "="*60)
    print("      CHECK WITH INTEGER OPTIMUM")
    print("="*60)
    
    model_int = Model("ilp")
    model_int.Params.OutputFlag = 0
    xi = model_int.addVars(n, lb=0, vtype=GRB.INTEGER, name="x")
    model_int.setObjective(quicksum(c[i] * xi[i] for i in range(n)), sense)
    for i in range(m):
        expr = quicksum(A[i, j] * xi[j] for j in range(n))
        if signs[i] == '<=': model_int.addConstr(expr <= b[i])
        elif signs[i] == '>=': model_int.addConstr(expr >= b[i])
        else: model_int.addConstr(expr == b[i])
    model_int.optimize()
    
    obj_int = None
    if model_int.status == GRB.OPTIMAL:
        obj_int = model_int.ObjVal
        sol_int = [xi[i].X for i in range(n)]
        print(f"Integer Optimum (Target): {obj_int:.4f}")
        print(f"Optimal Integer Solution: {sol_int}")
    else:
        print("Integer solution not found.")

    if generated_cuts and obj_int is not None:
        print("\nChoose a cut to verify:")
        for i, cut in enumerate(generated_cuts):
             print(f"{i+1}: Cut from Row {cut['id']} ({cut['var']})")
        
        try:
            inp = input("Number: ")
            if inp.strip():
                choice = int(inp)
                if 1 <= choice <= len(generated_cuts):
                    sel = generated_cuts[choice - 1]
                    
                    name_map = {vn: v for vn, v in zip(names_list, vars_gurobi)}
                    expr = quicksum(c * name_map[v] for c, v in sel['lhs'])
                    model.addConstr(expr >= sel['rhs'], "user_cut")
                    model.optimize()
                    
                    if model.status == GRB.OPTIMAL:
                        new_opt = model.ObjVal
                        print(f"New relaxed optimum: {new_opt:.4f}")
                        if abs(new_opt - obj_int) < 1e-4: print(">>> SUCCESS: Integer optimum reached!")
                        else: print(">>> Result: Not yet integer optimum (more cuts needed).")
                    else: print("Problem became infeasible.")
        except ValueError: pass

if __name__ == "__main__":
    A, b, signs, c, sense = parse_input(INPUT_FILE)
    if A is not None:
        solve_gomory(A, b, signs, c, sense)