import re
import numpy as np
import os

INPUT_FILE = "data.txt"
OUTPUT_FILE = "dual.txt"

def parse_primal(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} non trovato.")

    with open(filename, 'r') as f:
        lines = f.readlines()

    c = []
    constraints_data = [] 
    max_var_index = 0
    
    mode = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"): 
            continue

        if "target" in line: 
            mode = "target"
            continue
        elif "[CONSTRAINTS]" in line: 
            mode = "CONSTR"
            continue

        if mode == "target" and line.startswith("c:"):
            clean_line = line.split(":")[1].replace('[','').replace(']','').replace(" ", "")
            c = [float(x) for x in clean_line.split(",") if x]
            
        elif mode == "CONSTR":
            if ":" in line and ("<=" in line or ">=" in line):
                if "<=" in line: sep = "<="
                else: sep = ">="
                
                rest = line.split(":")[1].strip()
                lhs, rhs = rest.split(sep)
                rhs_val = float(rhs)
                                
                terms_raw = lhs.replace("-", "+-").split("+")
                row_dict = {} 
                
                for term in terms_raw:
                    term = term.strip().replace(" ", "")
                    if not term: continue
                    
                    coeff = 1.0
                    var_idx = -1
                    
                    if '*' in term:
                        parts = term.split('*')
                        try:
                            coeff = float(parts[0])
                        except ValueError:
                            continue
                        part_var = parts[1]
                    else:
                        part_var = term
                        if part_var.startswith("-"):
                            coeff = -1.0
                            part_var = part_var[1:]
                    
                    if 'x' in part_var:
                        var_str = part_var.replace('x', '')
                        if var_str.isdigit():
                            var_idx = int(var_str) - 1
                            if var_idx > max_var_index: max_var_index = var_idx
                    
                    if var_idx >= 0:
                        row_dict[var_idx] = row_dict.get(var_idx, 0) + coeff
                
                constraints_data.append((row_dict, rhs_val))

    n = max(len(c), max_var_index + 1)
    m = len(constraints_data)
    A = np.zeros((m, n))
    b = np.zeros(m)
    
    for i, (row_dict, rhs) in enumerate(constraints_data):
        b[i] = rhs
        for col_idx, val in row_dict.items():
            if col_idx < n:
                A[i, col_idx] = val

    return c, A, b

def write_dual(c, A, b, output_file):
    m, n = A.shape
    with open(output_file, "w") as f:
        f.write("target: \n")
        # Scrive la funzione obiettivo Min (che usa i termini noti del primale)
        obj_terms = []
        for i, val in enumerate(b):
            if abs(val) > 0:
                obj_terms.append(f"{val:.0f}*y{i+1}")
        f.write(f"min {' + '.join(obj_terms)}\n\n")
        
        f.write("[CONSTRAINTS]\n")
        
        for j in range(n):
            terms = []
            for i in range(m):
                val = A[i,j] 
                if abs(val) > 1e-9:
                    sign = "+" if val >= 0 else "-"
                    abs_val = abs(val)
                    var_name = f"y{i+1}"
                    
                    val_str = f"{abs_val:.2f}".rstrip('0').rstrip('.') if abs_val % 1 != 0 else f"{abs_val:.0f}"
                    if val_str == "1": term_str = var_name
                    else: term_str = f"{val_str}{var_name}"

                    if not terms and val < 0: 
                         terms.append(f"-{term_str}")
                    elif not terms:
                         terms.append(f"{term_str}")
                    else:
                         terms.append(f"{sign} {term_str}")
        
            slack_idx = m + j + 1
            terms.append(f"- y{slack_idx}")

            line_str = " ".join(terms)
            
            rhs_c = c[j] if j < len(c) else 0.0
            rhs_str = f"{rhs_c:.2f}".rstrip('0').rstrip('.') if rhs_c % 1 != 0 else f"{rhs_c:.0f}"
            
            f.write(f"{line_str} = {rhs_str}\n")
            
        f.write("\n# non neg\n")
        f.write("y >= 0")

if __name__ == "__main__":
    try:
        c, A, b = parse_primal(INPUT_FILE)
        write_dual(c, A, b, OUTPUT_FILE)
        print(f"SAVED in {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error: {e}")