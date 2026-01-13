import re
import numpy as np
from fractions import Fraction

def to_frac_str(val, max_denom=1000000):
    return str(Fraction(val).limit_denominator(max_denom))

def parse_value(val_str):
    val_str = val_str.strip()
    if '/' in val_str:
        n, d = val_str.split('/')
        return float(n) / float(d)
    return float(val_str)

def parse_data(filename):
    with open(filename, 'r') as f:
        content = f.read()

    all_var_indices = re.findall(r'x(\d+)', content)
    num_vars = 0
    if all_var_indices:
        num_vars = max(int(i) for i in all_var_indices)

    target_match = re.search(r'target:\s*max\s*(.*)', content)
    c_coeffs = [0.0] * num_vars
    
    if target_match:
        eq = target_match.group(1)
        matches = re.findall(r'([+\-]?\s*\d+(?:\.\d+)?)\*x(\d+)', eq)
        for coeff_str, var_idx in matches:
            idx = int(var_idx) - 1
            if 0 <= idx < num_vars:
                c_coeffs[idx] = float(coeff_str.replace(' ', ''))

    A_rows = {}
    b_vals = {}
    
    rows_pattern = re.findall(r'(\d+)\s*:\s*(.*?)\s*([<>=]+)\s*([+\-]?\d+(?:[./]\d+)?)', content)
    
    for idx_str, expr, op, rhs in rows_pattern:
        idx = int(idx_str)
        val_rhs = parse_value(rhs)
        
        mult = -1.0 if ">" in op else 1.0
        
        b_vals[idx] = val_rhs * mult
        
        row_coeffs = [0.0] * num_vars
        var_matches = re.findall(r'([+\-]?\s*\d+)\*x(\d+)', expr)
        for coeff_str, var_idx in var_matches:
            v_i = int(var_idx) - 1 
            if 0 <= v_i < num_vars:
                val = float(coeff_str.replace(' ', ''))
                row_coeffs[v_i] = val * mult
        
        A_rows[idx] = row_coeffs

    b_matches = re.findall(r'B:\s*\[(.*?)\]', content)
    base_indices = []
    if b_matches:
        base_indices = [int(x.strip()) for x in b_matches[-1].split(',')]

    return np.array(c_coeffs), A_rows, b_vals, base_indices

def main():
    input_file = 'data.txt'
    output_file = 'dual.txt'

    try:
        c, A_dict, b_dict, B_indices = parse_data(input_file)
    except Exception as e:
        print(f"Parsing error: {e}")
        return

    try:
        A_B_list = [A_dict[idx] for idx in B_indices]
        A_B = np.array(A_B_list)
        print("A_B^T (Float):\n", A_B.T)
    except KeyError as e:
        print(f"Error: basis idx {e} not found.")
        return

    try:
        A_B_inv = np.linalg.inv(A_B)
        
        y_B_vals = c @ A_B_inv
        
        A_B_inv_fracs = []
        for row in A_B_inv:
            frac_row = [to_frac_str(val) for val in row]
            A_B_inv_fracs.append(frac_row)

        y_B_fracs = [to_frac_str(val) for val in y_B_vals]
        
        print("\nA_B^-1:")
        for row in A_B_inv_fracs:
            print(f"  {row}")

        print(f"\nc: {c}")

    except np.linalg.LinAlgError:
        print("A_B is a singular matrix.")
        return

    all_indices = sorted(A_dict.keys())
    y_full_vals = {idx: 0.0 for idx in all_indices}
    y_full_fracs = {idx: "0" for idx in all_indices}
    
    for i, basis_idx in enumerate(B_indices):
        y_full_vals[basis_idx] = y_B_vals[i]
        y_full_fracs[basis_idx] = y_B_fracs[i]

    with open(output_file, 'w') as f:
        f.write("=== dual problem ===\n")
        
        obj_terms = []
        for idx in all_indices:
            val = b_dict[idx]
            if val != 0:
                val_str = to_frac_str(val) if val % 1 != 0 else str(int(val))
                obj_terms.append(f"{val_str}*y{idx}")
        f.write(f"Min W = {' + '.join(obj_terms)}\n\n")
        
        num_vars = len(c)
        for j in range(num_vars):
            constraint_terms = []
            for idx in all_indices:
                if idx in A_dict:
                    coeff = A_dict[idx][j]
                    if coeff != 0:
                        c_str = str(int(coeff)) if coeff % 1 == 0 else str(coeff)
                        constraint_terms.append(f"({c_str})*y{idx}")
            
            line = f"{' + '.join(constraint_terms)} = {int(c[j])}\n"
            f.write(line)
        
        f.write("\n=== dual solution y ===\n")
        f.write("y_B = A_B^-1 * c_T\n")
        
        f.write("\nA_B^-1 (Fractions):\n")
        for row in A_B_inv_fracs:
            f.write(str(row) + "\n")

        f.write("\ny (Fractions) = (")
        parts = [y_full_fracs[idx] for idx in all_indices]
        f.write(", ".join(parts))
        f.write(')\n')
        
        f.write("y (Floats)    = (")
        parts_float = [f"{y_full_vals[idx]:.4f}" for idx in all_indices]
        f.write(", ".join(parts_float))
        f.write(')')

    print(f"\nSaved in {output_file}")

if __name__ == "__main__":
    main()