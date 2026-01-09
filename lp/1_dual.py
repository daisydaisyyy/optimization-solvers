import re
import numpy as np

def parse_data(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # 1. Parsing Obiettivo (c)
    # Cerca pattern tipo: target: max - 1*x1 - 1*x2
    target_match = re.search(r'target:\s*max\s*(.*)', content)
    c_coeffs = []
    if target_match:
        eq = target_match.group(1)
        # Trova tutti i termini coefficiente*xN
        matches = re.findall(r'([+\-]?\s*\d+)\*x(\d+)', eq)
        # Ordina per indice variabile e estrai coeff
        matches.sort(key=lambda x: int(x[1])) 
        c_coeffs = [float(m[0].replace(' ', '')) for m in matches]

    # 2. Parsing Vincoli (A e b)
    # Cerca sezioni numerate. Es: 1 : - 100*x1 - 200*x2 <= -5000
    constraints = {}
    rows_pattern = re.findall(r'(\d+)\s*:\s*(.*?)\s*([<>=]+)\s*([+\-]?\d+)', content)
    
    A_rows = {}
    b_vals = {}
    
    num_vars = len(c_coeffs)
    
    for idx_str, expr, op, rhs in rows_pattern:
        idx = int(idx_str)
        b_vals[idx] = float(rhs)
        
        # Parsing coefficienti della riga
        row_coeffs = [0.0] * num_vars
        var_matches = re.findall(r'([+\-]?\s*\d+)\*x(\d+)', expr)
        for coeff_str, var_idx in var_matches:
            v_i = int(var_idx) - 1 # 0-based index
            if 0 <= v_i < num_vars:
                row_coeffs[v_i] = float(coeff_str.replace(' ', ''))
        
        A_rows[idx] = row_coeffs

    # 3. Parsing Base (B)
    # Prende l'ultima occorrenza di B se ce ne sono multiple
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
        print(f"Errore nel parsing del file: {e}")
        return

    # Costruisci matrice A_B basata sugli indici della base
    # Nota: B contiene indici 1-based delle righe (vincoli)
    try:
        A_B_list = [A_dict[idx] for idx in B_indices]
        A_B = np.array(A_B_list)
        print("A_B^T:\n",A_B.T)
    except KeyError as e:
        print(f"Errore: Indice di base {e} non trovato nei vincoli.")
        return

    # Calcolo soluzione duale y_B = A_B^-1 * c_T
    # c è trattato come vettore colonna per l'operazione
    try:
        A_B_inv = np.linalg.inv(A_B)
        print("\nA_B^-1:\n",A_B_inv)
        y_B_vals = c @ A_B_inv
        print("\nc:",c)
    except np.linalg.LinAlgError:
        print("Errore: La matrice A_B è singolare (non invertibile).")
        return

    # Mappa i valori calcolati nel vettore y completo (indicizzato per vincolo)
    # Inizializza tutto a 0
    all_indices = sorted(A_dict.keys())
    y_full = {idx: 0.0 for idx in all_indices}
    
    for i, basis_idx in enumerate(B_indices):
        y_full[basis_idx] = y_B_vals[i]

    with open(output_file, 'w') as f:
        f.write("=== dual problem ===\n")
        
        # target function
        obj_terms = []
        for idx in all_indices:
            val = b_dict[idx]
            if val != 0:
                obj_terms.append(f"{val}*y{idx}")
        f.write(f"Min W = {' + '.join(obj_terms)}\n\n")
        
        # dual constraints: A^T * y = c
        num_vars = len(c)
        for j in range(num_vars):
            constraint_terms = []
            for idx in all_indices:
                coeff = A_dict[idx][j]
                if coeff != 0:
                    constraint_terms.append(f"({coeff})*y{idx}")
            
            line = f"{' + '.join(constraint_terms)} = {c[j]}\n"
            f.write(line)
        
        f.write("\n=== dual solution y ===\n")
        f.write("y_B = A_B^-1 * c_T\n")
        f.write("y=(")
        for idx in all_indices:
            f.write(f"{y_full[idx]:.4f},")
        f.write(')')

    print(f"Operazione completata. Risultati salvati in {output_file}")

if __name__ == "__main__":
    main()