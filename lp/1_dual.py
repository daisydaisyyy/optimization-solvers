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
    constraints = []
    x_start = None

    mode = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"): 
            continue

        if "[PUNTO]" in line: 
            mode = "POINT"
            continue
        elif "[VINCOLI]" in line: 
            mode = "CONSTR"
            continue

        if mode == "POINT" and line.startswith("c:"):
            c = [float(x) for x in line.split(":")[1].replace('[','').replace(']','').split(",")]
        elif mode == "CONSTR":
            if ":" in line and ("<=" in line or ">=" in line):
                con_id = int(line.split(":")[0].strip())
                rest = line.split(":")[1].strip()
                if "<=" in rest: sep = "<="
                else: sep = ">="
                lhs, rhs = rest.split(sep)
                terms = lhs.replace("-", "+-").split("+")
                row = []
                for term in terms:
                    term = term.strip()
                    if not term: continue
                    if '*' in term:
                        coeff, var = term.split('*')
                        coeff = float(coeff)
                    else:
                        coeff = float(term)
                    row.append(coeff)
                constraints.append((row, float(rhs)))

    c = np.array(c)
    m = len(constraints)
    n = len(c)
    A = np.zeros((m, n))
    b = np.zeros(m)
    for i, (row, rhs) in enumerate(constraints):
        for j in range(len(row)):
            A[i, j] = row[j]
        b[i] = rhs

    return c, A, b

def write_dual(c, A, b, output_file):
    m, n = A.shape
    with open(output_file, "w") as f:
        f.write("[PUNTO]\n")
        f.write("c: [" + ", ".join(f"{val:.2f}" for val in b) + "]\n\n")
        f.write("[VINCOLI]\n")
        for j in range(n):
            terms = []
            for i in range(m):
                if abs(A[i,j]) > 1e-9:
                    terms.append(f"{A[i,j]:.2f}*y{i+1}")
            if terms:
                f.write(f"{j+1} : " + " + ".join(terms) + f" >= {c[j]:.2f}\n")
        f.write("\n# Non negatività\n")
        for i in range(m):
            f.write(f"{i+1} : 1*y{i+1} >= 0\n")

if __name__ == "__main__":
    c, A, b = parse_primal(INPUT_FILE)
    write_dual(c, A, b, OUTPUT_FILE)
    print(f"Duale scritto su {OUTPUT_FILE}")
