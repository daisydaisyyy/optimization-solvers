import re
import numpy as np
import os

INPUT_FILE = "data.txt"
OUTPUT_FILE = "dual.txt"

def format_num(val):
    if abs(val - round(val)) < 1e-9: return str(int(round(val)))
    return f"{val:.1f}"

def parse_primal(filename):
    if not os.path.exists(filename): raise FileNotFoundError("File not found")
    with open(filename, 'r') as f: lines = f.readlines()

    c = []
    constrs = []
    point = []
    is_min = True
    is_structural_section = True
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # non neg constraints
        if line.startswith("#") and "non" in line.lower() and "neg" in line.lower():
            is_structural_section = False
            continue
            
        if line.startswith("#"): continue
        
        if "target" in line:
            if "max" in line.lower(): is_min = False
            elif "min" in line.lower(): is_min = True
        
        if line.startswith("c:"):
            c = [float(x) for x in line.split(":")[1].strip(" []").split(",")]
            
        if "[POINT]" in line: pass
        elif not any(k in line for k in ["target", "c:", "CONSTRAINTS", "B:"]) and "," in line:
             try: point = [float(x) for x in line.split(',')]
             except: pass

        if ":" in line and any(x in line for x in ["=", "<=", ">="]):
            if "target" in line: continue

            # skip non-neg constraints
            if not is_structural_section: continue

            parts = line.split(":")
            rest = parts[1]
            sign = "="
            if "<=" in rest: sign = "<="; lhs, rhs = rest.split("<=")
            elif ">=" in rest: sign = ">="; lhs, rhs = rest.split(">=")
            else: lhs, rhs = rest.split("=")
            
            row_map = {}
            for t in lhs.replace("-", "+-").split("+"):
                t = t.strip()
                if not t: continue
                coeff = 1.0
                if "*" in t: 
                    co, var = t.split("*")
                    coeff = float(co.replace(" ", ""))
                    idx = int(re.search(r'x(\d+)', var).group(1)) - 1
                elif "x" in t:
                    if t.startswith("-"): coeff = -1.0
                    idx = int(re.search(r'x(\d+)', t).group(1)) - 1
                row_map[idx] = row_map.get(idx, 0) + coeff
            
            constrs.append({
                'A': row_map, 
                'sign': sign, 
                'b': float(rhs),
                'is_structural': is_structural_section # every constraints except non neg ones
            })

    return c, constrs, point, is_min

def write_dual(c, constrs, point, is_min, filename):
    n = len(c)
    m = len(constrs)
    
    with open(filename, "w") as f:
        target = "max" if is_min else "min"
        f.write(f"target: {target} ")
        obj = []
        for i, con in enumerate(constrs):
            if con['b'] != 0: obj.append(f"{con['b']:g}*y{i+1}")
        f.write(" + ".join(obj) + "\n\n[CONSTRAINTS]\n")
        
        for j in range(n):
            terms = []
            for i, con in enumerate(constrs):
                v = con['A'].get(j, 0)
                if v != 0: terms.append(f"{v:g}y{i+1}")
            
            rel = "<=" if is_min else ">="
            f.write(f"{' + '.join(terms)} {rel} {c[j]:g}\n")
            
        f.write("\n# Bounds\n")
        for i, con in enumerate(constrs):
            s = con['sign']
            if is_min:
                if s == "=": f.write(f"y{i+1} free\n")
                elif s == ">=": f.write(f"y{i+1} >= 0\n")
                elif s == "<=": f.write(f"y{i+1} <= 0\n")
            else:
                if s == "=": f.write(f"y{i+1} free\n")
                elif s == "<=": f.write(f"y{i+1} >= 0\n")
                elif s == ">=": f.write(f"y{i+1} <= 0\n")

    print(f"Dual saved to {filename}")

    if point:
        vals = list(point)
        for con in constrs:
            lhs = sum(con['A'].get(i,0)*point[i] for i in range(len(point)))
            
            if con['is_structural'] and con['sign'] in [">=", "<="]: # consider only disequality constraints to find the solution y
                 vals.append(lhs)
                 
        vec_str = "(" + ", ".join([format_num(v) for v in vals]) + ")"
        print(f"solution using the starting point: y = {vec_str}")

if __name__ == "__main__":
    try:
        c, constrs, pt, is_min = parse_primal(INPUT_FILE)
        write_dual(c, constrs, pt, is_min, OUTPUT_FILE)
    except Exception as e:
        print(f"Error: {e}")