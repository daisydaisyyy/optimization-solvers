import sympy as sp
import sys
import math

def get_constraints(filename='v.txt'):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    points = []
    for line in lines:
        if not line.strip(): continue
        parts = line.split(',')
        if len(parts) < 2: continue
        x_val = sp.sympify(parts[0].strip())
        y_val = sp.sympify(parts[1].strip())
        points.append((x_val, y_val))

    if len(points) < 3:
        print("Error: Need at least 3 points to form a polygon.")
        sys.exit(1)

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    points.sort(key=lambda p: float(sp.atan2(p[1] - cy, p[0] - cx)))

    x1, x2 = sp.symbols('x1 x2')
    constraints = []
    
    n = len(points)
    for i in range(n):
        p_curr = points[i]
        p_next = points[(i + 1) % n]

        A = p_curr[1] - p_next[1]
        B = p_next[0] - p_curr[0]
        C = -A * p_curr[0] - B * p_curr[1]

        expr = A * x1 + B * x2 + C

        check_val = expr.subs({x1: cx, x2: cy})

        if check_val > 0:
            expr = -expr
        
        constraints.append(str(expr).replace('**', '^'))

    print("g: " + ", ".join(constraints))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        get_constraints(sys.argv[1])
    else:
        get_constraints()