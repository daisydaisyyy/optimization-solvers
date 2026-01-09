import sympy as sp
import numpy as np
from scipy.optimize import linprog
import sys

def parse_pnl_linear(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    data = {}
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            data[key.strip()] = val.strip()

    vars_str = data.get('vars', '').split(',')
    vars_sym = [sp.symbols(v.strip()) for v in vars_str if v.strip()]
    local_dict = {str(v): v for v in vars_sym}

    f_str = data.get('f').replace('^', '**')
    f_expr = sp.sympify(f_str, locals=local_dict)

    g_str = data.get('g', '')
    A = []
    b = []
    
    if g_str:
        g_list = g_str.split(',')
        for g in g_list:
            if not g.strip(): continue
            expr = sp.sympify(g.strip().replace('^', '**'), locals=local_dict)
            coeffs = expr.as_coefficients_dict()
            const = float(coeffs.get(1, 0)) 
            
            row = []
            is_linear = True
            for v in vars_sym:
                try: d = sp.degree(expr, v)
                except: d = 0
                if d > 1: is_linear = False
                row.append(float(expr.coeff(v)))
            
            if not is_linear:
                print(f"ERROR: Non-linear constraint '{g.strip()}'"); sys.exit(1)
            A.append(row)
            b.append(-const)

    x0_str = data.get('x0', '').split(',')
    x0 = np.array([float(x.strip()) for x in x0_str])
    iters = int(float(data.get('step_size', '5')))

    return f_expr, np.array(A), np.array(b), x0, vars_sym, iters

def fmt_vec(v):
    vals = []
    for x in v:
        if abs(x) < 1e-9: x = 0.0
        if abs(x - round(x)) < 1e-9:
            vals.append(f"{int(round(x))}")
        else:
            vals.append(f"{x:.4g}")
    return f"({', '.join(vals)})"

def fmt_mat(M):
    if M.ndim == 1: M = M.reshape(1, -1)
    rows_str = []
    for row in M:
        rows_str.append("   " + fmt_vec(row).replace("(", "[").replace(")", "]"))
    return "\n".join(rows_str)

def analytic_line_search_bounded(f_sym, vars_sym, xk, direction, max_step):
    t = sp.symbols('t')
    x_param = [xk[i] + t * direction[i] for i in range(len(xk))]
    phi_t = f_sym.subs(dict(zip(vars_sym, x_param)))
    d_phi = sp.diff(phi_t, t)
    
    sol = sp.solve(d_phi, t)
    
    candidates = [0.0, max_step]
    if sol:
        for s in sol:
            val_t = float(sp.re(s))
            if 0 <= val_t <= max_step:
                candidates.append(val_t)
            
    best_t = 0.0
    best_val = float('inf')
    
    for cand in candidates:
        val = float(phi_t.subs(t, cand))
        if val < best_val:
            best_val = val
            best_t = cand
            
    return best_t, phi_t

def projected_gradient(filename='pnl.txt'):
    f_sym, A, b, xk, vars_sym, max_iter = parse_pnl_linear(filename)
    n_vars = len(vars_sym)

    print(f"=== PROJECTED GRADIENT (Detailed Calculations) ===")
    print(f"Function: {f_sym}")
    print(f"Start x^0: {fmt_vec(xk)}")
    print("=" * 60)

    grad_sym = [sp.diff(f_sym, v) for v in vars_sym]

    for k in range(max_iter):
        print(f"\n>>> ITERATION {k+1}")
        
        grad_val = np.array([float(g.subs(dict(zip(vars_sym, xk)))) for g in grad_sym])
        print(f"[1] Gradient ∇f(x^{k}): {fmt_vec(grad_val)}")
        
        print(f"[2] Identification of Active Constraints:")
        print(f"    Checking which constraints Ax <= b satisfy equality (residual ~ 0):")
        
        residuals = np.dot(A, xk) - b
        active_indices = []
        
        for i, res in enumerate(residuals):
            terms = []
            for j, coeff in enumerate(A[i]):
                if abs(coeff) > 1e-9:
                    terms.append(f"{coeff:g}*{vars_sym[j]}")
            lhs_str = " + ".join(terms).replace("+ -", "- ")
            
            is_active = abs(res) < 1e-6
            status = "ACTIVE" if is_active else "Inactive"
            
            val_ax = np.dot(A[i], xk)
            print(f"    g_{i}: {lhs_str} <= {b[i]:g}")
            print(f"       Calculation: {val_ax:g} - {b[i]:g} = {res:.4g} --> {status}")
            
            if is_active:
                active_indices.append(i)
                
        if not active_indices:
            print("    -> No active constraints (M empty).")
            M = np.array([])
        else:
            print(f"    -> Active constraints (indices): {active_indices}")
            M = A[active_indices]
            print(f"    Active constraints matrix M:")
            print(fmt_mat(M))
        
        if M.size == 0:
            H = np.eye(n_vars)
            print(f"[3] Projection Matrix H = I (Identity)")
        else:
            try:
                inv_term = np.linalg.pinv(np.dot(M, M.T))
                proj_part = np.dot(M.T, np.dot(inv_term, M))
                H = np.eye(n_vars) - proj_part
                print(f"[3] Projection Matrix H = I - M^T(MM^T)^-1 M:")
            except:
                print("Matrix inversion error.")
                return

        print(fmt_mat(H))

        dk = -np.dot(H, grad_val)
        dk = np.array([0.0 if abs(x) < 1e-9 else x for x in dk])
        
        print(f"[4] Direction d^{k} = -H * ∇f:")
        print(f"    d^{k} = {fmt_vec(dk)}")
        
        if np.linalg.norm(dk) < 1e-6:
            print("\n!!! STOP: Null direction (Constrained Stationary Point).")
            break

        print(f"[5] Maximum admissible step calculation (t_max):")
        max_t = float('inf')
        
        ad = np.dot(A, dk)
        b_ax = b - np.dot(A, xk)
        
        print(f"    Checking intersections with constraints:")
        for i in range(len(b)):
            if ad[i] > 1e-9:
                t_lim = b_ax[i] / ad[i]
                print(f"    Constraint {i}: t <= ({b_ax[i]:.4g}) / ({ad[i]:.4g}) = {t_lim:.4g}")
                if t_lim < max_t:
                    max_t = t_lim
            else:
                pass 
        
        max_t_str = "Infinite" if max_t > 1e10 else f"{max_t:.4g}"
        print(f"    -> t_max = {max_t_str}")

        calc_bound = max_t if max_t < 1e10 else 100.0
        
        tk, phi_expr = analytic_line_search_bounded(f_sym, vars_sym, xk, dk, calc_bound)
        if abs(tk - round(tk)) < 1e-9: tk = float(round(tk))
        
        phi_pretty = sp.N(phi_expr, 4)
        print(f"[6] Exact Line Search on [0, {max_t_str}]:")
        print(f"    Minimizing φ(t) = {phi_pretty}")
        print(f"    -> Optimal step t* = {tk:g}")
        
        xk_next = xk + tk * dk
        print(f"[7] New point x^{k+1}: {fmt_vec(xk_next)}")
        print("-" * 60)
        
        xk = xk_next

if __name__ == "__main__":
    projected_gradient()