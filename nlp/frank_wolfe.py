import sympy as sp
import numpy as np
from scipy.optimize import linprog
import sys

def parse_pnl_linear(filename):
    """Legge pnl.txt e restituisce f, A, b, x0 per problemi lineari."""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Errore: Il file '{filename}' non è stato trovato.")
        sys.exit(1)
    
    data = {}
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            data[key.strip()] = val.strip()

    vars_str = data.get('vars', '').split(',')
    vars_sym = [sp.symbols(v.strip()) for v in vars_str if v.strip()]
    local_dict = {str(v): v for v in vars_sym}

    # target function
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
                print(f"ERROR: Non linear constraint '{g.strip()}'"); sys.exit(1)
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

def analytic_line_search_bounded(f_sym, vars_sym, xk, direction):
    t = sp.symbols('t')
    # x(t) = xk + t * d
    x_param = [xk[i] + t * direction[i] for i in range(len(xk))]
    
    # substitute in f
    phi_t = f_sym.subs(dict(zip(vars_sym, x_param)))
    
    d_phi = sp.diff(phi_t, t) # derivative
    
    # solve d_phi = 0
    sol = sp.solve(d_phi, t)
    
    candidates = [0.0, 1.0]
    if sol:
        t_stat = float(sol[0])
        if 0 <= t_stat <= 1:
            candidates.append(t_stat)
            
    best_t = 0.0
    best_val = float('inf')
    
    for cand in candidates:
        val = float(phi_t.subs(t, cand))
        if val < best_val:
            best_val = val
            best_t = cand
            
    return best_t, phi_t

# --- Frank-Wolfe for min problem ---
def frank_wolfe_min(filename='pnl.txt'):
    f_sym, A, b, xk, vars_sym, max_iter = parse_pnl_linear(filename)
    
    print(f"f (minimize): {f_sym}")
    print(f"Starting point x0: {fmt_vec(xk)}")
    print("=" * 60)

    grad_sym = [sp.diff(f_sym, v) for v in vars_sym]

    for k in range(max_iter):
        print(f"\n>>> STEP {k+1}")
        
        # 1.  ∇f_x
        grad_val = np.array([float(g.subs(dict(zip(vars_sym, xk)))) for g in grad_sym])
        print(f"1) ∇f(x_{k}) = {fmt_vec(grad_val)}")
        
        # 2. LP problem solution
        # min ∇f(xk)^T * y  s.t. Ay <= b
        print(f"2) solve LP subproblem:")
        print(f"   min  {fmt_vec(grad_val)} * y")
        print(f"   s.t. {len(b)} linear constraints")
        
        res_lin = linprog(c=grad_val, A_ub=A, b_ub=b, bounds=(None, None), method='highs')
        
        if not res_lin.success:
            print("   ERROR: Linear problem is unlimited or impossible.")
            break
            
        yk = res_lin.x
        print(f"   -> opt solution LP (y^k): {fmt_vec(yk)}")
        
        # 3. discending direction
        direction = yk - xk
        print(f"3) Descending direction (d^k = y^k - x^k): {fmt_vec(direction)}")
        
        # 4. stationary check
        # gap = ∇f^T * d
        gap = np.dot(grad_val, direction)
        print(f"4) Stationary check (∇f^T * d): {gap:.5g}")
        
        if gap >= -1e-6: # to minimize, if the ∇ doesnt lower anymore
            print("   -> STOP: Il punto è stazionario (Gap ~= 0)")
            break

        # 5. find t that minimize f(xk + t*d)
        tk, phi_expr = analytic_line_search_bounded(f_sym, vars_sym, xk, direction)
        # clean to print
        if abs(tk - 1.0) < 1e-9: tk = 1.0
        if abs(tk) < 1e-9: tk = 0.0
        
        print(f"5) Find optimal step:")
        # phi(t) = restricted function
        phi_pretty = sp.N(phi_expr, 4) # pretty for print purposes lol
        print(f"   Minimize φ(t) = {phi_pretty}  su t in [0, 1]")
        print(f"   -> OPT STEP (t): {tk}")

        # 6. Update new point
        xk_next = xk + tk * direction
        print(f"6) NEW POINT x{k+1} = x{k} + t*d:")
        print(f"   x{k+1} = {fmt_vec(xk_next)}")
        
        xk = xk_next
        print("-" * 60)

if __name__ == "__main__":
    frank_wolfe_min()