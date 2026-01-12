import sympy as sp
import numpy as np
from scipy.optimize import linprog
from fractions import Fraction
import sys

def parsing(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: '{filename}' not found")
        sys.exit(1)
    
    data = {}
    mode = 'min' # default

    for line in lines:
        line_lower = line.lower().strip()
        if line_lower == 'max' or line_lower.startswith('max '):
            mode = 'max'
        elif line_lower == 'min' or line_lower.startswith('min '):
            mode = 'min'

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
            
            raw_g = g.strip().replace('^', '**')
            
            # handle standard inequalities to normalize to expression <= 0
            if '<=' in raw_g:
                lhs, rhs = raw_g.split('<=')
                # LHS <= RHS  -->  LHS - RHS <= 0
                expr = sp.sympify(lhs, locals=local_dict) - sp.sympify(rhs, locals=local_dict)
            elif '>=' in raw_g:
                lhs, rhs = raw_g.split('>=')
                # LHS >= RHS  -->  RHS - LHS <= 0
                expr = sp.sympify(rhs, locals=local_dict) - sp.sympify(lhs, locals=local_dict)
            else:
                # <= 0 if no operator found
                expr = sp.sympify(raw_g, locals=local_dict)

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
    x0 = np.array([float(sp.sympify(x.strip())) for x in x0_str])
    iters = int(float(data.get('step_size', '5')))

    return mode, f_expr, np.array(A), np.array(b), x0, vars_sym, iters

def to_frac(val):
    return str(Fraction(val).limit_denominator(1000))

def fmt_vec_frac(v):
    vals = [to_frac(x) for x in v]
    return f"({', '.join(vals)})"

def format_exp(coeffs, vars_sym):
    terms = []
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-9: continue
        frac = Fraction(c).limit_denominator(1000)
        sign = "+" if frac > 0 else "-"
        if len(terms) == 0: sign = "" if frac > 0 else "-"
        
        abs_frac = abs(frac)
        coeff_str = str(abs_frac) if abs_frac != 1 else ""
        terms.append(f"{sign} {coeff_str}{vars_sym[i]}")
    
    if not terms: return "0"
    return " ".join(terms).strip()

# find optimal step for min problems
def find_opt_step_min(f_sym, vars_sym, xk, direction):
    t = sp.symbols('t')
    # x(t) = xk + t * d
    x_param = [xk[i] + t * direction[i] for i in range(len(xk))]
    
    phi_t = f_sym.subs(dict(zip(vars_sym, x_param)))
    
    phi_expanded = sp.expand(phi_t)
    phi_disp = sp.nsimplify(phi_expanded, tolerance=1e-8, rational=True)
    print(f"   a) φ(t) = f(x^k + t^k*d^k):")
    print(f"      {phi_disp}")
    
    d_phi = sp.diff(phi_t, t)
    d_phi_disp = sp.nsimplify(sp.expand(d_phi), tolerance=1e-8, rational=True)
    print(f"   b) φ'(t): {d_phi_disp}")
    
    print(f"   c) Stationary points analysis:")
    print(f"      φ'(t) = 0 -> {d_phi_disp} = 0")
    
    c_t = sp.expand(d_phi).coeff(t)
    c_const = sp.expand(d_phi).subs(t, 0)
    
    if c_t != 0:
        try:
            val_a = sp.nsimplify(c_t, tolerance=1e-8, rational=True)
            val_b = sp.nsimplify(c_const, tolerance=1e-8, rational=True)
            val_rhs = -val_b
            
            print(f"      {val_a}*t = {val_rhs}")
            print(f"      t = {val_rhs} / {val_a}")
        except:
            pass 

    sol = sp.solve(d_phi, t)
    
    sol_formatted = []
    for s in sol:
        s_simp = sp.nsimplify(s, tolerance=1e-8, rational=True)
        sol_formatted.append(str(s_simp))
    
    print(f"      t: {', '.join(sol_formatted)}")
    
    candidates = [0.0, 1.0]
    if sol:
        for s in sol:
            try:
                val_s = float(s)
                if 0 < val_s < 1: 
                    candidates.append(val_s)
            except: pass
    
    candidates = sorted(list(set(candidates)))
    
    best_t = 0.0
    best_val = float('inf')
    
    print(f"   d) φ in [0, 1]:")
    for cand in candidates:
        val = float(phi_t.subs(t, cand))
        
        val_frac = str(Fraction(val).limit_denominator(1000))
        cand_str = str(cand)
        if cand == 0: cand_str = "0"
        elif cand == 1: cand_str = "1"
        else: cand_str = str(Fraction(cand).limit_denominator(1000))
            
        print(f"      φ({cand_str}) = {val_frac}")
        
        if val < best_val:
            best_val = val
            best_t = cand
            
    return best_t

# find optimal step for max problem
def find_opt_step_max(f_sym, vars_sym, xk, direction):
    t = sp.symbols('t')
    # x(t) = xk + t * d
    x_param = [xk[i] + t * direction[i] for i in range(len(xk))]
    
    phi_t = f_sym.subs(dict(zip(vars_sym, x_param)))
    
    phi_expanded = sp.expand(phi_t)
    phi_disp = sp.nsimplify(phi_expanded, tolerance=1e-8, rational=True)
    print(f"   a) φ(t) = f(x^k + t^k*d^k):")
    print(f"      {phi_disp}")
    
    d_phi = sp.diff(phi_t, t)
    d_phi_disp = sp.nsimplify(sp.expand(d_phi), tolerance=1e-8, rational=True)
    print(f"   b) φ'(t): {d_phi_disp}")

    print(f"   c) Stationary points analysis:")
    print(f"      φ'(t) = 0 -> {d_phi_disp} = 0")

    c_t = sp.expand(d_phi).coeff(t)
    c_const = sp.expand(d_phi).subs(t, 0)

    if c_t != 0:
        try:
            val_a = sp.nsimplify(c_t, tolerance=1e-8, rational=True)
            val_b = sp.nsimplify(c_const, tolerance=1e-8, rational=True)
            val_rhs = -val_b

            print(f"      {val_a}*t = {val_rhs}")
            print(f"      t = {val_rhs} / {val_a}")
        except:
            pass

    sol = sp.solve(d_phi, t)

    sol_formatted = []
    for s in sol:
        s_simp = sp.nsimplify(s, tolerance=1e-8, rational=True)
        sol_formatted.append(str(s_simp))

    print(f"      t: {', '.join(sol_formatted)}")

    candidates = [0.0, 1.0]
    if sol:
        for s in sol:
            try:
                val_s = float(s)
                if 0 < val_s < 1:
                    candidates.append(val_s)
            except: pass

    candidates = sorted(list(set(candidates)))

    best_t = 0.0
    best_val = -float('inf')

    print(f"   d) φ in [0, 1]:")
    for cand in candidates:
        val = float(phi_t.subs(t, cand))

        val_frac = str(Fraction(val).limit_denominator(1000))
        cand_str = str(cand)
        if cand == 0: cand_str = "0"
        elif cand == 1: cand_str = "1"
        else: cand_str = str(Fraction(cand).limit_denominator(1000))
            
        print(f"      φ({cand_str}) = {val_frac}")
        
        if val > best_val:
            best_val = val
            best_t = cand
            
    return best_t

def frank_wolfe_min(f_expr, A, b, xk, vars_sym, max_iter):
    print(f"f (minimize): {f_expr}")
    print(f"Starting point x0: {fmt_vec_frac(xk)}")
    print("=" * 60)

    grad_sym = [sp.diff(f_expr, v) for v in vars_sym]
    y_syms = [sp.symbols(f"y{i+1}") for i in range(len(vars_sym))]

    for k in range(max_iter):
        print(f"\n>>> STEP {k+1}")
        
        # 1.  ∇f_x
        grad_val = np.array([float(g.subs(dict(zip(vars_sym, xk)))) for g in grad_sym])
        print(f"1) ∇f(x_{k}) = {fmt_vec_frac(grad_val)}")
        
        # 2. LP problem solution
        print(f"2) solve LP subproblem (find y^k):")
        obj_str = format_exp(grad_val, y_syms)
        print(f"   min  {obj_str}")
        print("   s.t.")
        for row_idx, row in enumerate(A):
            lhs = format_exp(row, y_syms)
            rhs = str(Fraction(b[row_idx]).limit_denominator(1000))
            print(f"        {lhs} <= {rhs}")
            
        res_lin = linprog(c=grad_val, A_ub=A, b_ub=b, bounds=(None, None), method='highs')
        
        if not res_lin.success:
            print("   ERROR: Linear problem is unlimited or impossible.")
            break
            
        yk = res_lin.x
        print(f"   -> Solver found the opt vertex y^{k}:")
        print(f"      y^{k} = {fmt_vec_frac(yk)}")
        
        # 3. discending direction
        direction = yk - xk
        print(f"3) Desc direction (d^k = y^k - x^k):")
        for i, var_name in enumerate(vars_sym):
            y_frac = Fraction(yk[i]).limit_denominator(1000)
            x_frac = Fraction(xk[i]).limit_denominator(1000)
            res_frac = y_frac - x_frac
            print(f"   {var_name}: y_{i+1} - x_{i+1} = {y_frac} - ({x_frac}) = {res_frac}")
        print(f"   -> d^{k} = {fmt_vec_frac(direction)}")
        
        # 4. stationary check
        gap = np.dot(grad_val, direction)
        print(f"4) Stationary check (∇f^T * d): {to_frac(gap)}")
        
        if gap >= -1e-6:
            print("   -> STOP: Stationary point(∇f^T * d = 0)")
            break

        # 5. find t
        print(f"5) Find optimal step (Dettagli):")
        tk = find_opt_step_min(f_expr, vars_sym, xk, direction)
        
        if abs(tk - 1.0) < 1e-9: tk = 1.0
        if abs(tk) < 1e-9: tk = 0.0
        tk_frac = str(Fraction(tk).limit_denominator(1000))
        print(f"   -> OPT STEP (t): {tk} -> {tk_frac}")

        # 6. update new point
        xk_next = xk + tk * direction
        print(f"6) NEW POINT x^{k+1} = x^{k} + t^{k}*d^{k}:")
        
        t_frac = Fraction(tk).limit_denominator(1000)
        for i, var_name in enumerate(vars_sym):
            x_old_frac = Fraction(xk[i]).limit_denominator(1000)
            d_frac = Fraction(direction[i]).limit_denominator(1000)
            res_frac = Fraction(xk_next[i]).limit_denominator(1000)
            print(f"   Comp. {var_name}: {x_old_frac} + {t_frac} * ({d_frac}) = {res_frac}")
            
        print(f"   -> x^{k+1} = {fmt_vec_frac(xk_next)}")
        
        xk = xk_next
        print("-" * 60)

def frank_wolfe_max(f_expr, A, b, xk, vars_sym, max_iter):
    print(f"f (MAXIMIZE): {f_expr}")
    print(f"Starting point x0: {fmt_vec_frac(xk)}")
    print("=" * 60)

    grad_sym = [sp.diff(f_expr, v) for v in vars_sym]
    y_syms = [sp.symbols(f"y{i+1}") for i in range(len(vars_sym))]

    for k in range(max_iter):
        print(f"\n>>> STEP {k+1}")
        
        # 1.  ∇f_x
        grad_val = np.array([float(g.subs(dict(zip(vars_sym, xk)))) for g in grad_sym])
        print(f"1) ∇f(x_{k}) = {fmt_vec_frac(grad_val)}")
        
        # 2. LP problem solution (MAX)
        print(f"2) solve LP subproblem (trovare vertice y^k):")
        obj_str = format_exp(grad_val, y_syms)
        print(f"   max  {obj_str}")
        print("   s.t.")
        for row_idx, row in enumerate(A):
            lhs = format_exp(row, y_syms)
            rhs = str(Fraction(b[row_idx]).limit_denominator(1000))
            print(f"        {lhs} <= {rhs}")
        
        # linprog minimizes c*x. To maximize ∇f*y, minimize (-∇f)*y
        res_lin = linprog(c=-grad_val, A_ub=A, b_ub=b, bounds=(None, None), method='highs')
        
        if not res_lin.success:
            print("   ERROR: Linear problem is unlimited or impossible.")
            break
            
        yk = res_lin.x
        print(f"   -> Solver found the opt vertex y^{k}:")
        print(f"      y^{k} = {fmt_vec_frac(yk)}")
        
        # 3. discending direction
        direction = yk - xk
        print(f"3) Desc direction (d^k = y^k - x^k):")
        for i, var_name in enumerate(vars_sym):
            y_frac = Fraction(yk[i]).limit_denominator(1000)
            x_frac = Fraction(xk[i]).limit_denominator(1000)
            res_frac = y_frac - x_frac
            print(f"   {var_name}: y_{i+1} - x_{i+1} = {y_frac} - ({x_frac}) = {res_frac}")
        print(f"   -> d^{k} = {fmt_vec_frac(direction)}")
        
        # 4. stationary check
        gap = np.dot(grad_val, direction)
        print(f"4) Stationary check (∇f^T * d): {to_frac(gap)}")
        
        # 5. find t (MAXIMIZE)
        print(f"5) Find optimal step (Dettagli):")
        tk = find_opt_step_max(f_expr, vars_sym, xk, direction)
        
        if abs(tk - 1.0) < 1e-9: tk = 1.0
        if abs(tk) < 1e-9: tk = 0.0
        tk_frac = str(Fraction(tk).limit_denominator(1000))
        print(f"   -> OPT STEP (t): {tk} -> {tk_frac}")

        # 6. update new point
        xk_next = xk + tk * direction
        print(f"6) NEW POINT x{k+1} = x^{k} + t^{k}*d^{k }:")
        
        t_frac = Fraction(tk).limit_denominator(1000)
        for i, var_name in enumerate(vars_sym):
            x_old_frac = Fraction(xk[i]).limit_denominator(1000)
            d_frac = Fraction(direction[i]).limit_denominator(1000)
            res_frac = Fraction(xk_next[i]).limit_denominator(1000)
            print(f"   Comp. {var_name}: {x_old_frac} + {t_frac} * ({d_frac}) = {res_frac}")
            
        print(f"   -> x{k+1} = {fmt_vec_frac(xk_next)}")
        
        xk = xk_next
        print("-" * 60)

if __name__ == "__main__":
    filename = 'pnl.txt'
    mode, f_expr, A, b, x0, vars_sym, iters = parsing(filename)
    
    if mode == 'max':
        frank_wolfe_max(f_expr, A, b, x0, vars_sym, iters)
    else:
        frank_wolfe_min(f_expr, A, b, x0, vars_sym, iters)