import sympy as sp
import pandas as pd
import numpy as np
import sys

def parse_pnl_file(filename):
    """Reads the pnl.txt file and returns symbolic expressions."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
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

    raw_f = data['f'].replace('^', '**')
    f_expr = sp.sympify(raw_f, locals=local_dict)

    g_str = data.get('g', '')
    g_exprs = []
    if g_str:
        raw_gs = g_str.split(',')
        for g in raw_gs:
            g_clean = g.strip().replace('^', '**')
            if g_clean:
                g_exprs.append(sp.sympify(g_clean, locals=local_dict))

    return f_expr, g_exprs, vars_sym

def to_latex_frac(val):
    """Converts a value to a readable fractional string."""
    try:
        # Try to convert to rational
        r = sp.Rational(val)
        return str(r)
    except:
        return f"{float(val):.4f}"

def lkkt_solve(f, g, vars_sym):
    print("=== KKT Analysis (Fractional Output) ===")
    print(f"Function f(x): {f}")
    print(f"Constraints g(x) <= 0: {g}")
    print("-" * 40)

    n_g = len(g)
    lambdas = list(sp.symbols(f'lambda0:{n_g}'))
    
    # 1. Lagrangian Gradient
    grad_f = [sp.diff(f, v) for v in vars_sym]
    eqs_grad = list(grad_f)
    
    for i, g_i in enumerate(g):
        grad_g_i = [sp.diff(g_i, v) for v in vars_sym]
        for j in range(len(vars_sym)):
            eqs_grad[j] = eqs_grad[j] + lambdas[i] * grad_g_i[j]

    print("1. Stationary System (∇L = 0):")
    for eq in eqs_grad:
        print(f"   {eq} = 0")

    # 2. Complementarity
    eqs_slack = []
    print("\n2. Complementarity Conditions:")
    for i, g_i in enumerate(g):
        slack_eq = lambdas[i] * g_i
        eqs_slack.append(slack_eq)
        print(f"   {lambdas[i]} * ({g_i}) = 0")

    system_eqs = eqs_grad + eqs_slack
    system_vars = vars_sym + lambdas
    
    print("\nCalculating symbolic solutions...")
    try:
        solutions = sp.solve(system_eqs, system_vars, dict=True)
    except Exception as e:
        print(f"Solve error: {e}")
        return

    # 3. solution
    H_f = sp.hessian(f, vars_sym)
    results_data = []

    if not solutions:
        print("No solution found.")
        return

    for sol in solutions:
        vals_map_sym = {}
        vals_map_num = {}
        is_real = True
        
        # x
        for v in vars_sym:
            val_sym = sol.get(v, 0)
            try:
                val_num = float(val_sym.evalf()) if hasattr(val_sym, 'evalf') else float(val_sym)
                if isinstance(val_sym, sp.Expr) and not val_sym.is_real:
                     if abs(complex(val_sym).imag) > 1e-6:
                         is_real = False
                         break
                vals_map_sym[v] = val_sym
                vals_map_num[v] = val_num
            except:
                is_real = False

        if not is_real: continue

        for l in lambdas:
            val_sym = sol.get(l, 0)
            try:
                val_num = float(val_sym.evalf()) if hasattr(val_sym, 'evalf') else float(val_sym)
                vals_map_sym[l] = val_sym
                vals_map_num[l] = val_num
            except:
                vals_map_sym[l] = sp.S.Zero
                vals_map_num[l] = 0.0

        # Verify Constraints g(x) <= 0
        g_feasible = True
        for g_i in g:
            val_g = float(g_i.subs(vals_map_num))
            if val_g > 1e-6: 
                g_feasible = False
                break
        
        if not g_feasible: continue

        # Calculate f(x)
        f_sym_val = f.subs(vals_map_sym)
        f_num_val = float(f_sym_val)

        # Hessian Classification
        H_num = np.array(H_f.subs(vals_map_num)).astype(float)
        try:
            eigvals = np.linalg.eigvals(H_num)
            if np.all(eigvals > 1e-7):
                pt_type = "Convex (Local Min)"
            elif np.all(eigvals < -1e-7):
                pt_type = "Concave (Local Max)"
            elif np.any(eigvals > 1e-7) and np.any(eigvals < -1e-7):
                pt_type = "Saddle Point"
            else:
                pt_type = "Indeterminate"
        except:
            pt_type = "Hessian Error"

        row = {}
        row['sort_val'] = f_num_val 
        
        for v in vars_sym:
            row[str(v)] = to_latex_frac(vals_map_sym[v])
            
        for i, l in enumerate(lambdas):
            row[f'λ_{i}'] = to_latex_frac(vals_map_sym[l])
            
        row['f(x)'] = to_latex_frac(f_sym_val)
        row['Curvature Type'] = pt_type
        results_data.append(row)

    if not results_data:
        print("No feasible solution found.")
        return

    df = pd.DataFrame(results_data)


    # FIND GLOBAL MAX/MIN
    min_val = df['sort_val'].min()
    max_val = df['sort_val'].max()

    def tag_global(row):
        lbl = row['Curvature Type']
        val = row['sort_val']
        # numeric approx
        if abs(val - min_val) < 1e-6:
            return "GLOBAL MIN"
        if abs(val - max_val) < 1e-6 and max_val != min_val:
            return "GLOBAL MAX"
        return "Stationary"

    df['Categorization'] = df.apply(tag_global, axis=1)

    cols = [str(v) for v in vars_sym] + \
           [f'λ_{i}' for i in range(len(lambdas))] + \
           ['f(x)', 'Categorization']
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n=== Results ===")
    print(df[cols].to_string(index=False, col_space=10))

if __name__ == "__main__":
    f_e, g_e, v_s = parse_pnl_file('pnl.txt')
    lkkt_solve(f_e, g_e, v_s)