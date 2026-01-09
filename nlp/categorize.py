import sympy as sp
import pandas as pd
import numpy as np
import sys

def parse_pnl_file(filename):
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

    f_expr = sp.sympify(data['f'].replace('^', '**'), locals=local_dict)
    
    g_str = data.get('g', '')
    g_exprs = []
    if g_str:
        for g in g_str.split(','):
            if g.strip():
                g_exprs.append(sp.sympify(g.strip().replace('^', '**'), locals=local_dict))

    return f_expr, g_exprs, vars_sym

def to_latex_frac(val):
    try:
        return str(sp.Rational(val))
    except:
        return str(val)

def run_categorize():
    f_sym, g_syms, vars_sym = parse_pnl_file('pnl.txt')
    
    lams_sym = list(sp.symbols(f'λ_0:{len(g_syms)}'))
    
    L = f_sym + sum(l * g for l, g in zip(lams_sym, g_syms))
    
    grad_L_eqs = [sp.diff(L, v) for v in vars_sym]
    
    slack_eqs = [l * g for l, g in zip(lams_sym, g_syms)]
    
    system = grad_L_eqs + slack_eqs
    symbols_all = vars_sym + lams_sym
    
    print("Calculating KKT solutions...")
    try:
        solutions = sp.solve(system, symbols_all, dict=True)
    except Exception as e:
        print(f"Solver error: {e}")
        return
    
    H_matrix = sp.hessian(f_sym, vars_sym)
    
    results = []
    
    for sol in solutions:
        vals_map = {}
        is_real = True
        
        for s in symbols_all:
            val = sol.get(s, 0)
            if not val.is_real:
                is_real = False; break
            vals_map[s] = val
            
        if not is_real: continue

        g_ok = True
        for g in g_syms:
            if float(g.subs(vals_map)) > 1e-6:
                g_ok = False; break
        if not g_ok: continue

        l_vals = [vals_map[l] for l in lams_sym]
        
        all_pos_lambda = all(float(l) >= -1e-6 for l in l_vals)
        
        H_subs = H_matrix.subs(vals_map)
        eig_dict = H_subs.eigenvals()
        eig_sym = list(eig_dict.keys())
        
        eig_floats = []
        for e in eig_sym:
            try:
                eig_floats.append(float(e.evalf()))
            except:
                eig_floats.append(0.0)

        is_convex = np.all(np.array(eig_floats) > 1e-7)   
        is_concave = np.all(np.array(eig_floats) < -1e-7) 
        
        mL, mG, ML, MG, S = "NO", "NO", "NO", "NO", "NO"
        
        if is_convex:
            if all_pos_lambda: mL = "YES"
            else: ML = "YES"
        elif is_concave:
            if all_pos_lambda: ML = "YES"
            else: mL = "YES"
        else:
            S = "YES"

        res = {
            'x': [vals_map[v] for v in vars_sym],
            'l': l_vals,
            'f_val': float(f_sym.subs(vals_map)),
            'mL': mL, 'ML': ML, 'S': S,
            'vals_map': vals_map,
            'grad_eqs': grad_L_eqs,
            'eig_sym': eig_sym
        }
        results.append(res)

    if not results:
        print("No solution found.")
        return
        
    vals = [r['f_val'] for r in results]
    min_v, max_v = min(vals), max(vals)
    
    for r in results:
        r['mG'] = "YES" if abs(r['f_val'] - min_v) < 1e-6 else "NO"
        r['MG'] = "YES" if abs(r['f_val'] - max_v) < 1e-6 else "NO"

    print("="*95)
    print(f"{'x':<15} | {'λ':<35} | {'mL':<3} {'mG':<3} {'ML':<3} {'MG':<3} {'S':<3}")
    print("-" * 95)
    for r in results:
        x_str = f"({', '.join([to_latex_frac(x) for x in r['x']])})"
        l_str = "(" + ", ".join([to_latex_frac(l) for l in r['l']]) + ")"
        
        print(f"{x_str:<15} | {l_str:<35} | {r['mL']:<3} {r['mG']:<3} {r['ML']:<3} {r['MG']:<3} {r['S']:<3}")
    print("="*95 + "\n")

    print("STEPS TO CATEGORIZE POINTS:")
    
    for i, r in enumerate(results):
        print(f"\n" + "="*60)
        x_lbl = f"({', '.join([str(x) for x in r['x']])})"
        print(f">>> POINT {i+1}: {x_lbl}")
        
        print("    1. Stationarity equations (grad_f + sum λ*grad_g = 0):")
        print("       {")
        for eq in r['grad_eqs']:
            eq_vals = eq.subs(r['vals_map'])
            print(f"       {eq}  [Substitute x,λ] -> {eq_vals}")
        print("       }")

        print(f"\n    2. Verify Multipliers:")
        l_vals_fmt = [to_latex_frac(l) for l in r['l']]
        print(f"       λ = {l_vals_fmt}")
        
        neg_lams = [l for l in r['l'] if l < -1e-9]
        if neg_lams:
             print(f"       -> Negative λ found: {neg_lams}")
             print(f"       -> Pushing against constraint (Not a standard Min KKT)")
        else:
             print(f"       -> All λ >= 0. KKT Min condition satisfied.")

        print(f"\n    3. Hessian f(x):")
        
        eig_str = [str(e) for e in r['eig_sym']]
        
        print(f"       H = {H_matrix}")
        print(f"       Eigenvalues = {eig_str}")
        
        eig_floats = [float(e.evalf()) for e in r['eig_sym']]
        if np.all(np.array(eig_floats) > 0):
            print("       -> H is Positive Definite")
            if not neg_lams:
                print("       -> LOCAL MINIMUM")
            else:
                print("       -> LOCAL MAXIMUM")
        else:
             print("       -> H is indefinite or negative.")

    print("\n" + "="*80)
    print("FINAL CLASSIFICATION (Global Extrema & Saddles)")
    print("="*80)

    saddles = [r for r in results if r['S'] == 'YES']
    candidates = [r for r in results if r['S'] == 'NO']

    if saddles:
        print(">>> SADDLE POINTS:")
        for s in saddles:
            x_str = f"({', '.join([to_latex_frac(x) for x in s['x']])})"
            print(f"    x = {x_str} is a Saddle Point")
    else:
        print(">>> No Saddle Points found.")

    print("-" * 40)

    print(">>> GLOBAL SEARCH (Substitution Method):")
    if not candidates:
        print("    No valid candidates for global extrema.")
    else:
        print("    Comparing f(x) values of valid stationary points:")
        
        candidates.sort(key=lambda x: x['f_val'])
        
        for r in candidates:
            x_str = f"({', '.join([to_latex_frac(x) for x in r['x']])})"
            tipo = "Min" if r['mL'] == 'YES' else "Max"
            print(f"    x = {x_str:<15} -> f(x) = {r['f_val']:.4f}  [{tipo}]")

        min_glob = candidates[0]   
        max_glob = candidates[-1]  

        print("\n    CONCLUSION:")
        
        x_min = f"({', '.join([to_latex_frac(x) for x in min_glob['x']])})"
        print(f"    GLOBAL MINIMUM at x = {x_min} with value {min_glob['f_val']:.4f}")
        
        x_max = f"({', '.join([to_latex_frac(x) for x in max_glob['x']])})"
        print(f"    GLOBAL MAXIMUM at x = {x_max} with value {max_glob['f_val']:.4f}")

if __name__ == "__main__":
    run_categorize()