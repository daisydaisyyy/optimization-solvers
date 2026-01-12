import sympy as sp
import pandas as pd
import numpy as np
import sys
def parse(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    data = {}
    is_max = False
    for line in lines:
        if 'max' in line.lower(): is_max = True
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
                raw_expr = sp.sympify(g.strip().replace('^', '**'), locals=local_dict)
                
                if hasattr(raw_expr, 'lhs') and hasattr(raw_expr, 'rhs'):
                    g_exprs.append(raw_expr.lhs - raw_expr.rhs)
                else:
                    g_exprs.append(raw_expr)

    return f_expr, g_exprs, vars_sym, is_max

def solve():
    f_sym, g_syms, vars_sym, is_max = parse('pnl.txt')
    
    # setup Lagrangian
    lams_sym = list(sp.symbols(f'λ_0:{len(g_syms)}'))
    L = f_sym + sum(l * g for l, g in zip(lams_sym, g_syms))
    
    grad_L_eqs = [sp.diff(L, v) for v in vars_sym]
    slack_eqs = [l * g for l, g in zip(lams_sym, g_syms)]
    
    system = grad_L_eqs + slack_eqs
    symbols_all = vars_sym + lams_sym
    
    for v, eq in zip(vars_sym, grad_L_eqs):
        eq_str = str(eq).replace('**', '^')
        print(f"   ∂L/∂{v} : {eq_str} = 0")
    
    print("\n   (λ * g = 0):")
    for i, eq in enumerate(slack_eqs):
        eq_str = str(eq).replace('**', '^')
        print(f"   λ_{i} * g_{i} : {eq_str} = 0")
    
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
        
        res = {
            'x': [vals_map[v] for v in vars_sym],
            'l': l_vals,
            'f_val': f_sym.subs(vals_map),
            'vals_map': vals_map,
            'grad_eqs': grad_L_eqs,
            'hessian': H_matrix.subs(vals_map)
        }
        results.append(res)

    if not results:
        print("No solution found.")
        return
        
    print(f"\n{'='*80}")
    print(f"STATIONARY POINTS ({'MAX' if is_max else 'MIN'} PROBLEM)")
    print(f"{'='*80}")
    table_rows = []

    for i, r in enumerate(results):
        x_lbl = f"({', '.join([str(x) for x in r['x']])})"
        
        print(f"\n>>> POINT {i+1}: {x_lbl}")
        print(f"    f(x) = {float(r['f_val']):.4f}")
        
        # [1] Active Constraints
        print(f"\n    [1] ACTIVE CONSTRAINTS IDENTIFICATION:")
        active_indices = []
        for idx, g in enumerate(g_syms):
            val_g_sym = g.subs(r['vals_map'])
            val_g_num = float(val_g_sym)
            status = "INACTIVE"
            if abs(val_g_num) < 1e-6:
                status = "ACTIVE"
                active_indices.append(idx)
            g_str = str(g).replace('**', '^')
            print(f"       g_{idx}: {g_str:<20} = {val_g_sym} -> {status}")
            
        # [2]
        print(f"\n    [2] ∇f(x*):")
        grad_f_sym = [sp.diff(f_sym, v) for v in vars_sym]
        grad_f_sym_str = "(" + ", ".join([str(e).replace('**', '^') for e in grad_f_sym]) + ")"
        grad_f_val = [v.subs(r['vals_map']) for v in grad_f_sym]
        grad_f_str = f"({', '.join([str(v) for v in grad_f_val])})"
        print(f"       ∇f(x*) = {grad_f_sym_str} = {grad_f_str}")
        
        grad_g_vals = {}
        for idx in active_indices:
            g_expr = g_syms[idx]
            grad_g_sym = [sp.diff(g_expr, v) for v in vars_sym]
            grad_g_sym_str = "(" + ", ".join([str(e).replace('**', '^') for e in grad_g_sym]) + ")"
            grad_g_num = [v.subs(r['vals_map']) for v in grad_g_sym]
            grad_g_vals[idx] = grad_g_num
            g_vec_str = f"({', '.join([str(v) for v in grad_g_num])})"
            print(f"       ∇g_{idx}(x*) = {grad_g_sym_str} = {g_vec_str}")
            
        # [3] KKT System
        print(f"\n    [3] KKT SYSTEM (∇f + Σ λ_i ∇g_i = 0):")
        for dim, var in enumerate(vars_sym):
            eq_parts = [f"{grad_f_val[dim]}"] 
            for idx in active_indices:
                coeff = grad_g_vals[idx][dim]
                sign = "+" if coeff >= 0 else "-"
                eq_parts.append(f"{sign} {abs(coeff)}*λ_{idx}")
            eq_string = " ".join(eq_parts) + " = 0"
            print(f"         Eq({var}): {eq_string}")
            
        # [4] Solution
        print(f"\n    [4] SOLUTION (Multipliers):")
        l_vals = r['l']
        l_vec_full_str = "(" + ", ".join([str(l) for l in l_vals]) + ")"
        
        for idx in active_indices:
            val_l_sym = l_vals[idx]
            val_l_float = float(val_l_sym)
            print(f"       λ_{idx} (active) = {val_l_sym}  (approx {val_l_float:.4f})")
        print(f"\n       λ = {l_vec_full_str}")

        # [5] Classification Logic
        l_floats = [float(l) for l in l_vals]
        H = r['hessian']
        hessian_eig = []
        try:
            hessian_eig = [float(e) for e in H.eigenvals().keys()]
            print(f"       H Eigenvalues: {[f'{e:.4f}' for e in hessian_eig]}")
        except:
            print("       Could not compute eigenvalues.")

        is_mL = all(l >= -1e-7 for l in l_floats) and all(e > -1e-7 for e in hessian_eig)
        
        is_ML = all(l <= 1e-7 for l in l_floats) and all(e < 1e-7 for e in hessian_eig)
        
        is_S = not (is_mL or is_ML)

        if any(e > 1e-7 for e in hessian_eig) and any(e < -1e-7 for e in hessian_eig):
            is_mL = False
            is_ML = False
            is_S = True

        print(f"\n    [5] CONCLUSION:")
        if is_mL: print("       Candidate for LOCAL MINIMUM.")
        elif is_ML: print("       Candidate for LOCAL MAXIMUM.")
        else: print("       SADDLE POINT (or mixed).")

        table_rows.append({
            'x_fmt': x_lbl.replace('(', '').replace(')', '').replace(' ', ''),
            'l_fmt': l_vec_full_str.replace('(', '').replace(')', '').replace(' ', ''),
            'f_val': float(r['f_val']),
            'is_mL': is_mL,
            'is_ML': is_ML,
            'is_S': is_S
        })

    print("\n" + "="*80)

    g_min_val = min([x['f_val'] for x in table_rows if x['is_mL']], default=None)
    g_max_val = max([x['f_val'] for x in table_rows if x['is_ML']], default=None) 
    
    # header
    print(f"{'X':<15} {'λ':<20} {'mL':<6} {'mG':<6} {'ML':<6} {'MG':<6} {'S':<6}")
    print("-" * 75)

    for row in table_rows:
        is_loc_min = row['is_mL']
        is_glob_min = is_loc_min and (row['f_val'] == g_min_val)
        
        is_loc_max = row['is_ML']
        is_glob_max = is_loc_max and (row['f_val'] == g_max_val)
        
        is_saddle = row['is_S']

        s_mL = "YES" if is_loc_min else "NO"
        s_mG = "YES" if is_glob_min else "NO"
        s_ML = "YES" if is_loc_max else "NO"
        s_MG = "YES" if is_glob_max else "NO"
        s_S  = "YES" if is_saddle else "NO"

        
        print(f"{row['x_fmt']:<15} {row['l_fmt']:<20} {s_mL:<6} {s_mG:<6} {s_ML:<6} {s_MG:<6} {s_S:<6}")

    print("="*80)

if __name__ == "__main__":
    solve()