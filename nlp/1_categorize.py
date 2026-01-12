import sympy as sp
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
    
    print(f"\n{'='*80}")
    print(f"[0] Convexity/Concavity Check")
    print(f"{'='*80}")
    
    H_global = sp.hessian(f_sym, vars_sym)
    
    print("   1. H(f):")
    for row in range(H_global.rows):
        row_str = "[ " + ", ".join([str(H_global[row, col]).replace('**', '^') for col in range(H_global.cols)]) + " ]"
        print(f"      {row_str}")

    print("\n   2. Eigenvalues of H:")
    try:
        ev_dict = H_global.eigenvals()
        eig_signs = []
        is_constant_H = True
        
        for val, mult in ev_dict.items():
            val_str = str(val).replace('**', '^')
            print(f"      λ = {val_str} (multiplicity {mult})")
            
            sign = 0
            
            if val.is_number:
                val_f = float(val.evalf())
                if val_f > 1e-9: sign = 1
                elif val_f < -1e-9: sign = -1
                else: sign = 0
            else:
                is_constant_H = False
                if val.is_positive: sign = 1
                elif val.is_negative: sign = -1
                else: sign = None 
            
            if sign is not None:
                eig_signs.extend([sign] * mult)
            else:
                eig_signs.append(None)

        print("\n   3. Conclusion on f(x):")
        if not is_constant_H:
            print("      Hessian depends on variables (non-quadratic function).")
            print("      Convexity/Concavity might change depending on the region.")
        elif any(s is None for s in eig_signs):
             print("      Could not determine sign of eigenvalues symbolically.")
        else:
            if all(s >= 0 for s in eig_signs):
                if all(s > 0 for s in eig_signs):
                    print("      Strictly Convex (Positive Definite) -> Unique Global Minimum possible.")
                else:
                    print("      Convex (Positive Semidefinite) -> Minimum possible (convex region).")
            elif all(s <= 0 for s in eig_signs):
                if all(s < 0 for s in eig_signs):
                    print("      Strictly Concave (Negative Definite) -> Unique Global Maximum possible.")
                else:
                    print("      Concave (Negative Semidefinite) -> Maximum possible (concave region).")
            else:
                print("      Indefinite (Mixed signs) -> Saddle point geometry (Neither convex nor concave).")
                
    except Exception as e:
        print(f"      Could not calculate global properties: {e}")

    lams_sym = list(sp.symbols(f'λ_0:{len(g_syms)}'))
    L = f_sym + sum(l * g for l, g in zip(lams_sym, g_syms))
    
    grad_L_eqs = [sp.diff(L, v) for v in vars_sym]
    slack_eqs = [l * g for l, g in zip(lams_sym, g_syms)]
    
    system = grad_L_eqs + slack_eqs
    symbols_all = vars_sym + lams_sym
    
    print(f"\n{'-'*80}")
    print(f"Lagrangian Stationarity Conditions:")
    for v, eq in zip(vars_sym, grad_L_eqs):
        eq_str = str(eq).replace('**', '^')
        print(f"   ∂L/∂{v} : {eq_str} = 0")
    
    print("\nComplementary Slackness (λ * g = 0):")
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
            if g.subs(vals_map).evalf() > 1e-6:
                g_ok = False; break
        if not g_ok: continue

        l_vals = [vals_map[l] for l in lams_sym]
        
        res = {
            'x': [vals_map[v] for v in vars_sym],
            'l': l_vals,
            'f_val': f_sym.subs(vals_map),
            'vals_map': vals_map,
            'grad_eqs': grad_L_eqs,
            'hessian': H_matrix.subs(vals_map),
            'hessian_sym': H_matrix
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
        x_lbl = f"({', '.join([str(x).replace('**', '^') for x in r['x']])})"
        
        print(f"\n>>> POINT {i+1}: {x_lbl}")
        print(f"    f(x) = {r['f_val']}")
        
        print(f"\n    [1] ACTIVE CONSTRAINTS IDENTIFICATION:")
        active_indices = []
        for idx, g in enumerate(g_syms):
            val_g_sym = g.subs(r['vals_map'])
            status = "INACTIVE"
            if abs(val_g_sym) < 1e-6:
                status = "ACTIVE"
                active_indices.append(idx)
            g_str = str(g).replace('**', '^')
            print(f"       g_{idx}: {g_str:<20} = {val_g_sym} -> {status}")
            
        print(f"\n    [2] ∇f(x*):")
        grad_f_sym = [sp.diff(f_sym, v) for v in vars_sym]
        grad_f_sym_str = "(" + ", ".join([str(e).replace('**', '^') for e in grad_f_sym]) + ")"
        grad_f_val = [v.subs(r['vals_map']) for v in grad_f_sym]
        grad_f_str = f"({', '.join([str(v).replace('**', '^') for v in grad_f_val])})"
        print(f"       ∇f(x*) = {grad_f_sym_str} = {grad_f_str}")
        
        grad_g_vals = {}
        for idx in active_indices:
            g_expr = g_syms[idx]
            grad_g_sym = [sp.diff(g_expr, v) for v in vars_sym]
            grad_g_sym_str = "(" + ", ".join([str(e).replace('**', '^') for e in grad_g_sym]) + ")"
            grad_g_num = [v.subs(r['vals_map']) for v in grad_g_sym]
            grad_g_vals[idx] = grad_g_num
            g_vec_str = f"({', '.join([str(v).replace('**', '^') for v in grad_g_num])})"
            print(f"       ∇g_{idx}(x*) = {grad_g_sym_str} = {g_vec_str}")
            
        print(f"\n    [3] KKT SYSTEM (∇f + Σ λ_i ∇g_i = 0):")
        for dim, var in enumerate(vars_sym):
            eq_parts = [f"{grad_f_val[dim]}"] 
            for idx in active_indices:
                coeff = grad_g_vals[idx][dim]
                coeff_sym = coeff
                sign = "+"
                try:
                    if float(coeff) < 0:
                         sign = "-"
                         coeff_sym = -coeff
                except: pass
                
                eq_parts.append(f"{sign} ({coeff_sym})*λ_{idx}")
            eq_string = " ".join(eq_parts) + " = 0"
            print(f"         Eq({var}): {eq_string}")
            
        print(f"\n    [4] SOLUTION (Multipliers):")
        l_vals = r['l']
        l_vec_full_str = "(" + ", ".join([str(l).replace('**', '^') for l in l_vals]) + ")"
        
        for idx in active_indices:
            val_l_sym = l_vals[idx]
            print(f"       λ_{idx} (active) = {val_l_sym}")
        print(f"\n       λ = {l_vec_full_str}")

        print(f"\n    [5] H CALCULATION:")
        print(f"       a) Computing 2nd derivatives:")
        
        H_sym = r['hessian_sym']
        H_num = r['hessian']
        rows, cols = H_sym.shape
        
        for r_idx in range(rows):
            for c_idx in range(cols):
                var1 = vars_sym[r_idx]
                var2 = vars_sym[c_idx]
                d2f_sym = H_sym[r_idx, c_idx]
                d2f_val = H_num[r_idx, c_idx]
                
                d2f_sym_str = str(d2f_sym).replace('**', '^')
                d2f_val_str = str(d2f_val).replace('**', '^')
                
                if d2f_sym_str != d2f_val_str:
                     print(f"          ∂²f / (∂{var1} ∂{var2}) = {d2f_sym_str} = {d2f_val_str}")
                else:
                     print(f"          ∂²f / (∂{var1} ∂{var2}) = {d2f_sym_str}")
        
        print(f"\n       b) H:")
        for row in range(H_sym.rows):
            row_str = "[ " + ", ".join([str(H_sym[row, col]).replace('**', '^') for col in range(H_sym.cols)]) + " ]"
            print(f"          {row_str}")

        print(f"\n       c) Evaluated Hessian Matrix at x*:")
        for row in range(H_num.rows):
            row_str = "[ " + ", ".join([str(H_num[row, col]).replace('**', '^') for col in range(H_num.cols)]) + " ]"
            print(f"          {row_str}")

        hessian_eig = []
        hessian_eig_sym = []
        print(f"\n       d) Eigenvalues (roots of det(H - λI) = 0):")
        try:
            ev_dict = H_num.eigenvals()
            for val, mult in ev_dict.items():
                hessian_eig_sym.extend([val] * mult)
                hessian_eig.extend([float(val)] * mult)
                val_str = str(val).replace('**', '^')
                print(f"          λ_eig = {val_str} (multiplicity {mult})")
        except Exception as e:
            print(f"          Could not compute eigenvalues: {e}")

        l_floats = [float(l) for l in l_vals]
        
        is_mL = all(l >= -1e-7 for l in l_floats) and all(e > -1e-7 for e in hessian_eig)
        is_ML = all(l <= 1e-7 for l in l_floats) and all(e < 1e-7 for e in hessian_eig)
        is_S = not (is_mL or is_ML)

        if any(e > 1e-7 for e in hessian_eig) and any(e < -1e-7 for e in hessian_eig):
            is_mL = False
            is_ML = False
            is_S = True

        print(f"\n       e) Conclusion Logic:")
        if is_mL:
            print("          Eigenvalues >= 0 (Positive Semidefinite) AND λ suitable -> Local MINIMUM")
        elif is_ML:
            print("          Eigenvalues <= 0 (Negative Semidefinite) AND λ suitable -> Local MAXIMUM")
        else:
            print("          Mixed eigenvalues (Indefinite) -> SADDLE POINT")

        table_rows.append({
            'x_fmt': x_lbl.replace('(', '').replace(')', '').replace(' ', ''),
            'l_fmt': l_vec_full_str.replace('(', '').replace(')', '').replace(' ', ''),
            'f_val': r['f_val'],
            'is_mL': is_mL,
            'is_ML': is_ML,
            'is_S': is_S
        })

    print("\n" + "="*80)

    g_min_val = None
    min_cands = [x['f_val'] for x in table_rows if x['is_mL']]
    if min_cands:
        try:
             g_min_val = min(min_cands, key=lambda x: float(x))
        except: pass

    g_max_val = None
    max_cands = [x['f_val'] for x in table_rows if x['is_ML']]
    if max_cands:
         try:
            g_max_val = max(max_cands, key=lambda x: float(x))
         except: pass
    
    print(f"{'X':<20} {'λ':<20} {'mL':<6} {'mG':<6} {'ML':<6} {'MG':<6} {'S':<6}")
    print("-" * 80)

    for row in table_rows:
        is_loc_min = row['is_mL']
        try:
             is_glob_min = is_loc_min and (float(row['f_val']) == float(g_min_val)) if g_min_val is not None else False
        except: is_glob_min = False
        
        is_loc_max = row['is_ML']
        try:
            is_glob_max = is_loc_max and (float(row['f_val']) == float(g_max_val)) if g_max_val is not None else False
        except: is_glob_max = False
        
        is_saddle = row['is_S']

        s_mL = "YES" if is_loc_min else "NO"
        s_mG = "YES" if is_glob_min else "NO"
        s_ML = "YES" if is_loc_max else "NO"
        s_MG = "YES" if is_glob_max else "NO"
        s_S  = "YES" if is_saddle else "NO"

        print(f"{row['x_fmt']:<20} {row['l_fmt']:<20} {s_mL:<6} {s_mG:<6} {s_ML:<6} {s_MG:<6} {s_S:<6}")

    print("="*80)

if __name__ == "__main__":
    solve()