import sympy as sp
import numpy as np
import sys
from fractions import Fraction


def fmt_frac(val, limit_denom=1000):
    if abs(val) < 1e-9: return "0"
    try:
        f = Fraction(val).limit_denominator(limit_denom)
        if abs(float(f) - val) < 1e-5:
            if f.denominator == 1:
                return str(f.numerator)
            return f"{f.numerator}/{f.denominator}"
    except:
        pass
    return f"{val:.4g}"

def fmt_vec_frac(v):
    vals = [fmt_frac(x) for x in v]
    return f"({', '.join(vals)})"

def fmt_mat_frac(M):
    if M.size == 0: return "[] (Empty)"
    if M.ndim == 1: M = M.reshape(1, -1)
    rows = []
    for row in M:
        r_str = ",".join([fmt_frac(x) for x in row])
        rows.append(f"[{r_str}]")
    return "\n   ".join(rows)

def fmt_mat_inline(M):
    if M.size == 0: return "[]"
    rows = []
    for row in M:
        r_str = ", ".join([fmt_frac(x) for x in row])
        rows.append(f"[{r_str}]")
    return "[" + ", ".join(rows) + "]"


def analyze_convexity(f_sym, vars_sym):
    print("\n[CONVEXITY ANALYSIS]")
    hessian = [[sp.diff(f_sym, v1, v2) for v2 in vars_sym] for v1 in vars_sym]
    hessian_mat = sp.Matrix(hessian)
    
    print(f"1. Hessian Matrix Calculation ∇²f:")
    for i in range(len(vars_sym)):
        row_str = ", ".join([str(hessian_mat[i, j]) for j in range(len(vars_sym))])
        print(f"   | {row_str} |")

    try:
        eigenvals = hessian_mat.eigenvals()
        evals_list = list(eigenvals.keys())
        evals_float = [float(e) for e in evals_list]
        
        evals_str = ", ".join([fmt_frac(float(e)) for e in evals_list])
        print(f"2. Hessian Eigenvalues: {evals_str}")
        
        is_convex = all(e >= -1e-9 for e in evals_float)
        is_concave = all(e <= 1e-9 for e in evals_float)
        
        if is_convex and is_concave:
            msg = "convex and concave (linear)"
            nature = "convex_concave"
        elif is_convex:
            msg = "CONVEX (Eigenvalues >= 0)"
            nature = "convex"
        elif is_concave:
            msg = "CONCAVE (Eigenvalues <= 0)"
            nature = "concave"
        else:
            msg = "INDEFINITE (Mixed Eigenvalues)"
            nature = "indefinite"
            
        print(f"3. Conclusion: The function f is {msg}.")
        return nature
    except:
        print("   Unable to calculate numeric eigenvalues. Assuming indefinite.")
        return "unknown"


def parse_pnl_linear(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    data = {}
    # Default mode
    is_max = False 
    
    for line in lines:
        clean_line = line.strip().lower()
        if clean_line == 'max' or clean_line.startswith('max '):
            is_max = True
        elif clean_line == 'min' or clean_line.startswith('min '):
            is_max = False
            
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
            for v in vars_sym:
                row.append(float(expr.coeff(v)))
            A.append(row)
            b.append(-const)

    x0_str = data.get('x0', '').split(',')
    x0_vals = []
    for x in x0_str:
        if x.strip():
            val = float(sp.sympify(x.strip()))
            x0_vals.append(val)
    x0 = np.array(x0_vals)
    
    iters = int(float(data.get('step_size', '5')))
    return f_expr, np.array(A), np.array(b), x0, vars_sym, iters, is_max


def analytic_line_search_bounded(f_sym, vars_sym, xk, direction, max_step, is_max):
    t = sp.symbols('t')
    x_param = [xk[i] + t * direction[i] for i in range(len(xk))]
    
    phi_t = f_sym.subs(dict(zip(vars_sym, x_param)))
    d_phi = sp.diff(phi_t, t)
    
    sol = sp.solve(d_phi, t)
    
    candidates = [0.0, max_step]
    if sol:
        for s in sol:
            try:
                val_t = float(sp.re(s))
                if -1e-9 <= val_t <= max_step + 1e-9:
                    candidates.append(val_t)
            except: pass
            
    best_t = 0.0
    best_val = -float('inf') if is_max else float('inf')
    
    for cand in candidates:
        cand_clamped = max(0.0, min(cand, max_step))
        val = float(phi_t.subs(t, cand_clamped))
        
        if is_max:
            if val > best_val + 1e-9:
                best_val = val
                best_t = cand_clamped
        else:
            if val < best_val - 1e-9:
                best_val = val
                best_t = cand_clamped

    return best_t


def projected_gradient(filename='pnl.txt'):
    f_sym, A, b, xk, vars_sym, file_iter, is_max = parse_pnl_linear(filename)
    n_vars = len(vars_sym)
    
    prob_type = "MAXIMIZATION" if is_max else "MINIMIZATION"

    try:
        user_input = input(f"How many steps? (default {file_iter}): ")
        if user_input.strip():
            max_iter = int(user_input)
        else:
            max_iter = file_iter
    except ValueError:
        print(f"Not valid. Using default: {file_iter}")
        max_iter = file_iter

    print(f"\nFunction ({prob_type}): {f_sym}")
    print(f"Start x^0: {fmt_vec_frac(xk)}")
    
    convexity_status = analyze_convexity(f_sym, vars_sym)
    print("=" * 60)
    
    grad_sym = [sp.diff(f_sym, v) for v in vars_sym]
    
    k = 0
    
    while k < max_iter:
        print(f"\n>>> ITERATION {k+1}")
        
        print(f"[1] ∇f(x^{k}):")
        grad_list = []
        sorted_vars_for_sub = sorted(zip(vars_sym, xk), key=lambda x: len(str(x[0])), reverse=True)
        
        for i, expr in enumerate(grad_sym):
            val = float(expr.subs(dict(zip(vars_sym, xk))))
            grad_list.append(val)
            subst_str = str(expr)
            for v_sym, v_val in sorted_vars_for_sub:
                subst_str = subst_str.replace(str(v_sym), f"({fmt_frac(v_val)})")
            print(f"    ∂f/∂{vars_sym[i]}: {expr} = {subst_str} = {fmt_frac(val)}")
            
        grad_val = np.array(grad_list)
        print(f"    ∇f: {fmt_vec_frac(grad_val)}")
        
        J = []
        print(f"[2] Identification of Active Constraints:")
        for i in range(len(b)):
            lhs_val = np.dot(A[i], xk)
            rhs_val = b[i]
            lhs_terms = []
            for j, val in enumerate(xk):
                coeff = A[i][j]
                lhs_terms.append(f"({fmt_frac(coeff)})*({fmt_frac(val)})")
            lhs_expr = " + ".join(lhs_terms)
            print(f"    g_{i}: {lhs_expr} = {fmt_frac(lhs_val)}")
            print(f"         Compare: {fmt_frac(lhs_val)} <= {fmt_frac(rhs_val)}")

            if abs(lhs_val - rhs_val) < 1e-6:
                J.append(i)
                print(f"         -> ACTIVE")
            else:
                print(f"         -> NOT ACTIVE (Residual {fmt_frac(lhs_val - rhs_val)})")
        if not J: print("    No active constraints.")

        while True:
            print(f"    Active indices J: {J}")
            if not J: M = np.array([])
            else: M = A[J]
            print(f"    Matrix M:")
            print(f"   {fmt_mat_frac(M)}")

            # 3. H
            print(f"[3] Constructing Projection Matrix H:")
            if M.size == 0:
                print(f"    J empty -> H = I (Identity)")
                H = np.eye(n_vars)
            else:
                MT = M.T
                MMT = np.dot(M, MT)
                try: MMT_inv = np.linalg.pinv(MMT)
                except: print("Error inv(MMT)"); return

                print(f"    Intermediate Terms:")
                print(f"      M^T = \n   {fmt_mat_frac(MT)}")
                print(f"      M M^T = \n   {fmt_mat_frac(MMT)}")
                print(f"      (M M^T)^-1 = \n   {fmt_mat_frac(MMT_inv)}")

                proj_part = np.dot(MT, np.dot(MMT_inv, M))
                print(f"      M^T(MM^T)^-1 M (Projection Part) = \n   {fmt_mat_frac(proj_part)}")

                H = np.eye(n_vars) - proj_part
                print(f"    H = I - (M^T(MM^T)^-1 M):")

            print(f"   {fmt_mat_frac(H)}")

            # 4. direction
            # if max problem, d = H * grad; else d = -H * grad
            dk = np.dot(H, grad_val) if is_max else -np.dot(H, grad_val)
            dk = np.array([0.0 if abs(x) < 1e-9 else x for x in dk])

            print(f"[4] Direction d^{k} ({'Ascent' if is_max else 'Descent'}):")
            h_inline = fmt_mat_inline(H)
            g_inline = fmt_vec_frac(grad_val)
            res_inline = fmt_vec_frac(dk)

            sign_str = "+" if is_max else "-"
            print(f"    {sign_str} H * ∇f = {sign_str} {h_inline} * {g_inline} = {res_inline}")

            # --- d = 0 -> find u ---
            if np.linalg.norm(dk) < 1e-6:
                print(f"\n[3] d^{k} == 0. Go to step 4.")
                print(f"[4]  u:")

                if M.size == 0:
                    print("   ∇f = 0 and no active constraints.")
                    print(f"    STOP. x^k is a stationary point.")
                    return

                mmt = np.dot(M, M.T)
                mmt_inv = np.linalg.pinv(mmt)
                neg_mmt_inv = -mmt_inv
                m_grad = np.dot(M, grad_val)
                u = np.dot(neg_mmt_inv, m_grad)

                # max problem: multipliers must be neg (u <= 0)
                # if u < 0, constraint goes against the optimization

                u = np.array([0.0 if abs(val) < 1e-9 else val for val in u])
                u_str_parts = [f"u_{J[idx]}={fmt_frac(val)}" for idx, val in enumerate(u)]

                print(f"    u = -(MM^T)^-1 * M ∇f(x^k)")
                print(f"    u: [{', '.join(u_str_parts)}]")

                if np.all(u >= -1e-9):
                    print("    All u >= 0. KKT conditions satisfied.")
                    print(f"    STOP. x^{k} is a stationary point.")
                    return
                else:
                    min_u_idx = np.argmin(u)
                    constraint_idx_to_remove = J[min_u_idx]
                    val_remove = u[min_u_idx]
                    print(f"    Found u_{constraint_idx_to_remove} = {fmt_frac(val_remove)} < 0.")
                    print(f"    -> REMOVING constraint g_{constraint_idx_to_remove}.")
                    J.pop(min_u_idx)
                    print("    -> Recalculating H...")
                    print("-" * 30)
                    continue
            else:
                print(f"\n[5] d^{k} != 0 -> find t:")
                max_t = float('inf')
                ad = np.dot(A, dk)
                b_ax = b - np.dot(A, xk)
                has_limit = False

                for i in range(len(b)):
                    row_str_parts = []
                    for j, v in enumerate(vars_sym):
                        c = A[i][j]
                        if abs(c) > 1e-9:
                            val_str = fmt_frac(abs(c))
                            sign = "+" if c >= 0 else "-"
                            prefix = f" {sign} " if row_str_parts else ("-" if sign == "-" else "")
                            term = f"{v}" if val_str == "1" else f"{val_str}*{v}"
                            row_str_parts.append(f"{prefix}{term}")
                    lhs_str = "".join(row_str_parts) if row_str_parts else "0"
                    constr_desc = f"{lhs_str} <= {fmt_frac(b[i])}"
                    
                    den = ad[i]
                    num = b_ax[i]
                    
                    status_msg = ""
                    if abs(den) < 1e-9:
                         status_msg = "true for all t"
                    elif den > 0:
                         lim = num/den
                         status_msg = f"true if t <= {fmt_frac(lim)}"
                         if lim < max_t: max_t = lim; has_limit = True
                    else: # den < 0
                         lim = num/den
                         status_msg = f"true if t >= {fmt_frac(lim)}"
                    
                    print(f"    {constr_desc:<35} {status_msg}")
                
                max_t_str = "inf" if not has_limit else fmt_frac(max_t)
                print(f"\n    -> t_hat = {max_t_str}")

                calc_bound = max_t if has_limit else 100.0
                print(f"[6] Exact Line Search on [0, {max_t_str}]:")
                
                tk = analytic_line_search_bounded(f_sym, vars_sym, xk, dk, calc_bound, is_max)
                if abs(tk - round(tk)) < 1e-9: tk = float(round(tk))
                
                print(f"    -> Optimal step t* = {fmt_frac(tk)}")
                
                xk_next = xk + tk * dk
                print(f"[7] New point x^{k+1}:")
                print(f"     x^{k+1} = x^{k} + t* d^{k}")
                coords = []
                for i in range(n_vars):
                    term = f"{fmt_frac(xk[i])} + {fmt_frac(tk)}*({fmt_frac(dk[i])})"
                    res = fmt_frac(xk_next[i])
                    coords.append(f"{term} = {res}")
                print(f"    {',\t\t'.join(coords)}")
                print(f"    x^{k+1} = {fmt_vec_frac(xk_next)}")
                print("-" * 60)
                xk = xk_next
                k += 1
                break

if __name__ == "__main__":
    projected_gradient()