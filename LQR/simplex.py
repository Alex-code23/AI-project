# simplex_stepwise_fixed.py
import numpy as np
import pandas as pd
import datetime
import os

np.set_printoptions(precision=6, suppress=True)

def pretty_table(T, var_names, row_names):
    df = pd.DataFrame(T, index=row_names, columns=var_names + ["RHS"])
    return df.to_string(float_format=lambda v: f"{v:8.4f}")

def log_print(s, logfile=None):
    """Print and optionally append to logfile opened in UTF-8 (safe on Windows)."""
    print(s)
    if logfile:
        # Ensure directory exists
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        # Write with utf-8 and replace any unencodable chars to avoid crashes.
        with open(logfile, "a", encoding="utf-8", errors="replace") as f:
            f.write(s + "\n")

def simplex_max_stepwise(A, b, c, max_iters=50, tol=1e-12, logfile=None):
    """
    Simplexe primal (max c^T x s.t. A x <= b, x >= 0) avec affichage pas-à-pas des opérations.
    Retourne (history, final_tableau, basis).
    logfile: chemin optionnel pour sauvegarder la trace complète (UTF-8).
    """
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, "w", encoding="utf-8", errors="replace") as f:
            f.write(f"Simplex stepwise log started at {datetime.datetime.now().isoformat()}\n\n")

    def lprint(s):
        # Replace the unicode approx sign by ASCII tilde for portability in messages
        s_safe = s.replace("≈", "~")
        log_print(s_safe, logfile)

    m, n = A.shape
    A_ext = np.hstack([A, np.eye(m)])            # add slacks
    c_ext = np.concatenate([c, np.zeros(m)])
    N = n + m

    # build tableau
    T = np.zeros((m + 1, N + 1))
    T[:m, :N] = A_ext
    T[:m, -1] = b
    T[-1, :N] = -c_ext   # store -c in objective row for maximization

    basic = list(range(n, n + m))  # slack vars initial base
    var_names = [f"x{i+1}" for i in range(n)] + [f"s{j+1}" for j in range(m)]
    row_names = [f"constr_{i+1}" for i in range(m)] + ["objective"]

    history = []
    def extract_solution(T, basic):
        x = np.zeros(n)
        for i, bi in enumerate(basic):
            if bi < n:
                x[bi] = T[i, -1]
        obj = float(T[-1, -1])
        return x, obj

    # initial display
    lprint("=== SIMPLEX STEPWISE TRACE ===\n")
    lprint("Problem: maximize c^T x subject to A x <= b, x >= 0\n")
    lprint("Initial tableau (slack variables in base):")
    lprint(pretty_table(T, var_names, row_names))
    lprint(f"Initial base (indices) : {basic} -> {[var_names[i] for i in basic]}\n")
    x0, obj0 = extract_solution(T, basic)
    history.append((basic.copy(), x0.copy(), obj0))

    it = 0
    while it < max_iters:
        it += 1
        lprint("="*80)
        lprint(f"ITERATION {it} -- Current base: {[var_names[i] for i in basic]}")
        lprint("Current tableau:")
        lprint(pretty_table(T, var_names, row_names))
        x_curr, obj_curr = extract_solution(T, basic)
        lprint(f"Current primal x: {x_curr}, objective = {obj_curr:.6f}\n")

        # reduced costs
        red_costs = T[-1, :N]
        lprint("Reduced costs (objective row):")
        lprint(", ".join([f"{var_names[j]}:{red_costs[j]:+.6f}" for j in range(N)]))

        # entering candidates (reduced cost < 0 for max)
        entering_candidates = [j for j in range(N) if red_costs[j] < -tol]
        if not entering_candidates:
            lprint("\nNo negative reduced cost -> optimal solution reached.")
            break

        lprint(f"\nEntering candidates: {[var_names[j] for j in entering_candidates]}")
        entering = min(entering_candidates, key=lambda j: red_costs[j])  # most negative
        lprint(f"Chosen entering var: {var_names[entering]} (col {entering}, reduced cost {red_costs[entering]:+.6f})")

        # ratio test
        col = T[:m, entering].copy()
        rhs = T[:m, -1].copy()
        pos_rows = [i for i in range(m) if col[i] > tol]
        lprint("Entering column (coeffs by row): " + ", ".join([f"r{i}:{col[i]:.6f}" for i in range(m)]))
        if not pos_rows:
            lprint(" -> All entries in entering column <= 0 -> LP unbounded along this direction.")
            raise RuntimeError("Unbounded LP along entering var.")

        ratios = [(rhs[i] / col[i], i) for i in pos_rows]
        lprint("Ratio candidates (rhs/coeff): " + ", ".join([f"r{idx}:{val:.6f}" for val, idx in ratios]))
        ratios_sorted = sorted(ratios, key=lambda x: (x[0], x[1]))
        ratio, leaving_row = ratios_sorted[0]
        # detect tie / degeneracy
        equal_min = [r for r, idx in ratios_sorted if abs(r - ratio) <= 1e-12]
        if len([x for x in ratios_sorted if abs(x[0]-ratio)<=1e-12]) > 1:
            lprint(f" Degeneracy detected (multiple rows share min ratio = {ratio:.6f}). Tie-breaking by smallest index.")

        leaving_var = basic[leaving_row]
        pivot = T[leaving_row, entering]
        lprint(f"Chosen leaving row: row {leaving_row} (basic var {var_names[leaving_var]}), pivot = {pivot:.6f}, ratio = {ratio:.6f}\n")

        # Show leaving row BEFORE normalization
        old_row = T[leaving_row, :].copy()
        lprint("Leaving row (before normalization):")
        lprint(" | ".join([f"{val:.6f}" for val in old_row]) + f"   <-- {row_names[leaving_row]}")

        # Normalize leaving row
        if abs(pivot) < tol:
            lprint("Numerical pivot ~ 0 -> stopping.")
            raise RuntimeError("Numerical pivot ~ 0")
        norm_row = old_row / pivot
        lprint("\nLeaving row after normalization (divide by pivot):")
        lprint(" | ".join([f"{val:.6f}" for val in norm_row]))

        # Eliminate other rows (including objective)
        lprint("\nElimination operations (for each other row):")
        for r in range(m + 1):
            if r == leaving_row:
                continue
            factor = T[r, entering]
            if abs(factor) < 1e-15:
                lprint(f" Row {r}: factor = {factor:.6f} (~0) -> no operation, unchanged.")
                continue
            old_r = T[r, :].copy()
            new_r = old_r - factor * norm_row
            lprint(f" Row {r} (before): " + " | ".join([f"{v:.6f}" for v in old_r]))
            lprint(f"  -> Row {r} <- Row {r} - ({factor:.6f}) * normalized_row")
            lprint(f" Row {r} (after):  " + " | ".join([f"{v:.6f}" for v in new_r]))
            T[r, :] = new_r

        # set normalized leaving row and update base
        T[leaving_row, :] = norm_row
        basic[leaving_row] = entering

        # post-pivot log
        lprint("\nPost-pivot tableau:")
        lprint(pretty_table(T, var_names, row_names))
        x_new, obj_new = extract_solution(T, basic)
        lprint(f"Updated base: {[var_names[i] for i in basic]}")
        lprint(f"New solution (x): {x_new}, objective = {obj_new:.6f}\n")

        history.append((basic.copy(), x_new.copy(), obj_new))

    # final summary
    final_x, final_obj = extract_solution(T, basic)
    lprint("="*80)
    lprint("FINAL SUMMARY:")
    lprint(f"Final base: {[var_names[i] for i in basic]}")
    lprint(f"Final solution x: {final_x}")
    lprint(f"Final objective: {final_obj:.6f}")
    if logfile:
        lprint(f"Log saved to: {logfile}")

    # save history CSV
    hist_rows = []
    for idx, (bas, xvec, objv) in enumerate(history):
        hist_rows.append({
            "iter": idx,
            "basic": ",".join([var_names[i] for i in bas]),
            "obj": float(objv),
            "x1": float(xvec[0]) if len(xvec) > 0 else 0.0,
            "x2": float(xvec[1]) if len(xvec) > 1 else 0.0,
            "x3": float(xvec[2]) if len(xvec) > 2 else 0.0
        })
    csv_path = "LQR/simplex_stepwise_history.csv"
    pd.DataFrame(hist_rows).to_csv(csv_path, index=False)
    lprint(f"History CSV saved to: {csv_path}")
    return history, T, basic

# Example usage
if __name__ == "__main__":
    A = np.array([[1.0, 1.0, 1.0],      # x1 + x2 + x3 <= 4
                  [1.0, 0.0, 0.5],      # x1       + 0.5 x3 <= 2
                  [0.0, 1.0, 0.5]])     #      x2 + 0.5 x3 <= 3
    b = np.array([4.0, 2.0, 3.0])
    c = np.array([3.0, 2.0, 3.0])       # maximize 3 x1 + 2 x2 + 3 x3
    logfile = "LQR/simplex_stepwise_log.txt"
    history, final_T, final_basic = simplex_max_stepwise(A, b, c, max_iters=50, logfile=logfile)
    print("\nDone. See log:", logfile)
