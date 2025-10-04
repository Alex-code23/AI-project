# Demo: Empirical Rademacher complexity for three simple function classes
# - Constants in [-1,1]
# - Thresholds on the real line (binary outputs ±1)
# - Linear functions with ||w||_2 <= 1 (real-valued outputs)
#
# This code computes the empirical Rademacher complexity estimate
# \hat{R}_S(F) = E_sigma[ sup_{f in F} 1/n sum_i sigma_i f(x_i) ]
# by Monte Carlo over Rademacher vectors sigma (samples of ±1).
#
# It also displays histograms of the supremum values across sigma samples
# to illustrate how the expectation is formed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# Reproducibility
rng = np.random.default_rng(42)

def rademacher_samples(n, m):
    """Return an (m, n) array of Rademacher variables in {-1, +1}."""
    return rng.choice([-1, 1], size=(m, n))

def sup_constant(sigma):
    """Supremum over constant functions c in [-1,1]: (1/n) * |sum sigma|"""
    return np.abs(sigma.sum()) / sigma.size

def sup_thresholds_1d(sigma, x_sorted):
    """
    Supremum over 1D threshold functions that map to {-1, +1}.
    For sample x_sorted (sorted x), thresholds produce label vectors
    that are -1 on the left of the threshold and +1 on the right
    (or the opposite). We enumerate the n+1 threshold positions.
    """
    n = sigma.size
    best = -np.inf
    for k in range(n + 1):
        outputs = np.concatenate(( -np.ones(k), np.ones(n - k) ))
        val = (sigma * outputs).sum() / n
        val_flipped = (sigma * -outputs).sum() / n
        if val > best:
            best = val
        if val_flipped > best:
            best = val_flipped
    return best

def sup_linear_realvalued(sigma, X):
    """
    Supremum over linear functions f_w(x)=<w,x> with ||w||_2 <= 1.
    sup_w (1/n) sum_i sigma_i <w, x_i> = (1/n) * || sum_i sigma_i x_i ||_2
    """
    v = (sigma[:, None] * X).sum(axis=0)  # sum_i sigma_i x_i
    return np.linalg.norm(v) / sigma.size

# Main experiment parameters
n = 20          # number of sample points
d = 2           # dimension for linear class
m = 2000        # number of Monte Carlo draws for sigma

# Create sample S
x_1d = rng.random(n)           # points in [0,1]
order = np.argsort(x_1d)
x_1d_sorted = x_1d[order]

X_d = rng.normal(size=(n, d))  # d-dimensional inputs (zero mean)

# Draw many Rademacher samples
sigmas = rademacher_samples(n, m)  # shape (m, n)

# Compute supremums for each sigma
sup_const_vals = np.array([sup_constant(s) for s in sigmas])
sup_thresh_vals = np.array([sup_thresholds_1d(s[order], x_1d_sorted) for s in sigmas])
sup_lin_vals = np.array([sup_linear_realvalued(s, X_d) for s in sigmas])

# Monte Carlo estimates
R_const = sup_const_vals.mean()
R_thresh = sup_thresh_vals.mean()
R_lin = sup_lin_vals.mean()

# Report results in a DataFrame
results = pd.DataFrame({
    "class": ["constant [-1,1]", "1D thresholds (±1)", "linear ||w||_2 ≤ 1 (real-valued)"],
    "empirical_Rademacher_estimate": [R_const, R_thresh, R_lin],
    "n": [n, n, n],
    "d": [np.nan, np.nan, d]
})
BINS = 50

# plot un seul graphe
plt.figure(figsize=(8,5))
plt.hist(sup_const_vals, bins=BINS, alpha=0.5, label="constants [-1,1]")
plt.hist(sup_thresh_vals, bins=BINS, alpha=0.5, label="1D thresholds (±1)")
plt.hist(sup_lin_vals, bins=BINS, alpha=0.5, label="linear (||w||≤1)")          
plt.title("Supremum values over sigma — three function classes")
plt.xlabel("sup (1/n sum sigma_i f(x_i))")
plt.ylabel("count")
plt.legend()
plt.tight_layout()
plt.show()

