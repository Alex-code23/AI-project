# Code to simulate the marketing-client example using:
# 1) Gaussian copula-based generative model (classical)
# 2) Neural autoregressive copula (if PyTorch is available) - models conditional Beta distributions on U's
#
# The code:
# - Creates a synthetic "real" dataset (age, income, monthly_spend)
# - Fits empirical marginals (empirical CDF + quantile interpolation)
# - Fits a Gaussian copula and draws synthetic samples
# - (Optional) Trains small neural nets (PyTorch) to learn P(U2|U1) and P(U3|U1,U2) as Beta params,
#   then samples from the learned neural-copula
# - Shows comparison scatter plots and prints rejection rate (not used here) and sample summaries
#
# Note: this is a self-contained demo. If PyTorch is not installed, the neural-copula part will be skipped.
# The visualizations use matplotlib (no explicit color choices).

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression


np.random.seed(0)

# ---------- 0) Simulate a "real" dataset ----------
n = 2000
# Age: mixture to be realistic (some young, many middle-aged, some older)
age = np.clip(np.concatenate([
    np.random.normal(30, 6, int(0.25*n)),
    np.random.normal(45, 8, int(0.6*n)),
    np.random.normal(65, 6, int(0.15*n))
]), 18, 90)

# Income: log-normal-ish, depends on age (older -> higher mean) + noise
income = np.exp(np.random.normal(10 + 0.01*(age-40), 0.5, n))  # in arbitrary units

# Monthly spend: depends nonlinearly on income and age (e.g., younger spend more proportionally),
# plus heteroskedastic noise
monthly_spend = 0.2 * income * (1 - 0.001*(age-30)) + np.random.gamma(2, 50, n)
monthly_spend = np.maximum(monthly_spend, 10.0)

data = pd.DataFrame({
    "age": age,
    "income": income,
    "monthly_spend": monthly_spend
})


print("Dataset summary (real data):")
print(data.describe().T[["min","mean","50%","max"]])

# ---------- Helpers: empirical CDF and quantile interpolation ----------
def empirical_uniforms(x):
    """Return empirical CDF values in (0,1) using rank / (n+1)"""
    ranks = stats.rankdata(x, method="average")
    u = ranks / (len(x) + 1.0)
    return u

def empirical_quantile(u, x_sorted):
    """Invert empirical CDF by linear interpolation on the empirical quantile function"""
    # x_sorted must be sorted ascending
    n = len(x_sorted)
    # population quantile positions: i/(n+1) for i=1..n
    grid = np.arange(1, n+1) / (n + 1.0)
    return np.interp(u, grid, x_sorted)

# Fit marginals by empirical CDFs (we'll use the sorted arrays as quantile functions)
x_sorted_age = np.sort(data["age"].values)
x_sorted_income = np.sort(data["income"].values)
x_sorted_spend = np.sort(data["monthly_spend"].values)

# Transform to uniforms U via empirical ranks
U_age = empirical_uniforms(data["age"].values)
U_income = empirical_uniforms(data["income"].values)
U_spend = empirical_uniforms(data["monthly_spend"].values)

U = np.vstack([U_age, U_income, U_spend]).T

# ---------- 1) Gaussian copula model ----------
# Transform uniforms to standard normals
norm_ppf = stats.norm.ppf
Z = norm_ppf(U)  # shape (n,3) ; beware of exact 0/1, but our empirical method avoids it

# Estimate correlation matrix (empirical)
corr = np.cov(Z, rowvar=False)
# convert covariance to correlation
d = np.sqrt(np.diag(corr))
corr_matrix = corr / np.outer(d, d)

# Ensure positive definiteness
# small regularization if necessary
eps = 1e-6
eigvals = np.linalg.eigvalsh(corr_matrix)
if eigvals.min() < 1e-8:
    corr_matrix += np.eye(3) * (1e-6 - eigvals.min())

# Sampling from Gaussian copula
m = 2000  # number of synthetic samples to generate
z_synth = np.random.multivariate_normal(mean=[0,0,0], cov=corr_matrix, size=m)
u_synth = stats.norm.cdf(z_synth)  # back to uniforms

# Map uniforms back to original scales via empirical quantile functions
age_synth_gauss = empirical_quantile(u_synth[:,0], x_sorted_age)
income_synth_gauss = empirical_quantile(u_synth[:,1], x_sorted_income)
spend_synth_gauss = empirical_quantile(u_synth[:,2], x_sorted_spend)

synth_gauss = pd.DataFrame({
    "age": age_synth_gauss,
    "income": income_synth_gauss,
    "monthly_spend": spend_synth_gauss
})


print("\nGaussian copula: empirical correlation matrix of latent normals:")
print(corr_matrix)

# ---------- 2) Neural autoregressive copula (U1, then U2|U1, then U3|U1,U2) ----------
# We'll try to model conditionals U2|U1 and U3|U1,U2 with small neural networks that output Beta params.
# If torch is available, train; else skip.
use_nn = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    use_nn = True
except Exception as e:
    print("\nPyTorch not available in this environment — neural copula will be skipped.")
    use_nn = False

synth_nn = None
if use_nn:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare training data: U in (0,1)
    U_tr = U.astype(np.float32)
    U_train, U_val = train_test_split(U_tr, test_size=0.2, random_state=1)
    # Define small MLPs that output positive parameters (alpha,beta) for Beta via softplus
    class BetaParamNet(nn.Module):
        def __init__(self, input_dim, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2)  # outputs raw alpha,beta
            )
            self.softplus = nn.Softplus()
        def forward(self, x):
            out = self.net(x)
            # ensure positivity and not too close to zero
            out = self.softplus(out) + 1e-3
            return out  # alpha, beta

    # Create models
    net_u2 = BetaParamNet(input_dim=1).to(device)   # models U2|U1
    net_u3 = BetaParamNet(input_dim=2).to(device)   # models U3|U1,U2

    # Training setup
    optimizer = optim.Adam(list(net_u2.parameters()) + list(net_u3.parameters()), lr=1e-3)
    n_epochs = 120
    batch_size = 256

    # Helper: beta logpdf
    def beta_logpdf_torch(u, alpha, beta):
        # u in (0,1); alpha,beta >0
        # log pdf = (alpha-1)log u + (beta-1) log(1-u) - ln B(alpha,beta)
        # we use torch.lgamma for log gamma
        eps = 1e-8
        u = torch.clamp(u, eps, 1-eps)
        log_num = (alpha - 1.0) * torch.log(u) + (beta - 1.0) * torch.log(1.0 - u)
        log_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        return log_num - log_den

    # Training loop
    U_t = torch.from_numpy(U_train).to(device)
    n_train = U_t.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch = U_t[idx]
            u1 = batch[:,0:1]
            u2 = batch[:,1:2]
            u3 = batch[:,2:3]

            # forward nets
            alpha_beta_u2 = net_u2(u1)  # shape (b,2)
            alpha2 = alpha_beta_u2[:,0:1]
            beta2 = alpha_beta_u2[:,1:2]
            alpha_beta_u3 = net_u3(batch[:,0:2])  # using u1,u2 as inputs
            alpha3 = alpha_beta_u3[:,0:1]
            beta3 = alpha_beta_u3[:,1:2]

            # negative log-likelihood (we maximize logpdf -> minimize -sum logpdf)
            logp_u2 = beta_logpdf_torch(u2, alpha2, beta2)
            logp_u3 = beta_logpdf_torch(u3, alpha3, beta3)
            loss = - (logp_u2.sum() + logp_u3.sum()) / batch.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.cpu().detach().numpy()) * batch.shape[0]
        if (epoch+1) % 30 == 0 or epoch==0:
            print(f"Epoch {epoch+1}/{n_epochs}, avg neg log-lik: {total_loss / n_train:.4f}")

    # Sampling from neural autoregressive copula
    m = 2000
    with torch.no_grad():
        # sample u1 from empirical marginal (bootstrap or using uniform grid)
        # we sample u1 by sampling indices from training data uniformly (preserves marginal)
        idxs = np.random.randint(0, U_tr.shape[0], size=m)
        u1_samp = U_tr[idxs, 0:1].astype(np.float32)
        u1_t = torch.from_numpy(u1_samp).to(device)

        # sample u2 ~ Beta(alpha(u1), beta(u1))
        params_u2 = net_u2(u1_t)  # (m,2)
        alpha_u2 = params_u2[:,0].cpu().numpy()
        beta_u2 = params_u2[:,1].cpu().numpy()
        # draw beta samples (numpy)
        u2_samp = np.random.beta(alpha_u2, beta_u2)

        # sample u3 using net_u3 with inputs (u1,u2)
        inp_u3 = torch.from_numpy(np.hstack([u1_samp, u2_samp.reshape(-1,1)]).astype(np.float32)).to(device)
        params_u3 = net_u3(inp_u3)
        alpha_u3 = params_u3[:,0].cpu().numpy()
        beta_u3 = params_u3[:,1].cpu().numpy()
        u3_samp = np.random.beta(alpha_u3, beta_u3)

        u_synth_nn = np.vstack([u1_samp.reshape(-1), u2_samp.reshape(-1), u3_samp.reshape(-1)]).T

    # Map back to original scales via empirical quantiles
    age_synth_nn = empirical_quantile(u_synth_nn[:,0], x_sorted_age)
    income_synth_nn = empirical_quantile(u_synth_nn[:,1], x_sorted_income)
    spend_synth_nn = empirical_quantile(u_synth_nn[:,2], x_sorted_spend)

    synth_nn = pd.DataFrame({
        "age": age_synth_nn,
        "income": income_synth_nn,
        "monthly_spend": spend_synth_nn
    })


# ---------- 3) Visual comparison: pairwise scatter plots ----------
def pair_scatter_grid(real_df, synth_df, title_real="Real", title_synth="Synthetic"):
    vars_ = real_df.columns.tolist()
    k = len(vars_)
    plt.figure(figsize=(10,8))
    for i in range(k):
        for j in range(k):
            plt_idx = i*k + j + 1
            plt.subplot(k,k,plt_idx)
            if i == j:
                # histogram on diagonal
                plt.hist(real_df[vars_[i]].values, bins=20, alpha=0.6)
                plt.title(vars_[i])
                plt.xticks([], [])
                plt.yticks([], [])
            else:
                plt.scatter(real_df[vars_[j]].values, real_df[vars_[i]].values, s=4)
                if plt_idx == 2:
                    plt.ylabel(title_real)
                plt.xticks([], [])
                plt.yticks([], [])
    plt.suptitle(f"{title_real} - pairwise (upper-left block)")
    plt.tight_layout(rect=[0,0,1,0.96])

    plt.figure(figsize=(10,8))
    for i in range(k):
        for j in range(k):
            plt_idx = i*k + j + 1
            plt.subplot(k,k,plt_idx)
            if i == j:
                plt.hist(synth_df[vars_[i]].values, bins=20, alpha=0.6)
                plt.title(vars_[i])
                plt.xticks([], [])
                plt.yticks([], [])
            else:
                plt.scatter(synth_df[vars_[j]].values, synth_df[vars_[i]].values, s=4)
                if plt_idx == 2:
                    plt.ylabel(title_synth)
                plt.xticks([], [])
                plt.yticks([], [])
    plt.suptitle(f"{title_synth} - pairwise (upper-left block)")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

pair_scatter_grid(data, synth_gauss, title_real="Données réelles", title_synth="Synthétiques (Gaussian copula)")

if synth_nn is not None:
    pair_scatter_grid(data, synth_nn, title_real="Données réelles", title_synth="Synthétiques (Neural autoregressive copula)")

# ---------- 4) Print a few numerical checks ----------
def summary_stats(df):
    return df.describe().T[["mean","std"]]

print("\nReal data - mean and std:")
print(summary_stats(data))
print("\nGaussian copula synthetic - mean and std:")
print(summary_stats(synth_gauss))
if synth_nn is not None:
    print("\nNeural copula synthetic - mean and std:")
    print(summary_stats(synth_nn))

# Show pairwise Spearman correlations to check dependence structure
def pair_spearman(df):
    return df.corr(method="spearman")

print("\nSpearman correlations - real data:")
print(pair_spearman(data))
print("\nSpearman correlations - Gaussian copula synthetic:")
print(pair_spearman(synth_gauss))
if synth_nn is not None:
    print("\nSpearman correlations - Neural copula synthetic:")
    print(pair_spearman(synth_nn))

print("\nDone.")
