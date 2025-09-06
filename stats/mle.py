import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm, expon

# Pour reproductibilité
np.random.seed(42)

# =============================
# 1. Données simulées
# =============================
n = 50
data_bernoulli = bernoulli.rvs(0.7, size=n)       # p=0.7
data_normal = norm.rvs(loc=2, scale=1.5, size=n)  # N(mu=2, sigma=1.5)
data_expon = expon.rvs(scale=2, size=n)           # Exp(lambda=1/scale=0.5)

# =============================
# 2. Fonctions de log-vraisemblance
# =============================
def loglik_bernoulli(p, data):
    if p <= 0 or p >= 1:
        return -np.inf
    return np.sum(data*np.log(p) + (1-data)*np.log(1-p))

def loglik_normal(mu, data, sigma=1.5):
    return np.sum(norm.logpdf(data, loc=mu, scale=sigma))

def loglik_expon(lmbda, data):
    if lmbda <= 0:
        return -np.inf
    return np.sum(np.log(lmbda) - lmbda*data)

# =============================
# 3. Grilles de paramètres
# =============================
p_vals = np.linspace(0.01, 0.99, 100)
mu_vals = np.linspace(min(data_normal)-1, max(data_normal)+1, 100)
lmbda_vals = np.linspace(0.01, 2, 100)

loglik_p = [loglik_bernoulli(p, data_bernoulli) for p in p_vals]
loglik_mu = [loglik_normal(mu, data_normal) for mu in mu_vals]
loglik_lmbda = [loglik_expon(l, data_expon) for l in lmbda_vals]

# =============================
# 4. Trouver le maximum
# =============================
mle_p = p_vals[np.argmax(loglik_p)]
mle_mu = mu_vals[np.argmax(loglik_mu)]
mle_lmbda = lmbda_vals[np.argmax(loglik_lmbda)]

# =============================
# 5. Graphiques
# =============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bernoulli
axes[0].plot(p_vals, loglik_p, label="log-vraisemblance")
axes[0].axvline(mle_p, color="red", linestyle="--", label=f"MLE p={mle_p:.2f}")
axes[0].set_title("Bernoulli : log-vraisemblance en fonction de p")
axes[0].set_xlabel("p")
axes[0].set_ylabel("log L(p)")
axes[0].legend()

# Normal (mu inconnu, sigma connu)
axes[1].plot(mu_vals, loglik_mu, label="log-vraisemblance")
axes[1].axvline(mle_mu, color="red", linestyle="--", label=f"MLE mu={mle_mu:.2f}")
axes[1].set_title("Normale : log-vraisemblance en fonction de mu (σ connu)")
axes[1].set_xlabel("mu")
axes[1].set_ylabel("log L(mu)")
axes[1].legend()

# Exponentielle (lambda inconnu)
axes[2].plot(lmbda_vals, loglik_lmbda, label="log-vraisemblance")
axes[2].axvline(mle_lmbda, color="red", linestyle="--", label=f"MLE λ={mle_lmbda:.2f}")
axes[2].set_title("Exponentielle : log-vraisemblance en fonction de λ")
axes[2].set_xlabel("lambda")
axes[2].set_ylabel("log L(λ)")
axes[2].legend()

plt.tight_layout()
plt.show()

# Résultats numériques
mle_p, mle_mu, mle_lmbda
