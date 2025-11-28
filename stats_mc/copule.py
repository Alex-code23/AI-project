"""
best_copula_yfinance.py
But : télécharger des actions depuis yfinance, estimer plusieurs copules et choisir la "meilleure" par AIC.
Usage basique : éditer la liste `TICKERS` et les dates, puis lancer.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats, special
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# ---------------------------
# 1. paramètres utilisateur
# ---------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]   # modifie si besoin
START = "2021-01-01"
END = "2024-12-31"
SIM_SAMPLES = 20000   # pour simulation/backtest
np.random.seed(42)

# ---------------------------
# 2. fonctions utilitaires
# ---------------------------
def download_adjclose(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    # si un ticker seul, data est Series -> convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how='all', axis=0)
    return data

def log_returns(price_df):
    return np.log(price_df).diff().dropna()

def emp_transform(df):
    """
    Transforme chaque colonne en U(0,1) via rangs empiriques :
    U = rank(x) / (n+1)
    Retourne numpy array shape (n_obs, n_vars)
    """
    ranks = df.rank(method="average")
    n = len(df)
    U = (ranks / (n + 1)).values
    return U

# ---------------------------
# 3. Copule gaussienne
# ---------------------------
def fit_gaussian_copula(U):
    """
    Estimation simple : transformer par quantile normal puis estimer matrice de corrélation R.
    On calcule log-vraisemblance multivariée normale (corr matrix).
    """
    n, d = U.shape
    Z = stats.norm.ppf(U)         # scores normaux
    # correction numérique des inf/sup
    Z = np.clip(Z, -10, 10)
    R = np.corrcoef(Z.T)         # matrice de corrélation estimée
    invR = np.linalg.inv(R)
    sign, logdet = np.linalg.slogdet(R) # permet de calculer log(det(R)) de façon stable qui gère les petits déterminants
    if sign <= 0:
        raise ValueError("Mat det non positive pour R (numéro stable?)")
    # log-likelihood :
    const = -0.5 * d * np.log(2 * np.pi)
    ll = 0.0
    for i in range(n):
        z = Z[i,:]
        ll += const - 0.5 * logdet - 0.5 * (z @ invR @ z)
    params = {"R": R}
    return ll, params

# ---------------------------
# 4. Copule t multivariée (profil sur nu)
# ---------------------------
def multivariate_t_logpdf(x, R, nu):
    """
    Log-pdf du t multivarié (moyenne 0, corr matrix R, df=nu)
    x: vecteur (d,)
    Formule basée sur densité du t multivarié
    """
    d = len(x)
    sign, logdetR = np.linalg.slogdet(R)
    if sign <= 0:
        return -np.inf
    invR = np.linalg.inv(R)
    term1 = special.gammaln((nu + d) / 2) - special.gammaln(nu / 2)
    term2 = -0.5 * logdetR - (d / 2) * np.log(nu * np.pi)
    quad = 1 + (x @ invR @ x) / nu
    term3 = - ((nu + d) / 2) * np.log(quad)
    return term1 + term2 + term3

def fit_t_copula_profile_nu(U, nu_grid=None):
    """
    Profil sur nu :
    - pour chaque nu testé, transformer U par t.ppf(u, nu) -> T
    - estimer corr(T) -> R_nu
    - calculer log-likelihood multivarié t (avec R_nu, nu)
    - retourner nu* et R*
    """
    if nu_grid is None:
        nu_grid = np.concatenate((np.linspace(2.1, 10, 17), np.linspace(11, 60, 10)))
    n, d = U.shape
    best = {"nu": None, "R": None, "ll": -np.inf}
    for nu in nu_grid:
        # transformer
        try:
            T = stats.t.ppf(U, df=nu)
        except Exception:
            continue
        # corr
        T = np.clip(T, -1e6, 1e6)
        R = np.corrcoef(T.T)
        # compute loglik
        invR = np.linalg.pinv(R)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            continue
        ll = 0.0
        for i in range(n):
            x = T[i, :]
            ll += multivariate_t_logpdf(x, R, nu)
        if ll > best["ll"]:
            best = {"nu": nu, "R": R, "ll": ll}
    if best["nu"] is None:
        raise RuntimeError("Échec estimation t-copule sur la grille")
    params = {"R": best["R"], "nu": best["nu"]}
    return best["ll"], params

# ---------------------------
# 5. Copules archimédiennes (estimation par Kendall tau)
#    - On calcule taus par paires et on récupère un theta moyen
#    - Utilise composite log-likelihood (somme des log densités bivariées) comme critère approximatif
#    NB: densités bivariées pour Clayton/Gumbel/Frank sont codées ci-dessous (bivarié)
# ---------------------------
def kendall_tau_matrix(U):
    d = U.shape[1]
    taus = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            tau, _ = stats.kendalltau(U[:, i], U[:, j])
            taus[i, j] = taus[j, i] = tau
    return taus

def clayton_theta_from_tau(tau):
    # tau = theta / (theta + 2) => theta = 2 tau / (1 - tau)
    return 2 * tau / (1 - tau)

def gumbel_theta_from_tau(tau):
    # tau = 1 - 1/theta => theta = 1/(1 - tau)
    return 1.0 / (1.0 - tau)

# bivariate density approximations: we implement log density for Clayton and Gumbel (bivarié)
def clayton_bivariate_logpdf(u, v, theta):
    # densité bivariée de Clayton
    if theta <= 0:
        return -np.inf
    a = (theta + 1) * (u * v) ** (- (theta + 1))
    b = (u ** (-theta) + v ** (-theta) - 1) ** (-2 - 1/theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = (1 + theta) * (u * v) ** (- (1 + theta)) * (u ** (-theta) + v ** (-theta) - 1) ** (- (2 + 1/theta))
        logd = np.log(dens)
    if np.isneginf(logd) or np.isnan(logd):
        return -1e9
    return logd

def gumbel_bivariate_logpdf(u, v, theta):
    # densité bivariée de Gumbel (formule peut être instable en bords)
    if theta < 1:
        return -np.inf
    # Using numerical stable form via logs
    try:
        a = (-np.log(u)) ** theta
        b = (-np.log(v)) ** theta
        A = (a + b) ** (1.0 / theta)
        C = np.exp(-A)
        # derivatives
        part1 = C * (a + b) ** (1.0/theta - 2)
        # approximate density (may be numerically delicate)
        logd = np.log(C) + np.log(part1)  # rough
    except Exception:
        return -1e9
    if np.isnan(logd):
        return -1e9
    return logd

def fit_archimedean_via_kendall(U, family='clayton'):
    """
    Estime un theta global comme moyenne des thetas paires issues de Kendall tau.
    Retourne un composite log-likelihood (somme des log dens bivariées) comme score.
    """
    n, d = U.shape
    taus = kendall_tau_matrix(U)
    thetas = []
    for i in range(d):
        for j in range(i+1, d):
            tau = taus[i,j]
            if np.isnan(tau):
                continue
            if family == 'clayton':
                theta_ij = clayton_theta_from_tau(tau)
            elif family == 'gumbel':
                theta_ij = gumbel_theta_from_tau(tau)
            else:
                raise ValueError("family not supported")
            thetas.append(theta_ij)
    if len(thetas) == 0:
        raise RuntimeError("Pas de paires valides pour estimer theta")
    theta_hat = float(np.median(thetas))  # mediane plus robuste que moyenne
    # composite log-likelihood (somme sur paires)
    comp_ll = 0.0
    for i in range(d):
        for j in range(i+1, d):
            u = U[:, i]
            v = U[:, j]
            if family == 'clayton':
                vec = [clayton_bivariate_logpdf(u[k], v[k], theta_hat) for k in range(n)]
            elif family == 'gumbel':
                vec = [gumbel_bivariate_logpdf(u[k], v[k], theta_hat) for k in range(n)]
            comp_ll += np.sum(vec)
    params = {"theta": theta_hat}
    return comp_ll, params

# ---------------------------
# 6. Comparaison AIC / BIC
# ---------------------------
def compute_aic(ll, k):
    return 2 * k - 2 * ll

def compute_bic(ll, k, n):
    return k * np.log(n) - 2 * ll

# ---------------------------
# 7. Simulation pour contrôle (ex : gaussienne)
# ---------------------------
def simulate_gaussian_copula(R, n):
    d = R.shape[0]
    z = np.random.multivariate_normal(mean=np.zeros(d), cov=R, size=n)
    u = stats.norm.cdf(z)
    return u

def simulate_t_copula(R, nu, n):
    d = R.shape[0]
    # simulate multivariate t: z ~ N(0,R), g ~ chi2(nu)/nu
    z = np.random.multivariate_normal(mean=np.zeros(d), cov=R, size=n)
    g = np.random.chisquare(nu, size=(n,1)) / nu
    t = z / np.sqrt(g)
    u = stats.t.cdf(t, df=nu)
    return u

# ---------------------------
# 8. pipeline principal
# ---------------------------
def main():
    print("Downloading data...")
    prices = download_adjclose(TICKERS, START, END)
    print("Prices shape:", prices.shape)
    returns = log_returns(prices)
    returns = returns.dropna(axis=1, how='any')  # remove tickers with too many NaN
    tickers = returns.columns.tolist()
    print("Using tickers:", tickers)
    U = emp_transform(returns)
    n, d = U.shape
    print("Observations:", n, "Dimension:", d)

    results = []

    # Gaussian copula
    try:
        ll_gauss, params_gauss = fit_gaussian_copula(U)
        k_gauss = d*(d-1)//2   # nb param corr matrix (approx)
        aic_gauss = compute_aic(ll_gauss, k_gauss)
        bic_gauss = compute_bic(ll_gauss, k_gauss, n)
        results.append(("Gaussian", ll_gauss, k_gauss, aic_gauss, bic_gauss, params_gauss))
        print("Gaussian LL:", ll_gauss, "AIC:", aic_gauss)
    except Exception as e:
        print("Gaussian estimation failed:", e)

    # t copula (profil sur nu)
    try:
        print("Estimating t-copule (profil sur nu)...")
        ll_t, params_t = fit_t_copula_profile_nu(U)
        k_t = d*(d-1)//2 + 1   # correlation params + nu
        aic_t = compute_aic(ll_t, k_t)
        bic_t = compute_bic(ll_t, k_t, n)
        results.append(("t", ll_t, k_t, aic_t, bic_t, params_t))
        print("t-copule LL:", ll_t, "nu:", params_t["nu"], "AIC:", aic_t)
    except Exception as e:
        print("t-copule estimation failed:", e)

    # Clayton (archimédienne) estimation via Kendall + composite ll
    try:
        ll_clay, params_clay = fit_archimedean_via_kendall(U, family='clayton')
        k_clay = 1
        aic_clay = compute_aic(ll_clay, k_clay)
        bic_clay = compute_bic(ll_clay, k_clay, n)
        results.append(("Clayton (composite)", ll_clay, k_clay, aic_clay, bic_clay, params_clay))
        print("Clayton composite LL:", ll_clay, "theta:", params_clay["theta"], "AIC:", aic_clay)
    except Exception as e:
        print("Clayton estimation failed:", e)

    # Gumbel
    try:
        ll_gumb, params_gumb = fit_archimedean_via_kendall(U, family='gumbel')
        k_gumb = 1
        aic_gumb = compute_aic(ll_gumb, k_gumb)
        bic_gumb = compute_bic(ll_gumb, k_gumb, n)
        results.append(("Gumbel (composite)", ll_gumb, k_gumb, aic_gumb, bic_gumb, params_gumb))
        print("Gumbel composite LL:", ll_gumb, "theta:", params_gumb["theta"], "AIC:", aic_gumb)
    except Exception as e:
        print("Gumbel estimation failed:", e)

    # Résumé et sélection
    df_res = pd.DataFrame([{
        "model": r[0],
        "loglik": r[1],
        "n_params": r[2],
        "AIC": r[3],
        "BIC": r[4],
        "params": r[5]
    } for r in results])
    df_res = df_res.sort_values("AIC")
    print("\nComparaison des modèles (triée par AIC) :\n")
    print(df_res[["model", "loglik", "n_params", "AIC", "BIC"]])

    best = df_res.iloc[0]
    print("\nMeilleur modèle selon AIC:", best["model"])
    print("Paramètres:", best["params"])

    # Backtest / check: comparer matrice de Kendall de données vs simulation
    print("\nBacktesting: simulation selon le meilleur modèle pour comparer Kendall taus...")
    if best["model"] == "Gaussian":
        R = best["params"]["R"]
        Usim = simulate_gaussian_copula(R, SIM_SAMPLES)
    elif best["model"] == "t":
        R = best["params"]["R"]
        nu = best["params"]["nu"]
        Usim = simulate_t_copula(R, nu, SIM_SAMPLES)
    else:
        # pour archimédiennes bivariées, on fait une simulation bivariée par paire — approximatif
        print("Simulation directe non implémentée pour ce modèle (archimédienne multivariable).")
        Usim = None

    if Usim is not None:
        tau_emp = kendall_tau_matrix(U)
        tau_sim = kendall_tau_matrix(pd.DataFrame(Usim))
        print("Kendall tau (empirique) :\n", np.round(tau_emp, 3))
        print("Kendall tau (simulé)   :\n", np.round(tau_sim, 3))
        # Visual quick plot: empirical vs simulated scatter for first two assets
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.scatter(U[:,0], U[:,1], s=4, alpha=0.4)
        plt.title("Empirical (U space) {} vs {}".format(tickers[0], tickers[1]))
        plt.subplot(1,2,2)
        plt.scatter(Usim[:,0], Usim[:,1], s=4, alpha=0.4)
        plt.title("Simulated by best copula")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
