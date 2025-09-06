import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.robust.scale import huber

# -----------------------------
# 1. Génération des données
# -----------------------------
np.random.seed(123)
n = 50
data = norm.rvs(loc=0, scale=1, size=n)  # données normales centrées
data_with_outliers = np.concatenate([data, [10, 12, -9]])  # ajout de gros outliers

# -----------------------------
# 2. Estimateurs
# -----------------------------

# Moyenne
mean_est = np.mean(data_with_outliers)

# Médiane
median_est = np.median(data_with_outliers)

# MLE pour mu (sigma=1 connu) → c'est la moyenne
mle_est = mean_est

# Huber estimator (robuste)
# statsmodels.robust.scale.huber renvoie (mu, scale)
huber_est, _ = huber(data_with_outliers)

# -----------------------------
# 3. Visualisation
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(data_with_outliers, bins=20, alpha=0.6, color="skyblue", edgecolor="black", density=True, label="Données avec outliers")

# Lignes verticales pour les estimateurs
ax.axvline(mean_est, color="red", linestyle="--", linewidth=2, label=f"Moyenne / MLE = {mean_est:.2f}")
ax.axvline(median_est, color="green", linestyle="--", linewidth=2, label=f"Médiane = {median_est:.2f}")
ax.axvline(huber_est, color="purple", linestyle="--", linewidth=2, label=f"Huber = {huber_est:.2f}")

ax.set_title("Comparaison de M-estimateurs avec outliers")
ax.legend()
plt.show()

mean_est, median_est, huber_est
