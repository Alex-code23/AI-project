# Re-run: calcule la rétropropagation de Riccati (P_t), les gains K_t, et simule la trajectoire forward.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paramètres du problème LQR (exemple simple)
np.random.seed(0)
n = 2   # dimension de l'état
m = 1   # dimension de l'action
T = 45  # horizon

A = np.array([[1.1, 0.1],
              [0.0, 0.95]])
B = np.array([[0.0],
              [1.0]])
Q = np.eye(n) * 1.0
R = np.eye(m) * 1.0

# Condition terminale pour P_T
P = np.zeros((T+1, n, n))
K = np.zeros((T, m, n))
P[T] = Q.copy()  # on prend Q_T = Q pour cet exemple

# Récurrence de Riccati (backward)
for t in reversed(range(T)):
    Pt1 = P[t+1]
    S = R + B.T @ Pt1 @ B             # m x m
    # résoudre S K^T = B^T P_{t+1} A  => K = S^{-1} B^T P_{t+1} A
    Kt = np.linalg.solve(S, B.T @ Pt1 @ A)  # m x n
    P[t] = Q + A.T @ Pt1 @ A - A.T @ Pt1 @ B @ Kt
    K[t] = Kt

# Simulation forward avec la loi u_t = -K_t x_t
x0 = np.array([2.0, 0.0])  # état initial
xs = np.zeros((T+1, n))
us = np.zeros((T, m))
xs[0] = x0.copy()

for t in range(T):
    us[t] = -K[t] @ xs[t]
    xs[t+1] = A @ xs[t] + B @ us[t]

# Préparer un DataFrame pour afficher P_t (éléments triangulaires) et K_t
rows = []
for t in range(T):
    Pt = P[t]
    Kt = K[t]
    row = {
        "t": t,
        "P00": float(Pt[0,0]),
        "P01": float(Pt[0,1]),
        "P11": float(Pt[1,1]),
        "K0_0": float(Kt[0,0]),
        "K0_1": float(Kt[0,1]),
        "x0": float(xs[t,0]),
        "x1": float(xs[t,1]),
        "u": float(us[t,0])
    }
    rows.append(row)

df = pd.DataFrame(rows).set_index("t")


# Afficher quelques matrices finales et trajectoire
print("P_0 (matrice de coût au temps 0) :\n", P[0])
print("\nK_0 (gain au temps 0) :\n", K[0])
print("\nÉtats finaux (x_T) :", xs[-1])



# Tracer plusieurs subplots pour visualiser l'évolution de tous les paramètres : états, commandes, gains K_t, et éléments de P_t

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# (1) États
axs[0,0].plot(np.arange(T+1), xs[:,0], label="x[0]")
axs[0,0].plot(np.arange(T+1), xs[:,1], label="x[1]")
axs[0,0].set_title("Trajectoire des états x_t")
axs[0,0].set_xlabel("time step")
axs[0,0].legend()
axs[0,0].grid(True)

# (2) Commande
axs[0,1].plot(np.arange(T), us[:,0], label="u(t)")
axs[0,1].set_title("Commande optimale u_t")
axs[0,1].set_xlabel("time step")
axs[0,1].legend()
axs[0,1].grid(True)

# (3) Gains K_t
axs[1,0].plot(np.arange(T), K[:,0,0], label="K[0,0]")
axs[1,0].plot(np.arange(T), K[:,0,1], label="K[0,1]")
axs[1,0].set_title("Gains de feedback K_t")
axs[1,0].set_xlabel("time step")
axs[1,0].legend()
axs[1,0].grid(True)

# (4) Éléments de P_t
axs[1,1].plot(np.arange(T), P[:-1,0,0], label="P[0,0]")
axs[1,1].plot(np.arange(T), P[:-1,0,1], label="P[0,1]")
axs[1,1].plot(np.arange(T), P[:-1,1,1], label="P[1,1]")
axs[1,1].set_title("Éléments de la matrice de Riccati P_t")
axs[1,1].set_xlabel("time step")
axs[1,1].legend()
axs[1,1].grid(True)

plt.suptitle("Analyse complète du LQR (Backward + Forward)", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()
