# Visualization comparing Q-learning (Bellman residual) direction vs full gradient of Bellman error
import numpy as np
import matplotlib.pyplot as plt

# Toy single-state two-action MDP parameters
r0, r1 = 1.0, 0.6
gamma = 0.9

# Define Bellman optimality operator T(Q) for 2 actions in a single-state MDP
def T_of_Q(q):
    # q: array-like shape (2,)
    m = np.max(q)
    return np.array([r0 + gamma * m, r1 + gamma * m])

# Grid of Q-values (q0, q1)
grid_min, grid_max, N = -1.0, 3.0, 41
q0_vals = np.linspace(grid_min, grid_max, N)
q1_vals = np.linspace(grid_min, grid_max, N)
Q0, Q1 = np.meshgrid(q0_vals, q1_vals)

# Preallocate fields
res_norm = np.zeros_like(Q0)
semi_u = np.zeros_like(Q0)  # semi-gradient (T-Q) x-component
semi_v = np.zeros_like(Q1)  # semi-gradient y-component
full_u = np.zeros_like(Q0)  # full gradient direction x
full_v = np.zeros_like(Q1)  # full gradient direction y

for i in range(N):
    for j in range(N):
        q = np.array([Q0[i,j], Q1[i,j]])
        Tq = T_of_Q(q)
        residual = Tq - q  # Bellman residual (T(Q) - Q)
        res_norm[i,j] = np.linalg.norm(residual)
        # semi-gradient direction (as used in Q-learning / bootstrapping): delta = T(q) - q
        semi = residual.copy()
        semi_u[i,j], semi_v[i,j] = semi[0], semi[1]
        # full gradient of 0.5 * ||T(q) - q||^2 (includes dT/dq)
        # compute Jacobian J_T (2x2) piecewise depending on argmax
        arg = 0 if q[0] >= q[1] else 1
        # derivative dT_i/dq_j = gamma if q_j is argmax, else 0
        J = np.zeros((2,2))
        J[:, arg] = gamma  # same column for both rows
        # full gradient = (J^T - I) @ (T - Q)
        full_grad = (J.T - np.eye(2)).dot(residual)
        # We will plot -full_grad direction (descent direction): negative gradient points towards decreasing loss
        fd = -full_grad
        full_u[i,j], full_v[i,j] = fd[0], fd[1]

# Normalize vectors for quiver plots (visual clarity)
def normalize_field(u, v, eps=1e-8):
    mag = np.sqrt(u**2 + v**2) + eps
    return u/mag, v/mag, mag

s_u, s_v, s_mag = normalize_field(semi_u, semi_v)
f_u, f_v, f_mag = normalize_field(full_u, full_v)

# Plot 1: contour of residual norm with semi-gradient quiver
plt.figure(figsize=(7,6))
CS = plt.contourf(Q0, Q1, res_norm, levels=40)
plt.colorbar(CS, label='||T(Q) - Q|| (Bellman residual norm)')
plt.quiver(Q0, Q1, s_u, s_v, scale=30, width=0.004)
plt.title("Semi-gradient / Bellman-residual direction (Q-learning style)")
plt.xlabel("q0")
plt.ylabel("q1")
plt.scatter([],[], label=f"params: r0={r0}, r1={r1}, gamma={gamma}")
plt.legend(loc='upper left', framealpha=0.9)
plt.grid(False)
plt.gca().set_aspect('equal', adjustable='box')

# Plot 2: contour of residual norm with full-gradient descent quiver
plt.figure(figsize=(7,6))
CS2 = plt.contourf(Q0, Q1, res_norm, levels=40)
plt.colorbar(CS2, label='||T(Q) - Q|| (Bellman residual norm)')
plt.quiver(Q0, Q1, f_u, f_v, scale=30, width=0.004)
plt.title("Negative full gradient of 0.5 ||T(Q)-Q||^2 (true gradient descent)")
plt.xlabel("q0")
plt.ylabel("q1")
plt.scatter([],[], label=f"params: r0={r0}, r1={r1}, gamma={gamma}")
plt.legend(loc='upper left', framealpha=0.9)
plt.grid(False)
plt.gca().set_aspect('equal', adjustable='box')

# Mark fixed points where residual ~ 0
tol = 1e-3
fixed_mask = res_norm < tol
fixed_points = np.column_stack((Q0[fixed_mask], Q1[fixed_mask]))
if fixed_points.size > 0:
    plt.figure(figsize=(6,6))
    plt.contourf(Q0, Q1, res_norm, levels=40)
    plt.colorbar(label='||T(Q) - Q||')
    plt.scatter(fixed_points[:,0], fixed_points[:,1], marker='x', s=50)
    plt.title("Fixed points (where T(Q)=Q)")
    plt.xlabel("q0")
    plt.ylabel("q1")
    plt.gca().set_aspect('equal', adjustable='box')

plt.show()

# For clarity print a short explanation summary
print("Legend / interpretation:")
print("- Contours show the norm of the Bellman residual ||T(Q)-Q||. Fixed points satisfy residual=0.")
print("- Top figure: arrows show the semi-gradient / residual direction T(Q)-Q (what Q-learning bootstrapping follows).")
print("- Middle figure: arrows show the negative FULL gradient of 0.5||T(Q)-Q||^2 (the true descent direction including dT/dQ).")
print("- Differences between the two vector fields explain why Q-learning updates (bootstrapping) are NOT generally equal to gradient descent steps on the Bellman error.")

