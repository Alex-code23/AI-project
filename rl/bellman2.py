# Multi-state MDP comparison: semi-gradient vs full-gradient vs regression-style
# - MDP with S states and A actions (here S=3, A=2)
# - Q is shape (S, A). We'll flatten to vector of length S*A for math.
# - T(Q)[s,a] = R[s,a] + gamma * sum_{s'} P[s,a,s'] * max_a' Q[s',a']
# - Full gradient uses Jacobian dT/dQ that depends on argmax at each next-state.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# MDP definition
S = 3
A = 2
gamma = 0.9

# Transition probabilities P[s,a,s'] (stochastic rows sum to 1 over s')
P = np.zeros((S, A, S))
# specify some structured transitions for interpretability
P[0,0] = [0.8, 0.2, 0.0]
P[0,1] = [0.1, 0.7, 0.2]
P[1,0] = [0.0, 0.9, 0.1]
P[1,1] = [0.3, 0.3, 0.4]
P[2,0] = [0.2, 0.0, 0.8]
P[2,1] = [0.25, 0.25, 0.5]

# Rewards R[s,a]
R = np.array([
    [1.0, 0.0],   # state 0: a0 gives 1, a1 gives 0
    [0.5, 0.2],   # state 1
    [0.0, 1.5]    # state 2: a1 is attractive
])

# helper to flatten/unflatten
def pack(q_mat):
    return q_mat.reshape(-1)

def unpack(q_vec):
    return q_vec.reshape(S, A)

# Bellman optimality operator T(Q)
def T_of_Q(q_vec):
    q = unpack(q_vec)
    q_next_max = np.max(q, axis=1)   # max_a' Q[s', a']
    T = np.zeros_like(q)
    for s in range(S):
        for a in range(A):
            T[s,a] = R[s,a] + gamma * np.dot(P[s,a], q_next_max)
    return pack(T)

# Jacobian of T wrt Q (size (S*A, S*A))
# For each (s,a) and each (s',a'), dT_{(s,a)}/dQ_{(s',a')} = gamma * P[s,a,s'] * I[a' == argmax_a' Q[s']]
def J_T(q_vec):
    q = unpack(q_vec)
    argmax_actions = np.argmax(q, axis=1)  # for each next-state s', which action is argmax
    J = np.zeros((S*A, S*A))
    for s in range(S):
        for a in range(A):
            row = s * A + a
            for sp in range(S):
                a_star = argmax_actions[sp]
                col = sp * A + a_star
                J[row, col] += gamma * P[s, a, sp]
    return J

# Norm of Bellman residual
def residual_norm(q_vec):
    r = T_of_Q(q_vec) - q_vec
    return np.linalg.norm(r)

# Methods parameters
alpha = 0.6   # semi-gradient step size (bootstrapping)
eta = 0.05    # full-gradient descent step size
lr_reg = 0.6  # regression style toward frozen target
n_steps = 60

# initial Q (random small values)
q0 = np.array([0.5, 0.2, 0.2, 0.1, -0.2, 0.0])  # shape S*A (for S=3,A=2)
# ensure length matches S*A
assert q0.size == S*A

# storage for trajectories
traj_semi = [q0.copy()]
traj_full = [q0.copy()]
traj_reg = [q0.copy()]

q_semi = q0.copy()
q_full = q0.copy()
q_reg = q0.copy()

for t in range(n_steps):
    # semi-gradient: q <- q + alpha * (T(q) - q)
    delta = T_of_Q(q_semi) - q_semi
    q_semi = q_semi + alpha * delta
    traj_semi.append(q_semi.copy())
    # full-gradient: grad L = (J^T - I) @ (T - Q); descent step
    residual = T_of_Q(q_full) - q_full
    J = J_T(q_full)
    grad_L = (J.T - np.eye(S*A)).dot(residual)
    q_full = q_full - eta * grad_L
    traj_full.append(q_full.copy())
    # regression-style: frozen target y = T(q_reg) then move toward it
    y = T_of_Q(q_reg.copy())
    q_reg = q_reg + lr_reg * (y - q_reg)
    traj_reg.append(q_reg.copy())

traj_semi = np.array(traj_semi)
traj_full = np.array(traj_full)
traj_reg = np.array(traj_reg)

# Plot evolution of each Q component over iterations for each method
iters = np.arange(n_steps+1)
plt.figure(figsize=(12,8))
component_labels = []
for s in range(S):
    for a in range(A):
        component_labels.append(f"Q(s={s},a={a})")

colors = plt.cm.tab10(np.arange(S*A))

for i in range(S*A):
    plt.plot(iters, traj_semi[:,i], linestyle='-', color=colors[i], alpha=0.6)
    plt.plot(iters, traj_full[:,i], linestyle='--', color=colors[i], alpha=0.9)
    plt.plot(iters, traj_reg[:,i], linestyle=':', color=colors[i], alpha=0.9)
# create legend entries only once per component
handles = []
for i in range(S*A):
    handles.append(plt.Line2D([0],[0], color=colors[i], lw=2))
plt.legend(handles, component_labels, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xlabel("Iteration")
plt.ylabel("Q value")
plt.title("Evolution of Q components (solid=semi, dashed=full-gradient, dotted=regression)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot residual norms over iterations
res_semi = [residual_norm(q) for q in traj_semi]
res_full = [residual_norm(q) for q in traj_full]
res_reg = [residual_norm(q) for q in traj_reg]

plt.figure(figsize=(8,5))
plt.plot(iters, res_semi, label='semi-gradient', linestyle='-')
plt.plot(iters, res_full, label='full-gradient', linestyle='--')
plt.plot(iters, res_reg, label='regression-style', linestyle=':')
plt.xlabel("Iteration")
plt.ylabel("||T(Q) - Q||")
plt.title("Bellman residual norm over iterations")
plt.legend()
plt.grid(True)
plt.show()

# Print final Q matrices and residuals
print("Final Q (semi-gradient):\n", unpack(traj_semi[-1]))
print("Final Q (full-gradient):\n", unpack(traj_full[-1]))
print("Final Q (regression):\n", unpack(traj_reg[-1]))
print("\nFinal residual norms: semi={:.6f}, full={:.6f}, reg={:.6f}".format(res_semi[-1], res_full[-1], res_reg[-1]))
