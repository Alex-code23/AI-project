# Comparison of three methods on toy single-state two-action MDP
# 1) Semi-gradient iterative updates: q <- q + alpha * (T(q) - q)
# 2) Full gradient descent on L(q) = 0.5 * ||T(q) - q||^2
# 3) Regression-style (fit q to frozen targets y = T(q_prev)): q <- q + lr_reg * (y - q)

import numpy as np
import matplotlib.pyplot as plt

# Toy MDP parameters (single state, two actions)
r0, r1 = 1.0, 0.5
gamma = 0.9

def T_of_Q(q):
    m = np.max(q)
    return np.array([r0 + gamma * m, r1 + gamma * m]) # Bellman optimality operator

def J_T(q):
    # Jacobian of T wrt q (2x2) depending on argmax
    arg = 0 if q[0] >= q[1] else 1
    J = np.zeros((2,2))
    J[:, arg] = gamma
    return J

# Parameters
alpha = 0.5    # semi-gradient step size
eta = 0.1      # full-gradient descent step size (on the loss)
lr_reg = 0.3   # regression learning rate toward frozen targets
n_steps = 30

# Starting points
starts = [
    # np.array([2.0, 0.0]),
    # np.array([0.0, 2.0]),
    # np.array([0.5, 0.5]),
    np.array([-0.5, 4.0])
]

# Prepare contour background for Bellman residual norm
grid_min, grid_max, N = -1.0, 10.0, 201
q0_vals = np.linspace(grid_min, grid_max, N)
q1_vals = np.linspace(grid_min, grid_max, N)
Q0, Q1 = np.meshgrid(q0_vals, q1_vals)
res_norm = np.zeros_like(Q0)
for i in range(N):
    for j in range(N):
        q = np.array([Q0[i,j], Q1[i,j]])
        res_norm[i,j] = np.linalg.norm(T_of_Q(q) - q)

# Run trajectories
traj_semi = []
traj_full = []
traj_reg = []

for s in starts:
    # initialize
    q_semi = s.astype(float).copy()
    q_full = s.astype(float).copy()
    q_reg = s.astype(float).copy()
    t_semi = [q_semi.copy()]
    t_full = [q_full.copy()]
    t_reg = [q_reg.copy()]
    for t in range(n_steps):
        # semi-gradient update (bootstrapping)
        delta = T_of_Q(q_semi) - q_semi
        q_semi = q_semi + alpha * delta
        t_semi.append(q_semi.copy())
        # full gradient descent on L = 0.5 ||T(q) - q||^2
        residual = T_of_Q(q_full) - q_full
        J = J_T(q_full)
        grad_L = (J.T - np.eye(2)).dot(residual)  # gradient of 0.5||T-q||^2
        q_full = q_full - eta * grad_L
        t_full.append(q_full.copy())
        # regression-style: compute frozen target y = T(q_prev_reg) and do one step toward it
        y = T_of_Q(q_reg.copy())  # frozen target using current q_reg before update
        q_reg = q_reg + lr_reg * (y - q_reg)
        t_reg.append(q_reg.copy())
    traj_semi.append(np.array(t_semi))
    traj_full.append(np.array(t_full))
    traj_reg.append(np.array(t_reg))

# Plotting
plt.figure(figsize=(9,8))
CS = plt.contourf(Q0, Q1, res_norm, levels=40)
plt.colorbar(CS, label='||T(Q) - Q|| (Bellman residual norm)')
plt.title("Comparison: Semi-gradient vs Full-gradient vs Regression-style")
plt.xlabel("q0")
plt.ylabel("q1")
plt.gca().set_aspect('equal', adjustable='box')

labels = ['semi-gradient', 'full-gradient', 'regression']
markers = ['o', 's', 'D']

for i in range(len(starts)):
    # semi
    traj = traj_semi[i]
    plt.plot(traj[:,0], traj[:,1], marker=markers[0], linewidth=0.3, markersize=2, label=f"{labels[0]} start {i}" if i==0 else None)
    plt.scatter(traj_semi[i][-1,0], traj_semi[i][-1,1], marker='*', s=80)
    # full
    traj = traj_full[i]
    plt.plot(traj[:,0], traj[:,1], marker=markers[1], linewidth=0.3, markersize=2, label=f"{labels[1]} start {i}" if i==0 else None)
    plt.scatter(traj_full[i][-1,0], traj_full[i][-1,1], marker='*', s=80)
    # reg
    traj = traj_reg[i]
    plt.plot(traj[:,0], traj[:,1], marker=markers[2], linewidth=0.3, markersize=2, label=f"{labels[2]} start {i}" if i==0 else None)
    plt.scatter(traj_reg[i][-1,0], traj_reg[i][-1,1], marker='*', s=80)
    # annotate only first and last for clarity
    plt.scatter(starts[i][0], starts[i][1], marker='x', s=80)


plt.legend(loc='upper left', framealpha=0.9)
plt.show()

# Print summary final positions and residuals
print("Summary of final Q values after {} steps:".format(n_steps))
print("{:>10s} {:>20s} {:>20s} {:>20s}".format("start", "semi-gradient", "full-gradient", "regression"))
for i, s in enumerate(starts):
    fs = traj_semi[i][-1]
    ff = traj_full[i][-1]
    fr = traj_reg[i][-1]
    rs = np.linalg.norm(T_of_Q(fs) - fs)
    rf = np.linalg.norm(T_of_Q(ff) - ff)
    rr = np.linalg.norm(T_of_Q(fr) - fr)
    print("{:>10s} {:>20s} {:>20s} {:>20s}".format(str(s.tolist()), 
                                                   np.array2string(fs, precision=4), 
                                                   np.array2string(ff, precision=4),
                                                   np.array2string(fr, precision=4)))
    print(" " * 12 + "residuals: semi={:.6f}, full={:.6f}, reg={:.6f}".format(rs, rf, rr))

# Also compute and print average residual over trajectory for each method
def avg_res(traj_list):
    res = []
    for traj in traj_list:
        vals = [np.linalg.norm(T_of_Q(q) - q) for q in traj]
        res.append(np.mean(vals))
    return np.mean(res)

print("\nAverage residual over trajectories (lower better):")
print(" semi-gradient avg:", avg_res(traj_semi))
print(" full-gradient avg :", avg_res(traj_full))
print(" regression avg    :", avg_res(traj_reg))
