import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# --- Environnement GridWorld (identique) ---
class GridWorld:
    def __init__(self, size=5, start=(0,0), goal=(4,4), max_steps=100):
        self.size = size
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.reset()
    
    def reset(self):
        self.pos = tuple(self.start)
        self.steps = 0
        return self.pos
    
    def step(self, action):
        # action: 0=up,1=right,2=down,3=left
        r, c = self.pos
        if action == 0:
            r = max(0, r-1)
        elif action == 1:
            c = min(self.size-1, c+1)
        elif action == 2:
            r = min(self.size-1, r+1)
        elif action == 3:
            c = max(0, c-1)
        self.pos = (r,c)
        self.steps += 1
        done = False
        reward = -1.0
        if self.pos == self.goal:
            reward = 10.0
            done = True
        if self.steps >= self.max_steps:
            done = True
        return self.pos, reward, done, {}

# --- Model-based Dyna-style learning ---
def dyna_model_based(env, episodes=300, alpha=0.5, gamma=0.98, eps_start=1.0, eps_end=0.05, n_planning=20, vi_interval=20):
    n = env.size
    n_actions = 4
    # Q table
    Q = np.zeros((n,n,n_actions))
    # model: counts and reward sums for (s,a)->s'
    trans_counts = defaultdict(lambda: np.zeros((n,n), dtype=np.int32))  # key (r,c,a) -> matrix counts over s' cells
    reward_sums = defaultdict(float)  # key (r,c,a) -> sum of rewards observed
    reward_counts = defaultdict(int)  # key (r,c,a) -> count of reward observations
    observed_sa = set()
    visits = np.zeros((n,n))
    rewards_hist = []
    eps = eps_start
    eps_decay = (eps_start - eps_end) / (episodes * 0.6)
    last_traj = []
    # for plotting model uncertainty: we'll compute entropy of P_hat(s'|s,a) averaged over a
    def get_model_probs(key):
        counts = trans_counts[key].reshape(-1).astype(float)
        s = counts.sum()
        if s == 0:
            return None
        probs = counts / s
        return probs.reshape(n,n)
    # Value iteration on the learned model (deterministic from counts -> probs)
    def value_iteration_from_model(gamma=0.98, tol=1e-4, maxiter=1000):
        V = np.zeros((n,n))
        # if no model at all, return zeros
        for it in range(maxiter):
            delta = 0.0
            V_new = np.zeros_like(V)
            for i in range(n):
                for j in range(n):
                    if (i,j) == env.goal:
                        V_new[i,j] = 0.0  # terminal value baseline (we treat immediate reward at transition to terminal)
                        continue
                    vals = []
                    for a in range(n_actions):
                        key = (i,j,a)
                        probs = get_model_probs(key)
                        if probs is None:
                            # if unknown transition, assume uniform random next state (conservative)
                            vals.append(0.0)
                        else:
                            # expected reward estimate
                            r_hat = (reward_sums[key] / reward_counts[key]) if reward_counts[key] > 0 else 0.0
                            ev = 0.0
                            # expected next value
                            ev += np.sum(probs * V)
                            vals.append(r_hat + gamma * ev)
                    V_new[i,j] = max(vals)
                    delta = max(delta, abs(V_new[i,j] - V[i,j]))
            V = V_new
            if delta < tol:
                break
        # greedy policy from V using model
        policy = np.zeros((n,n), dtype=int)
        for i in range(n):
            for j in range(n):
                best_a = 0
                best_val = -1e9
                for a in range(n_actions):
                    key = (i,j,a)
                    probs = get_model_probs(key)
                    if probs is None:
                        val = 0.0
                    else:
                        r_hat = (reward_sums[key] / reward_counts[key]) if reward_counts[key] > 0 else 0.0
                        val = r_hat + gamma * np.sum(probs * V)
                    if val > best_val:
                        best_val = val
                        best_a = a
                policy[i,j] = best_a
        return V, policy
    
    for ep in range(episodes):
        s = env.reset()
        total_r = 0.0
        traj = [s]
        done = False
        while not done:
            r0,c0 = s
            # epsilon-greedy w.r.t Q
            if random.random() < eps:
                a = random.randint(0, n_actions-1)
            else:
                a = int(np.argmax(Q[r0,c0,:]))
            s2, r, done, _ = env.step(a)
            r1,c1 = s2
            # Update model counts & rewards
            key = (r0,c0,a)
            trans_counts[key][r1,c1] += 1
            reward_sums[key] += r
            reward_counts[key] += 1
            observed_sa.add(key)
            # Real Q update (like Q-learning)
            best_next = np.max(Q[r1,c1,:])
            td_target = r + gamma * best_next * (0.0 if done else 1.0)
            Q[r0,c0,a] += alpha * (td_target - Q[r0,c0,a])
            visits[r0,c0] += 1
            total_r += r
            s = s2
            traj.append(s)
            # Planning: sample n_planning (s,a) from observed_sa, simulate s' from model probs, do Q-update on simulated transition
            if len(observed_sa) > 0:
                for _ in range(n_planning):
                    # sample a previously observed (s,a) pair
                    key_sim = random.choice(list(observed_sa))
                    i0,j0,a0 = key_sim
                    probs = get_model_probs(key_sim)
                    if probs is None:
                        continue
                    # sample next state according to empirical probs
                    flat = probs.reshape(-1)
                    idx = np.random.choice(len(flat), p=flat)
                    i1 = idx // n
                    j1 = idx % n
                    # reward estimate
                    r_hat = (reward_sums[key_sim] / reward_counts[key_sim]) if reward_counts[key_sim] > 0 else 0.0
                    best_next_sim = np.max(Q[i1,j1,:])
                    td_target_sim = r_hat + gamma * best_next_sim
                    Q[i0,j0,a0] += alpha * (td_target_sim - Q[i0,j0,a0])
        rewards_hist.append(total_r)
        last_traj = traj
        # epsilon decay
        if eps > eps_end:
            eps -= eps_decay
            if eps < eps_end:
                eps = eps_end
        # optional: run value iteration on current model occasionally to extract policy estimate
        if (ep+1) % vi_interval == 0 or ep == episodes-1:
            V_est, policy_est = value_iteration_from_model(gamma=gamma)
        if (ep+1) % max(1, episodes//6) == 0:
            avg = np.mean(rewards_hist[-(episodes//6):])
            print(f"Episode {ep+1}/{episodes}, recent avg reward: {avg:.2f}, eps={eps:.3f}, observed_sa={len(observed_sa)}")
    # final value/policy
    V_est, policy_est = value_iteration_from_model(gamma=gamma)
    return Q, V_est, policy_est, trans_counts, reward_sums, reward_counts, visits, rewards_hist, last_traj

# --- Run training ---
env = GridWorld(size=5, start=(0,0), goal=(4,4), max_steps=50)
Q_mb, V_est, policy_est, trans_counts, reward_sums, reward_counts, visits, rewards, traj = dyna_model_based(
    env, episodes=1000, alpha=0.6, gamma=0.999, eps_start=1.0, eps_end=0.05, n_planning=15, vi_interval=15
)

# --- Prepare visualizations: one figure with 2x3 subplots ---
fig, axes = plt.subplots(2,2, figsize=(15,9))
ax1, ax2, ax5, ax6 = axes.flatten()

# Subplot 1: Reward per episode + moving average
ax1.plot(np.arange(1,len(rewards)+1), rewards, label="reward par épisode")
window = 20
ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
ax1.plot(np.arange(window, len(rewards)+1), ma, label=f"moyenne mobile ({window})")
ax1.set_xlabel("Épisode")
ax1.set_ylabel("Reward total")
ax1.set_title("Reward par épisode (Model-Based Dyna)")
ax1.grid(True)
ax1.legend()

# Subplot 2: Heatmap de la valeur estimée par Value Iteration sur le modèle
im2 = ax2.imshow(V_est.T, origin='lower', interpolation='nearest', extent=[0, env.size, 0, env.size])
ax2.set_title("Valeur estimée V (Value Iteration sur modèle)")
ax2.set_xlabel("col")
ax2.set_ylabel("row")
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)


# Subplot 5: Nombre de visites par case
im5 = ax5.imshow(visits, origin='lower', interpolation='nearest', extent=[0, env.size, 0, env.size])
ax5.set_title("Nombre de visites par case (train)")
fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# Subplot 6: TRAJECTOIRE CORRIGEE
traj_arr = np.array(traj)
# x = column index, y = row index -> matches heatmaps where x increases to the right and y increases upward
x = traj_arr[:,1] + 0.5
y = traj_arr[:,0] + 0.5
ax6.plot(x, y, marker='o', linewidth=2)
ax6.scatter([env.start[1]+0.5], [env.start[0]+0.5], marker='s', label='start', zorder=5)
ax6.scatter([env.goal[1]+0.5], [env.goal[0]+0.5], marker='*', label='goal', zorder=5)
ax6.set_xlim(0, env.size)
ax6.set_ylim(0, env.size)
ax6.set_aspect('equal')
# IMPORTANT: do NOT invert yaxis -> origin lower (consistent with heatmaps)
ax6.set_title("Trajectoire finale (x=col, y=row)")
ax6.legend()

plt.tight_layout()
plt.show()

# --- Résumé simple ---
print("Résumé:")
print(f"Récompense moyenne (tous épisodes) : {np.mean(rewards):.2f}")
print(f"Récompense moyenne (derniers 50 épisodes) : {np.mean(rewards[-50:]):.2f}")
most_visited = np.unravel_index(np.argmax(visits), visits.shape)
print(f"Case la plus visitée (row,col) = {most_visited}, visites = {visits[most_visited]}")
