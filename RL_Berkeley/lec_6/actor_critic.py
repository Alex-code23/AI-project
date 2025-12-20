import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- Environment: simple chain ----
class ChainMDP:
    def __init__(self, L=6, max_steps=10):
        self.L = L
        self.max_steps = max_steps
    def reset(self):
        self.s = 0
        self.t = 0
        return self.s
    def step(self, a):
        # a: 0 left, 1 right
        if a == 1:
            self.s = min(self.s + 1, self.L-1)
        else:
            self.s = max(0, self.s - 1)
        self.t += 1
        done = (self.s == self.L-1) or (self.t >= self.max_steps)
        r = 1.0 if self.s == self.L-1 else 0.0
        return self.s, r, done, {}

# ---- Policy and helpers ----
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def sample_action(logits, rng):
    probs = softmax(logits)
    a = rng.choice(len(probs), p=probs)
    return a, probs

def grad_log_softmax(logits, action):
    probs = softmax(logits)
    g = -probs
    g[action] += 1.0
    return g

# ---- Algorithms ----
def run_reinforce(env, episodes=400, lr=0.1, gamma=0.99, seed=0):
    rng = np.random.RandomState(seed)
    # per-state logits (policy)
    logits = np.zeros((env.L, 2))
    returns = []
    for ep in range(episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        done = False
        while not done:
            a, _ = sample_action(logits[s], rng)
            s2, r, done, _ = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            s = s2
        # reward-to-go
        T = len(rewards)
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + gamma * running
            G[t] = running
        # policy update (batch / episodic)
        for t in range(T):
            s_t = states[t]; a_t = actions[t]; g_t = G[t]
            grad = grad_log_softmax(logits[s_t], a_t)
            logits[s_t] += lr * grad * g_t  # ascend
        returns.append(sum(rewards))
    return np.array(returns)

def run_actor_critic_td(env, episodes=400, lr_actor=0.05, lr_critic=0.1, gamma=0.99, seed=0):
    rng = np.random.RandomState(seed)
    logits = np.zeros((env.L, 2))
    V = np.zeros(env.L)  # state-value table (critic)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            a, _ = sample_action(logits[s], rng)
            s2, r, done, _ = env.step(a)
            ep_return += r
            # TD(0) critic update
            td_target = r + (0.0 if done else gamma * V[s2])
            delta = td_target - V[s]
            V[s] += lr_critic * delta
            # actor update using advantage = delta (one-step advantage)
            grad = grad_log_softmax(logits[s], a)
            logits[s] += lr_actor * grad * delta
            s = s2
        returns.append(ep_return)
    return np.array(returns), V

def run_n_step_ac(env, episodes=400, n=4, lr_actor=0.05, lr_critic=0.1, gamma=0.99, seed=0):
    rng = np.random.RandomState(seed)
    logits = np.zeros((env.L, 2))
    V = np.zeros(env.L)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        done = False
        ep_return = 0.0
        while not done:
            a, _ = sample_action(logits[s], rng)
            s2, r, done, _ = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            ep_return += r
            s = s2
        T = len(rewards)
        # compute n-step returns for each time step and update
        for t in range(T):
            # compute G_t^n: sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n V(s_{t+n})
            G = 0.0
            for k in range(n):
                if t + k < T:
                    G += (gamma**k) * rewards[t+k]
                else:
                    break
            if t + n < T:
                G += (gamma**n) * V[states[t+n]]
            else:
                # episode ended before t+n, no bootstrap
                pass
            s_t = states[t]; a_t = actions[t]
            adv = G - V[s_t]
            # critic regression towards G (simple gradient step)
            V[s_t] += lr_critic * (G - V[s_t])
            grad = grad_log_softmax(logits[s_t], a_t)
            logits[s_t] += lr_actor * grad * adv
        returns.append(ep_return)
    return np.array(returns), V

def run_gae(env, episodes=400, lr_actor=0.05, lr_critic=0.1, gamma=0.99, lam=0.95, seed=0):
    rng = np.random.RandomState(seed)
    logits = np.zeros((env.L, 2))
    V = np.zeros(env.L)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        done = False
        ep_return = 0.0
        while not done:
            a, _ = sample_action(logits[s], rng)
            s2, r, done, _ = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            ep_return += r
            s = s2
        T = len(rewards)
        # compute deltas and advantages
        deltas = np.zeros(T)
        for t in range(T):
            s_t = states[t]
            s_tp1 = states[t+1] if t+1 < T else None
            v_tp1 = 0.0 if (t+1==T) else V[s_tp1]
            deltas[t] = rewards[t] + gamma * v_tp1 - V[s_t]
        # GAE advantages
        A = np.zeros(T)
        adv = 0.0
        for t in reversed(range(T)):
            adv = deltas[t] + gamma * lam * adv
            A[t] = adv
        # update critic towards n-step/MC return (simple: use reward-to-go as target)
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + gamma * running
            G[t] = running
        # apply updates
        for t in range(T):
            s_t = states[t]; a_t = actions[t]
            # critic update towards G[t]
            V[s_t] += lr_critic * (G[t] - V[s_t])
            grad = grad_log_softmax(logits[s_t], a_t)
            logits[s_t] += lr_actor * grad * A[t]
        returns.append(ep_return)
    return np.array(returns), V

# ---- Run multiple seeds and aggregate ----
def multi_run(fn, runs=8, **kwargs):
    all_runs = []
    for i in range(runs):
        out = fn(seed=i, **kwargs)
        if isinstance(out, tuple):
            rets = out[0]
        else:
            rets = out
        all_runs.append(rets)
    arr = np.array(all_runs)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return mean, std, arr

env = ChainMDP(L=6, max_steps=10)
episodes = 600
runs = 8

mean_reinf, std_reinf, _ = multi_run(lambda **kw: run_reinforce(env, episodes=episodes, lr=0.08, gamma=0.99, **kw), runs=runs)
mean_ac, std_ac, _ = multi_run(lambda **kw: run_actor_critic_td(env, episodes=episodes, lr_actor=0.04, lr_critic=0.2, gamma=0.99, **kw)[0], runs=runs)
mean_n1, std_n1, _ = multi_run(lambda **kw: run_n_step_ac(env, episodes=episodes, n=1, lr_actor=0.04, lr_critic=0.2, gamma=0.99, **kw)[0], runs=runs)
mean_n4, std_n4, _ = multi_run(lambda **kw: run_n_step_ac(env, episodes=episodes, n=4, lr_actor=0.04, lr_critic=0.2, gamma=0.99, **kw)[0], runs=runs)
mean_gae, std_gae, _ = multi_run(lambda **kw: run_gae(env, episodes=episodes, lr_actor=0.04, lr_critic=0.2, gamma=0.99, lam=0.95, **kw)[0], runs=runs)

# smoothing: cumulative average to see stable performance
def cumavg(x):
    return np.cumsum(x) / (np.arange(len(x)) + 1)

plt.figure(figsize=(10,5))
plt.plot(cumavg(mean_reinf), label='REINFORCE (MC reward-to-go)')
plt.plot(cumavg(mean_ac), label='Actor-Critic TD(0) (1-step)')
plt.plot(cumavg(mean_n4), label='n-step AC (n=4)')
plt.plot(cumavg(mean_gae), label='GAE (lambda=0.95)')
plt.fill_between(np.arange(episodes), cumavg(mean_ac)-std_ac, cumavg(mean_ac)+std_ac, alpha=0.15)
plt.xlabel('Episode')
plt.ylabel('Cumulative average episodic return')
plt.title('Comparison: REINFORCE vs Actor-Critic variants\n(Chain MDP)')
plt.legend()
plt.savefig("RL_Berkeley/lec_6/plot/chain_compare.png")

# show a second figure focused on variance (raw episode returns with band)
plt.figure(figsize=(10,5))
plt.plot(mean_reinf, label='REINFORCE')
plt.plot(mean_ac, label='AC TD(0)')
plt.plot(mean_n4, label='n=4 AC')
plt.plot(mean_gae, label='GAE λ=0.95')
plt.fill_between(np.arange(episodes), mean_reinf-std_reinf, mean_reinf+std_reinf, alpha=0.12)
plt.xlabel('Episode')
plt.ylabel('Episodic return')
plt.title('Raw episodic returns (mean ± std across runs)')
plt.legend()
plt.savefig("RL_Berkeley/lec_6/plot/chain_returns.png")



