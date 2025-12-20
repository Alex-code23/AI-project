import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ------------------ Helper functions ------------------
def softmax_logits_to_probs(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1, keepdims=True)

def sample_action_from_logits(logits):
    probs = softmax_logits_to_probs(logits)
    return np.random.choice(len(probs), p=probs), probs

# ------------------ Experiment 1: 2-armed Bernoulli bandit ------------------
def run_bandit_reinforce(n_steps=2000, lr=0.1, baseline=False, seed=0):
    rng = np.random.RandomState(seed)
    # True reward probs for the two arms
    true_p = np.array([0.3, 0.6])
    logits = np.zeros(2)  # policy logits (parameters)
    avg_rewards = []
    baseline_val = 0.0
    alpha_baseline = 0.01  # for running mean baseline
    for t in range(n_steps):
        # sample action
        action, probs = sample_action_from_logits(logits)
        r = rng.rand() < true_p[action]
        # REINFORCE gradient estimate: grad log pi(a) * R
        # For softmax with logits, grad w.r.t logits = one-hot(a) - probs
        grad_log = np.zeros_like(logits)
        grad_log[action] = 1.0
        grad_log -= probs
        R = float(r)
        if baseline:
            # subtract baseline (running mean estimate)
            R = R - baseline_val
            baseline_val += alpha_baseline * (float(r) - baseline_val)
        logits += lr * grad_log * R
        if not baseline:
            # still track running mean for plotting convenience
            baseline_val += alpha_baseline * (float(r) - baseline_val)
        avg_rewards.append(float(r))
    return avg_rewards, logits

# run multiple seeds to show variance
n_runs = 10
steps = 2000
results_no_baseline = np.zeros((n_runs, steps))
results_baseline = np.zeros((n_runs, steps))

for i in range(n_runs):
    r_nb, _ = run_bandit_reinforce(n_steps=steps, lr=0.1, baseline=False, seed=i)
    r_b, _ = run_bandit_reinforce(n_steps=steps, lr=0.1, baseline=True, seed=i)
    results_no_baseline[i] = r_nb
    results_baseline[i] = r_b

# smoothing for plotting cumulative average reward
def cumulative_average(arr):
    return np.cumsum(arr) / (np.arange(1, arr.shape[-1]+1))

cumavg_no_baseline = np.array([cumulative_average(run) for run in results_no_baseline])
cumavg_baseline = np.array([cumulative_average(run) for run in results_baseline])

mean_no_baseline = cumavg_no_baseline.mean(axis=0)
std_no_baseline = cumavg_no_baseline.std(axis=0)
mean_baseline = cumavg_baseline.mean(axis=0)
std_baseline = cumavg_baseline.std(axis=0)

plt.figure(figsize=(8,4))
plt.plot(mean_no_baseline, label='REINFORCE (no baseline)')
plt.plot(mean_baseline, label='REINFORCE (running mean baseline)')
plt.fill_between(np.arange(steps), mean_no_baseline-std_no_baseline, mean_no_baseline+std_no_baseline, alpha=0.2)
plt.fill_between(np.arange(steps), mean_baseline-std_baseline, mean_baseline+std_baseline, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('Cumulative average reward')
plt.title('2-armed bandit: baseline reduces variance of learning curves')
plt.legend()
plt.savefig("RL_Berkeley/lec_5/plot/bandit_reinforce.png")

# ------------------ Experiment 2: Episodic chain MDP ------------------
# A simple chain: states 0..(L-1), agent starts at 0, goal at L-1. Actions: left=0, right=1.
# If reaches goal -> reward 1 and episode ends; else 0 reward. Episode length limited.
class ChainMDP:
    def __init__(self, L=5, max_steps=10):
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

def run_policy_gradient_chain(reward_to_go=True, n_episodes=2000, lr=0.1, seed=0):
    rng = np.random.RandomState(seed)
    env = ChainMDP(L=6, max_steps=10)
    # parameterize policy as logits per state for two actions
    logits_table = np.zeros((env.L, 2))  # learn per-state logits
    avg_returns = []
    for ep in range(n_episodes):
        # generate trajectory
        states = []
        actions = []
        rewards = []
        s = env.reset()
        done = False
        while not done:
            logits = logits_table[s]
            a, probs = sample_action_from_logits(logits)
            s2, r, done, _ = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s2
        T = len(rewards)
        # compute returns
        returns = np.zeros(T)
        if reward_to_go:
            for t in range(T):
                returns[t] = sum(rewards[t:])
        else:
            total = sum(rewards)
            returns[:] = total
        # policy gradient update: for each time step, grad log pi * G_t
        for t in range(T):
            s_t = states[t]
            a_t = actions[t]
            logits = logits_table[s_t]
            probs = softmax_logits_to_probs(logits)
            grad_log = np.zeros_like(logits)
            grad_log[a_t] = 1.0
            grad_log -= probs
            logits_table[s_t] += lr * grad_log * returns[t]
        avg_returns.append(sum(rewards))
    return avg_returns

# Run both variants
runs = 8
episodes = 800
rtg_runs = np.array([run_policy_gradient_chain(reward_to_go=True, n_episodes=episodes, lr=0.1, seed=i) for i in range(runs)])
full_runs = np.array([run_policy_gradient_chain(reward_to_go=False, n_episodes=episodes, lr=0.1, seed=i) for i in range(runs)])

mean_rtg = rtg_runs.mean(axis=0)
std_rtg = rtg_runs.std(axis=0)
mean_full = full_runs.mean(axis=0)
std_full = full_runs.std(axis=0)

plt.figure(figsize=(8,4))
plt.plot(mean_full, label='REINFORCE (full-trajectory return)')
plt.plot(mean_rtg, label='REINFORCE (reward-to-go)')
plt.fill_between(np.arange(episodes), mean_full-std_full, mean_full+std_full, alpha=0.2)
plt.fill_between(np.arange(episodes), mean_rtg-std_rtg, mean_rtg+std_rtg, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('Return (episodic)')
plt.title('Chain MDP: reward-to-go improves credit assignment and reduces variance')
plt.legend()
plt.savefig("RL_Berkeley/lec_5/plot/chain_reinforce.png")


# ------------------ Experiment 3: Importance Sampling demo ------------------
# Estimate expected reward of a target policy using samples from behavior policy.
# Simple one-step MDP for clarity (could be extended): action space 3 arms, reward = Bernoulli with arm-specific probs.
def importance_sampling_demo(n_samples=2000, seed=0):
    rng = np.random.RandomState(seed)
    true_p = np.array([0.2, 0.5, 0.8])
    # target and behavior logits
    logits_target = np.array([0.0, 0.0, 0.0])  # uniform target
    logits_behavior = np.array([1.0, -1.0, -2.0])  # biased behavior policy
    probs_target = softmax_logits_to_probs(logits_target)
    probs_behavior = softmax_logits_to_probs(logits_behavior)
    estimates_is = []
    estimates_naive = []
    estimates_self_normalized = []
    for i in range(n_samples):
        # sample according to behavior policy
        a = rng.choice(3, p=probs_behavior)
        r = float(rng.rand() < true_p[a])
        # importance weight = pi_target(a) / pi_behavior(a)
        w = probs_target[a] / probs_behavior[a]
        # unbiased IS estimate of expected reward (single-sample)
        estimates_is.append(w * r)
        estimates_naive.append(r)  # naive (biased) estimate if you pretend samples were from target
        # self-normalized IS estimate accumulates weights
        # here we compute running self-normalized estimate
        if i == 0:
            ws = w
            numer = w * r
        else:
            ws += w
            numer += w * r
        estimates_self_normalized.append(numer / ws)
    # convert to arrays and compute running mean estimates
    estimates_is = np.array(estimates_is)
    estimates_naive = np.array(estimates_naive)
    # build running averages
    running_is = np.cumsum(estimates_is) / (np.arange(1, n_samples+1))
    running_naive = np.cumsum(estimates_naive) / (np.arange(1, n_samples+1))
    running_sn = np.array(estimates_self_normalized)
    return running_is, running_naive, running_sn, probs_target, probs_behavior, true_p

running_is, running_naive, running_sn, p_t, p_b, true_p = importance_sampling_demo(n_samples=2000, seed=1)

plt.figure(figsize=(8,4))
plt.plot(running_naive, label='Naive (uses behavior as if target)')
plt.plot(running_is, label='Unbiased IS estimate (per-sample)')
plt.plot(running_sn, label='Self-normalized IS (running)')
# horizontal line: true expected reward under target policy
true_expected = (p_t * true_p).sum()
plt.axhline(true_expected, linestyle='--')
plt.xlabel('Samples from behavior')
plt.ylabel('Estimate of expected reward under target policy')
plt.title('Importance Sampling: unbiased but high variance; self-normalized reduces variance')
plt.legend()
plt.savefig("RL_Berkeley/lec_5/plot/importance_sampling.png")
