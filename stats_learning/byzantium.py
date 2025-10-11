import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# Parameters
np.random.seed(1)
N = 31  # number of drones (odd helps for majority)
rounds = 10
trials = 50
true_value = 25.0  # true scalar observation (e.g., temperature)
obs_noise_sigma = 0.1  # sensor noise for honest drones
byzantine_behavior = "extreme"  # "random" or "extreme" (send far values)
byzantine_extreme_offset = 20.0

# Strategies implementations:
def strategy_no_filter(received_values, i, history):
    # simple average of received values
    return np.mean(received_values)

def strategy_median(received_values, i, history):
    return np.median(received_values)

def strategy_trimmed_mean(received_values, i, history, trim_frac=0.2):
    # remove top and bottom trim_frac proportion and average rest
    arr = np.sort(received_values)
    n = len(arr)
    k = int(np.floor(trim_frac * n))
    if 2*k >= n:
        return np.mean(arr)
    trimmed = arr[k:n-k]
    return np.mean(trimmed)

def strategy_reputation_weighted(received_values, i, history, reputations):
    # reputations is an array of length n giving weights for senders
    # received_values is array in same order as reputations
    weights = reputations + 1e-9
    return np.dot(received_values, weights) / np.sum(weights)

# Helper to simulate one trial
def run_trial(N, byzantine_indices, strategy_name):
    # initial observations: honest observe true + noise; byzantines may craft values
    # We'll run synchronous rounds where in each round everyone broadcasts current estimate
    # and updates using the chosen strategy.
    # Initialize estimates from initial observations
    estimates = np.zeros(N)
    is_byz = np.zeros(N, dtype=bool)
    is_byz[byzantine_indices] = True
    # initial observation
    for i in range(N):
        if is_byz[i]:
            if byzantine_behavior == "random":
                estimates[i] = true_value + np.random.randn()*10
            else:
                # extreme adversary sends a far away initial value
                estimates[i] = true_value + (np.random.choice([-1,1]) * byzantine_extreme_offset) + np.random.randn()*2
        else:
            estimates[i] = true_value + np.random.randn()*obs_noise_sigma

    # Keep history of estimates for reputation calculation
    history = [estimates.copy()]
    # reputations initial equal
    reputations = np.ones(N)

    # Function to get byzantine message given sender index and round
    def byzantine_message(sender, round_idx):
        # Can be more sophisticated; here some send random extremes, some try to imitate but slightly shift
        if byzantine_behavior == "random":
            return true_value + np.random.randn()*15
        else:
            # adversary sends extreme value away from true
            direction = -1 if (sender % 2 == 0) else 1
            return true_value + direction * byzantine_extreme_offset + np.random.randn()*2

    # run rounds
    estimates_over_rounds = [estimates.copy()]
    for r in range(rounds):
        # broadcast: each drone receives values from all (including itself)
        received_matrix = np.zeros((N, N))
        for sender in range(N):
            msg = byzantine_message(sender, r) if is_byz[sender] else estimates[sender]
            received_matrix[:, sender] = msg  # everyone receives same message from sender (fully connected)
        new_estimates = np.zeros(N)
        # compute reputations if needed (based on last round similarity)
        if strategy_name == "reputation":
            # update reputations: compare each sender's last message to median of others
            last_msgs = received_matrix[0, :]  # all rows identical here
            median_val = np.median(last_msgs[~is_byz]) if np.any(~is_byz) else np.median(last_msgs)
            # reputation proportional to closeness to median of honest messages (unknown in reality, but here we approximate using previous values)
            reputations = 1.0 / (1.0 + np.abs(last_msgs - median_val))
            reputations = reputations / np.mean(reputations)  # normalize around 1
        # Update each node according to strategy
        for i in range(N):
            received = received_matrix[i, :]  # values from all senders
            if is_byz[i]:
                # byzantine may update arbitrarily; we keep them sending their crafted values
                # but for bookkeeping, let them keep their last sent message
                new_estimates[i] = byzantine_message(i, r)
            else:
                if strategy_name == "no_filter":
                    new_estimates[i] = strategy_no_filter(received, i, history)
                elif strategy_name == "median":
                    new_estimates[i] = strategy_median(received, i, history)
                elif strategy_name == "trimmed":
                    new_estimates[i] = strategy_trimmed_mean(received, i, history, trim_frac=0.4)
                elif strategy_name == "reputation":
                    new_estimates[i] = strategy_reputation_weighted(received, i, history, reputations)
                else:
                    # fallback to mean
                    new_estimates[i] = np.mean(received)
        estimates = new_estimates
        history.append(estimates.copy())
        estimates_over_rounds.append(estimates.copy())
    return np.array(estimates_over_rounds), is_byz

# Run experiments varying fraction of Byzantines and strategies
fractions = np.linspace(0.0, 0.4, 9)  # 0% to 40%
strategies = ["no_filter", "median", "trimmed", "reputation"]
results = []

# We'll collect final RMSE (root mean squared error) among honest nodes for each strategy
for frac in fractions:
    f = float(frac)
    byz_count = int(np.floor(f * N))
    if byz_count == 0 and f>0:
        byz_count = 1
    for strat in strategies:
        rmse_list = []
        for t in range(trials):
            # choose random byzantine indices
            byz_indices = np.random.choice(N, size=byz_count, replace=False) if byz_count>0 else np.array([], dtype=int)
            est_rounds, is_byz = run_trial(N, byz_indices, strat)
            final_estimates = est_rounds[-1]
            # compute RMSE among honest nodes
            honest = ~is_byz
            rmse = np.sqrt(np.mean((final_estimates[honest] - true_value)**2)) if np.any(honest) else np.nan
            rmse_list.append(rmse)
        results.append({
            "byz_fraction": f,
            "byz_count": byz_count,
            "strategy": strat,
            "rmse_mean": np.nanmean(rmse_list),
            "rmse_std": np.nanstd(rmse_list)
        })

df_results = pd.DataFrame(results)

# Plot RMSE vs Byzantine fraction for each strategy
plt.figure(figsize=(8,5))
for strat in strategies:
    sub = df_results[df_results["strategy"]==strat]
    plt.errorbar(sub["byz_fraction"], sub["rmse_mean"], yerr=sub["rmse_std"], label=strat, marker='o')
plt.xlabel("Fraction de Byzantins")
plt.ylabel("RMSE finale (drones honnêtes)")
plt.title("Performance des stratégies face aux Byzantins")
plt.legend()
plt.grid(True)
plt.show()

# Convergence plot for a representative scenario (e.g., 20% Byzantines) - show mean across trials of honest RMSE per round
rep_frac = 0.1
byz_count = int(np.floor(rep_frac * N))
roundwise_rmse = {s: np.zeros(rounds+1) for s in strategies}
for strat in strategies:
    accum = np.zeros(rounds+1)
    for t in range(trials):
        byz_indices = np.random.choice(N, size=byz_count, replace=False) if byz_count>0 else np.array([], dtype=int)
        est_rounds, is_byz = run_trial(N, byz_indices, strat)
        # compute RMSE among honest at each round
        honest = ~is_byz
        per_round = np.sqrt(np.mean((est_rounds[:, honest] - true_value)**2, axis=1))
        accum += per_round
    roundwise_rmse[strat] = accum / trials

# Plot convergence
plt.figure(figsize=(8,5))
x = np.arange(rounds+1)
for strat in strategies:
    plt.plot(x, roundwise_rmse[strat], label=strat, marker='o')
plt.xlabel("Round")
plt.ylabel("RMSE (drones honnêtes)")
plt.title(f"Convergence des stratégies (fraction Byzantins={rep_frac})")
plt.legend()
plt.grid(True)
plt.show()

# Also show a density/histogram of final honest estimates for one scenario and strategies
plt.figure(figsize=(8,5))
byz_count = int(np.floor(0.2 * N))
final_vals = {}
for strat in strategies:
    vals = []
    for t in range(trials):
        byz_indices = np.random.choice(N, size=byz_count, replace=False) if byz_count>0 else np.array([], dtype=int)
        est_rounds, is_byz = run_trial(N, byz_indices, strat)
        honest = ~is_byz
        vals.extend(list(est_rounds[-1, honest]))
    final_vals[strat] = np.array(vals)

# Plot histograms in separate subplots (one per strategy) to obey "one chart at a time"
for strat in strategies:
    plt.figure(figsize=(6,3))
    plt.hist(final_vals[strat], bins=30, density=True)
    plt.axvline(true_value, linestyle='--')
    plt.title(f"Distribution des estimations finales (honnêtes) - {strat}")
    plt.xlabel("Estimation finale")
    plt.ylabel("Densité")
    plt.grid(True)
    plt.show()

print("Simulation terminée. Le tableau interactif 'Résultats - RMSE par stratégie et fraction Byzantines' contient les métriques numériques.")

