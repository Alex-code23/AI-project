from matplotlib import pyplot as plt
import numpy as np

# ----- Environnement ligne -----
n_states = 10
n_actions = 3  # 0=gauche, 1=sur place, 2=droite
objectif_state = 5

def step(state, action):
    if action == 0: # 0 means left
        next_state = max(0, state - 1)
        if state - 1 < 0:
            return next_state, -10, True
    elif action == 1: # 1 means  sur place
        next_state = state
    else: # 2 means right
        next_state = min(n_states - 1, state + 1)
        if state + 1 >= n_states:
            return next_state, -10, True

    done = (next_state == objectif_state) and state == next_state # objectif et reste sur place
    reward = 25 if done else -0.1

    # if done:
    #     print(state, next_state, objectif_state)
    return next_state, reward, done

# ----- Politique -----
def softmax(logits):
    """Ici, la politique est softmax.
    Evite overflow"""
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


# ----- REINFORCE -----
np.random.seed(0)
theta = np.zeros((n_states, n_actions))    # logits par (état, action)
alpha = 0.1
gamma = 0.99
n_episodes = 2000

print("Paramètres initiaux (theta) :")
print(theta)

# Storage
record_theta = np.zeros((n_episodes, n_states, n_actions))
record_probs  = np.zeros((n_episodes, n_states, n_actions))
record_returns = np.zeros(n_episodes)
record_lengths = np.zeros(n_episodes)

for ep in range(n_episodes):
    # reset
    s = np.random.randint(n_states)       # we begin at state 0
    traj_states = []
    traj_actions = []
    traj_probs = []
    traj_rewards = []
    done = False

    # Génération d'un épisode
    idx = 0
    while not done:
        probs = softmax(theta[s])
        a = np.random.choice(n_actions, p=probs)        # we choose action depends on probs
        s_next, r, done = step(s, a)                # we determine next step

        if idx >= 50:
            r = -200
            done = True

        traj_states.append(s)
        traj_actions.append(a)
        traj_probs.append(probs)
        traj_rewards.append(r)

        s = s_next
        # print(f"ep={ep}, action={a}, state={s}")
        idx += 1

    # Retours G_t
    G = 0.0
    returns = []
    # On calcule la récompense a chaque moment
    for r in reversed(traj_rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    total_return = sum(traj_rewards)  # in this env it's 1 if reached, else 0
    record_returns[ep] = total_return
    record_lengths[ep] = len(traj_rewards)

    # Normalisation pour réduire la variance
    R = np.array(returns)
    if R.std() > 1e-8:
        R = (R - R.mean()) / (R.std() + 1e-8)


    # Mise à jour paramètre par pas de temps
    for s, a, probs, Gt in zip(traj_states, traj_actions, traj_probs, R):
        one_hot = np.zeros(n_actions)
        one_hot[a] = 1.0                        # sert à représenter l’action choisie comme un vecteur binaire
        grad_logpi = one_hot - probs            # ∇ log π
        """
        one_hot = [0.0, 1.0, 0.0]
        grad_logpi = one_hot - probs = [ -0.2, 0.2 ]
        Ce vecteur [ -0.2, 0.2 ] dit :
        baisser un peu le logit de gauche (car elle n'a pas été choisie)
        augmenter celui de droite (car elle a été choisie et a rapporté 
        """
        theta[s] += alpha * Gt * grad_logpi     # ascent sur J

    # Record diagnostics
    record_theta[ep] = theta.copy()
    record_probs[ep] = np.array([softmax(theta[s]) for s in range(n_states)])



# Affichage
print("Paramètres finaux (theta) :")
print(theta)
print("Politique apprise (probas gauche/droite) :")
for s in range(n_states):
    print(f"État {s} → {softmax(theta[s])}")


# ---- Utilities: moving averages for smoothing ----
def moving_average(x, w=25):
    if len(x) < w:
        return np.convolve(x, np.ones(len(x))/len(x), mode='same')
    return np.convolve(x, np.ones(w)/w, mode='same')

# ---- Utilities: moving averages + confidence interval ----
def moving_average_with_ci(x, w=25, ci=2.56):
    """Retourne la moyenne glissante et l'intervalle de confiance à 95%"""
    ma = []
    ci_low, ci_high = [], []
    for i in range(len(x)):
        if i < w:
            window = x[:i+1]
        else:
            window = x[i-w+1:i+1]
        mean = np.mean(window)
        std = np.std(window)
        # erreur standard = std/sqrt(n)
        sem = std / np.sqrt(len(window))
        ma.append(mean)
        ci_low.append(mean - ci * sem)
        ci_high.append(mean + ci * sem)
    return np.array(ma), np.array(ci_low), np.array(ci_high)

ma_returns = moving_average(record_returns, w=25)
ma_lengths = moving_average(record_lengths, w=25)

fig, axs = plt.subplots(4, 2, figsize=(18, 8))
(ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8) = axs

# Plot 1: Return per episode (top-left)
ax1.plot(record_returns, label='Return per episode', linewidth=0.8)
ax1.plot(ma_returns, label='Moving average (w=25)', linewidth=2)
ax1.set_xlabel('Episode', fontsize=5)
ax1.set_ylabel('Total return', fontsize=5)
ax1.set_title('Return evolution per episode', fontsize=13)
ax1.legend(fontsize=5)
ax1.grid(True)

# Plot 2: Episode length per episode (top-right)
# ax2.plot(record_lengths, label='Episode length', linewidth=0.8)
# ax2.plot(ma_lengths, label='Moving average length (w=25)', linewidth=2)
# ax2.set_xlabel('Episode', fontsize=5)
# ax2.set_ylabel('Episode length (timesteps)', fontsize=5)
# ax2.set_title('Episode length evolution', fontsize=13)
# ax2.legend(fontsize=5)
# ax2.grid(True)

# ---- Moving average + confidence interval pour record_lengths ----
ma_lengths, ci_low, ci_high = moving_average_with_ci(record_lengths, w=55)
ax2.plot(record_lengths, label='Episode length', linewidth=0.8, alpha=0.1)
ax2.plot(ma_lengths, label='Moving average length (w=25)', linewidth=2, color='C1')
ax2.fill_between(range(len(ma_lengths)), ci_low, ci_high, color='C2', alpha=0.6, label='95% CI')
ax2.set_xlabel('Episode', fontsize=5)
ax2.set_ylabel('Episode length (timesteps)', fontsize=5)
ax2.set_title('Episode length evolution', fontsize=13)
ax2.legend(fontsize=5)
ax2.grid(True)

# Plot 3: Theta logits for action "stay put" by state (bottom-left)
for s in range(n_states):
    ax3.plot(record_theta[:, s, 1], label=f'State {s}')
ax3.set_xlabel('Episode', fontsize=5)
ax3.set_ylabel('Theta (logit for "stay pu")', fontsize=5)
ax3.set_title('Theta logits for "stay pu" (one line per state)', fontsize=13)
ax3.legend(ncol=2, fontsize=5)
ax3.grid(True)

# Plot 4: Probability of choosing "stay put" by state (bottom-right)
# for s in range(n_states):
#     ax4.plot(record_probs[:, s, 1], label=f'State {s}')
# ax4.set_xlabel('Episode', fontsize=5)
# ax4.set_ylabel('P(stay put)', fontsize=5)
# ax4.set_title('Probability of choosing "stay put" by state', fontsize=13)
# ax4.legend(ncol=2, fontsize=5)
# ax4.grid(True)

# Plot avec bande de confiance 
for s in range(n_states):
    ma, ci_low, ci_high = moving_average_with_ci(record_probs[:, s, 1], w=55)
    ax4.plot(ma, label=f'State {s}')  # moyenne glissante
    ax4.fill_between(range(len(ma)), ci_low, ci_high, alpha=0.8)  # bande CI
ax4.set_xlabel('Episode', fontsize=5)
ax4.set_ylabel('P(stay put)', fontsize=5)
ax4.set_title('Probability of choosing "stay put" by state', fontsize=13)
ax4.legend(ncol=2, fontsize=5)
ax4.grid(True)



# Plot 5: Theta logits for action "left" by state (bottom-left)
for s in range(n_states):
    ax5.plot(record_theta[:, s, 0], label=f'State {s}')
ax5.set_xlabel('Episode', fontsize=5)
ax5.set_ylabel('Theta (logit for "left")', fontsize=5)
ax5.set_title('Theta logits for "left" (one line per state)', fontsize=13)
ax5.legend(ncol=2, fontsize=5)
ax5.grid(True)

# Plot 6: Probability of choosing "left" by state (bottom-right)
for s in range(n_states):
    ax6.plot(record_probs[:, s, 0], label=f'State {s}')
ax6.set_xlabel('Episode', fontsize=5)
ax6.set_ylabel('P(left)', fontsize=5)
ax6.set_title('Probability of choosing "left" by state', fontsize=13)
ax6.legend(ncol=2, fontsize=5)
ax6.grid(True)

# Plot 7: Theta logits for action "right" by state (bottom-left)
for s in range(n_states):
    ax7.plot(record_theta[:, s, 2], label=f'State {s}')
ax7.set_xlabel('Episode', fontsize=5)
ax7.set_ylabel('Theta (logit for "right")', fontsize=5)
ax7.set_title('Theta logits for "right" (one line per state)', fontsize=13)
ax7.legend(ncol=2, fontsize=5)
ax7.grid(True)

# Plot 8: Probability of choosing "right" by state (bottom-right)
for s in range(n_states):
    ax8.plot(record_probs[:, s, 2], label=f'State {s}')
ax8.set_xlabel('Episode', fontsize=5)
ax8.set_ylabel('P(right)', fontsize=5)
ax8.set_title('Probability of choosing "right" by state', fontsize=13)
ax8.legend(ncol=2, fontsize=5)
ax8.grid(True)

fig.suptitle("REINFOCEMENT policy algo")
plt.tight_layout()
plt.show()

