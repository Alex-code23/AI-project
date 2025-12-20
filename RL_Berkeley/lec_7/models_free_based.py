import numpy as np
import random
import matplotlib.pyplot as plt

# --- 1. L'ENVIRONNEMENT (Large GridWorld 7x7) ---
class LargeGridWorld:
    def __init__(self):
        self.h, self.w = 7, 7
        # S: Start, G: Goal (+1), #: Wall, .: Empty, -: Trap (-1)
        self.grid = [
            ["S", ".", ".", ".", ".", ".", "."],
            [".", "#", "#", "#", ".", "#", "."],
            [".", ".", ".", ".", ".", "#", "."],
            [".", "#", "#", "#", ".", ".", "."],
            [".", ".", ".", "#", "#", "#", "."],
            [".", "#", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", "G"]
        ]
        self.start = (0, 0)
        self.pos = self.start
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_map = {'UP': (-1,0), 'DOWN': (1,0), 'LEFT': (0,-1), 'RIGHT': (0,1)}
        self.n_states = self.h * self.w
        self.n_actions = 4

    def reset(self):
        self.pos = self.start
        return self._get_state_idx(self.pos)

    def _get_state_idx(self, pos):
        return pos[0] * self.w + pos[1]

    def step(self, action_idx):
        y, x = self.pos
        move = self.actions[action_idx]
        dy, dx = self.action_map[move]
        
        ny, nx = y + dy, x + dx
        
        # Murs et limites
        if ny < 0 or ny >= self.h or nx < 0 or nx >= self.w:
            ny, nx = y, x
        elif self.grid[ny][nx] == '#':
            ny, nx = y, x
            
        self.pos = (ny, nx)
        cell = self.grid[ny][nx]
        
        # Récompense un peu plus punitive pour le temps passé
        reward = -0.01 
        done = False
        
        if cell == 'G':
            reward = 1.0
            done = True
        elif cell == '-': # Optionnel : des pièges
            reward = -1.0
            done = True
            
        return self._get_state_idx(self.pos), reward, done

# --- 2. AGENT MODEL-FREE (Q-Learning) ---
def train_model_free(episodes, alpha=0.5, gamma=0.95, epsilon=0.1):
    env = LargeGridWorld()
    q_table = np.zeros((env.n_states, env.n_actions))
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        steps = 0
        
        while not done and steps < 200: # Limite de pas pour éviter les boucles infinies au début
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, done = env.step(action)
            
            # Q-Learning update
            target = reward + gamma * (np.max(q_table[next_state]) if not done else 0)
            q_table[state, action] += alpha * (target - q_table[state, action])
            
            state = next_state
            total_r += reward
            steps += 1
            
        rewards.append(total_r)
    return rewards

# --- 3. AGENT MODEL-BASED (Simple Dyna) ---
def train_model_based(episodes, gamma=0.95, epsilon=0.1, planning_steps=20):
    env = LargeGridWorld()
    
    # Modèle du monde
    model_trans_counts = np.zeros((env.n_states, env.n_actions, env.n_states))
    model_reward_sum = np.zeros((env.n_states, env.n_actions))
    model_sa_counts = np.zeros((env.n_states, env.n_actions))
    
    # V-table pour la planification
    V = np.zeros(env.n_states)
    rewards_history = []
    
    # On suit les états visités pour ne planifier que sur ce qu'on connait (optimisation)
    visited_states = set()

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        steps = 0
        
        while not done and steps < 200:
            visited_states.add(state)
            
            # --- A. Sélection d'action (basée sur V actuel) ---
            q_vals = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                if model_sa_counts[state, a] > 0:
                    r_est = model_reward_sum[state, a] / model_sa_counts[state, a]
                    p_trans = model_trans_counts[state, a] / model_sa_counts[state, a]
                    val_next = np.sum(p_trans * V)
                    q_vals[a] = r_est + gamma * val_next
                else:
                    q_vals[a] = 1.0 # Optimism in the face of uncertainty (encourager l'exploration)

            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_vals)

            # --- B. Action réelle ---
            next_state, reward, done = env.step(action)
            total_r += reward
            steps += 1

            # --- C. Mise à jour du Modèle ---
            model_trans_counts[state, action, next_state] += 1
            model_reward_sum[state, action] += reward
            model_sa_counts[state, action] += 1
            
            state = next_state

            # --- D. Planification (Value Iteration sur le modèle) ---
            # On simule la propagation de la valeur dans notre tête
            for _ in range(planning_steps):
                # On met à jour un échantillon d'états visités pour aller vite
                # Dans un vrai Dyna-Q, on prendrait des (s,a) au hasard dans le buffer
                states_to_update = list(visited_states) 
                random.shuffle(states_to_update)
                
                for s in states_to_update[:30]: # On en met à jour 30 par step de planning
                    best_val = -float('inf')
                    for a in range(env.n_actions):
                        if model_sa_counts[s, a] > 0:
                            count = model_sa_counts[s, a]
                            r_hat = model_reward_sum[s, a] / count
                            v_next = np.sum((model_trans_counts[s, a] / count) * V)
                            val = r_hat + gamma * v_next
                        else:
                            val = 1.0 # Optimisme
                        
                        if val > best_val:
                            best_val = val
                    V[s] = best_val

        rewards_history.append(total_r)
    return rewards_history

# --- 4. COMPARAISON ---
if __name__ == "__main__":
    # On augmente le nombre d'épisodes car la grille est plus grande
    n_episodes = 100 
    
    print("Entraînement Model-Free (Q-Learning)...")
    mf_rewards = train_model_free(n_episodes)

    print("Entraînement Model-Based (Learned Model)...")
    mb_rewards = train_model_based(n_episodes, planning_steps=30)

    # Lissage
    def smooth(data, k=10):
        return np.convolve(data, np.ones(k)/k, mode='same')

    plt.figure(figsize=(10, 6))
    plt.plot(smooth(mf_rewards), label='Model-Free (Q-Learning)', color='red', alpha=0.6)
    plt.plot(smooth(mb_rewards), label='Model-Based (Dyna)', color='green', linewidth=2)
    plt.xlabel('Épisodes')
    plt.ylabel('Récompense Cumulée')
    plt.title('Comparaison sur Large GridWorld (7x7)')
    plt.legend()
    plt.grid(True)
    plt.savefig('RL_Berkeley/lec_7/plot/comparison_large.png')

