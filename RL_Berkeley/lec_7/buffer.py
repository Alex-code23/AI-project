import numpy as np
import random
import matplotlib.pyplot as plt

# --- 1. ENVIRONNEMENT (SimpleGridWorld) ---
class SimpleGridWorld:
    def __init__(self):
        self.height = 4
        self.width = 4
        # S: Start, G: Goal (+1), H: Hole (-1), .: Empty, #: Wall
        self.grid = [
            ["S", ".", ".", "."],
            [".", "#", ".", "-"],
            [".", "#", ".", "."],
            [".", ".", ".", "+"]
        ]
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_map = {
            'UP': (-1, 0), 'DOWN': (1, 0),
            'LEFT': (0, -1), 'RIGHT': (0, 1)
        }

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action_idx):
        action_name = self.actions[action_idx]
        dy, dx = self.action_map[action_name]
        y, x = self.agent_pos
        
        ny, nx = y + dy, x + dx
        
        # Gestion des limites et des murs
        if ny < 0 or ny >= self.height or nx < 0 or nx >= self.width:
            ny, nx = y, x
        elif self.grid[ny][nx] == '#':
            ny, nx = y, x

        self.agent_pos = (ny, nx)
        cell = self.grid[ny][nx]
        
        reward = -0.01 # Coût par étape pour encourager la rapidité
        done = False
        
        if cell == '+':
            reward = 1.0
            done = True
        elif cell == '-':
            reward = -1.0
            done = True
            
        return self.agent_pos, reward, done

# --- 2. FONCTIONS D'AIDE ---
def get_action(q_table, state, epsilon):
    y, x = state
    # Exploration Epsilon-Greedy
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[y, x])

# --- 3. AGENT SANS BUFFER (Online Q-Learning) ---
def train_no_buffer(episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    env = SimpleGridWorld()
    q_table = np.zeros((env.height, env.width, 4))
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = get_action(q_table, state, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # --- UPDATE IMMÉDIAT (Online) ---
            # On utilise uniquement l'échantillon qu'on vient de voir
            y, x = state
            ny, nx = next_state
            
            old_val = q_table[y, x, action]
            next_max = np.max(q_table[ny, nx]) if not done else 0.0
            target = reward + gamma * next_max
            
            # Mise à jour
            q_table[y, x, action] = old_val + alpha * (target - old_val)
            
            state = next_state
            
        rewards_history.append(total_reward)
    return rewards_history

# --- 4. AGENT AVEC BUFFER (Experience Replay) ---
def train_with_buffer(episodes, alpha=0.1, gamma=0.99, epsilon=0.1, buffer_size=1000, batch_size=10):
    env = SimpleGridWorld()
    q_table = np.zeros((env.height, env.width, 4))
    rewards_history = []
    
    # Le Buffer de rejeu
    buffer = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = get_action(q_table, state, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 1. Stockage dans le buffer
            buffer.append((state, action, reward, next_state, done))
            if len(buffer) > buffer_size:
                buffer.pop(0) # On retire les plus vieux si le buffer est plein
            
            # 2. Apprentissage par Batch (si on a assez de données)
            if len(buffer) >= batch_size:
                # Échantillonnage aléatoire (casse les corrélations temporelles)
                mini_batch = random.sample(buffer, batch_size)
                
                for (b_state, b_action, b_reward, b_next_state, b_done) in mini_batch:
                    by, bx = b_state
                    bny, bnx = b_next_state
                    
                    old_val = q_table[by, bx, b_action]
                    next_max = np.max(q_table[bny, bnx]) if not b_done else 0.0
                    target = b_reward + gamma * next_max
                    
                    q_table[by, bx, b_action] = old_val + alpha * (target - old_val)
            
            state = next_state
            
        rewards_history.append(total_reward)
    return rewards_history

# --- 5. LANCEUR ET VISUALISATION ---
if __name__ == "__main__":
    episodes = 200
    
    print("Entraînement Sans Buffer...")
    res_no_buffer = train_no_buffer(episodes)
    
    print("Entraînement Avec Buffer...")
    res_buffer = train_with_buffer(episodes)

    # Lissage pour le graphique (moyenne mobile)
    window = 10
    def smooth(data):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(smooth(res_no_buffer), label='Sans Buffer (Online)', color='red', alpha=0.7)
    plt.plot(smooth(res_buffer), label='Avec Buffer (Replay)', color='blue', alpha=0.7)
    
    plt.xlabel('Épisodes')
    plt.ylabel('Récompense Cumulée (Lissée)')
    plt.title('Performance: Q-Learning Online vs Experience Replay')
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde ou affichage
    plt.savefig('RL_Berkeley/lec_7/plot/comparison_buffer.png')
 