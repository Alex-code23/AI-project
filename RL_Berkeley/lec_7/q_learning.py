import numpy as np
import random
import time

# --- 1. L'ENVIRONNEMENT (Simple GridWorld) ---
# S: Start, G: Goal (+1), H: Hole (-1), .: Empty (0)
class SimpleGridWorld:
    def __init__(self):
        self.height = 3
        self.width = 4
        self.grid = [
            ["S", ".", ".", "G"],
            [".", "#", ".", "H"],
            [".", ".", ".", "."]
        ]
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        # Mapping actions to movements (dy, dx)
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
        
        # Calcul nouvelle position
        ny, nx = y + dy, x + dx
        
        # Vérifier les murs (limites de la grille)
        if ny < 0 or ny >= self.height or nx < 0 or nx >= self.width:
            ny, nx = y, x # Reste sur place
        
        # Vérifier le mur obstacle '#'
        if self.grid[ny][nx] == '#':
            ny, nx = y, x

        self.agent_pos = (ny, nx)
        cell = self.grid[ny][nx]
        
        reward = 0
        done = False
        
        if cell == 'G':
            reward = 1
            done = True
        elif cell == 'H':
            reward = -1
            done = True
        else:
            reward = -0.01 # Petite pénalité pour encourager le chemin court
            
        return self.agent_pos, reward, done

# --- 2. L'AGENT ET L'ALGORITHME (Q-Learning) ---

def train_q_learning(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    env = SimpleGridWorld()
    # Initialisation de la Q-Table (State x Action)
    # 3x4 états, 4 actions
    q_table = np.zeros((env.height, env.width, len(env.actions)))

    print("--- Début de l'entraînement (Q-Learning) ---")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            y, x = state
            
            # --- Exploration vs Exploitation (Slide 18) ---
            # "Epsilon-greedy" policy 
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3) # Exploration
            else:
                action = np.argmax(q_table[y, x]) # Exploitation (arg max Q)

            # --- Exécution de l'action (Sampling) ---
            # "take some action a_i and observe (s_i, a_i, s_i', r_i)" [cite: 223]
            next_state, reward, done = env.step(action)
            ny, nx = next_state
            
            # --- Mise à jour Q-Learning (Slide 17) ---
            # Formule: Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
            # Target y_i = r + gamma * max_a' Q(s', a') 
            
            old_value = q_table[y, x, action]
            next_max = np.max(q_table[ny, nx])
            
            # Note: Si c'est l'état terminal, il n'y a pas de next_max (valeur future = 0)
            target = reward + (gamma * next_max if not done else 0)
            
            # Update rule [cite: 225]
            new_value = old_value + alpha * (target - old_value)
            q_table[y, x, action] = new_value
            
            state = next_state

    print("--- Entraînement terminé ---")
    return q_table

# --- 3. VISUALISATION ---

def print_policy(q_table):
    """ Affiche la meilleure action à prendre pour chaque case """
    actions_symbols = ['^', 'v', '<', '>']
    h, w, _ = q_table.shape
    
    print("\nPolitique apprise (Meilleure action par état):")
    print("S = Start, T = Trap, G = Goal, # = Wall")
    print("-" * 20)
    
    grid_vis = [
        ["S", " ", " ", "G"],
        [" ", "#", " ", "T"],
        [" ", " ", " ", " "]
    ]

    for y in range(h):
        row_str = "|"
        for x in range(w):
            if grid_vis[y][x] in ['G', 'T', '#']:
                row_str += f" {grid_vis[y][x]} |"
            else:
                best_action_idx = np.argmax(q_table[y, x])
                symbol = actions_symbols[best_action_idx]
                row_str += f" {symbol} |"
        print(row_str)
    print("-" * 20)

def print_values(q_table):
    """ Affiche la Value Function V(s) = max_a Q(s,a) """
    h, w, _ = q_table.shape
    print("\nFonction de Valeur V(s) (approx):")
    for y in range(h):
        row_vals = []
        for x in range(w):
            v_s = np.max(q_table[y, x])
            row_vals.append(f"{v_s:5.2f}")
        print(" | ".join(row_vals))

# --- MAIN ---
if __name__ == "__main__":
    # Lancement de l'apprentissage
    q_table_trained = train_q_learning(episodes=1000)
    
    # Affichage des résultats
    print_values(q_table_trained)
    print_policy(q_table_trained)