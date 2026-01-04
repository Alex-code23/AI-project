import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
torch.manual_seed(42)
np.random.seed(42)

# 1. Données : Digits (Images 8x8 de chiffres)
digits = load_digits()
X = digits.data
y = digits.target

# Normalisation et Split
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conversion en Tenseurs
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# 2. Modèle : MLP Simple
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(64, 64) # 8x8 pixels = 64 features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10) # 10 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Fonctions Utilitaires
def get_weights_vector(model):
    """Concatène tous les poids en un seul vecteur 1D."""
    vec = []
    for param in model.parameters():
        vec.append(param.view(-1))
    return torch.cat(vec)

def get_predictions(model, X):
    """Récupère les prédictions sur le test set."""
    model.eval()
    with torch.no_grad():
        output = model(X)
        preds = output.argmax(dim=1)
    return preds

# 4. Entraînement
def train_experiment(epochs=15, save_checkpoints=False):
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    saved_weights = []
    saved_preds = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        
        if save_checkpoints:
            saved_weights.append(get_weights_vector(model).clone().detach())
            saved_preds.append(get_predictions(model, X_test_t))
            
    final_weights = get_weights_vector(model).clone().detach()
    final_preds = get_predictions(model, X_test_t)
    
    if save_checkpoints:
        return saved_weights, saved_preds
    else:
        return final_weights, final_preds

# --- Exécution des Expériences ---

# Expérience A : Au sein d'une trajectoire (Checkpoints époque 1 à 15)
print("Expérience A : Analyse intra-trajectoire...")
traj_weights, traj_preds = train_experiment(epochs=30, save_checkpoints=True)

# Expérience B : Entre trajectoires (5 modèles indépendants)
print("Expérience B : Analyse inter-modèles...")
indep_weights = []
indep_preds = []
for i in range(10):
    w, p = train_experiment(epochs=30, save_checkpoints=False)
    indep_weights.append(w)
    indep_preds.append(p)

# --- Calcul des Matrices ---

def compute_cosine_matrix(weights_list):
    n = len(weights_list)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos = F.cosine_similarity(weights_list[i].unsqueeze(0), weights_list[j].unsqueeze(0))
            matrix[i, j] = cos.item()
    return matrix

def compute_disagreement_matrix(preds_list):
    n = len(preds_list)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Taux de désaccord = moyenne(pred_i != pred_j)
            diff = (preds_list[i] != preds_list[j]).float().mean()
            matrix[i, j] = diff.item()
    return matrix

# 1. Matrices Intra-Trajectoire
mat_sim_within = compute_cosine_matrix(traj_weights)
mat_dis_within = compute_disagreement_matrix(traj_preds)

# 2. Matrices Inter-Modèles
mat_sim_across = compute_cosine_matrix(indep_weights)
mat_dis_across = compute_disagreement_matrix(indep_preds)

# --- Affichage des Heatmaps ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot 1: Similarité Poids (Intra)
sns.heatmap(mat_sim_within, ax=axes[0, 0], cmap="viridis", annot=False, vmin=0, vmax=1)
axes[0, 0].set_title("Similarité Cosinus des Poids\n(Même Trajectoire : Epoch 1 -> 30)")
axes[0, 0].set_ylabel("Epoch")

# Plot 2: Désaccord (Intra)
sns.heatmap(mat_dis_within, ax=axes[0, 1], cmap="magma", annot=False, vmin=0, vmax=0.3)
axes[0, 1].set_title("Désaccord des Prédictions\n(Même Trajectoire : Epoch 1 -> 30)")

# Plot 3: Similarité Poids (Inter)
sns.heatmap(mat_sim_across, ax=axes[1, 0], cmap="viridis", annot=True, fmt=".2f", vmin=0, vmax=1)
axes[1, 0].set_title("Similarité Cosinus des Poids\n(5 Modèles Indépendants)")
axes[1, 0].set_xlabel("Model ID")
axes[1, 0].set_ylabel("Model ID")

# Plot 4: Désaccord (Inter)
sns.heatmap(mat_dis_across, ax=axes[1, 1], cmap="magma", annot=True, fmt=".2f", vmin=0, vmax=0.3)
axes[1, 1].set_title("Désaccord des Prédictions\n(5 Modèles Indépendants)")
axes[1, 1].set_xlabel("Model ID")

plt.tight_layout()
plt.show() # Affiche les heatmaps