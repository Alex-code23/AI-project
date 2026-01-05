import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# 1. Préparation des données (Digits Dataset)
# ID: Classes 0-4 (Chiffres 0 à 4)
# OOD Train (Auxiliaire pour Energy Bounded): Classes 5-7
# OOD Test (Vraiment inconnu): Classes 8-9

digits = load_digits()
X, y = digits.data, digits.target

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Création des masques
mask_id = y < 5
mask_ood_aux = (y >= 5) & (y < 8)
mask_ood_test = y >= 8

X_id, y_id = X[mask_id], y[mask_id]
X_ood_aux = X[mask_ood_aux] # Pas d'étiquettes utilisées pour l'entrainement OOD, juste les données
X_ood_test = X[mask_ood_test]

# Split Train/Test pour ID
X_train_id, X_test_id, y_train_id, y_test_id = train_test_split(X_id, y_id, test_size=0.2, random_state=42)

# 2. Implémentation MLP Simple avec NumPy (Pour pouvoir personnaliser la Loss)
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.logits = np.dot(self.a1, self.W2) + self.b2
        return self.logits

    def softmax(self, logits):
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def compute_energy(self, logits, T=1.0):
        # E(x) = -T * log(sum(exp(f(x)/T)))
        return -T * np.log(np.sum(np.exp(logits/T), axis=1))

    def train_step_classic(self, X, y_idx):
        # Forward
        logits = self.forward(X)
        probs = self.softmax(logits)
        
        # Cross Entropy Loss Gradient
        m = X.shape[0]
        grad_logits = probs
        grad_logits[range(m), y_idx] -= 1
        grad_logits /= m

        # Backprop
        self._backprop(X, grad_logits)
        
        # Loss value (for monitoring)
        log_likelihood = -np.log(probs[range(m), y_idx])
        loss = np.sum(log_likelihood) / m
        return loss

    def train_step_energy_bounded(self, X_in, y_idx, X_out, m_in, m_out, weight_energy=0.1):
        # Forward ID
        logits_in = self.forward(X_in)
        probs_in = self.softmax(logits_in)
        
        # Forward OOD (Auxiliary)
        # Need to re-compute forward for OOD specifically to get activations
        # We'll do a combined forward or separate. Separate is easier for grad handling in this simple implementation.
        # But to update weights, we need to accumulate gradients.
        
        # 1. Gradients from Cross Entropy on ID
        m = X_in.shape[0]
        grad_logits_ce = probs_in.copy()
        grad_logits_ce[range(m), y_idx] -= 1
        grad_logits_ce /= m
        
        # 2. Gradients from Energy Loss
        # L_energy = E[(max(0, E_in - m_in))^2] + E[(max(0, m_out - E_out))^2]
        
        # Energy ID
        E_in = self.compute_energy(logits_in)
        diff_in = E_in - m_in
        mask_in = diff_in > 0
        loss_energy_in = np.mean(np.square(np.maximum(0, diff_in)))
        
        # dL/dE_in = 2 * (E_in - m_in) if > 0 else 0
        d_loss_d_E_in = 2 * diff_in * mask_in / m
        # dE/df = -softmax(f)
        d_E_in_d_logits = -probs_in
        
        # Chain rule: dL/df = dL/dE * dE/df
        grad_logits_energy_in = d_loss_d_E_in[:, np.newaxis] * d_E_in_d_logits
        
        # Energy OOD
        # We need separate forward pass for OOD to get their logits
        z1_out = np.dot(X_out, self.W1) + self.b1
        a1_out = np.maximum(0, z1_out)
        logits_out = np.dot(a1_out, self.W2) + self.b2
        probs_out = self.softmax(logits_out)
        E_out = self.compute_energy(logits_out)
        
        m_out_curr = X_out.shape[0]
        diff_out = m_out - E_out
        mask_out = diff_out > 0
        loss_energy_out = np.mean(np.square(np.maximum(0, diff_out)))
        
        # dL/dE_out = -2 * (m_out - E_out) if > 0 else 0   <-- Derivative of (m-E)^2 wrt E is 2(m-E)*(-1)
        d_loss_d_E_out = -2 * diff_out * mask_out / m_out_curr
        d_E_out_d_logits = -probs_out
        grad_logits_energy_out = d_loss_d_E_out[:, np.newaxis] * d_E_out_d_logits
        
        # Combine gradients
        # For ID data: CE + weight * EnergyIn
        total_grad_logits_in = grad_logits_ce + weight_energy * grad_logits_energy_in
        
        # For OOD data: weight * EnergyOut
        total_grad_logits_out = weight_energy * grad_logits_energy_out
        
        # Backprop accumulate
        # We need to manually do backprop for both and sum grads
        
        # Backprop ID
        d_W2_in = np.dot(self.a1.T, total_grad_logits_in)
        d_b2_in = np.sum(total_grad_logits_in, axis=0, keepdims=True)
        d_a1_in = np.dot(total_grad_logits_in, self.W2.T)
        d_z1_in = d_a1_in * (self.z1 > 0)
        d_W1_in = np.dot(X_in.T, d_z1_in)
        d_b1_in = np.sum(d_z1_in, axis=0, keepdims=True)
        
        # Backprop OOD
        d_W2_out = np.dot(a1_out.T, total_grad_logits_out)
        d_b2_out = np.sum(total_grad_logits_out, axis=0, keepdims=True)
        d_a1_out = np.dot(total_grad_logits_out, self.W2.T)
        d_z1_out = d_a1_out * (z1_out > 0)
        d_W1_out = np.dot(X_out.T, d_z1_out)
        d_b1_out = np.sum(d_z1_out, axis=0, keepdims=True)
        
        # Update
        self.W2 -= self.lr * (d_W2_in + d_W2_out)
        self.b2 -= self.lr * (d_b2_in + d_b2_out)
        self.W1 -= self.lr * (d_W1_in + d_W1_out)
        self.b1 -= self.lr * (d_b1_in + d_b1_out)
        
        return loss_energy_in + loss_energy_out

    def _backprop(self, X, grad_logits):
        d_W2 = np.dot(self.a1.T, grad_logits)
        d_b2 = np.sum(grad_logits, axis=0, keepdims=True)
        d_a1 = np.dot(grad_logits, self.W2.T)
        d_z1 = d_a1 * (self.z1 > 0)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

# 3. Entraînement
# Modèle Classique
mlp_classic = MLP(input_dim=64, hidden_dim=32, output_dim=5, lr=0.01) # 5 classes (0-4)
epochs = 50
batch_size = 32

# Training loop classic
loss_history_classic = []
for epoch in range(epochs):
    perm = np.random.permutation(len(X_train_id))
    for i in range(0, len(X_train_id), batch_size):
        idx = perm[i:i+batch_size]
        X_batch = X_train_id[idx]
        y_batch = y_train_id[idx]
        loss = mlp_classic.train_step_classic(X_batch, y_batch)
    loss_history_classic.append(loss)

# Fine-tuning Energy Bounded
# On part d'une copie du modèle classique
import copy
mlp_energy = copy.deepcopy(mlp_classic)
# Paramètres de marge (basés sur les énergies observées du modèle classique)
# Calculons l'énergie moyenne sur le train ID pour calibrer m_in
logits_train = mlp_classic.forward(X_train_id)
energies_train = mlp_classic.compute_energy(logits_train)
m_in = np.percentile(energies_train, 80) # On veut que E < m_in (un peu au dessus de la moyenne)
m_out = m_in - 5 # On veut que E_out > m_out. Note: Energie = -logsumexp. Plus confiant = Energie basse (négative ou petite). 
# Attn: L'article utilise Negative Energy pour le score (-E).
# L'article dit: "lower for observed data and higher for unobserved ones" (Section 1).
# Donc E_in doit être BAS, E_out doit être HAUT.
# m_in : seuil supérieur pour ID. Penalité si E_in > m_in.
# m_out : seuil inférieur pour OOD. Penalité si E_out < m_out.
# On veut E_in < m_in et E_out > m_out.
# Donc il faut un "gap". Disons m_in = -20, m_out = -5.

# Calibration simple
mean_E_in = np.mean(energies_train)
m_in = mean_E_in - 2 # Un peu plus bas que la moyenne pour forcer à baisser encore (ou proche)
m_out = mean_E_in + 5 # Plus haut pour forcer les OOD à monter

print(f"Marges calibrées : m_in={m_in:.2f}, m_out={m_out:.2f}")

# Training loop Energy Fine-tuning
for epoch in range(20): # Moins d'époques pour le fine-tuning
    perm = np.random.permutation(len(X_train_id))
    perm_ood = np.random.permutation(len(X_ood_aux))
    
    # On itère sur les batchs
    num_batches = min(len(X_train_id), len(X_ood_aux)) // batch_size
    for i in range(num_batches):
        idx = perm[i*batch_size : (i+1)*batch_size]
        idx_ood = perm_ood[i*batch_size : (i+1)*batch_size]
        
        X_batch = X_train_id[idx]
        y_batch = y_train_id[idx]
        X_batch_ood = X_ood_aux[idx_ood]
        
        mlp_energy.train_step_energy_bounded(X_batch, y_batch, X_batch_ood, m_in, m_out, weight_energy=0.1)

# 4. Évaluation et Comparaison
# On évalue sur le Test set ID et le Test set OOD (qu'on n'a jamais vu)

def get_scores(model, X):
    logits = model.forward(X)
    return -model.compute_energy(logits) # Score = -Energie (Plus haut = In-distribution)

scores_id_classic = get_scores(mlp_classic, X_test_id)
scores_ood_classic = get_scores(mlp_classic, X_ood_test)

scores_id_energy = get_scores(mlp_energy, X_test_id)
scores_ood_energy = get_scores(mlp_energy, X_ood_test)

# AUROC
y_true = np.concatenate([np.ones(len(scores_id_classic)), np.zeros(len(scores_ood_classic))])
auroc_classic = roc_auc_score(y_true, np.concatenate([scores_id_classic, scores_ood_classic]))
auroc_energy = roc_auc_score(y_true, np.concatenate([scores_id_energy, scores_ood_energy]))

print(f"AUROC Classic: {auroc_classic:.4f}")
print(f"AUROC Energy Bounded: {auroc_energy:.4f}")

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(scores_id_classic, bins=30, alpha=0.5, label='ID (Test)', density=True)
plt.hist(scores_ood_classic, bins=30, alpha=0.5, label='OOD (Test)', density=True)
plt.title(f"Classic Model\nAUROC: {auroc_classic:.3f}")
plt.xlabel("Negative Energy Score")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(scores_id_energy, bins=30, alpha=0.5, label='ID (Test)', density=True)
plt.hist(scores_ood_energy, bins=30, alpha=0.5, label='OOD (Test)', density=True)
plt.title(f"Energy Bounded Model\nAUROC: {auroc_energy:.3f}")
plt.xlabel("Negative Energy Score")
plt.legend()

plt.tight_layout()
plt.savefig('THALES/plot_EB/comparison_energy.png')
print("Graphique sauvegardé sous comparison_energy.png")