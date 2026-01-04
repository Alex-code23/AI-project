import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration pour reproductibilité
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. Génération du Dataset "Toy"
# ==========================================
def generate_data(n_samples=50):
    x = np.random.uniform(-4, 4, n_samples)
    noise = np.random.normal(0, 3, n_samples)
    y = x**3 + noise
    x_train = torch.from_numpy(x).float().unsqueeze(1)
    y_train = torch.from_numpy(y).float().unsqueeze(1)
    return x_train, y_train

x_train, y_train = generate_data(n_samples=50)
x_test = torch.linspace(-6, 6, 200).unsqueeze(1) 

# ==========================================
# 2. Définition du Modèle Probabiliste
# ==========================================
class ProbabilisticNN(nn.Module):
    def __init__(self, hidden_size=50, dropout_rate=0.0):
        super(ProbabilisticNN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 2) 

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        mean = out[:, 0:1]
        variance = nn.functional.softplus(out[:, 1:2]) + 1e-6
        return mean, variance

# ==========================================
# 3. Fonction de Perte (NLL)
# ==========================================
def gaussian_nll_loss(mean, variance, target):
    loss = 0.5 * torch.log(variance) + 0.5 * (target - mean)**2 / variance
    return loss.mean()

# ==========================================
# 4. Fonctions d'Entraînement avec ADVERSARIAL TRAINING
# ==========================================
def train_model(model, x_train, y_train, epochs=2000, lr=0.01, adversarial=False, epsilon=0.1):
    """
    Entraîne un modèle unique, optionnellement avec Adversarial Training (FGSM).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Copie des données pour ne pas modifier l'original lors du requires_grad
    x_data = x_train.clone().detach()
    y_data = y_train.clone().detach()

    for epoch in range(epochs):
        model.train()
        
        if adversarial:
            # --- Étape 1 : Génération de l'exemple adverse ---
            optimizer.zero_grad()
            x_data.requires_grad = True
            
            mean, variance = model(x_data)
            loss = gaussian_nll_loss(mean, variance, y_data)
            loss.backward() # Calcul du gradient dL/dx
            
            # Fast Gradient Sign Method (FGSM)
            data_grad = x_data.grad.data
            x_adv = x_data + epsilon * data_grad.sign()
            
            # On détache pour l'étape suivante (on ne veut pas dériver par rapport à la génération)
            x_adv = x_adv.detach()
            
            # --- Étape 2 : Entraînement conjoint (Clean + Adversarial) ---
            optimizer.zero_grad()
            x_data.requires_grad = False # Plus besoin de gradients sur x
            
            # Passage sur données propres
            mean_clean, var_clean = model(x_data)
            loss_clean = gaussian_nll_loss(mean_clean, var_clean, y_data)
            
            # Passage sur données adverses
            mean_adv, var_adv = model(x_adv)
            loss_adv = gaussian_nll_loss(mean_adv, var_adv, y_data)
            
            # Somme des pertes (comme suggéré dans le papier)
            total_loss = loss_clean + loss_adv
            total_loss.backward()
            optimizer.step()
            
        else:
            # Entraînement standard
            optimizer.zero_grad()
            mean, variance = model(x_data)
            loss = gaussian_nll_loss(mean, variance, y_data)
            loss.backward()
            optimizer.step()
            
    return model

def train_ensemble(n_models, x_train, y_train, adversarial=False):
    models = []
    mode_str = "+ AT" if adversarial else ""
    print(f"Entraînement Ensemble (M={n_models} {mode_str})...")
    
    for i in range(n_models):
        model = ProbabilisticNN(hidden_size=50, dropout_rate=0.0)
        # Epsilon fixé à 0.1 (~1% de la plage totale des données [-6, 6])
        train_model(model, x_train, y_train, adversarial=adversarial, epsilon=0.1)
        models.append(model)
    return models

# ==========================================
# 5. Prédiction Ensemble
# ==========================================
def predict_ensemble(models, x_test):
    means = []
    variances = []
    with torch.no_grad():
        for model in models:
            mean, var = model(x_test)
            means.append(mean)
            variances.append(var)
    
    means_stack = torch.stack(means)
    vars_stack = torch.stack(variances)
    
    ensemble_mean = means_stack.mean(dim=0)
    ensemble_var = (vars_stack + means_stack**2).mean(dim=0) - ensemble_mean**2
    
    return ensemble_mean.numpy(), ensemble_var.numpy()

# ==========================================
# 6. Baseline Bayésienne
# ==========================================
def predict_mc_dropout(model, x_test, n_samples=5):
    means = []
    variances = []
    model.train() 
    with torch.no_grad():
        for _ in range(n_samples):
            mean, var = model(x_test)
            means.append(mean)
            variances.append(var)
    
    means_stack = torch.stack(means)
    vars_stack = torch.stack(variances)
    mc_mean = means_stack.mean(dim=0)
    mc_var = (vars_stack + means_stack**2).mean(dim=0) - mc_mean**2
    return mc_mean.numpy(), mc_var.numpy()

# ==========================================
# 7. Exécution
# ==========================================

results = {}

# 1. Ensemble Standard M=1 (Single Model)
models_m1 = train_ensemble(1, x_train, y_train, adversarial=False)
results['M1'] = predict_ensemble(models_m1, x_test)

# 2. Ensemble Standard M=5
models_m5 = train_ensemble(5, x_train, y_train, adversarial=False)
results['M5'] = predict_ensemble(models_m5, x_test)

# 3. Ensemble M=5 AVEC Adversarial Training (AT)
models_m5_at = train_ensemble(5, x_train, y_train, adversarial=True)
results['M5_AT'] = predict_ensemble(models_m5_at, x_test)

# 4. MC Dropout
print("Entraînement MC Dropout...")
mc_model = ProbabilisticNN(hidden_size=50, dropout_rate=0.1)
train_model(mc_model, x_train, y_train, adversarial=False)
mc_results = predict_mc_dropout(mc_model, x_test)


# ==========================================
# 8. Visualisation (Les 4 Graphiques)
# ==========================================

# Dossier de sauvegarde
if not os.path.exists('THALES/plot_NN_Ensemble'):
    os.makedirs('THALES/plot_NN_Ensemble')

y_true = x_test.numpy()**3

# --- PLOT 1 : Comparaison Visuelle des Régressions ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

plot_configs = [
    ('M1', "Single Model (M=1)"),
    ('M5', "Deep Ensemble (M=5)"),
    ('M5_AT', "Deep Ensemble (M=5) + Adversarial Training"),
    ('MC', "MC Dropout")
]

for i, (key, title) in enumerate(plot_configs):
    ax = axes[i]
    if key == 'MC':
        mean, var = mc_results
    else:
        mean, var = results[key]
        
    std = np.sqrt(var)
    
    ax.plot(x_test.numpy(), y_true, 'k--', alpha=0.6, label=r'Vérité ($x^3$)')
    ax.scatter(x_train.numpy(), y_train.numpy(), c='red', s=40, zorder=10, label='Train')
    ax.plot(x_test.numpy(), mean, 'b-', linewidth=2, label='Prédiction')
    ax.fill_between(x_test.numpy().flatten(), 
                    (mean - 3*std).flatten(), (mean + 3*std).flatten(), 
                    color='blue', alpha=0.2, label=r'Incertitude ($3\sigma$)')
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(-150, 150)
    if i == 0: ax.legend()

plt.tight_layout()
plt.savefig('THALES/plot_NN_Ensemble/comparaison_regressions.png')
print("Graphique 1 sauvegardé.")

# --- PLOT 2 : Analyse Incertitude (OOD) ---
plt.figure(figsize=(10, 6))
x_flat = x_test.numpy().flatten()

# On trace les écarts-types
plt.plot(x_flat, np.sqrt(results['M1'][1]).flatten(), '--', label='Single Model')
plt.plot(x_flat, np.sqrt(results['M5'][1]).flatten(), linewidth=2, label='Ensemble M=5')
plt.plot(x_flat, np.sqrt(results['M5_AT'][1]).flatten(), linewidth=2, color='purple', label='Ensemble M=5 + AT')
plt.plot(x_flat, np.sqrt(mc_results[1]).flatten(), 'k:', linewidth=2, label='MC Dropout')

plt.axvspan(-4, 4, color='gray', alpha=0.15, label='Zone Train')
plt.title("Comparaison des Incertitudes (Robustesse OOD)")
plt.xlabel("Input x")
plt.ylabel(r"Incertitude Prédite ($\sigma$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('THALES/plot_NN_Ensemble/analyse_incertitude.png')
print("Graphique 2 sauvegardé.")

# --- PLOT 3 : Métriques (RMSE & NLL) ---
# Génération données bruitées pour NLL
np.random.seed(101)
y_test_noisy = y_true.flatten() + np.random.normal(0, 3, x_test.shape[0])

metrics_data = {'RMSE': [], 'NLL': [], 'Labels': []}
for key, label in plot_configs:
    if key == 'MC': m, v = mc_results
    else: m, v = results[key]
    
    m, v = m.flatten(), v.flatten()
    rmse = np.sqrt(np.mean((m - y_true.flatten())**2))
    nll = 0.5 * np.log(v) + 0.5 * (y_test_noisy - m)**2 / v
    
    metrics_data['RMSE'].append(rmse)
    metrics_data['NLL'].append(np.mean(nll))
    metrics_data['Labels'].append(label.replace("Deep Ensemble ", "").replace("Adversarial Training", "AT"))

fig, ax1 = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(metrics_data['Labels']))
width = 0.35

ax1.bar(x_pos - width/2, metrics_data['RMSE'], width, label='RMSE', color='skyblue')
ax1.set_ylabel('RMSE')
ax2 = ax1.twinx()
ax2.bar(x_pos + width/2, metrics_data['NLL'], width, label='NLL', color='salmon')
ax2.set_ylabel('NLL')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics_data['Labels'], rotation=15)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Performances : RMSE vs NLL")
plt.tight_layout()
plt.savefig('THALES/plot_NN_Ensemble/metrics_comparison.png')
print("Graphique 3 sauvegardé.")

# --- PLOT 4 : Corrélation Erreur vs Incertitude ---
plt.figure(figsize=(10, 6))
for key, label in plot_configs:
    if key == 'MC': m, v = mc_results
    else: m, v = results[key]
    
    std = np.sqrt(v).flatten()
    err = np.abs(m.flatten() - y_true.flatten())
    plt.scatter(std, err, alpha=0.5, s=15, label=label.replace("Deep Ensemble ", "Ens "))

plt.xlabel('Incertitude Prédite')
plt.ylabel('Erreur Absolue')
plt.title('Calibration : L\'incertitude prédit-elle l\'erreur ?')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('THALES/plot_NN_Ensemble/error_vs_uncertainty.png')
print("Graphique 4 sauvegardé.")