import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import copy
import os

# Configuration
np.random.seed(42)
torch.manual_seed(42)

# 1. Dataset : Two Moons (Classification)
# Simple dataset to visualize decision boundaries
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train = torch.from_numpy(X).float()
y_train = torch.from_numpy(y).long()

# 2. Model: Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.net(x)

# 3. Training Function
def train_model(model, X, y, epochs=500, lr=0.01, save_at_epoch=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    saved_state = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if save_at_epoch is not None and epoch == save_at_epoch:
            saved_state = copy.deepcopy(model.state_dict())
            
    return model, saved_state

# --- Experiment ---

# A. Train Independent Models (Ensemble Members) to find different Modes
print("Training Model 1 (and saving trajectory checkpoint)...")
model_1 = SimpleMLP()
# Save 'trajectory' point slightly before the end (e.g., epoch 450 vs 500)
# to simulate "functions along an optimization trajectory"
model_1, state_1_mid = train_model(model_1, X_train, y_train, epochs=500, save_at_epoch=450)
state_1_end = copy.deepcopy(model_1.state_dict())

print("Training Model 2 (Independent Initialization)...")
model_2 = SimpleMLP()
model_2, _ = train_model(model_2, X_train, y_train, epochs=500)
state_2_end = copy.deepcopy(model_2.state_dict())

# B. Loss Landscape Interpolation Function
# "Linear Mode Connectivity" test
def get_loss_interpolation(model_class, state_a, state_b, X, y, alphas=np.linspace(0, 1, 20)):
    losses = []
    criterion = nn.CrossEntropyLoss()
    temp_model = model_class()
    
    for alpha in alphas:
        # Interpolate weights: theta = (1-alpha)*theta_a + alpha*theta_b
        interpolated_state = {}
        for key in state_a.keys():
            interpolated_state[key] = (1 - alpha) * state_a[key] + alpha * state_b[key]
        
        temp_model.load_state_dict(interpolated_state)
        with torch.no_grad():
            output = temp_model(X)
            loss = criterion(output, y)
            losses.append(loss.item())
    return alphas, losses

# C. Calculate Interpolations
# 1. Between Independent Modes (Ensemble)
alphas, loss_ensemble = get_loss_interpolation(SimpleMLP, state_1_end, state_2_end, X_train, y_train)

# 2. Between Trajectory Points (Same Mode)
# Note: Usually trajectory points are close, let's see if there is a barrier.
# If they are in the same basin, the loss should be linear or flat, no bump.
alphas_traj, loss_traj = get_loss_interpolation(SimpleMLP, state_1_mid, state_1_end, X_train, y_train)


# --- Plotting ---

if not os.path.exists('THALES/plot_Deep_Ensemble'):
    os.makedirs('THALES/plot_Deep_Ensemble')

# Plot 1: Loss Landscapes
plt.figure(figsize=(10, 6))
plt.plot(alphas, loss_ensemble, 'r-o', linewidth=2, label='Ensemble (Mode 1 -> Mode 2)')
plt.plot(alphas, loss_traj, 'g--^', linewidth=2, label='Trajectory (Epoch 450 -> Epoch 500)')
plt.title("Perspective du Paysage de Perte (Loss Landscape)\nBarrière de perte entre solutions")
plt.xlabel("Interpolation Coefficient alpha")
plt.ylabel("Loss (CrossEntropy)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('THALES/plot_Deep_Ensemble/loss_landscape_barrier.png')

# Plot 2: Decision Boundaries (Visualizing Diversity)
# Create a grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

# Get predictions
model_1.load_state_dict(state_1_end)
with torch.no_grad():
    Z1 = torch.softmax(model_1(grid_tensor), dim=1)[:, 1].reshape(xx.shape)

model_2.load_state_dict(state_2_end)
with torch.no_grad():
    Z2 = torch.softmax(model_2(grid_tensor), dim=1)[:, 1].reshape(xx.shape)

# Difference map
diversity = np.abs(Z1.numpy() - Z2.numpy())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Model 1
axes[0].contourf(xx, yy, Z1, cmap=plt.cm.RdBu, alpha=0.8)
axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
axes[0].set_title("Modèle 1 (Mode A)")

# Model 2
axes[1].contourf(xx, yy, Z2, cmap=plt.cm.RdBu, alpha=0.8)
axes[1].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
axes[1].set_title("Modèle 2 (Mode B)")

# Diversity (Difference)
diff_plot = axes[2].contourf(xx, yy, diversity, cmap='viridis', alpha=0.8)
axes[2].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', alpha=0.2)
axes[2].set_title("Diversité (Différence Absolue)\nZones de désaccord = Incertitude")
plt.colorbar(diff_plot, ax=axes[2])

plt.savefig('THALES/plot_Deep_Ensemble/decision_boundary_diversity.png')

print("Graphs generated: THALES/plot_Deep_Ensemble/loss_landscape_barrier.png, THALES/plot_Deep_Ensemble/decision_boundary_diversity.png")