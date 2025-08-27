from matplotlib import pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class LowLevelRNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = nn.RNNCell(dim, dim)
    def forward(self, zL, zH, x):
        return self.rnn(zL + zH + x)

class HighLevelRNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = nn.RNNCell(dim, dim)
    def forward(self, zH, zL):
        return self.rnn(zH + zL)

class HRM(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.L = LowLevelRNN(dim)
        self.H = HighLevelRNN(dim)
        self.output_head = nn.Linear(dim, output_dim) 
    
    def forward_cycle(self, x, zH, zL, T=3):
        # Fast updates for L, slow update for H
        for t in range(T-1):
            with torch.no_grad():  # No gradient for intermediate L
                zL = self.L(zL, zH, x)
        # Last step: allow gradients (1-step gradient)
        zL = self.L(zL, zH, x)
        zH = self.H(zH, zL)
        y_hat = self.output_head(zH)
        return zH, zL, y_hat

if __name__ == "__main__":
    torch.manual_seed(0)
    input_dim, output_dim = 2, 2
    # Training loop with deep supervision
    model = HRM(dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Larger dummy batch (batch size 64)
    batch_size = 512
    X = torch.rand((batch_size, input_dim))
    weights = torch.randn(input_dim)
    bias = torch.randn(1)
    y_linear = X @ weights + bias
    
    median = y_linear.median()
    y = (y_linear > median).long().view(-1)  # shape (batch_size,)

    # Afficher la répartition des classes
    uniq, counts = torch.unique(y, return_counts=True)
    print("Classes et effectifs:", list(zip(uniq.tolist(), counts.tolist())))

    print(X.shape) 
    print(y.shape)  

    zH = torch.zeros(batch_size, input_dim)
    zL = torch.zeros(batch_size, input_dim)

    N_cycles = 20
    n_epochs = 40
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        for cycle in range(N_cycles):
            zH, zL, y_hat = model.forward_cycle(X, zH, zL, T=5)
            loss = criterion(y_hat, y)
            # Detach hidden states to prevent full BPTT
            zH = zH.detach()
            zL = zL.detach()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # --- Préparer prédictions sur les points d'entraînement ---
    model.eval()
    with torch.no_grad():
        _, _, logits = model.forward_cycle(X, zH, zL, T=5)
        preds = logits.argmax(dim=1)

    # Accuracy sur le batch
    acc = (preds == y).float().mean().item()
    print(f"Train accuracy (batch): {acc*100:.2f}%")

    # --- Grille pour frontière de décision sur les 2 premières dims ---
    x_min, x_max = X[:, 0].min().item() - 0.1, X[:, 0].max().item() + 0.1
    y_min, y_max = X[:, 1].min().item() - 0.1, X[:, 1].max().item() + 0.1

    grid_res = 100  # résolution (ajuste si lent)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]

    # Compléter les autres dimensions avec la moyenne empirique de X
    X_mean = X.mean(dim=0)
    grid_torch = torch.zeros((grid_points_2d.shape[0], input_dim), dtype=torch.float32)
    grid_torch[:, :2] = torch.from_numpy(grid_points_2d).float()
    # remplir les dims 2.. avec la moyenne (broadcast)
    if input_dim > 2:
        grid_torch[:, 2:] = X_mean[2:].unsqueeze(0).repeat(grid_torch.shape[0], 1)

    # Prédictions sur la grille
    with torch.no_grad():
        _, _, grid_logits = model.forward_cycle(grid_torch, torch.zeros_like(grid_torch), torch.zeros_like(grid_torch), T=5)
        grid_preds = grid_logits.argmax(dim=1).cpu().numpy()

    Z = grid_preds.reshape(xx.shape)

    # --- Plot ---
    plt.figure(figsize=(8, 7))
    # frontière
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    # vrais labels (cercles)
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), s=80, edgecolors='k', cmap=plt.cm.Paired, label='True')
    # prédictions du modèle sur les mêmes points (croix)
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=preds.numpy(), marker='x', s=100, cmap=plt.cm.Paired, label='Predicted')
    plt.xlabel('X[:,0]')
    plt.ylabel('X[:,1]')
    plt.title(f'2D view (dims 0 & 1) — Train acc: {acc*100:.2f}%')
    # créer une légende explicite
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', label='True (class 0/1)', markersize=8),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='k', label='Predicted (x)', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()