
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from loss_class import LossVisualization

# -------------------------
# CONFIG - modifie ici
# -------------------------
SEED = 42
HIDDEN = 256            # taille de la couche cachée
LR = 1e-3
EPOCHS = 1000
GRID_N = 220
RADIUS = 150.0           # échelle des directions (zoom)
SURFACE_CMAP = 'terrain'   # colormap pour la surface 3D
CONTOUR_CMAP = 'terrain'   # colormap pour le contour
TRAJECTORY_COLOR = 'black'
TRAJECTORY_MARKER = 'o'

# -------------------------

# -------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)

X1, y1 = make_circles(n_samples=5000, noise=0.10, factor=0.4, random_state=SEED)
X2, y2 = make_moons(n_samples=5000, noise=0.10, random_state=SEED+1)


# concaténation
X = np.vstack([X1, X2])
y = np.hstack([y1, y2])

# shuffle
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# standardize
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# -------------------------
# Définition du MLP
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2*hidden),
            nn.ReLU(),
            nn.Linear(2*hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleMLP()

# -------------------------
# utilitaires pour vectoriser / fixer les paramètres
# -------------------------
def get_param_vector(model):
    vecs = []
    for p in model.parameters():
        vecs.append(p.detach().cpu().view(-1))
    return torch.cat(vecs)


def set_param_vector(model, vec):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(vec[pointer:pointer+numel].view_as(p))
        pointer += numel

# -------------------------
# Entraînement et enregistrement de la trajectoire
# -------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

trajectory = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_loss = criterion(val_logits, y_val_t).item()

    trajectory.append(get_param_vector(model).detach().clone())
    val_losses.append(val_loss)

    if (epoch+1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  val_loss={val_loss:.4f}")

trajectory = torch.stack(trajectory)
base_vec = trajectory[-1].clone()


# fin de la classe LossVisualization

lv = LossVisualization(
    model_class=SimpleMLP,
    set_param_fn=set_param_vector,
    criterion=criterion,
    X_val=X_val_t,
    y_val=y_val_t,
    trajectory=trajectory,
    base_vec=base_vec,           
    radius=RADIUS,
    grid_n=GRID_N,
    surface_cmap=SURFACE_CMAP,
    contour_cmap=CONTOUR_CMAP,
    traj_color=TRAJECTORY_COLOR,
    traj_marker=TRAJECTORY_MARKER,
)

lv.compute_directions(seed=42)
lv.evaluate_grid()
lv.project_trajectory()
lv.plot()
