# Nécessite : torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# --- réseau "deterministic" wrapper pour poids échantillonnés ---
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden, out):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, out)
    def forward(self, x, weights):
        # weights is dict with sampled tensors keyed by layer name
        x = F.linear(x, weights['fc1.weight'], weights['fc1.bias'])
        x = F.relu(x)
        x = F.linear(x, weights['fc2.weight'], weights['fc2.bias'])
        return x

# --- Paramétrisation du posterior (mu et logvar pour chaque paramètre) ---
class BayesParams(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.param_shapes = {}
        for name, p in model.named_parameters():
            safe_name = name.replace('.', '_')
            self.param_shapes[name] = p.shape
            mu = nn.Parameter(torch.randn_like(p) * 0.01)
            logvar = nn.Parameter(torch.full_like(p, -5.0))  # init petit sigma
            self.register_parameter(safe_name + "_mu", mu)
            self.register_parameter(safe_name + "_logvar", logvar)
    def sample_weights(self, nsamples=1):
        samples = []
        for _ in range(nsamples):
            w = {}
            for name, shape in self.param_shapes.items():
                safe_name = name.replace('.', '_')
                mu = getattr(self, safe_name + "_mu")
                logvar = getattr(self, safe_name + "_logvar")
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(mu)
                w[name] = mu + std * eps
            samples.append(w)
        return samples
    def kl_divergence(self, sigma_p):
        # KL(Q||P) where P = N(0, sigma_p^2 I), Q factorized
        kl = 0.0
        # S'assurer que sigma_p est un tenseur pour les opérations torch
        if not isinstance(sigma_p, torch.Tensor):
            sigma_p = torch.tensor(sigma_p)
        for name in self.param_shapes:
            safe_name = name.replace('.', '_')
            mu = getattr(self, safe_name + "_mu")
            logvar = getattr(self, safe_name + "_logvar")
            var_q = logvar.exp()
            # per-dim KL
            kl += (torch.log(sigma_p.to(mu.device)) - 0.5*logvar).sum()
            kl += 0.5 * (var_q + mu.pow(2)).sum() / (sigma_p**2)
            kl -= 0.5 * mu.numel()
        return kl

# --- Entraînement ---
def train_pac_bayes(model, bayes_params, dataloader, epochs=10,
                    nsamples=3, lambda_coef=1.0, sigma_p=1.0, lr=1e-3, n_total=None):
    opt = torch.optim.Adam(bayes_params.parameters(), lr=lr)
    if n_total is None:
        n_total = len(dataloader.dataset)
    for epoch in range(epochs):
        model.train()
        for x, y in dataloader:
            samples = bayes_params.sample_weights(nsamples)
            losses = []
            for w in samples:
                logits = model(x, w)
                loss = F.cross_entropy(logits, y, reduction='mean')  # non-bornée: pratique courante
                losses.append(loss)
            emp_risk = torch.stack(losses).mean()
            kl = bayes_params.kl_divergence(sigma_p)
            objective = emp_risk + (lambda_coef / n_total) * kl
            opt.zero_grad()
            objective.backward()
            opt.step()

# --- Jeu de données et visualisation ---
def get_data(n_samples=200):
    """Génère un jeu de données synthétique (moons)."""
    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.int64))
    return X, y

def plot_decision_boundary(model, bayes_params, X, y, n_samples_plot=100):
    """Affiche la frontière de décision du modèle bayésien."""
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    with torch.no_grad():
        samples = bayes_params.sample_weights(nsamples=n_samples_plot)
        predictions = []
        for w in samples:
            logits = model(grid_tensor, w)
            probs = F.softmax(logits, dim=1)[:, 1]  # Probabilité de la classe 1
            predictions.append(probs.reshape(xx.shape))
        
        mean_prediction = torch.stack(predictions).mean(0)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, mean_prediction.numpy(), cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    plt.title("Frontière de décision bayésienne")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


if __name__ == '__main__':
    # --- Paramètres ---
    N_SAMPLES_DATA = 500
    HIDDEN_DIM = 16
    EPOCHS = 100
    LR = 0.01
    N_SAMPLES_TRAIN = 20 # Nombre d'échantillons de poids par étape d'entraînement

    # --- Préparation des données ---
    X_train, y_train = get_data(N_SAMPLES_DATA)
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- Initialisation du modèle et des paramètres bayésiens ---
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))
    net = SimpleNet(input_dim, HIDDEN_DIM, output_dim)
    bayes_params = BayesParams(net)

    # --- Entraînement ---
    train_pac_bayes(net, bayes_params, dataloader, epochs=EPOCHS, lr=LR, nsamples=N_SAMPLES_TRAIN, n_total=N_SAMPLES_DATA)

    # --- Visualisation ---
    plot_decision_boundary(net, bayes_params, X_train, y_train)
