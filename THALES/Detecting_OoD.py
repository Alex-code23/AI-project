import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegressionCV
import os

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"Utilisation de l'appareil : {DEVICE}")

# ==========================================
# 1. Modèle et Données
# ==========================================

class SimpleCNN(nn.Module):
    """Un CNN simple pour extraire des features."""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Nous avons besoin des features de l'avant-dernière couche pour Mahalanobis
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = self.pool(self.relu(self.conv3(out)))
        out = out.view(out.size(0), -1) # Flatten
        feature = self.relu(self.fc1(out)) # Penultimate features
        logits = self.fc2(feature)
        return logits, feature

# Préparation des données (CIFAR-10)
# Nous allons utiliser les classes 0-4 comme "In-Distribution" et 5-9 comme "OOD"
# pour simuler un scénario OOD réaliste sans télécharger d'autres datasets.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Transformation pour MNIST (OOD) : Resize 32x32, 3 channels
transform_mnist = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

# Téléchargement (si nécessaire) et chargement
train_dataset_full = datasets.CIFAR10(root='THALES/data', train=True, download=True, transform=transform)
test_dataset_full = datasets.CIFAR10(root='THALES/data', train=False, download=True, transform=transform)
ood_dataset = datasets.MNIST(root='THALES/data', train=False, download=True, transform=transform_mnist)

# In-Distribution : Tout CIFAR-10
train_loader = DataLoader(train_dataset_full, batch_size=64, shuffle=True)
test_loader_in = DataLoader(test_dataset_full, batch_size=64, shuffle=False)

# Out-of-Distribution : MNIST
test_loader_ood = DataLoader(ood_dataset, batch_size=64, shuffle=False)

# ==========================================
# 2. Entraînement Rapide
# ==========================================
model = SimpleCNN(num_classes=10).to(DEVICE) # 10 classes pour CIFAR-10 complet
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Entraînement du modèle (5 epochs)...")
for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 3. Implémentation Mahalanobis
# ==========================================

def get_mahalanobis_params(model, loader):
    """Calcule la moyenne par classe et la matrice de covariance liée."""
    model.eval()
    features = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            _, feat = model(images)
            features.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())
            
    features = np.concatenate(features)
    labels = np.concatenate(labels_list)
    
    # Calcul des moyennes par classe
    class_means = []
    classes = np.unique(labels)
    for c in classes:
        class_means.append(np.mean(features[labels == c], axis=0))
    class_means = np.array(class_means)
    
    # Calcul de la covariance liée (moyenne des covariances)
    # (Centrer les données par rapport à leur moyenne de classe)
    centered_features = []
    for i in range(len(labels)):
        centered_features.append(features[i] - class_means[labels[i]])
    centered_features = np.array(centered_features)
    
    covariance = np.cov(centered_features, rowvar=False)
    
    # Inversion de la covariance (pour la distance de Mahalanobis)
    # Ajout d'un epsilon pour la stabilité numérique si nécessaire
    precision = np.linalg.inv(covariance + 1e-6 * np.eye(covariance.shape[0]))
    
    return class_means, precision

print("Calcul des paramètres de Mahalanobis...")
sample_means, precision = get_mahalanobis_params(model, train_loader)
sample_means = torch.from_numpy(sample_means).float().to(DEVICE)
precision = torch.from_numpy(precision).float().to(DEVICE)

def get_mahalanobis_score(model, images, means, precision, magnitude=0.0):
    """
    Calcule le score de Mahalanobis.
    Inclut l'Input Pre-processing si magnitude > 0.
    """
    model.eval()
    images = images.clone().detach().requires_grad_(True)
    
    # 1. Forward pass pour obtenir les features
    logits, features = model(images)
    
    # 2. Calculer la distance de Mahalanobis pour chaque classe
    # Distance = (f - mu)^T * Sigma^-1 * (f - mu)
    batch_size = features.size(0)
    num_classes = means.size(0)
    
    # Reshape pour broadcasting: (batch, classes, features)
    features_exp = features.unsqueeze(1).expand(batch_size, num_classes, -1)
    means_exp = means.unsqueeze(0).expand(batch_size, num_classes, -1)
    
    diff = features_exp - means_exp # (B, C, F)
    
    # Calcul efficace : diag(diff * precision * diff^T)
    # On veut juste le terme diagonal pour chaque classe
    # (B, C, F) @ (F, F) -> (B, C, F)
    temp = torch.matmul(diff, precision) 
    # Somme sur l'axe des features après multiplication élément par élément
    dists = torch.sum(temp * diff, dim=2) # (B, C)
    
    # Le score est l'opposé de la distance minimale (Max score = plus proche)
    # On prend la classe la plus proche pour le calcul du gradient (Input pre-processing)
    min_dist, min_idx = torch.min(dists, dim=1)
    
    if magnitude > 0:
        # Input Pre-processing: Ajouter du bruit pour réduire la distance à la classe prédite
        # On veut MINIMISER la distance, donc on descend le gradient de la distance
        loss = min_dist.sum()
        loss.backward()
        
        # Le gradient par rapport à l'image
        gradient = images.grad.data
        # Perturbation adverse INVERSE (on aide le modèle)
        images_processed = images - magnitude * torch.sign(gradient)
        
        # Recalculer le score avec l'image prétraitée
        with torch.no_grad():
            _, features_p = model(images_processed)
            features_exp_p = features_p.unsqueeze(1).expand(batch_size, num_classes, -1)
            diff_p = features_exp_p - means_exp
            temp_p = torch.matmul(diff_p, precision)
            dists_p = torch.sum(temp_p * diff_p, dim=2)
            min_dist_p, _ = torch.min(dists_p, dim=1)
            
        return -min_dist_p # Score négatif (plus grand = mieux)
    
    return -min_dist

def get_softmax_score(model, images):
    """Baseline: Maximum Softmax Probability."""
    with torch.no_grad():
        logits, _ = model(images)
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
    return max_probs

# ==========================================
# 4. Génération d'Attaques Adverses (FGSM)
# ==========================================
def generate_fgsm(model, images, labels, epsilon=0.05):
    images = images.clone().detach().requires_grad_(True)
    logits, _ = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    return perturbed_images

# ==========================================
# 5. Évaluation et Collecte de Scores
# ==========================================
results = {
    "In-Dist": {"Softmax": [], "Mahalanobis": []},
    "OOD": {"Softmax": [], "Mahalanobis": []},
    "Adversarial": {"Softmax": [], "Mahalanobis": []}
}

print("Évaluation In-Distribution...")
for images, _ in test_loader_in:
    images = images.to(DEVICE)
    results["In-Dist"]["Softmax"].extend(get_softmax_score(model, images).cpu().numpy())
    results["In-Dist"]["Mahalanobis"].extend(get_mahalanobis_score(model, images, sample_means, precision, magnitude=0.01).cpu().numpy())

print("Évaluation OOD (MNIST)...")
for images, _ in test_loader_ood:
    images = images.to(DEVICE)
    results["OOD"]["Softmax"].extend(get_softmax_score(model, images).cpu().numpy())
    results["OOD"]["Mahalanobis"].extend(get_mahalanobis_score(model, images, sample_means, precision, magnitude=0.01).cpu().numpy())

print("Évaluation Adversarial (FGSM sur In-Dist)...")
# On génère des attaques seulement sur un sous-ensemble pour aller vite
limit = 0
for images, labels in test_loader_in:
    if limit > 10: break # Limite pour la démo
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    adv_images = generate_fgsm(model, images, labels, epsilon=0.1)
    
    results["Adversarial"]["Softmax"].extend(get_softmax_score(model, adv_images).cpu().numpy())
    results["Adversarial"]["Mahalanobis"].extend(get_mahalanobis_score(model, adv_images, sample_means, precision, magnitude=0.01).cpu().numpy())
    limit += 1

# ==========================================
# 6. Visualisation (Graphiques)
# ==========================================
if not os.path.exists('THALES/plots_mahalanobis'):
    os.makedirs('THALES/plots_mahalanobis')

def plot_histogram(scores_in, scores_out, title, filename, label_in="In-Dist", label_out="Out"):
    plt.figure(figsize=(8, 6))
    plt.hist(scores_in, bins=50, alpha=0.5, label=label_in, density=True, color='blue')
    plt.hist(scores_out, bins=50, alpha=0.5, label=label_out, density=True, color='red')
    plt.title(title)
    plt.xlabel("Score de Confiance")
    plt.ylabel("Densité")
    plt.legend()
    plt.savefig(f'THALES/plots_mahalanobis/{filename}')
    plt.close()

def plot_roc(scores_in, scores_out, label, ax):
    y_true = np.concatenate([np.ones(len(scores_in)), np.zeros(len(scores_out))])
    y_scores = np.concatenate([scores_in, scores_out])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')
    return roc_auc

# 1. Histogrammes : In vs OOD
plot_histogram(results["In-Dist"]["Softmax"], results["OOD"]["Softmax"], 
               "Baseline (Softmax): In-Dist vs OOD", "hist_softmax_ood.png", label_out="OOD (MNIST)")
plot_histogram(results["In-Dist"]["Mahalanobis"], results["OOD"]["Mahalanobis"], 
               "Mahalanobis: In-Dist vs OOD", "hist_mahalanobis_ood.png", label_out="OOD (MNIST)")

# 2. Histogrammes : In vs Adversarial
plot_histogram(results["In-Dist"]["Softmax"], results["Adversarial"]["Softmax"], 
               "Baseline (Softmax): In-Dist vs Adversarial", "hist_softmax_adv.png", label_out="Adversarial (FGSM)")
plot_histogram(results["In-Dist"]["Mahalanobis"], results["Adversarial"]["Mahalanobis"], 
               "Mahalanobis: In-Dist vs Adversarial", "hist_mahalanobis_adv.png", label_out="Adversarial (FGSM)")

# 3. Courbes ROC Comparatives
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ROC pour OOD
plot_roc(results["In-Dist"]["Softmax"], results["OOD"]["Softmax"], "Softmax", ax1)
plot_roc(results["In-Dist"]["Mahalanobis"], results["OOD"]["Mahalanobis"], "Mahalanobis", ax1)
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Détection OOD (CIFAR-10 vs MNIST)')
ax1.legend(loc="lower right")

# ROC pour Adversarial
plot_roc(results["In-Dist"]["Softmax"], results["Adversarial"]["Softmax"], "Softmax", ax2)
plot_roc(results["In-Dist"]["Mahalanobis"], results["Adversarial"]["Mahalanobis"], "Mahalanobis", ax2)
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Détection Adversarial (FGSM)')
ax2.legend(loc="lower right")

plt.tight_layout()
plt.savefig('THALES/plots_mahalanobis/roc_curves.png')
print("Graphiques sauvegardés dans THALES/plots_mahalanobis/")