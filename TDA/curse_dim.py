"""
But : fournir un pipeline d'expériences pour analyser la "curse of dimensionality".
- Génère plusieurs jeux de données synthétiques (gaussien, uniform, clusters).
- Mesure la concentration des distances (moyenne, écart-type, CV).
- Mesure contraste relatif (min vs max, nearest/farthest).
- Estime dimension intrinsèque via PCA et Levina-Bickel (k-NN MLE).
- Calcule la participation ratio (mesure d'homogénéité spectrale).
- Teste la performance d'un k-NN supervisé entre deux classes.
- Produit des graphiques sauvegardés en png et un résumé CSV.

"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# Utils : génération de données
# -----------------------
def make_gaussian_clusters(n_samples, dim, n_clusters=2, separation=2.0, seed=None):
    """Génère n_clusters de Gaussiennes isotropes séparées.
    Renvoie X (n_samples, dim), y (n_samples,)
    separation permet de contrôler la distance entre les centres.
    """
    rng = np.random.RandomState(seed)
    samples_per = n_samples // n_clusters
    X = []
    y = []
    for k in range(n_clusters):
        center = rng.normal(loc=0.0, scale=separation, size=dim) + k * separation
        Xk = rng.normal(loc=0.0, scale=3.0, size=(samples_per, dim)) + center
        X.append(Xk)
        y.append(np.full(samples_per, k, dtype=int))
    X = np.vstack(X)
    y = np.concatenate(y)
    # shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]

def make_uniform_cube(n_samples, dim, low=0.0, high=1.0, seed=None):
    rng = np.random.RandomState(seed)
    X = rng.uniform(low=low, high=high, size=(n_samples, dim))
    return X, np.zeros(n_samples, dtype=int)

def make_manifold_synthetic(n_samples, dim, intrinsic_dim=2, noise=0.05, seed=None):
    """
    Crée un point-cloud qui a une faible dimension intrinsèque :
    - On génère des points dans un espace intrinseque_dim,
    - puis on les propage linéairement dans dim dimensions + bruit.
    """
    rng = np.random.RandomState(seed)
    Z = rng.normal(size=(n_samples, intrinsic_dim))
    A = rng.normal(size=(intrinsic_dim, dim))
    X = Z.dot(A) + noise * rng.normal(size=(n_samples, dim))
    return X, np.zeros(n_samples, dtype=int)

# -----------------------
# Mesures principales
# -----------------------
def distance_stats(X, sample=2000):
    """Calcule statistiques sur les distances entre points.
       Pour grande N, on échantillonne `sample` points pour réduire complexité."""
    n = X.shape[0]
    if n > sample:
        idx = np.random.choice(n, sample, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    nbrs = NearestNeighbors(n_neighbors=Xs.shape[0], algorithm='auto').fit(Xs)
    dists = nbrs.kneighbors(Xs, return_distance=True)[0]
    # dists includes 0 on diagonal; collect upper triangular distances (excluding self)
    # simpler : take for each row the distances[1:] (excluding self 0)
    all_pairwise = dists[:, 1:].ravel()
    mean = np.mean(all_pairwise)
    std = np.std(all_pairwise)
    med = np.median(all_pairwise)
    # nearest neighbor distribution:
    nn = dists[:, 1]  # 1st neighbor (excluding self)
    far = dists[:, -1]  # farthest within sampled set
    stats = {
        'pairwise_mean': float(mean),
        'pairwise_std': float(std),
        'pairwise_median': float(med),
        'nn_mean': float(np.mean(nn)),
        'nn_std': float(np.std(nn)),
        'far_mean': float(np.mean(far)),
        'far_std': float(np.std(far)),
        'cv_pairwise': float(std / mean) if mean != 0 else np.nan,
        'rel_contrast': float((np.mean(far) - np.mean(nn)) / np.mean(nn)) if np.mean(nn) != 0 else np.nan
    }
    return stats

def pca_intrinsic_dim(X, variance_threshold=0.9):
    """Retourne le nombre de composantes PCA nécessaires pour atteindre variance_threshold
       et la fraction cumulée d'explained variance."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(Xs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    participation_ratio = (np.sum(pca.explained_variance_)**2) / np.sum(pca.explained_variance_**2)
    return {
        'pca_n_components_%.0f' % (variance_threshold*100): n_comp,
        'pca_explained_variance_first': float(pca.explained_variance_ratio_[0]),
        'pca_participation_ratio': float(participation_ratio),
        'pca_cumvar': list(cumvar[:min(10, len(cumvar))])  # premières valeurs utiles
    }

def levina_bickel_intrinsic_dim(X, k=10):
    """
    Estimation de la dimension intrinsèque par la méthode MLE (Levina & Bickel).
    Formula: for each point i, let r_1..r_k be distances to k nearest neighbors (sorted increasing)
    m_i = [ (1/(k-1)) * sum_{j=1..k-1} log(r_k / r_j) ]^{-1}
    retour: moyenne m_i
    """
    n = X.shape[0]
    if n <= k:
        return np.nan
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    dists, _ = nbrs.kneighbors(X)  # includes self-distance 0 at col 0
    # drop the 0th column
    d = dists[:, 1:]  # shape (n, k)
    # ensure strictly positive distances
    d[d <= 0] = 1e-12
    rk = d[:, -1]
    logs = np.log(rk[:, None] / d[:, :-1])  # shape (n, k-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        m_i = (np.sum(logs, axis=1) / (k - 1))
        # inverse
        m_i = 1.0 / m_i
    # Filter out inf/nan
    m_i = m_i[np.isfinite(m_i) & (m_i > 0)]
    if len(m_i) == 0:
        return np.nan
    return float(np.mean(m_i))

def knn_classification_score(X, y, k=5, cv=5):
    """Evalue performance k-NN via cross-val stratifiée."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=k)
    cvsplit = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    try:
        scores = cross_val_score(knn, Xs, y, cv=cvsplit, scoring='accuracy', n_jobs=1)
        return float(np.mean(scores)), float(np.std(scores))
    except Exception:
        return np.nan, np.nan

# -----------------------
# Expérimentation : boucle sur dimensions
# -----------------------
def run_experiment(dims, n_samples=2000, seeds=42, outdir="cod_results"):
    """
    dims : liste d'entiers (dimensions à tester)
    n_samples : nombre de points
    """
    os.makedirs(outdir, exist_ok=True)
    results = []
    for dim in tqdm(dims, desc="Dims"):
        seed = seeds
        # 3 types de jeux :
        Xg, yg = make_gaussian_clusters(n_samples=n_samples, dim=dim, n_clusters=6, separation=0.5, seed=seed)
        Xu, yu = make_uniform_cube(n_samples=n_samples, dim=dim, seed=seed)
        Xm, ym = make_manifold_synthetic(n_samples=n_samples, dim=dim, intrinsic_dim=min(3, dim), noise=0.05, seed=seed)

        # pour chacun, calculs
        for kind, X, y in [('gaussian', Xg, yg), ('uniform', Xu, yu), ('manifold', Xm, ym)]:
            ds = distance_stats(X, sample=min(1000, len(X)))
            pca = pca_intrinsic_dim(X, variance_threshold=0.9)
            lb = levina_bickel_intrinsic_dim(X, k=10 if n_samples > 20 else max(2, n_samples-1))
            if True : #kind == 'gaussian':
                knn_mean, knn_std = knn_classification_score(X, y, k=5, cv=5)
            else:
                knn_mean, knn_std = (np.nan, np.nan)
            row = {
                'dim': dim,
                'kind': kind,
                'n_samples': n_samples,
                'distance_pairwise_mean': ds['pairwise_mean'],
                'distance_pairwise_std': ds['pairwise_std'],
                'distance_pairwise_cv': ds['cv_pairwise'],
                'nn_mean': ds['nn_mean'],
                'far_mean': ds['far_mean'],
                'rel_contrast': ds['rel_contrast'],
                'pca_n_components_90': pca['pca_n_components_90'],
                'pca_first_ev': pca['pca_explained_variance_first'],
                'participation_ratio': pca['pca_participation_ratio'],
                'levina_bickel_id': lb,
                'knn_acc_mean': knn_mean,
                'knn_acc_std': knn_std
            }
            results.append(row)
    df = pd.DataFrame(results)
    csvpath = os.path.join(outdir, "cod_summary.csv")
    df.to_csv(csvpath, index=False)
    print(f"[+] Résultats sauvegardés : {csvpath}")
    return df

# -----------------------
# Visualisations
# -----------------------
def plot_metrics(df, outdir="cod_results"):
    os.makedirs(outdir, exist_ok=True)
    # 1) CV des distances vs dimension (par kind)
    plt.figure(figsize=(8,5))
    for kind, g in df.groupby('kind'):
        plt.plot(g['dim'], g['distance_pairwise_cv'], label=kind, marker='o')
    plt.xlabel('Dimension')
    plt.ylabel('Coefficient de variation (std/mean) des distances')
    plt.title('Concentration des distances vs dimension')
    plt.legend()
    p1 = os.path.join(outdir, "cv_vs_dim.png")
    plt.savefig(p1, bbox_inches='tight')
    plt.close()

    # 2) Relative contrast vs dimension
    plt.figure(figsize=(8,5))
    for kind, g in df.groupby('kind'):
        plt.plot(g['dim'], g['rel_contrast'], label=kind, marker='o')
    plt.xlabel('Dimension')
    plt.ylabel('Contraste relatif (far_mean - nn_mean) / nn_mean')
    plt.title('Contraste relatif des distances vs dimension')
    plt.legend()
    p2 = os.path.join(outdir, "rel_contrast_vs_dim.png")
    plt.savefig(p2, bbox_inches='tight')
    plt.close()

    # 3) PCA n_components_90 vs dimension
    plt.figure(figsize=(8,5))
    for kind, g in df.groupby('kind'):
        plt.plot(g['dim'], g['pca_n_components_90'], label=kind, marker='o')
    plt.xlabel('Dimension')
    plt.ylabel('Nombre composantes PCA pour 90% variance')
    plt.title('Composantes nécessaires (PCA) vs dimension')
    plt.legend()
    p3 = os.path.join(outdir, "pca90_vs_dim.png")
    plt.savefig(p3, bbox_inches='tight')
    plt.close()

    # 4) Levina-Bickel ID estimate vs dimension
    plt.figure(figsize=(8,5))
    for kind, g in df.groupby('kind'):
        plt.plot(g['dim'], g['levina_bickel_id'], label=kind, marker='o')
    plt.xlabel('Dimension')
    plt.ylabel('Estimation dimension intrinsèque (Levina-Bickel)')
    plt.title('Dimension intrinsèque estimée vs dimension')
    plt.legend()
    p4 = os.path.join(outdir, "levina_vs_dim.png")
    plt.savefig(p4, bbox_inches='tight')
    plt.close()

    # 5) k-NN accuracy (pour gaussian) vs dimension
    g = df[df['kind']=='gaussian']
    plt.figure(figsize=(8,5))
    plt.plot(g['dim'], g['knn_acc_mean'], label='k-NN accuracy', marker='o')
    plt.fill_between(g['dim'], g['knn_acc_mean'] - g['knn_acc_std'], g['knn_acc_mean'] + g['knn_acc_std'], alpha=0.2)
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy (k-NN 5-fold CV)')
    plt.title('Performance supervisée (k-NN) vs dimension pour 2 clusters gaussiens')
    plt.ylim(0,1.05)
    p5 = os.path.join(outdir, "knn_acc_vs_dim.png")
    plt.savefig(p5, bbox_inches='tight')
    plt.close()

    print(f"[+] Graphiques sauvegardés dans {outdir}:")
    for p in [p1,p2,p3,p4,p5]:
        print("   -", os.path.basename(p))

# -----------------------
# Exemple main
# -----------------------
def main():
    # grille de dimensions (modifiable)
    dims = [2, 5, 10, 20, 50, 100]
    # nombre de points (garder raisonnable sinon calcul pairwise coûte cher)
    n_samples = 4000
    outdir = "TDA/cod_results"
    df = run_experiment(dims=dims, n_samples=n_samples, seeds=0, outdir=outdir)
    plot_metrics(df, outdir=outdir)
    print("\nRésumé (extrait) :")
    print(df.groupby(['kind','dim']).agg({
        'distance_pairwise_cv':'mean',
        'rel_contrast':'mean',
        'pca_n_components_90':'mean',
        'levina_bickel_id':'mean',
        'knn_acc_mean':'mean'
    }).reset_index().head(20))

if __name__ == "__main__":
    main()
