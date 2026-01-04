import numpy as np
import pandas as pd
import yfinance as yf
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
# ==============================================================================
# 1. RÉCUPÉRATION DES DONNÉES (CAC 40 pour l'exemple)
# ==============================================================================
def get_stock_data():
    # Liste des tickers du CAC 40 (Liste représentative)
    tickers = [
        "AIR.PA", "AI.PA", "ALO.PA", "MT.AS", "CS.PA", "BNP.PA", "EN.PA", "CAP.PA", 
        "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "EL.PA", "RMS.PA", "KER.PA", "OR.PA", 
        "LR.PA", "MC.PA", "ML.PA", "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", 
        "SGO.PA", "SAN.PA", "SU.PA", "GLE.PA", "STLAP.PA", "STMPA.PA", "TEP.PA", 
        "HO.PA", "TTE.PA", "VIE.PA", "DG.PA", "VIV.PA"
    ]
    
    print(f"📥 Téléchargement de {len(tickers)} actions...")
    # On prend 2 ans d'historique
    # auto_adjust=False pour garantir la présence de 'Adj Close' et éviter les warnings
    data = yf.download(tickers, start="2022-01-01", progress=False, auto_adjust=False)['Adj Close']
    
    # Nettoyage : On supprime les colonnes (tickers) qui ont échoué (tout est NaN)
    data = data.dropna(axis=1, how='all')
    if data.empty: raise ValueError("Aucune donnée récupérée. Vérifiez la connexion.")
    
    # Calcul des rendements logarithmiques
    returns = np.log(data / data.shift(1)).dropna()
    return returns

# ==============================================================================
# 2. CŒUR MATHÉMATIQUE : MARCHENKO-PASTUR & NETTOYAGE
# ==============================================================================
def fit_marchenko_pastur(var, q, pts=100):
    """
    Génère la courbe théorique de densité des valeurs propres (le Bruit).
    Formula: Marchenko-Pastur PDF
    """
    # Bornes théoriques du bruit
    lambda_min = var * (1 - np.sqrt(1/q))**2
    lambda_max = var * (1 + np.sqrt(1/q))**2
    
    valeurs = np.linspace(lambda_min, lambda_max, pts)
    
    # Fonction de densité
    def pdf(x):
        return (q / (2 * np.pi * var * x)) * np.sqrt((lambda_max - x) * (x - lambda_min))
    
    densite = pdf(valeurs)
    # On nettoie les NaN (cas où x est hors bornes à cause de l'arrondi)
    densite[np.isnan(densite)] = 0
    
    return valeurs, densite, lambda_max

def denoise_correlation_matrix(returns):
    """
    Prend les rendements, nettoie la matrice via RMT, et retourne la propre.
    """
    # 1. Matrice de corrélation empirique (bruitée)
    corr_matrix = returns.corr().values
    
    # 2. Décomposition en Valeurs Propres (Eigenvalues) et Vecteurs Propres (Eigenvectors)
    # eVal = les forces des corrélations, eVec = les directions
    eVal, eVec = np.linalg.eigh(corr_matrix)
    
    # T = nombre de jours, N = nombre d'actions
    T, N = returns.shape
    Q = T / N # Ratio de qualité (plus Q est grand, moins il y a de bruit)
    sigma2 = 1 # Variance théorique pour une matrice de corrélation
    
    # 3. Calcul de la borne Marchenko-Pastur (Lambda Max)
    # Tout ce qui est inférieur à lambda_max est considéré comme du bruit aléatoire
    lambda_max_theorique = sigma2 * (1 + np.sqrt(1/Q))**2
    
    # 4. Filtrage (Denoising)
    # On identifie les valeurs propres qui sont du Signal (> Lambda Max)
    # On remplace les valeurs de Bruit par leur moyenne pour conserver la Trace de la matrice
    
    eVal_clean = eVal.copy()
    
    # Masque du bruit
    mask_noise = eVal <= lambda_max_theorique
    
    # Calcul de la moyenne des valeurs propres de bruit
    noise_mean = np.mean(eVal[mask_noise])
    
    # Remplacement
    eVal_clean[mask_noise] = noise_mean
    
    # 5. Reconstruction de la Matrice (Clean Matrix)
    # C_clean = V * Lambda_clean * V_transposé
    corr_clean = np.dot(eVec, np.dot(np.diag(eVal_clean), eVec.T))
    
    # Remise à 1 de la diagonale (normalisation obligatoire après reconstruction)
    np.fill_diagonal(corr_clean, 1)
    
    return corr_matrix, corr_clean, eVal, lambda_max_theorique, Q

# ==============================================================================
# 3. VISUALISATION ET EXÉCUTION
# ==============================================================================
def plot_rmt_spectrum(eVal, lambda_max, Q):
    """
    Affiche l'histogramme des valeurs propres vs la théorie.
    C'est LE graphique pour voir si le marché est bruité.
    """
    plt.figure(figsize=(12, 6))
    
    # Histogramme des vraies valeurs propres (Empirique)
    plt.hist(eVal, bins=50, density=True, alpha=0.6, color='blue', label='Distribution Empirique (Réalité)')
    
    # Courbe théorique de Marchenko-Pastur (Bruit Pur)
    x_mp, y_mp, _ = fit_marchenko_pastur(var=1, q=Q)
    plt.plot(x_mp, y_mp, color='red', linewidth=2, label='Marchenko-Pastur (Bruit Théorique)')
    
    plt.axvline(lambda_max, color='green', linestyle='--', label=rf'Seuil Signal/Bruit ($\lambda_{{max}}={lambda_max:.2f}$)')
    
    plt.title('Spectre des Valeurs Propres : Séparation du Bruit et du Signal')
    plt.xlabel(r'Valeur Propre ($\lambda$)')
    plt.ylabel('Densité de probabilité')
    plt.legend()
    plt.xlim(0, max(lambda_max * 1.5, 3)) # Zoom sur la partie gauche
    plt.show()

def visualiser_clusters_portfolio(returns, corr_clean):
    """
    Utilise la matrice nettoyée pour trier les actions par clusters réels.
    """
    plt.figure(figsize=(12, 10))
    
    # 1. Calcul de la distance mathématique (basée sur la corrélation nettoyée)
    # D = sqrt(2(1 - rho))
    dist = np.sqrt(2 * (1 - corr_clean))
    
    # 2. Linkage (On lie les éléments les plus proches)
    linkage = sch.linkage(dist, method='ward')
    
    # 3. Dendrogramme (L'arbre généalogique des actions)
    # C'est ici qu'on voit les clusters se former
    dendrogram = sch.dendrogram(linkage, labels=returns.columns, leaf_rotation=90)
    plt.title("Dendrogramme : Les vraies familles d'actions (basé sur RMT)")
    plt.show()
    
    # 4. Matrice de Corrélation Triée (Quasi-Diagonalisation)
    # On réordonne les actions pour mettre les familles ensemble
    ind = sch.fcluster(linkage, 0.5 * dist.max(), 'distance')
    columns_sorted = [returns.columns[i] for i in np.argsort(ind)]
    
    # Récupération des données triées
    df_sorted = returns[columns_sorted].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_sorted, cmap='coolwarm', center=0)
    plt.title("Matrice de Corrélation Triée (Les carrés rouges sont les clusters)")
    plt.show()

# --- LANCEMENT ---
if __name__ == "__main__":
    # 1. Données
    returns = get_stock_data()
    
    # 2. Calculs RMT
    corr_bruit, corr_clean, eigenvalues, seuil_bruit, Q_ratio = denoise_correlation_matrix(returns)
    
    # 3. Affichage
    print("\n📊 RÉSULTATS RMT :")
    print(f"Ratio Q (T/N) : {Q_ratio:.2f}")
    print(f"Seuil de bruit (Lambda Max) : {seuil_bruit:.2f}")
    
    nb_signal = np.sum(eigenvalues > seuil_bruit)
    pct_info = (nb_signal / len(eigenvalues)) * 100
    
    print(f"Nombre de valeurs propres considérées comme 'Signal' : {nb_signal} sur {len(eigenvalues)}")
    print(f"Cela signifie que seulement {pct_info:.1f}% des modes de corrélation sont réels !")
    print("Le reste (la grande majorité) est du bruit statistique.")

    # 4. Comparaison visuelle (Heatmap rapide)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(corr_bruit, cmap='coolwarm')
    ax[0].set_title("Matrice Bruitée (Originale)")
    ax[1].imshow(corr_clean, cmap='coolwarm')
    ax[1].set_title("Matrice Nettoyée (RMT)")
    plt.show()
    
    # 5. Graphique Spectral (Le plus important)
    plot_rmt_spectrum(eigenvalues, seuil_bruit, Q_ratio)

    # Cluster
    visualiser_clusters_portfolio(returns, corr_clean)