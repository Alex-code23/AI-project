import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURATION & IMPORTATION
# ==============================================================================
def recuperer_donnees(tickers, jours=365*3):
    """Récupère les prix de clôture ajustés de manière robuste."""
    start_date = (datetime.now() - timedelta(days=jours)).strftime('%Y-%m-%d')
    print(f"📥 Récupération des données depuis {start_date}...")
    
    # On télécharge tout le DataFrame brut d'abord
    # auto_adjust=False garantit qu'on a bien 'Adj Close' et 'Close' séparés
    df = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
    
    # Vérification si le téléchargement est vide
    if df.empty:
        raise ValueError("❌ Aucune donnée téléchargée. Vérifiez votre connexion internet ou les Tickers.")

    # --- SÉLECTION ROBUSTE DE LA COLONNE ---
    # Cas 1 : La structure standard (MultiIndex)
    if 'Adj Close' in df.columns:
        return df['Adj Close'].dropna()
    
    # Cas 2 : Parfois yfinance ne renvoie que 'Close' (si auto_adjust est activé par défaut)
    elif 'Close' in df.columns:
        print("⚠️ Note : Utilisation de 'Close' au lieu de 'Adj Close'.")
        return df['Close'].dropna()
        
    # Cas 3 : Si la structure est inversée (Ticker au niveau 1)
    # On essaie de récupérer n'importe quelle colonne qui ressemble à un prix
    else:
        try:
            return df.xs('Adj Close', axis=1, level=0).dropna()
        except KeyError:
            return df.xs('Close', axis=1, level=0).dropna()

# ==============================================================================
# 2. MATHÉMATIQUES DE PORTEFEUILLE (Markowitz)
# ==============================================================================
def optimiser_portefeuille(data):
    """
    Simule 5000 allocations différentes pour trouver celle qui minimise la Variance
    tout en maximisant l'Espérance (Ratio de Sharpe).
    """
    # Calcul des rendements journaliers (Log Returns pour les propriétés mathématiques)
    # Formule : ln(Pt / Pt-1)
    returns = np.log(data / data.shift(1))
    
    # --- Indicateurs Statistiques Annualisés ---
    # Moyenne (Espérance) * 252 jours de bourse
    mean_returns = returns.mean() * 252
    # Matrice de Covariance (Le cœur de la réduction de variance)
    cov_matrix = returns.cov() * 252
    
    # --- Simulation de Monte Carlo (Poids Aléatoires) ---
    nb_simulations = 5000
    all_weights = np.zeros((nb_simulations, len(data.columns)))
    ret_arr = np.zeros(nb_simulations)
    vol_arr = np.zeros(nb_simulations)
    sharpe_arr = np.zeros(nb_simulations)

    print(f"🧮 Simulation de {nb_simulations} portefeuilles...")

    for i in range(nb_simulations):
        # Poids aléatoires qui s'additionnent à 1
        weights = np.array(np.random.random(len(data.columns)))
        weights = weights / np.sum(weights)
        all_weights[i, :] = weights

        # Espérance de rendement du portefeuille (R)
        # R = Somme(Poids * Rendement Moyen)
        ret_arr[i] = np.sum(mean_returns * weights)

        # Variance et Volatilité du portefeuille (σ)
        # Var = Transposée(W) * MatriceCov * W
        # C'est ici que la magie de la diversification opère (les covariances négatives réduisent le total)
        var = np.dot(weights.T, np.dot(cov_matrix, weights))
        vol_arr[i] = np.sqrt(var)

        # Ratio de Sharpe (Rendement / Risque)
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]

    # --- Résultat : Le Meilleur Portefeuille ---
    max_sharpe_idx = sharpe_arr.argmax()
    best_weights = all_weights[max_sharpe_idx, :]
    
    print("\n🏆 MEILLEURE ALLOCATION (Optimisation Variance/Rendement) :")
    for ticker, weight in zip(data.columns, best_weights):
        print(f"  - {ticker} : {weight*100:.1f}%")
        
    return ret_arr, vol_arr, sharpe_arr, max_sharpe_idx

# ==============================================================================
# 3. PROJECTION FUTURE (Mouvement Brownien Géométrique)
# ==============================================================================
def simulation_monte_carlo(data, ticker, jours_futurs=15,jours_historique=120, simulations=1000):
    """
    Projette le futur d'une action selon ses stats passées (Drift & Volatilité).
    Formule : St = St-1 * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """
    if ticker not in data.columns:
        return None, 0.0, 0.0, 0.0
        
    # On ne garde que les X derniers jours pour le calcul des paramètres (Mu, Sigma)
    prices = data[ticker].dropna()
    if len(prices) > jours_historique:
        prices = prices.iloc[-jours_historique:]
        
    if len(prices) < 10:
        return None, 0.0, 0.0, 0.0
        
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Paramètres statistiques
    mu = returns.mean() # Drift (Tendance moyenne)
    sigma = returns.std() # Volatilité (Variance écart-type)
    start_price = data[ticker].iloc[-1]
    
    # Création de la matrice de simulation
    # dt = 1 jour
    sim_data = np.zeros((jours_futurs, simulations))
    sim_data[0] = start_price
    
    # print(f"\n🎲 Lancement de {simulations} futurs possibles pour {ticker}...")
    
    for t in range(1, jours_futurs):
        # Z est la composante aléatoire (Loi Normale)
        Z = np.random.normal(0, 1, simulations)
        
        # Formule mathématique du prix futur
        drift = (mu - 0.5 * sigma**2)
        shock = sigma * Z
        
        sim_data[t] = sim_data[t-1] * np.exp(drift + shock)
        
    # --- Analyse des résultats ---
    final_prices = sim_data[-1]
    esperance_finale = np.mean(final_prices)
    pire_cas_5pct = np.percentile(final_prices, 5) # VaR 95%
    meilleur_cas_95pct = np.percentile(final_prices, 95)
    
    return sim_data, esperance_finale, pire_cas_5pct, meilleur_cas_95pct

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Sélection diversifiée : 
    # Air Liquide (Solide), LVMH (Luxe), Total (Rendement), Interparfums (PME Croissance), Valneva (PME Volatile)
    mes_actions = ["AIR.PA", "MC.PA", "TTE.PA", "ITP.PA", "VLA.PA"]

    # 1. Récupération
    data = recuperer_donnees(mes_actions)

    # 2. Optimisation de Portefeuille (Frontière Efficiente)
    ret, vol, sharpe, best_idx = optimiser_portefeuille(data)

    # Graphique 1 : La Frontière Efficiente
    plt.figure(figsize=(10, 6))
    plt.scatter(vol, ret, c=sharpe, cmap='viridis', marker='o', s=10, alpha=0.5)
    plt.colorbar(label='Ratio de Sharpe (Rentabilité/Risque)')
    plt.scatter(vol[best_idx], ret[best_idx], c='red', s=100, edgecolors='black', label='Portfolio Optimal')
    plt.title('Frontière Efficiente : Réduction de la Variance par Diversification')
    plt.xlabel('Volatilité Annualisée (Risque)')
    plt.ylabel('Rendement Espéré Annualisé')
    plt.legend()
    plt.show()

    # 3. Simulation Monte Carlo sur la valeur la plus volatile (ex: Valneva ou Interparfums)
    ticker_focus = "AIR.PA" 
    sims, esp, pire, meilleur = simulation_monte_carlo(data, ticker_focus)

    # Graphique 2 : Le Cône d'Incertitude
    plt.figure(figsize=(10, 6))
    plt.plot(sims, color='gray', alpha=0.1) # Les 1000 scénarios
    plt.axhline(y=data[ticker_focus].iloc[-1], color='blue', linestyle='--', label='Prix Actuel')
    plt.axhline(y=esp, color='green', linewidth=2, label=f'Espérance Moyenne ({esp:.1f}€)')
    plt.axhline(y=pire, color='red', linestyle='--', label=f'Scénario Pessimiste 95% ({pire:.1f}€)')
    plt.title(f'Simulation Monte Carlo sur 1 an : {ticker_focus}')
    plt.xlabel('Jours Futurs')
    plt.ylabel('Prix (€)')
    plt.legend()
    plt.show()

    print(f"\n🔮 RÉSULTATS MONTE CARLO POUR {ticker_focus} (1 AN) :")
    print(f"Prix actuel : {data[ticker_focus].iloc[-1]:.2f} €")
    print(f"Espérance mathématique : {esp:.2f} €")
    print(f"Risque (VaR 95%) : Il y a 5% de chances que le prix tombe sous {pire:.2f} €")