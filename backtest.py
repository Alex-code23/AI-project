import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from gestionnaire import GestionnairePortefeuille

# Configuration
START_DATE = "2024-01-01"  # Date de début du backtest
INITIAL_CAPITAL = 10000  # 10k€ de départ
REBALANCE_DAYS = 90      # Rebalancement tous les mois (30 jours)
PAST_DAYS = 120         
BENCHMARK = "^FCHI"      # CAC 40 pour comparaison

def run_backtest():
    print(f"🚀 Démarrage du Backtest ({START_DATE} -> Aujourd'hui)...")
    
    # 1. Initialisation
    # On crée une instance vide juste pour récupérer la liste des tickers de l'univers
    temp_bot = GestionnairePortefeuille({}) 
    universe_tickers = temp_bot.tickers
    
    # On télécharge TOUT l'historique une seule fois (Optimisation vitesse)
    print("📥 Téléchargement global des données (Portefeuille + Benchmark)...")
    all_tickers = list(set(universe_tickers + [BENCHMARK]))
    full_data = yf.download(all_tickers, start="2023-01-01", progress=True, auto_adjust=False)['Adj Close']
    
    # Nettoyage des tickers invalides
    full_data = full_data.dropna(axis=1, how='all')
    valid_tickers = [t for t in universe_tickers if t in full_data.columns]
    
    # 2. Boucle Temporelle
    current_date = pd.Timestamp(START_DATE)
    end_date = pd.Timestamp.now()
    
    portfolio = {} # Portefeuille vide au début (tout en cash)
    cash = INITIAL_CAPITAL
    
    history = [] # Pour stocker la courbe de valeur
    
    # On avance dans le temps par pas de REBALANCE_DAYS
    while current_date < end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n📅 --- Période : {date_str} ---")
        
        # Vérifier si on a des données à cette date (sinon on recule d'un jour ou deux pour trouver un jour ouvré)
        # Simple hack : on prend la dernière donnée dispo avant ou égale à current_date
        available_data = full_data[full_data.index <= current_date]
        if available_data.empty:
            current_date += timedelta(days=1)
            continue
            
        last_prices = available_data.iloc[-1]
        
        # --- A. Calcul Valeur Portefeuille ---
        valeur_actions = sum(portfolio.get(t, 0) * last_prices.get(t, 0) for t in portfolio)
        total_value = cash + valeur_actions
        
        # --- B. Lancement du Gestionnaire (IA + Math) ---
        # On instancie le bot avec le portefeuille actuel
        bot = GestionnairePortefeuille(portfolio, part_dividende=0.40)
        
        # Injection des données coupées à la date actuelle (Simulation du passé)
        # On passe 'full_data' en cache pour éviter le retéléchargement
        bot.f0_recuperer_donnees(date_limite=date_str, data_cache=full_data[valid_tickers])
        
        # Si pas assez de données (ex: début 2024), on saute
        if len(bot.hist_data) < 60:
            print("⚠️ Pas assez d'historique pour l'analyse. On attend.")
            current_date += timedelta(days=REBALANCE_DAYS)
            continue

        # Exécution de la stratégie
        try:
            bot.f1_analyse_ia_et_risque(jours_futurs=REBALANCE_DAYS, jours_historique=PAST_DAYS)
            bot.f2_optimisation_math()
            
            # --- C. Exécution des Ordres (Virtuel) ---
            # Le bot nous donne des poids cibles (bot.final_weights)
            # On rebalance tout le portefeuille selon ces poids
            
            new_portfolio = {}
            new_cash = total_value # On remet tout en "pot commun" pour redistribuer
            
            # Frais de transaction simulés (ex: 0.2% par rebalancement global)
            transaction_cost = total_value * 0.002
            new_cash -= transaction_cost
            
            for t, weight in bot.final_weights.items():
                if weight > 0 and t in last_prices and not np.isnan(last_prices[t]):
                    price = last_prices[t]
                    amount_to_invest = (total_value - transaction_cost) * weight
                    qty = int(amount_to_invest / price)
                    
                    if qty > 0:
                        new_portfolio[t] = qty
                        cost = qty * price
                        new_cash -= cost
            
            portfolio = new_portfolio
            cash = new_cash

            print(f"💰 Valeur Actuelle : {total_value:.2f} €")
            
        except Exception as e:
            print(f"❌ Erreur stratégie à {date_str}: {e}")
        
        # --- D. Sauvegarde pour le graphique ---
        # Valeur du Benchmark (Normalisée base 100 ou base Capital Initial)
        bench_price = available_data[BENCHMARK].iloc[-1]
        
        history.append({
            'Date': current_date,
            'Portfolio_Value': total_value,
            'Benchmark_Price': bench_price
        })
        
        # Avance rapide
        current_date += timedelta(days=REBALANCE_DAYS)

    # 3. Analyse des Résultats
    df_res = pd.DataFrame(history)
    
    # Normalisation du Benchmark pour qu'il commence au même montant que le portefeuille
    start_bench = df_res['Benchmark_Price'].iloc[0]
    df_res['Benchmark_Value'] = df_res['Benchmark_Price'] * (INITIAL_CAPITAL / start_bench)
    
    # Calcul Performance Finale
    final_val = df_res['Portfolio_Value'].iloc[-1]
    perf_algo = ((final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    final_bench = df_res['Benchmark_Value'].iloc[-1]
    perf_bench = ((final_bench - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print("\n" + "="*50)
    print(f"🏁 RÉSULTATS BACKTEST ({START_DATE} -> Aujourd'hui)")
    print("="*50)
    print(f"Capital Initial : {INITIAL_CAPITAL} €")
    print(f"Capital Final   : {final_val:.2f} €")
    print(f"Performance Moi  : {perf_algo:+.2f} %")
    print(f"Performance CAC40: {perf_bench:+.2f} %")
    
    if perf_algo > perf_bench:
        print("🏆 J'ai battu le marché !")
    else:
        print("🐢 J'ai sous-performé le marché.")

    # 4. Visualisation
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 6))
    
    plt.plot(df_res['Date'], df_res['Portfolio_Value'], label='Portefeuille IA', color='blue', linewidth=2)
    plt.plot(df_res['Date'], df_res['Benchmark_Value'], label='CAC 40 (Benchmark)', color='gray', linestyle='--', alpha=0.7)
    
    plt.title(f"Backtest Performance : IA vs Marché ({START_DATE} - Now)")
    plt.xlabel("Date")
    plt.ylabel("Valeur du Portefeuille (€)")
    plt.legend()
    plt.fill_between(df_res['Date'], df_res['Portfolio_Value'], df_res['Benchmark_Value'], alpha=0.1, color='blue')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()
