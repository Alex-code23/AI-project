import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime

# --- IMPORTATION DE VOS MODULES PRÉCÉDENTS ---
try:
    import RMT as rmt
    import pred_AI as ai
    from risk import simulation_monte_carlo

except ImportError:
    print("⚠️ Modules RMT/IA non trouvés. Mode simulation activé.")
    # Mock functions pour que le code tourne même sans les fichiers
    class rmt:
        @staticmethod
        def denoise_correlation_matrix(returns):
            return returns.corr().values # Fallback sur corrélation classique
    class ai:
        @staticmethod
        def predict_return(ticker):
            return np.random.uniform(0.05, 0.20) # Simulation prédiction IA

class GestionnairePortefeuille:
    TRADING_DAYS = 252

    def __init__(self, portfolio_actuel, part_dividende=0.40):
        """
        portfolio_actuel : Dict {'Ticker': Quantité}
        part_dividende : % cible pour la poche rendement (ex: 0.40 pour 40%)
        """
        self.portfolio = portfolio_actuel
        self.target_yield_ratio = part_dividende
        
        # Univers d'investissement élargi (CAC 40 + existants)
        universe = [
            "AIR.PA", "AI.PA", "ALO.PA", "MT.AS", "CS.PA", "BNP.PA", "EN.PA", "CAP.PA", 
            "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "EL.PA", "RMS.PA", "KER.PA", "OR.PA", 
            "LR.PA", "MC.PA", "ML.PA", "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", 
            "SGO.PA", "SAN.PA", "SU.PA", "GLE.PA", "STLAP.PA", "STMPA.PA", "TEP.PA", 
            "HO.PA", "TTE.PA", "VIE.PA", "DG.PA", "VIV.PA"
        ]
        # On combine avec le portefeuille actuel pour ne rien oublier
        self.tickers = list(set(list(portfolio_actuel.keys()) + universe))
        self.capital_total = 0
        self.prices = {}
        self.infos = {}

    def f0_recuperer_donnees(self, date_limite=None, data_cache=None):
        """
        date_limite : (str) Date 'YYYY-MM-DD' pour simuler le passé (Backtest).
        data_cache : (DataFrame) Données déjà téléchargées pour éviter les appels API répétés.
        """
        if data_cache is not None:
            data = data_cache.copy()
        else:
            print("📥 Récupération des données marché...")
            data = yf.download(self.tickers, period="2y", progress=False, auto_adjust=False)['Adj Close']
        
        # Nettoyage : on ne garde que les tickers qui ont des données
        data = data.dropna(axis=1, how='all')
        self.tickers = [t for t in self.tickers if t in data.columns]
        self.hist_data = data[self.tickers]
        
        # Si on est en mode Backtest, on coupe les données après la date limite
        if date_limite:
            self.hist_data = self.hist_data[self.hist_data.index <= date_limite]

        # Récupération des infos fondamentales (Dividendes & Prix actuel)
        for t in self.tickers:
            try:
                stock = yf.Ticker(t)
                last_price = self.hist_data[t].iloc[-1]
                self.prices[t] = last_price
                
                # Calcul de la valeur actuelle du portefeuille
                qty = self.portfolio.get(t, 0)
                self.capital_total += last_price * qty
                
                # Info Dividende (Yield)
                yield_val = stock.info.get('dividendYield', 0)
                if yield_val is None: yield_val = 0
                
                # --- INDICATEURS FONDAMENTAUX & TECHNIQUES ---
                pe = stock.info.get('trailingPE', None)
                growth = stock.info.get('revenueGrowth', None)
                roe = stock.info.get('returnOnEquity', None)

                # Calcul RSI (14 jours)
                delta = self.hist_data[t].diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50

                self.infos[t] = {
                    'price': last_price,
                    'yield': yield_val,
                    'is_dividend_stock': yield_val > 0.05, # Seuil arbitraire 3% pour être "Yield"
                    'pe': pe, 'growth': growth, 'roe': roe, 'rsi': current_rsi
                }
            except Exception as e:
                print(f"Erreur sur {t}: {e}")

        # print(f"💰 Valeur Portefeuille Actuel : {self.capital_total:.2f} €")

    def f1_analyse_ia_et_risque(self, jours_futurs=15, jours_historique=120):
        print("🧠 Lancement de MC (Prédictions & RMT)...")
        
        # 1. Prédictions de rendement (Expected Returns via IA)
        self.expected_returns = {}
        for t in self.tickers:
            # appelle Montecarlo
            _, future_price, _, _ = simulation_monte_carlo(self.hist_data, t, jours_futurs=jours_futurs, jours_historique=jours_historique, simulations=1000)
            
            current_price = self.prices.get(t, 0)
            if current_price > 0:
                self.expected_returns[t] = (future_price - current_price) / current_price
            else:
                self.expected_returns[t] = 0.0
            

        # 2. Nettoyage de la Matrice de Risque (RMT)
        # On restreint l'historique à la fenêtre demandée (ex: 120 jours) pour la cohérence
        data_window = self.hist_data.iloc[-jours_historique:]
        returns = np.log(data_window / data_window.shift(1)).dropna()
        # On utilise la fonction RMT codée précédemment
        rmt_res = rmt.denoise_correlation_matrix(returns)
        
        if isinstance(rmt_res, tuple):
            _, self.clean_cov_matrix, _, _, _ = rmt_res
        else:
            self.clean_cov_matrix = rmt_res
        
        # On doit reconvertir la corrélation nettoyée en covariance pour l'optimiseur
        # Cov = Corr_clean * std(i) * std(j)
        std_devs = returns.std() * np.sqrt(self.TRADING_DAYS)
        D = np.diag(std_devs)
        # Approximation : On utilise la matrice RMT pour la structure, 
        # et les volatilités historiques pour l'échelle.
        self.clean_cov_matrix = np.dot(D, np.dot(self.clean_cov_matrix, D))

    def f2_optimisation_math(self):
        """
        L'étape cruciale : On sépare l'univers en 2 et on optimise.
        """
        print("📐 Optimisation sous contraintes...")
        
        # --- FILTRAGE INTELLIGENT (Santé & Opportunité) ---
        active_tickers = []
        for t in self.tickers:
            info = self.infos.get(t, {})
            
            # 1. CRITÈRES DE VENTE (Santé financière dégradée)
            # Exclure si croissance très négative ou PER aberrant
            if info.get('growth') is not None and info['growth'] < -0.05:
                print(f"❌ {t} exclu : Croissance négative ({info['growth']:.1%})")
                continue
            if info.get('pe') is not None and info['pe'] > 80:
                print(f"❌ {t} exclu : Survalorisé (PER {info['pe']:.1f})")
                continue

            # 2. CRITÈRES D'ACHAT (Opportunité)
            # Ajustement du rendement espéré (IA) selon RSI et Fondamentaux
            score = 1.0
            rsi = info.get('rsi', 50)
            
            if rsi < 30: score *= 1.15       # Survendue -> Rebond probable
            elif rsi > 70: score *= 0.85     # Surachetée -> Risque correction
            
            if info.get('roe') and info['roe'] > 0.15: score *= 1.05    # Qualité
            if info.get('growth') and info['growth'] > 0.10: score *= 1.05 # Croissance
            
            self.expected_returns[t] *= score
            active_tickers.append(t)
        
        # Séparation des tickers (sur la liste filtrée)
        tickers_yield = [t for t in active_tickers if self.infos[t]['is_dividend_stock']]
        tickers_growth = [t for t in active_tickers if not self.infos[t]['is_dividend_stock']]
        
        # Fonction interne d'optimisation (Ratio de Sharpe)
        def optimiser_sous_poche(sub_tickers, objective='sharpe'):
            if not sub_tickers: return {}
            
            n = len(sub_tickers)
            # Indices dans la matrice globale
            indices = [self.tickers.index(t) for t in sub_tickers]
            
            # Sous-matrices
            sub_cov = self.clean_cov_matrix[np.ix_(indices, indices)]
            sub_rets = np.array([self.expected_returns[t] for t in sub_tickers])
            
            def get_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(sub_cov, weights)))
                
            def get_sharpe(weights):
                ret = np.sum(sub_rets * weights)
                vol = get_volatility(weights)
                return - (ret / vol) # On minimise l'opposé du Sharpe
            
            # Contrainte : Somme poids = 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.0, 1.0) for _ in range(n)) # Pas de vente à découvert (PEA)
            
            init_guess = [1/n] * n
            
            res = minimize(get_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            
            return dict(zip(sub_tickers, res.x))

        # --- OPTIMISATION POCHE RENDEMENT (X %) ---
        # Ici on pourrait minimiser la volatilité pure au lieu du Sharpe pour sécuriser
        w_yield = optimiser_sous_poche(tickers_yield)
        
        # --- OPTIMISATION POCHE CROISSANCE (100-X %) ---
        w_growth = optimiser_sous_poche(tickers_growth)
        
        # Fusion et Pondération Globale
        self.final_weights = {}
        for t in self.tickers:
            if t in w_yield:
                self.final_weights[t] = w_yield[t] * self.target_yield_ratio
            elif t in w_growth:
                self.final_weights[t] = w_growth[t] * (1 - self.target_yield_ratio)
            else:
                self.final_weights[t] = 0.0

    def f3_generer_ordres(self):
        print("\n" + "="*50)
        print("🤖 RAPPORT DE REBALANCEMENT (15 JOURS)")
        print("="*50)
        
        df_ordres = []
        
        for t in self.tickers:
            target_amount = self.capital_total * self.final_weights[t]
            current_amount = self.prices[t] * self.portfolio.get(t, 0)
            diff = target_amount - current_amount
            
            # On force un nombre entier d'actions (troncature) car on ne peut pas acheter 2.4 actions
            nb_shares_diff = int(diff / self.prices[t])
            montant_reel = nb_shares_diff * self.prices[t]
            
            action = "CONSERVER"
            if nb_shares_diff > 0: action = "🟢 ACHETER"
            elif nb_shares_diff < 0: action = "🔴 VENDRE"
            
            if action != "CONSERVER":
                df_ordres.append({
                    'Ticker': t,
                    'Action': action,
                    'Quantité': abs(nb_shares_diff),
                    'Montant Est.': round(abs(montant_reel), 2),
                    'Poids Cible (%)': round(self.final_weights[t]*100, 1),
                    'IA Return Prev.': round(self.expected_returns[t]*100, 1)
                })
        
        if not df_ordres:
            print("✅ Portefeuille déjà optimal. Rien à faire.")
        else:
            df = pd.DataFrame(df_ordres)
            print(df.to_string(index=False))
            
            print("\n💡 Note :")
            print(f"- Objectif Rendement : {self.target_yield_ratio*100}% du capital")
            print("- Les pondérations sont basées sur la covariance nettoyée (RMT).")

        # Estimation de la valeur future du portefeuille optimisé
        valeur_future = sum(self.capital_total * self.final_weights[t] * (1 + self.expected_returns[t]) for t in self.tickers)
        plus_value = valeur_future - self.capital_total

        print("\n" + "="*50)
        print("🔮 ESTIMATION FINALE DU PORTEFEUILLE (15 JOURS)")
        print(f"Valeur Actuelle : {self.capital_total:.2f} €")
        print(f"Valeur Projetée : {valeur_future:.2f} €")
        print(f"Gain Espéré     : {plus_value:+.2f} € ({plus_value/self.capital_total*100:+.2f}%)")
        print("="*50)

# ==============================================================================
# EXECUTION (A LANCER TOUS LES 15 JOURS)
# ==============================================================================
if __name__ == "__main__":
    # 1. Portefeuille actuel 
    mon_portefeuille = {
        "AIR.PA": 10,   # Air Liquide
        "TTE.PA": 50,   # TotalEnergies (Dividende)
        "MC.PA": 2,     # LVMH
        "BNP.PA": 30,   # BNP (Dividende)
        "ITP.PA": 15,   # Interparfums (Croissance)
        "VLA.PA": 100   # Valneva (Spéculatif)
    }

    # 2. Configuration : Je veux 40% de dividendes sûrs, 60% de performance 
    bot = GestionnairePortefeuille(mon_portefeuille, part_dividende=0.40)
    # 3. Lancement du processus
    bot.f0_recuperer_donnees()
    bot.f1_analyse_ia_et_risque()
    bot.f2_optimisation_math()
    bot.f3_generer_ordres()