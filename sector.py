import yfinance as yf
import pandas as pd
import time

def analyser_investissement_futur(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # On a besoin du Cash Flow (pour CAPEX) et Income Stmt (pour R&D et Revenue)
        cashflow = stock.cashflow
        income = stock.income_stmt
        
        if cashflow.empty or income.empty:
            return None
            
        # --- 1. CAPEX (Investissement physique) ---
        # "Capital Expenditure" est souvent négatif dans les comptes (sortie d'argent)
        # On le passe en positif pour le calcul
        try:
            capex_n = abs(cashflow.loc['Capital Expenditure'].iloc[0])
            capex_n_1 = abs(cashflow.loc['Capital Expenditure'].iloc[1])
        except:
            capex_n = 0
            capex_n_1 = 0
            
        # --- 2. R&D (Investissement intellectuel) ---
        try:
            rnd_n = income.loc['Research And Development'].iloc[0]
        except:
            rnd_n = 0 # Beaucoup d'entreprises n'ont pas de R&D
            
        # --- 3. REVENU ---
        try:
            revenue_n = income.loc['Total Revenue'].iloc[0]
        except:
            return None

        # --- CALCUL DES INDICATEURS AVANCÉS ---
        
        # Intensité de l'investissement (% du CA réinvesti)
        # Plus c'est haut, plus l'entreprise parie sur l'avenir
        intensite_investissement = ((capex_n + rnd_n) / revenue_n) * 100
        
        # Accélération du CAPEX (Est-ce qu'ils investissent PLUS qu'avant ?)
        croissance_capex = 0
        if capex_n_1 > 0:
            croissance_capex = ((capex_n - capex_n_1) / capex_n_1) * 100

        return {
            'Ticker': ticker,
            'Nom': info.get('longName', ticker),
            'Secteur': info.get('sector', 'N/A'),
            'Intensité Investissement (%)': round(intensite_investissement, 2),
            'Croissance CAPEX 1an (%)': round(croissance_capex, 2),
            'R&D / Revenue (%)': round((rnd_n / revenue_n * 100), 2) if revenue_n else 0,
            'PER (Est.)': info.get('forwardPE', 'N/A') # Si PER élevé, le marché a déjà vu le coup
        }

    except Exception as e:
        return None

def scan_secteurs_porteurs(liste_tickers):
    print("🕵️ Recherche des secteurs qui investissent massivement...\n")
    resultats = []
    
    for ticker in liste_tickers:
        res = analyser_investissement_futur(ticker)
        if res:
            resultats.append(res)
        time.sleep(0.1)
        
    df = pd.DataFrame(resultats)
    
    if df.empty: return
    
    # On agrège par SECTEUR pour voir la tendance globale
    # On fait la moyenne des indicateurs par secteur
    df_secteurs = df.groupby('Secteur')[['Intensité Investissement (%)', 'Croissance CAPEX 1an (%)']].mean()
    
    # On trie par Intensité d'Investissement
    df_secteurs = df_secteurs.sort_values(by='Intensité Investissement (%)', ascending=False)
    
    print("--- CLASSEMENT DES SECTEURS PAR INVESTISSEMENT FUTUR ---")
    print(df_secteurs)
    
    print("\n--- TOP 5 ENTREPRISES QUI PRÉPARENT DEMAIN ---")
    # On cherche forte intensité ET forte croissance du CAPEX
    top_cies = df.sort_values(by='Intensité Investissement (%)', ascending=False).head(5)
    print(top_cies[['Nom', 'Secteur', 'Intensité Investissement (%)', 'Croissance CAPEX 1an (%)']].to_string(index=False))

# --- LISTE DE TEST DIVERSIFIÉE (Tech, Energie, Industrie, Pharma) ---
liste_test = [
    "TTE.PA", "AIR.PA", "SAN.PA", "MC.PA", # France
    "NVDA", "TSLA", "LLY", "AMZN", # US (souvent en avance sur les tendances)
    "ASML.AS", "SAP.DE", "SIE.DE", # Europe Tech/Indus
    "VLA.PA", "SOIT.PA", "NEOEN.PA" # Growth France
]

scan_secteurs_porteurs(liste_test)