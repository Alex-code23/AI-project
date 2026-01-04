import yfinance as yf
import pandas as pd
import time

def analyser_societe_complete(ticker):
    """
    Récupère l'historique + les infos statiques + calcule la dynamique
    et retourne une ligne par année.
    """
    print(f"🔄 Analyse : {ticker}...", end="\r")
    data_rows = []
    
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Infos "Snapshot" (Statiques)
        try:
            info = stock.info
        except:
            info = {}
            
        nom = info.get('longName', ticker)
        secteur = info.get('sector', 'N/A')
        pays = info.get('country', 'N/A')
        prix = info.get('currentPrice', 'N/A')
        devise = info.get('currency', 'EUR')
        cap_market = round(info.get('marketCap', 0) / 1_000_000, 1)
        per = info.get('trailingPE', 'N/A')

        # 2. Récupération Historique
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        
        if income_stmt.empty:
            return []

        # --- 3. CALCUL DYNAMIQUE (3 dernières années) ---
        croissance_ca_3ans = 0
        croissance_rn_3ans = 0
        tendance = "Stable"
        
        try:
            if len(income_stmt.columns) >= 3:
                ca_recent = income_stmt.iloc[:, 0].get('Total Revenue', 0)
                ca_vieux = income_stmt.iloc[:, 2].get('Total Revenue', 0)
                rn_recent = income_stmt.iloc[:, 0].get('Net Income', 0)
                rn_vieux = income_stmt.iloc[:, 2].get('Net Income', 0)

                if ca_vieux > 0:
                    croissance_ca_3ans = ((ca_recent - ca_vieux) / ca_vieux) * 100
                    if croissance_ca_3ans > 20: tendance = "🚀 Hyper-Croissance"
                    elif croissance_ca_3ans > 5: tendance = "✅ Croissance"
                    elif croissance_ca_3ans < -5: tendance = "⚠️ Déclin"
                if abs(rn_vieux) > 0:
                    croissance_rn_3ans = ((rn_recent - rn_vieux) / abs(rn_vieux)) * 100
                
                
        except:
            pass

        # --- 4. BOUCLE SUR CHAQUE ANNÉE ---
        dates_disponibles = income_stmt.columns
        
        for date in dates_disponibles:
            annee = date.year
            
            try:
                ca = income_stmt.loc['Total Revenue', date] if 'Total Revenue' in income_stmt.index else 0
                resultat_net = income_stmt.loc['Net Income', date] if 'Net Income' in income_stmt.index else 0
            except KeyError:
                continue 
            
            equity = 0
            dette = 0
            if not balance_sheet.empty and date in balance_sheet.columns:
                equity = balance_sheet.loc['Stockholders Equity', date] if 'Stockholders Equity' in balance_sheet.index else 0
                dette = balance_sheet.loc['Total Debt', date] if 'Total Debt' in balance_sheet.index else 0

            marge_nette = (resultat_net / ca * 100) if ca != 0 else 0
            roe = (resultat_net / equity * 100) if equity != 0 else 0
            gearing = (dette / equity) if equity != 0 else 0

            ligne = {
                'Ticker': ticker,
                'Nom': nom,
                'Secteur': secteur,
                'Pays': pays,
                'Prix Actuel': prix,
                'Devise': devise,
                'Capitalisation (M)': cap_market,
                'Année': annee,
                'Chiffre d\'Affaires (M)': round(ca / 1_000_000, 1),
                'Résultat Net (M)': round(resultat_net / 1_000_000, 1),
                'Marge Nette (%)': round(marge_nette, 2),
                'ROE (%)': round(roe, 2),
                'Dette/Equity': round(gearing, 2),
                'PER (Trailing)': per,
                'Croissance CA 3ans (%)': round(croissance_ca_3ans, 2),
                'Croissance Bénéfice 3ans (%)': round(croissance_rn_3ans, 2),
                'Tendance': tendance
            }
            data_rows.append(ligne)
            
        return data_rows

    except Exception as e:
        return []

def generer_csv_final(liste_tickers, nom_fichier):
    toutes_les_donnees = []
    total = len(liste_tickers)
    
    print(f"🚀 Démarrage de l'analyse sur {total} sociétés européennes...")
    print("☕ Prenez un café, cela va prendre environ 3-4 minutes.\n")
    
    for i, ticker in enumerate(liste_tickers):
        print(f"[{i+1}/{total}] ", end="")
        lignes = analyser_societe_complete(ticker)
        if lignes:
            toutes_les_donnees.extend(lignes)
        time.sleep(0.25) # Pause légèrement augmentée pour 80 requêtes
        
    df = pd.DataFrame(toutes_les_donnees)
    
    if df.empty:
        print("❌ Aucune donnée récupérée.")
        return

    # Tri et Sélection Colonnes
    df = df.sort_values(by=['Tendance', 'Ticker', 'Année'], ascending=[True, True, False])
    
    colonnes_ordre = [
        'Ticker', 'Nom', 'Secteur', 'Pays', 'Prix Actuel', 'Devise', 
        'Capitalisation (M)', 'Année', 
        'Chiffre d\'Affaires (M)', 'Résultat Net (M)', 
        'Marge Nette (%)', 'ROE (%)', 'Dette/Equity', 
        'PER (Trailing)', 
        'Croissance CA 3ans (%)', 'Croissance Bénéfice 3ans (%)', 'Tendance'
    ]
    
    # Filtrage des colonnes existantes
    cols_existantes = [c for c in colonnes_ordre if c in df.columns]
    df = df[cols_existantes]

    if not nom_fichier.endswith('.csv'):
        nom_fichier = f"{nom_fichier}.csv"
    df.to_csv(nom_fichier, index=False, sep=';', encoding='utf-8-sig')
    print(f"\n\n✅ TERMINÉ ! Fichier généré : {nom_fichier}")

# ==============================================================================
# LISTES DES VALEURS (Tickers Yahoo Finance)
# ==============================================================================

# LISTE 1 : PEA CLASSIQUE (Grandes valeurs Européennes)
# Mix de pays : FR, DE (Allemagne), IT (Italie), ES (Espagne), PL (Pologne)
actions_pea_europe = [
    # --- FRANCE (CAC 40 Leaders) ---
    "MC.PA", "OR.PA", "TTE.PA", "AIR.PA", "SAN.PA", "BNP.PA", "SU.PA", "AI.PA",
    "KER.PA", "EL.PA", "DG.PA", "BN.PA", 
    
    # --- ALLEMAGNE (DAX - Tech, Auto, Industrie) ---
    "SAP.DE",       # Tech (Logiciel)
    "SIE.DE",       # Siemens (Industrie)
    "ALV.DE",       # Allianz (Assurance)
    "DTE.DE",       # Deutsche Telekom
    "MBG.DE",       # Mercedes-Benz
    "BMW.DE",       # BMW
    "ADS.DE",       # Adidas
    "BAS.DE",       # BASF (Chimie)
    "DHL.DE",       # DHL Group (Logistique)
    "MUV2.DE",      # Munich Re (Assurance)

    # --- ITALIE (FTSE MIB - Luxe, Finance, Energie) ---
    "RACE.MI",      # Ferrari (Luxe/Auto)
    "ISP.MI",       # Intesa Sanpaolo (Banque)
    "ENEL.MI",      # Enel (Energie/Util)
    "ENI.MI",       # Eni (Pétrole/Gaz)
    "UCG.MI",       # Unicredit (Banque)
    "MONC.MI",      # Moncler (Luxe)
    "PRY.MI",       # Prysmian (Câbles)
    "STLAM.MI",     # Stellantis (Auto)

    # --- ESPAGNE (IBEX 35 - Retail, Tourisme, Bank) ---
    "ITX.MC",       # Inditex (Zara - Retail)
    "IBE.MC",       # Iberdrola (Energie Verte)
    "SAN.MC",       # Santander (Banque)
    "BBVA.MC",      # BBVA (Banque)
    "AMS.MC",       # Amadeus (Tech/Tourisme)
    "TEF.MC",       # Telefonica

    # --- POLOGNE (WIG - Croissance Est) ---
    "CDR.WA",       # CD Projekt (Jeux Vidéo - Cyberpunk/Witcher)
    "DNP.WA",       # Dino Polska (Retail/Supermarché - Forte croissance)
    "ALE.WA",       # Allegro (E-commerce - Le "Amazon" polonais)
    "PKO.WA",       # PKO Bank (Banque)
    "KGH.WA"        # KGHM (Mines/Cuivre)
]

# LISTE 2 : PEA-PME (Petites & Moyennes Capitalisations)
# Focus France (plus simple pour l'éligibilité) + quelques européennes
actions_pea_pme = [
    # --- TECH & ESN (Services Numériques) ---
    "NRO.PA",       # Neurones
    "WAVE.PA",      # Wavestone
    "SII.PA",       # SII
    "AUB.PA",       # Aubay
    "INF.PA",       # Infotel
    "SWP.PA",       # Sword Group
    "ALVGA.PA",     # Visiativ

    # --- INDUSTRIE & MATÉRIAUX ---
    "ALTHE.PA",     # Thermador Groupe (Distribution)
    "MRN.PA",       # Mersen (Matériaux avancés)
    "JAC.PA",       # Jacquet Metals
    "EXE.PA",       # Exel Industries (Agricole)
    "MAN.PA",       # Manitou (Chariots élévateurs)
    "PIG.PA",       # Haulotte (Nacelles)
    "SEQ.PA",       # Sequana Medical

    # --- SANTÉ & BIOTECH ---
    "VIRP.PA",      # Virbac (Santé animale - Grosse PME)
    "VLA.PA",       # Valneva (Vaccins)
    "ABVX.PA",      # Abivax (Biotech)
    "GBT.PA",       # Guerbet (Imagerie médicale)
    "IPH.PA",       # Innate Pharma
    
    # --- CONSO & LOISIRS ---
    "ITP.PA",       # Interparfums
    "BEN.PA",       # Beneteau (Bateaux)
    "TRI.PA",       # Trigano (Camping-cars)
    "CATANA.PA",    # Catana Group (Catamarans)
    "ALMDO.PA",     # LDLC (Commerce info)
    "BIG.PA",       # Bigben Interactive

    # --- FINANCE & IMMO ---
    "ABCA.PA",      # ABC Arbitrage
    "IDIP.PA",      # IDI (Private Equity)
    "TKO.PA",       # Tikehau Capital (Gestion d'actifs - Borderline PME selon date)
    "NXI.PA",       # Nexity (Immo - Vérifier éligibilité selon capitalisation)

    # --- ÉNERGIE VERTE ---
    "VLTSA.PA",     # Voltalia
    "ALRS.PA"       # Altamir
]

# Fusion des deux listes pour le scan
liste_complete = actions_pea_europe + actions_pea_pme

if __name__ == "__main__":
    generer_csv_final(actions_pea_europe, "PEA_Europe.csv")
    generer_csv_final(actions_pea_pme, "PEA_PME.csv")
