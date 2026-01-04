import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration du style des graphiques
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def charger_donnees(fichiers):
    """Charge et concatène les fichiers CSV spécifiés."""
    dfs = []
    for f in fichiers:
        path = f
        # Gestion de la double extension potentielle si test.py n'a pas été corrigé
        if not os.path.exists(path) and os.path.exists(path + ".csv"):
            path = path + ".csv"
            
        if os.path.exists(path):
            print(f"📂 Chargement de {path}...")
            try:
                # Le séparateur est ';' dans le script de génération
                df = pd.read_csv(path, sep=';')
                dfs.append(df)
            except Exception as e:
                print(f"❌ Erreur lors de la lecture de {path}: {e}")
        else:
            print(f"⚠️ Fichier introuvable : {f}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def nettoyer_donnees(df):
    """S'assure que les colonnes numériques sont bien typées."""
    cols_num = [
        'Capitalisation (M)', 'Chiffre d\'Affaires (M)', 'Résultat Net (M)', 
        'Marge Nette (%)', 'ROE (%)', 'Dette/Equity', 'PER (Trailing)', 
        'Croissance CA 3ans (%)', 'Croissance Bénéfice 3ans (%)'
    ]
    
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Nettoyage des emojis dans la colonne Tendance pour éviter les warnings Matplotlib
    if 'Tendance' in df.columns:
        df['Tendance'] = df['Tendance'].astype(str).str.replace('[🚀✅⚠️\ufe0f]', '', regex=True).str.strip()

    return df

def plot_vue_densemble(df_latest, output_dir):
    """Affiche la répartition par secteur et pays."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Répartition par Secteur
    secteurs = df_latest['Secteur'].value_counts()
    sns.barplot(x=secteurs.values, y=secteurs.index, ax=axes[0], palette="viridis", hue=secteurs.index, legend=False)
    axes[0].set_title("Répartition par Secteur")
    axes[0].set_xlabel("Nombre d'entreprises")
    
    # 2. Répartition par Pays
    pays = df_latest['Pays'].value_counts()
    sns.barplot(x=pays.values, y=pays.index, ax=axes[1], palette="rocket", hue=pays.index, legend=False)
    axes[1].set_title("Répartition par Pays")
    axes[1].set_xlabel("Nombre d'entreprises")
    
    plt.tight_layout()
    path = os.path.join(output_dir, "vue_densemble.png")
    plt.savefig(path)
    plt.close()

def plot_croissance_rentabilite(df_latest, output_dir):
    """Scatter plot : Croissance vs Marge Nette (Matrice de performance)."""
    plt.figure(figsize=(12, 8))
    
    # Filtrage des valeurs extrêmes pour la lisibilité du graphique
    data = df_latest[
        (df_latest['Croissance CA 3ans (%)'] > -50) & 
        (df_latest['Croissance CA 3ans (%)'] < 150) &
        (df_latest['Marge Nette (%)'] > -20) & 
        (df_latest['Marge Nette (%)'] < 60)
    ]
    
    sns.scatterplot(
        data=data, 
        x='Croissance CA 3ans (%)', 
        y='Marge Nette (%)',
        hue='Secteur',
        size='Capitalisation (M)',
        sizes=(50, 600),
        alpha=0.7,
        palette="deep"
    )
    
    # Lignes médianes pour diviser en 4 quadrants
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.title("Matrice de Performance : Croissance (3 ans) vs Rentabilité")
    plt.xlabel("Croissance CA 3 ans (%)")
    plt.ylabel("Marge Nette (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "croissance_rentabilite.png")
    plt.savefig(path)
    plt.close()

def plot_qualite_valorisation(df_latest, output_dir):
    """Scatter plot : ROE vs PER (Chercher la qualité à bon prix)."""
    plt.figure(figsize=(12, 8))
    
    # On garde les PER positifs et raisonnables (< 60)
    data = df_latest[
        (df_latest['PER (Trailing)'] > 0) & 
        (df_latest['PER (Trailing)'] < 60) &
        (df_latest['ROE (%)'] > 0) &
        (df_latest['ROE (%)'] < 60)
    ]
    
    sns.scatterplot(
        data=data,
        x='ROE (%)',
        y='PER (Trailing)',
        hue='Secteur',
        style='Tendance',
        s=120,
        palette="Set2"
    )
    
    plt.title("Qualité (ROE) vs Valorisation (PER)")
    plt.xlabel("ROE (%) - Rentabilité des capitaux propres")
    plt.ylabel("PER (Price Earning Ratio)")
    
    # Zone "Idéale" (ROE élevé, PER faible)
    plt.axvspan(15, 60, ymin=0, ymax=0.33, color='green', alpha=0.1, label='Zone "Value & Quality"')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    path = os.path.join(output_dir, "qualite_valorisation.png")
    plt.savefig(path)
    plt.close()

def analyser_evolution_ticker(df, ticker, output_dir):
    """Affiche l'évolution historique CA et Résultat Net pour une entreprise donnée."""
    data = df[df['Ticker'] == ticker].sort_values('Année')
    
    if data.empty:
        print(f"Pas de données trouvées pour {ticker}")
        return

    nom = data.iloc[0]['Nom']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Barres pour le CA
    color_ca = 'tab:blue'
    ax1.set_xlabel('Année')
    ax1.set_ylabel('Chiffre d\'Affaires (M€)', color=color_ca)
    ax1.bar(data['Année'], data['Chiffre d\'Affaires (M)'], color=color_ca, alpha=0.5, label='Chiffre d\'Affaires')
    ax1.tick_params(axis='y', labelcolor=color_ca)
    ax1.grid(False)
    
    # Ligne pour le Résultat Net
    ax2 = ax1.twinx()  
    color_rn = 'tab:red'
    ax2.set_ylabel('Résultat Net (M€)', color=color_rn)  
    ax2.plot(data['Année'], data['Résultat Net (M)'], color=color_rn, marker='o', linewidth=3, label='Résultat Net')
    ax2.tick_params(axis='y', labelcolor=color_rn)
    ax2.grid(False)
    
    plt.title(f"Dynamique Financière : {nom} ({ticker})")
    
    # Légende unifiée
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.tight_layout()
    path = os.path.join(output_dir, f"evolution_{ticker}.png")
    plt.savefig(path)
    plt.close()

def main():
    # Liste des fichiers générés par test.py
    fichiers = ["PEA_Europe.csv", "PEA_PME.csv"]
    
    print("🚀 Démarrage de l'analyse visuelle...")
    df = charger_donnees(fichiers)
    
    if df.empty:
        print("❌ Aucune donnée chargée. Vérifiez que vous avez lancé test.py avant.")
        return
        
    df = nettoyer_donnees(df)
    
    # Création du dossier de sortie
    output_dir = "visualisations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 Dossier créé : {output_dir}")
    
    # --- PRÉPARATION DES DONNÉES SNAPSHOT ---
    # Pour les graphiques comparatifs, on ne veut qu'une ligne par entreprise (la plus récente)
    df_snapshot = df.sort_values('Année', ascending=False).groupby('Ticker').first().reset_index()
    
    print(f"✅ Données prêtes : {len(df_snapshot)} entreprises uniques analysées.\n")
    
    # 1. VUE D'ENSEMBLE
    print("📊 Génération : Répartition Secteurs & Pays...")
    plot_vue_densemble(df_snapshot, output_dir)
    
    # 2. ANALYSE PERFORMANCE
    print("📊 Génération : Croissance vs Rentabilité...")
    plot_croissance_rentabilite(df_snapshot, output_dir)
    
    # 3. ANALYSE VALORISATION
    print("📊 Génération : Qualité vs Prix...")
    plot_qualite_valorisation(df_snapshot, output_dir)
    
    # 4. FOCUS INDIVIDUEL (Exemples automatiques)
    
    # A. La plus grosse capitalisation
    top_cap = df_snapshot.sort_values('Capitalisation (M)', ascending=False).iloc[0]
    print(f"\n🔎 Zoom sur la plus grosse capitalisation : {top_cap['Nom']}")
    analyser_evolution_ticker(df, top_cap['Ticker'], output_dir)
    
    # B. La plus forte croissance (parmi celles qui font > 100M de CA pour éviter les anomalies)
    df_growth = df_snapshot[df_snapshot['Chiffre d\'Affaires (M)'] > 100]
    if not df_growth.empty:
        top_growth = df_growth.sort_values('Croissance CA 3ans (%)', ascending=False).iloc[0]
        print(f"🔎 Zoom sur la plus forte croissance (>100M CA) : {top_growth['Nom']}")
        analyser_evolution_ticker(df, top_growth['Ticker'], output_dir)

    print(f"\n✅ Tous les graphiques ont été sauvegardés dans le dossier '{output_dir}'")

if __name__ == "__main__":
    main()
