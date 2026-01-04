import feedparser
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

class SentinelleLicornes:
    def __init__(self):
        print("🧠 Chargement des cerveaux artificiels (Modèles BERT)...")
        
        # 1. Le modèle NER (Named Entity Recognition)
        # Il sert à repérer "Mistral AI" ou "Stripe" dans une phrase.
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        
        # 2. Le modèle FinBERT (Sentiment Financier)
        # Entraîné spécifiquement sur des textes financiers, pas sur des avis de films.
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def scanner_flux_rss(self):
        """Récupère les dernières news Tech/Startup"""
        urls = [
            "http://feeds.feedburner.com/TechCrunch/", # Bible des startups
            "https://www.wired.com/feed/category/business/latest/rss",
            "https://venturebeat.com/feed/"
        ]
        
        articles = []
        print("📡 Scan des médias en cours...")
        
        for url in urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]: # On prend les 10 plus récents par flux
                articles.append({
                    'titre': entry.title,
                    'link': entry.link
                })
        return articles

    def analyser_articles(self, articles):
        resultats = []
        
        print(f"🔬 Analyse profonde de {len(articles)} articles...")
        
        for art in articles:
            text = art['titre']
            
            # --- ETAPE A : Extraction des Entités (Qui ?) ---
            # On cherche les entités de type 'ORG' (Organisation/Entreprise)
            entities = self.ner_pipeline(text)
            compagnies = [e['word'] for e in entities if e['entity_group'] == 'ORG']
            
            # Si aucune entreprise n'est citée, on passe
            if not compagnies:
                continue
                
            # --- ETAPE B : Analyse de Sentiment (Positif/Négatif ?) ---
            sentiment_output = self.sentiment_pipeline(text)[0]
            score = sentiment_output['score']
            label = sentiment_output['label']
            
            # Mapping FinBERT : Neutral=0, Positive=1, Negative=-1
            valeur_sentiment = 0
            if label == 'Positive': valeur_sentiment = 1 * score
            elif label == 'Negative': valeur_sentiment = -1 * score
            
            for cie in compagnies:
                # Nettoyage basique du nom
                cie_clean = cie.replace(" Inc", "").strip()
                resultats.append({
                    'Entreprise': cie_clean,
                    'Titre': text,
                    'Sentiment_Score': valeur_sentiment,
                    'Confiance_IA': score
                })
                
        return pd.DataFrame(resultats)

    def detecter_pepites(self, df):
        if df.empty:
            print("❌ Aucune entreprise détectée dans les news récentes.")
            return

        # --- ETAPE C : Agrégation et Scoring ---
        # On groupe par entreprise pour voir celles qui font le plus de bruit positif
        analyse = df.groupby('Entreprise').agg({
            'Sentiment_Score': 'mean', # Moyenne du sentiment
            'Titre': 'count'           # Nombre de mentions (Buzz)
        }).rename(columns={'Titre': 'Mentions'})
        
        # Score "Licorne" = Sentiment * Log(Mentions + 1)
        # On pondère la qualité par la quantité
        analyse['Unicorn_Score'] = analyse['Sentiment_Score'] * np.log1p(analyse['Mentions'])
        
        # Tri des résultats
        top_pepites = analyse.sort_values(by='Unicorn_Score', ascending=False)
        
        print("\n🦄 --- RADAR À LICORNES (Détection NLP) ---")
        print(top_pepites.head(10))
        
        # Affichage d'un exemple concret
        top_cie = top_pepites.index[0]
        print(f"\n💡 Pourquoi {top_cie} ?")
        exemples = df[df['Entreprise'] == top_cie]['Titre'].values
        for t in exemples[:3]:
            print(f"  - {t}")

# --- EXECUTION DU PIPELINE ---
if __name__ == "__main__":
    bot = SentinelleLicornes()
    
    # 1. Lire la presse
    news = bot.scanner_flux_rss()
    
    # 2. Analyser le texte
    df_resultats = bot.analyser_articles(news)
    
    # 3. Sortir le classement
    bot.detecter_pepites(df_resultats)