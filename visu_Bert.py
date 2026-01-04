from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def extraction_vecteurs_mathematiques():
    # Chargement du modèle brut (sans la couche de classification)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    phrases = [
        "OpenAI releases GPT-5 with massive capabilities.", # Startup IA
        "Anthropic raises 4 billion dollars.",             # Startup IA
        "McDonalds sells burgers.",                        # Rien à voir
    ]
    
    print("\n🧮 TRANSFORMATION TEXTE -> VECTEURS (Espace 768D)")
    
    vectors = []
    
    for text in phrases:
        # 1. Tokenization (Transformation en ID numériques)
        inputs = tokenizer(text, return_tensors="pt")
        
        # 2. Passage dans le Transformer
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 3. Récupération du "Hidden State" (Le vecteur sémantique)
        # On prend la moyenne des vecteurs de chaque mot pour avoir le sens de la phrase
        # Shape finale : (768,)
        sentence_vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        vectors.append(sentence_vector)
        
        print(f"Phrase : '{text}'")
        print(f" -> Vecteur (5 premiers chiffres) : {sentence_vector[:5]}... [Taille totale: {len(sentence_vector)}]")

    # 4. Calcul de similarité (Corrélation Cosinus)
    # Mathématiquement : (A . B) / (||A|| * ||B||)
    vec_openai = vectors[0]
    vec_anthropic = vectors[1]
    vec_mcdonalds = vectors[2]
    
    sim_ia = np.dot(vec_openai, vec_anthropic) / (np.linalg.norm(vec_openai) * np.linalg.norm(vec_anthropic))
    sim_burger = np.dot(vec_openai, vec_mcdonalds) / (np.linalg.norm(vec_openai) * np.linalg.norm(vec_mcdonalds))
    
    print("\n📐 ANALYSE DES CORRÉLATIONS INVISIBLES :")
    print(f"Similarité mathématique OpenAI <-> Anthropic : {sim_ia:.4f} (Très forte)")
    print(f"Similarité mathématique OpenAI <-> McDonalds : {sim_burger:.4f} (Faible)")

if __name__ == "__main__":
    extraction_vecteurs_mathematiques()