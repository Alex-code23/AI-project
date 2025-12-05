import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Import des nouvelles fonctions de visualisation
from visualization_utils import (
    plot_training_history, 
    plot_label_distribution, 
    get_embeddings, 
    plot_latent_space)

#Configuration du device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {device}")

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================

class SocialMediaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        texts: Liste de strings (les posts)
        labels: Liste d'entiers (0 ou 1)
        tokenizer: L'objet CamembertTokenizer
        max_len: Longueur max de la séquence (512 max pour CamemBERT)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # Tokenisation "à la volée"
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,    # Ajoute <s> et </s>
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',       # Pad jusqu'à max_len
            truncation=True,            # Coupe si trop long
            return_attention_mask=True,
            return_tensors='pt',        # Retourne des tenseurs PyTorch
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 2. L'ARCHITECTURE
# ==========================================

class InfluencerClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(InfluencerClassifier, self).__init__()
        # Chargement du backbone CamemBERT
        self.camembert = CamembertModel.from_pretrained('camembert-base')
        
        # Dropout pour régularisation
        self.drop = nn.Dropout(p=0.3)
        
        # Couche de classification
        self.out = nn.Linear(self.camembert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Passage dans CamemBERT
        outputs = self.camembert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # On récupère le pooling du token <s> (pour la classification)
        # outputs.last_hidden_state est de taille (Batch, Seq_Len, Hidden)
        # On prend tout le batch, token index 0, toutes les dimensions cachées
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        
        output = self.drop(pooled_output)
        return self.out(output)

# ==========================================
# 3. FONCTIONS D'ENTRAÎNEMENT + ÉVALUATION
# ==========================================

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        # Backward pass
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        
        # Gradient clipping (pour éviter l'explosion des gradients)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()

    return (correct_predictions.double() / n_examples).cpu(), np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval() # Mode évaluation (désactive dropout)
    losses = []
    correct_predictions = 0

    with torch.no_grad(): # Pas de calcul de gradient
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return (correct_predictions.double() / n_examples).cpu(), np.mean(losses)

# ==========================================
# 4. EXÉCUTION PRINCIPALE 
# ==========================================

if __name__ == '__main__':
    # --- A. Création de fausses données ---
    # 1. Exemples INFLUENCEURS (Vocabulaire : promo, engagement, marques, impératif)
    influencer_texts = [
        "J'adore ce nouveau rouge à lèvres ! #ad #makeup",
        "Utilisez mon code PROMO20 pour -20% sur tout le site.",
        "Lien en bio pour découvrir la nouvelle collection !",
        "Grand jeu concours ! Taguez deux amis en commentaire pour gagner un iPhone.",
        "Merci à @DanielWellington pour cette magnifique montre. #partenariat",
        "Swipe up pour lire mon dernier article sur le blog !",
        "Unboxing de mes derniers achats chez Sephora, c'est une dinguerie.",
        "Petit code promo juste pour vous : MARIE15 (-15% sur la commande).",
        "Collaboration commerciale avec ma marque préférée ❤️",
        "Retrouvez ma morning routine complète en story à la une.",
        "Ce produit a changé ma vie, je ne peux plus m'en passer. #skincare",
        "Rendez-vous ce soir à 18h pour un live FAQ !",
        "Profitez des soldes avec mon lien affilié en description.",
        "Je vous présente mon indispensable de l'été. #sponsored",
        "Gagnez un voyage à Bali ! Conditions sous la photo.",
        "Testing de la nouvelle gamme, mon avis sincère en vidéo.",
        "N'oubliez pas de liker et d'enregistrer le post pour soutenir !",
        "Dernières heures pour profiter de l'offre flash ⚡",
        "En partenariat avec la marque, on vous offre un bon d'achat.",
        "Cliquez ici pour shopper mon look !"
    ]

    # 2. Exemples NON-INFLUENCEURS (Vocabulaire : quotidien, questions, avis perso, neutre)
    non_influencer_texts = [
        "Promenade au parc ce matin, il fait beau.",
        "Je déteste les lundis matins...",
        "Recette de crêpes : farine, lait, oeufs.",
        "Quelqu'un sait si la ligne 13 fonctionne aujourd'hui ?",
        "Trop hâte d'être en week-end pour dormir.",
        "Mon chat a encore renversé son bol d'eau...",
        "Super soirée hier avec les collègues, on a bien ri.",
        "Je cherche un plombier sur Paris, des recommandations ?",
        "Le dernier film Marvel est vraiment décevant, je trouve.",
        "J'ai enfin fini mon livre, c'était trop bien.",
        "Il pleut encore, c'est déprimant ce temps.",
        "Qui est chaud pour un foot ce soir ?",
        "Révision pour les partiels... envoyez de la force.",
        "Regardez ce que j'ai cuisiné ce midi !",
        "Joyeux anniversaire maman ❤️",
        "Mon train a 2h de retard, merci la SNCF.",
        "Photo de vacances de l'année dernière, ça me manque.",
        "Besoin de conseils pour choisir une nouvelle télé.",
        "Rien de prévu ce dimanche, ça fait du bien.",
        "Incroyable le match d'hier soir !"
    ]

    # Fusion des listes
    raw_texts = influencer_texts + non_influencer_texts
    
    # Création des labels correspondants (1 pour influenceur, 0 pour non-influenceur)
    labels = ([1] * len(influencer_texts)) + ([0] * len(non_influencer_texts))

    # On peut multiplier le tout pour simuler un volume plus gros si nécessaire
    raw_texts = raw_texts
    labels = labels

    # --- B. Paramètres ---
    MAX_LEN = 64
    BATCH_SIZE = 4 # Petit batch car CamemBERT est lourd
    EPOCHS = 3
    LEARNING_RATE = 2e-5 # Très petit LR pour le fine-tuning

    # --- C. Tokenizer & DataLoaders ---
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    
    # Split Train/Test
    df_train_txt, df_test_txt, df_train_lbl, df_test_lbl = train_test_split(
        raw_texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = SocialMediaDataset(df_train_txt, df_train_lbl, tokenizer, MAX_LEN)
    test_dataset = SocialMediaDataset(df_test_txt, df_test_lbl, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- D. Initialisation Modèle ---
    model = InfluencerClassifier(n_classes=2)
    model = model.to(device)

    # Optimiseur spécifique (AdamW avec weight decay fix)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Fonction de perte
    loss_fn = nn.CrossEntropyLoss().to(device)

    # --- E. Boucle d'entraînement ---
    print("Début de l'entraînement...")
    
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, len(df_train_txt)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model, test_loader, loss_fn, device, len(df_test_txt)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        # Sauvegarde des métriques
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

    # ==========================================
    # 5. VISUALISATIONS
    # ==========================================
    print("Affichage des visualisations...")
    
    # 5.1. Historique de l'entraînement
    plot_training_history(history)

    # 5.2. Distribution des données
    plot_label_distribution(df_train_lbl, df_test_lbl)


    # ==========================================
    # 5. INFÉRENCE 
    # ==========================================
    
    def predict_text(text, model, tokenizer):
        encoded_text = tokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        
        probability = torch.nn.functional.softmax(output, dim=1)
        
        return prediction.item(), probability[0][1].item() # Retourne la classe et la proba d'être classe 1

    # Test final
    nouveau_tweet = "Incroyable cette crème, allez voir le lien ! #skincare"
    prediction, proba = predict_text(nouveau_tweet, model, tokenizer)
    
    classe = "Influenceur" if prediction == 1 else "Non-Influenceur"
    print(f"Texte : '{nouveau_tweet}'")
    print(f"Prédiction : {classe} (Probabilité Influenceur : {proba:.4f})")

    # ==========================================
    # 6. VISUALISATION DE L'ESPACE LATENT
    # ==========================================
    # On utilise le test_loader pour visualiser comment le modèle sépare les classes
    print("\nExtraction des embeddings pour la visualisation de l'espace latent...")
    embeddings, labels = get_embeddings(model, test_loader, device)
    plot_latent_space(embeddings, labels)