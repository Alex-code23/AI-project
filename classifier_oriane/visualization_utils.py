import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
import pandas as pd

def plot_training_history(history):
    """
    Affiche les courbes d'accuracy et de loss pour l'entraînement et la validation.
    
    history: un dictionnaire contenant les listes: 
             'train_acc', 'val_acc', 'train_loss', 'val_loss'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Courbe de la loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Évolution de la Loss')
    ax1.set_xlabel('Époques')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Courbe de l'accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Évolution de l\'Accuracy')
    ax2.set_xlabel('Époques')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.suptitle("Historique de l'entraînement")
    plt.show()

def plot_label_distribution(train_labels, test_labels):
    """
    Affiche la distribution des classes dans les ensembles d'entraînement et de test.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x=train_labels)
    plt.title("Distribution des labels (Train)")
    plt.xlabel("Classe (0: Non-Influencer, 1: Influencer)")
    plt.ylabel("Nombre d'exemples")

    plt.subplot(1, 2, 2)
    sns.countplot(x=test_labels)
    plt.title("Distribution des labels (Test)")
    plt.xlabel("Classe (0: Non-Influencer, 1: Influencer)")
    plt.ylabel("Nombre d'exemples")

    plt.tight_layout()
    plt.show()

def get_embeddings(model, data_loader, device):
    """
    Extrait les embeddings (sorties de CamemBERT avant classification) et les labels.
    """
    model = model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            # On récupère la sortie de CamemBERT
            outputs = model.camembert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
            all_embeddings.append(pooled_output.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

def plot_latent_space(embeddings, labels):
    """
    Visualise les embeddings en 2D avec t-SNE, colorés par classe.
    """
    print("Calcul de la projection t-SNE... (cela peut prendre un moment)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    df['Classe'] = df['label'].apply(lambda x: 'Influencer' if x == 1 else 'Non-Influencer')

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='Classe', palette='viridis', alpha=0.7)
    plt.title('Visualisation de l\'espace latent (t-SNE)')
    plt.xlabel('Composante t-SNE 1')
    plt.ylabel('Composante t-SNE 2')
    plt.legend()
    plt.show()
