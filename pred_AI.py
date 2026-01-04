import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION DU MATÉRIEL (GPU/CPU) ---
# PyTorch permet d'utiliser la carte graphique pour accélérer les calculs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Utilisation du processeur : {device}")

# ==============================================================================
# 1. PRÉPARATION DE LA DATA (Identique, mais conversion en Tenseurs)
# ==============================================================================
def preparer_donnees_pytorch(ticker, jours_back=60):
    # Téléchargement
    data = yf.download(ticker, start="2015-01-01", progress=False, auto_adjust=False)['Adj Close'].values
    data = data.reshape(-1, 1)

    # Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x_data, y_data = [], []
    len_train = len(scaled_data) - 200 # On garde 200 jours pour le test
    
    # Création des séquences
    for i in range(jours_back, len(scaled_data)):
        x_data.append(scaled_data[i-jours_back:i, 0])
        y_data.append(scaled_data[i, 0])
        
    x_data, y_data = np.array(x_data), np.array(y_data)
    
    # Reshape [Samples, Time Steps, Features]
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

    # --- SPÉCIFIQUE PYTORCH : Conversion en Tenseurs ---
    # On sépare Train et Test ici pour créer les tenseurs
    x_train = torch.from_numpy(x_data[:len_train-jours_back]).float().to(device)
    y_train = torch.from_numpy(y_data[:len_train-jours_back]).float().to(device)
    
    x_test = torch.from_numpy(x_data[len_train-jours_back:]).float().to(device)
    y_test_real = y_data[len_train-jours_back:] # Gardé en numpy pour affichage
    
    return x_train, y_train, x_test, y_test_real, scaler, data

# ==============================================================================
# 2. L'ARCHITECTURE DU MODÈLE (La Classe PyTorch)
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Couche LSTM
        # batch_first=True signifie que nos données sont (Batch, Seq, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Couche Fully Connected (Sortie)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialisation des états cachés (h0) et cellulaires (c0) à zéro
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Propagation avant (Forward pass)
        # out contient les sorties de tous les pas de temps
        out, _ = self.lstm(x, (h0, c0))
        
        # On ne s'intéresse qu'à la sortie du DERNIER pas de temps (Many-to-One)
        out = out[:, -1, :] 
        
        # Passage dans la couche linéaire pour obtenir le prix
        out = self.fc(out)
        return out

# ==============================================================================
# 3. BOUCLE D'ENTRAÎNEMENT ET PRÉDICTION
# ==============================================================================
def train_and_predict(ticker):
    jours_memoire = 60
    input_size = 1   # On a 1 seule feature (le prix de clôture)
    hidden_size = 50 # Nombre de neurones dans le LSTM
    num_layers = 2   # Nombre de couches empilées
    output_size = 1  # On prédit 1 seule valeur (le prix)
    num_epochs = 50  # Nombre d'itérations
    learning_rate = 0.01

    print(f"🧠 Préparation des données pour {ticker}...")
    x_train, y_train, x_test, y_test_real, scaler, full_data = preparer_donnees_pytorch(ticker, jours_memoire)
    
    # Instanciation du modèle
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    
    # Fonction de perte (Loss) et Optimiseur
    criterion = nn.MSELoss() # Erreur Quadratique Moyenne
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("🔥 Démarrage de l'entraînement...")
    
    # --- BOUCLE D'ENTRAÎNEMENT MANUELLE ---
    model.train() # Mode entraînement
    for epoch in range(num_epochs):
        # 1. Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train.view(-1, 1)) # view assure la bonne forme
        
        # 2. Backward pass (Rétropropagation) et Optimisation
        optimizer.zero_grad() # On remet les gradients à zéro
        loss.backward()       # Calcul des gradients
        optimizer.step()      # Mise à jour des poids
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    # --- PRÉDICTION (INFERENCE) ---
    print("🔮 Génération des prédictions...")
    model.eval() # Mode évaluation (désactive le dropout)
    with torch.no_grad(): # On ne calcule pas les gradients pour la prédiction (gain mémoire)
        predictions = model(x_test)
        # On ramène les tenseurs vers le CPU et en numpy pour l'affichage
        predictions = predictions.cpu().numpy()
        
    # Inversion de la normalisation pour retrouver les prix en Euros
    predictions = scaler.inverse_transform(predictions)
    y_test_real_inv = scaler.inverse_transform(y_test_real.reshape(-1, 1))

    # --- VISUALISATION ---
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real_inv, color='black', label='Prix Réel')
    plt.plot(predictions, color='blue', label='Prédiction PyTorch LSTM')
    plt.title(f'PyTorch LSTM : Prédiction sur {ticker}')
    plt.xlabel('Jours (Test Set)')
    plt.ylabel('Prix (€)')
    plt.legend()
    plt.show()

    # Prédiction pour DEMAIN
    last_sequence = torch.from_numpy(scaler.transform(full_data[-jours_memoire:].reshape(-1, 1))).float().to(device)
    # Ajout de la dimension batch (1, 60, 1)
    last_sequence = last_sequence.unsqueeze(0)
    
    with torch.no_grad():
        pred_tomorrow = model(last_sequence)
        pred_tomorrow = pred_tomorrow.cpu().numpy()
        price_tomorrow = scaler.inverse_transform(pred_tomorrow)
        
    print(f"\n💰 L'IA (PyTorch) estime le prix de demain à : {price_tomorrow[0][0]:.2f} €")

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    train_and_predict("AIR.PA")