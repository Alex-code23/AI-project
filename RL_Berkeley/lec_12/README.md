# CS 285 : Model-Based RL Part 3 - Advanced Model Learning (Lecture 12)

Ce document résume la troisième partie du cours sur le **Model-Based RL**.
Après avoir vu comment *utiliser* un modèle (Planification) et comment l'*entraîner* (Supervised Learning de base), ce cours s'attaque à deux problèmes critiques qui font échouer les approches naïves :
1.  **L'Incertitude :** Le modèle est souvent trop confiant dans les zones inconnues ("Model Exploitation").
2.  **La Dimensionnalité :** Apprendre la dynamique directement sur des pixels (Images) est inefficace et difficile.

---

## 1. Gestion de l'Incertitude (Uncertainty Estimation)

L'optimiseur (le planificateur) cherche les états où la récompense prédite est maximale. Si le modèle fait une erreur positive dans une zone inexplorée (hallucination), le planificateur va foncer vers cet état.

### [cite_start]Types d'Incertitude [cite: 19]
1.  **Incertitude Aléatorique (Statistical) :** Bruit inhérent au système (ex: le vent, un résultat de dé). On ne peut pas la réduire avec plus de données.
    * *Solution :* Apprendre une distribution de sortie $p_\theta(s_{t+1}|s_t, a_t)$ (ex: Gaussienne) au lieu d'une valeur ponctuelle déterministe.
2.  **Incertitude Épistémique (Model Uncertainty) :** Le modèle ne connaît pas la réponse car il manque de données d'entraînement dans cette région. Elle diminue avec plus de données.
    * *C'est celle qui nous intéresse pour éviter le "Model Exploitation".*

### [cite_start]La Solution Pratique : Ensembles (Bootstrap) [cite: 20]
Les Réseaux de Neurones Bayésiens (BNN) sont théoriquement idéaux mais difficiles à entraîner. L'approximation standard en Deep RL est l'utilisation d'**ensembles** :
1.  Entraîner $N$ modèles indépendants $\theta_1, \dots, \theta_N$ sur le même dataset (avec ré-échantillonnage ou initialisation aléatoire différente).
2.  Leur moyenne donne la prédiction.
3.  Leur variance donne l'estimation de l'incertitude épistémique.

**Utilisation pour la Planification :**
* **Approche Robuste :** Pénaliser les actions où la variance des modèles est élevée (évite les zones inconnues).
* **Approche Optimiste (Exploration) :** Favoriser la variance élevée pour découvrir de nouvelles dynamiques.

---

## 2. Modèles Latents (Latent State Models)

Pour des observations visuelles (images $o_t \in \mathbb{R}^{64 \times 64 \times 3}$), prédire le pixel suivant ($o_{t+1}$) est extrêmement difficile et souvent inutile (on se fiche de la couleur du ciel, on veut savoir la position du robot).

### [cite_start]L'Architecture Latente [cite: 31, 32]
L'idée est d'apprendre un espace d'état compact $s_t$ (latent) qui résume $o_t$.

Le modèle se décompose en 3 parties :
1.  **Encodeur (Representation) :** $s_t = E(o_t)$. Compresse l'image en un vecteur latent.
2.  **Dynamique (Dynamics) :** $s_{t+1} = g(s_t, a_t)$. Prédit le futur dans l'espace latent.
3.  **Décodeur (Reconstruction/Reward) :** $o_{t+1} \approx D(s_{t+1})$ ou $r_{t+1} \approx R(s_{t+1})$.

### [cite_start]Entraînement : Variational Auto-Encoders (VAE) [cite: 38]
On maximise la borne inférieure de la vraisemblance (ELBO - Evidence Lower Bound).
* *Loss de Reconstruction :* Le modèle doit pouvoir recréer l'image (garantit que $s_t$ contient l'info).
* *Loss de Dynamique :* Le modèle latent doit prédire correctement $s_{t+1}$.
* *Régularisation KL :* Force l'espace latent à être structuré/lisse.

### Exemples Notables
* [cite_start]**E2C (Embed to Control)[cite: 37]:** Apprend un espace latent où la dynamique est localement linéaire, permettant l'utilisation de iLQR pour planifier.
* [cite_start]**World Models (Ha & Schmidhuber, 2018)[cite: 42]:** Entraîne un VAE massif pour compresser le monde, puis un RNN pour "rêver" le futur dans l'espace latent, et enfin une petite politique qui apprend dans le rêve.

---

## 3. End-to-End & Value Equivalence

Faut-il vraiment reconstruire les pixels (Décodeur) ? C'est lourd et parfois le modèle gaspille des ressources à reconstruire des détails inutiles (arrière-plan).

### [cite_start]Observation vs État [cite: 27]
* **Observation ($o_t$) :** Ce que la caméra voit (pixels, haute dimension, partiel).
* **État ($s_t$) :** Résumé suffisant pour prédire le futur.

### [cite_start]Value Prediction Networks / MuZero [cite: 46]
L'idée est de ne **jamais reconstruire l'observation**. On force l'espace latent à être bon uniquement pour prédire :
1.  La récompense future.
2.  La valeur future ($V^\pi$).

L'espace latent n'est plus contraint par la reconstruction visuelle, mais par l'utilité pour la tâche (Task-aware latent space).

---

## ✅ Résumé des Architectures Model-Based

| Architecture | Principe | Avantage | Inconvénient |
| :--- | :--- | :--- | :--- |
| **Global Network** | $s_{t+1} = f_\theta(s_t, a_t)$ (Réseau Dense) | Simple, efficace pour états de basse dimension. | Ne passe pas à l'échelle pour les images. |
| **Ensembles** | $\{f_{\theta_1}, \dots, f_{\theta_N}\}$ | Donne une estimation de l'incertitude (crucial). | Coûteux (N fois l'entraînement et l'inférence). |
| **Latent Models (VAE)** | Encoder $\rightarrow$ Dynamique Latente $\rightarrow$ Decoder | Gère les images, planification rapide dans l'espace latent. | Difficile à entraîner (équilibrer reconstruction vs dynamique). |
| **Value Equivalence** | Encoder $\rightarrow$ Dynamique $\rightarrow$ Value/Reward | Très performant (SOTA sur Atari/Go), pas de reconstruction inutile. | Signal d'apprentissage plus rare (seulement reward/valeur). |

---
*Source: CS 285 Lecture 12 Slides, Instructor: Sergey Levine, UC Berkeley.*