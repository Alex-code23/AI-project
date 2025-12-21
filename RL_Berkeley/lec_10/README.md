# CS 285 : Model-Based Reinforcement Learning (Part 1) (Lecture 10)

Ce document résume le cours sur le **Model-Based RL (MBRL)**. Contrairement aux méthodes Model-Free (PG, Actor-Critic, Q-Learning) qui apprennent directement une politique ou une valeur, MBRL apprend un **modèle de la dynamique** de l'environnement, puis utilise ce modèle pour **planifier** ou optimiser une trajectoire.

## 🏗️ Le Principe Fondamental

L'objectif est d'apprendre la fonction de transition $f(s_t, a_t)$ telle que :
$$s_{t+1} = f(s_t, a_t)$$

Une fois ce modèle appris (souvent un réseau de neurones paramétré par $\phi$), on formule le problème de contrôle comme un problème d'optimisation :

$$\arg\max_{a_1, \dots, a_T} \sum_{t=1}^T r(s_t, a_t) \quad \text{sujet à } s_{t+1} = f_\phi(s_t, a_t)$$

---

## ⚠️ Le Problème du Décalage de Distribution (Distribution Mismatch)

L'algorithme naïf (Version 0.5/1.0) consiste à collecter des données aléatoires, entraîner le modèle $f_\phi$, puis planifier. Cela échoue souvent à cause du **Covariate Shift** :

1.  Le modèle est entraîné sur des données $p_{\text{train}}(s)$.
2.  La politique planifiée induit une nouvelle distribution de visite $p_{\pi}(s)$.
3.  Petite erreur sur $f_\phi$ $\rightarrow$ l'agent visite des états légèrement différents $\rightarrow$ le modèle ne connaît pas ces états $\rightarrow$ l'erreur explose.

**Théorie :** Si le modèle a une erreur $\epsilon$ à chaque pas, l'erreur totale sur la trajectoire croît en **$O(T^2)$** (quadratique en l'horizon).

---

## 🛠️ Solutions Algorithmiques

### 1. DAgger pour la Dynamique (Dataset Aggregation)
Pour corriger le décalage de distribution, on force le modèle à apprendre sur les états que la politique actuelle visite.
1.  Entraîner le modèle $f_\phi$ sur le dataset $\mathcal{D}$.
2.  Utiliser le modèle pour planifier une politique $\pi_\phi$.
3.  Exécuter $\pi_\phi$ dans le vrai environnement pour générer de nouvelles transitions $(s, a, s')$.
4.  Ajouter ces données à $\mathcal{D}$ et recommencer.

### 2. Model-Predictive Control (MPC)
Au lieu d'exécuter toute la séquence planifiée (Open Loop), on utilise une approche à **horizon fuyant (Closed Loop)** pour corriger les erreurs du modèle en temps réel.
1.  Observer l'état $s_t$.
2.  Optimiser la séquence d'actions $\{a_t, \dots, a_{t+H}\}$ qui maximise la récompense prédite par le modèle.
3.  Exécuter **seulement la première action** $a_t$.
4.  Observer le nouvel état réel $s_{t+1}$.
5.  Répéter.

---

## 🧠 Optimisation et Planification (Comment choisir les actions ?)

Une fois le modèle $f_\phi$ appris, comment trouver la séquence d'actions optimale ? On ne peut pas toujours utiliser la descente de gradient (Backpropagation through time) car les gradients explosent/disparaissent sur de longues horizons.

### Méthodes sans Gradient (Gradient-Free Optimization) :
1.  **Random Shooting :** Générer $N$ séquences d'actions aléatoires, évaluer leur récompense cumulée avec le modèle, choisir la meilleure. (Simple mais inefficace en haute dimension).
2.  **CEM (Cross-Entropy Method) :** Méthode itérative.
    * Échantillonner des actions depuis une distribution (ex: Gaussienne).
    * Sélectionner les $K$ meilleures séquences ("élites").
    * Mettre à jour la moyenne et la variance de la distribution pour se rapprocher des élites.
    * Répéter.

---

## 🔮 Incertitude et "Model Exploitation"

Les réseaux de neurones généralisent mal hors de leur distribution d'entraînement.
**Le problème :** L'optimiseur (le planificateur) va chercher des actions pour lesquelles le modèle prédit (à tort) une récompense énorme ("Model Exploitation"). Le modèle "hallucine" des gains.

**La Solution : Estimer l'Incertitude (Epistemic Uncertainty)**
L'agent doit savoir ce qu'il ne sait pas.
* **Bootstrap Ensembles :** Entraîner $N$ modèles indépendants $f_{\phi_1}, \dots, f_{\phi_N}$ sur les mêmes données (avec ré-échantillonnage).
* **Utilisation :** Lors de la planification, on utilise la moyenne des prédictions, ou on pénalise les actions où les modèles sont en désaccord (forte variance).

---

## 🖼️ Modèles Complexes (Images)

Pour les observations visuelles (pixels), on ne peut pas prédire directement $s_{t+1}$ (vecteur d'état inconnu). On utilise des **Video Prediction Models** (ex: Convolutional LSTM, Stochastic Variational Video Prediction) pour prédire les futures frames, puis on optimise par rapport à une fonction de récompense définie sur les pixels ou un but visuel.

---

## ✅ Avantages et ❌ Inconvénients du MBRL

| Avantages | Inconvénients |
| :--- | :--- |
| **Sample Efficiency :** Extrêmement efficace. Un modèle apprend la physique du monde bien plus vite qu'une politique n'apprend à maximiser un score. (Ex: 10x à 100x moins de données que le Model-Free). | **Complexité de calcul :** La planification (MPC/CEM) est coûteuse en temps de calcul à l'exécution (inference time). |
| **Transférabilité :** Le modèle de dynamique est agnostique à la tâche (reward function). Si la tâche change, le modèle reste valide. | **Biais Asymptotique :** Si le modèle n'est pas parfait, la performance finale sera limitée par la qualité du modèle ("Model Bias"). Le Model-Free finit souvent par être meilleur asymptotiquement. |
| **Sécurité :** Permet de prédire des états dangereux avant de les atteindre. | **Model Exploitation :** Nécessite une bonne gestion de l'incertitude pour éviter d'exploiter les erreurs du modèle. |

---
*Source: CS 285 Lecture 10 Slides, Instructor: Sergey Levine, UC Berkeley.*