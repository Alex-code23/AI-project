# CS 285 : Offline Reinforcement Learning Part 2 (Lecture 16)

Ce document résume la deuxième partie du cours sur l'**Offline RL**.
Après avoir couvert les méthodes Model-Free (contraintes de politique, CQL) dans le cours précédent, ce cours se concentre sur deux avancées majeures :
1.  **Model-Based Offline RL :** Utiliser des modèles de dynamique pour mieux généraliser, en gérant l'incertitude.
2.  **Sequence Modeling (Transformers) :** Traiter le RL comme un problème de prédiction de séquence supervisé à grande échelle.

---

## 🏗️ 1. Model-Based Offline RL

Les méthodes Model-Free (comme CQL) sont très stables mais restent parfois "collées" aux données d'entraînement. Les méthodes Model-Based ont le potentiel de mieux généraliser en apprenant la physique du monde, mais elles souffrent du même problème de **décalage de distribution** : le modèle hallucine des transitions optimistes pour les actions hors distribution (OOD).

### Le Problème : L'Exploitation du Modèle
Si on apprend un modèle $T_\phi(s'|s,a)$ sur le dataset statique et qu'on planifie avec, l'agent va chercher les actions où le modèle prédit (à tort) des états futurs très avantageux.
* **Erreur du modèle :** L'erreur est faible sur les données $\mathcal{D}$, mais élevée ailleurs.
* **Conséquence :** La politique apprise diverge vers des zones où le modèle est faux.

### La Solution : Pénalité d'Incertitude (MOPO / MOREL)
Pour empêcher l'agent d'aller là où le modèle n'est pas fiable, on modifie la fonction de récompense dans le processus de planification (ou d'apprentissage de politique).

1.  **Ensemble de Modèles :** Entraîner un ensemble de $N$ modèles dynamiques $\{T_{\theta_1}, \dots, T_{\theta_N}\}$ pour estimer l'incertitude épistémique (variance des prédictions).
    $$u(s, a) = \text{Var}(s' | s, a)$$
2.  **MDP Pénalisé :** On construit un MDP artificiel où la récompense est pénalisée par cette incertitude :
    $$\tilde{r}(s, a) = r(s, a) - \lambda \cdot u(s, a)$$
3.  **Optimisation :** On apprend une politique (ou on planifie) pour maximiser cette récompense pénalisée.

**Résultat :** L'agent est "pessimiste". Il préfère une action avec une récompense moyenne mais certaine, plutôt qu'une action avec une récompense potentiellement énorme mais incertaine.

---

## 📜 2. RL as Sequence Modeling (Transformers)

Au lieu d'utiliser la programmation dynamique (Bellman/Q-Learning), peut-on traiter le RL comme un simple problème de **prédiction de séquence** (comme GPT pour le langage) ?

### Decision Transformer (DT)
L'idée est de modéliser la trajectoire comme une séquence de tokens :
$$\tau = (\dots, R_t, s_t, a_t, R_{t+1}, s_{t+1}, a_{t+1}, \dots)$$
Où $R_t$ est le **Return-to-go** (la somme des récompenses futures espérées).

* **Entraînement :** On entraîne un Transformer (GPT) de manière supervisée pour prédire le prochain token (surtout l'action $a_t$).
  $$a_t \sim P(a_t | R_t, s_t, a_{t-1}, \dots)$$
* **Inférence (Test) :** On donne à l'agent l'état actuel $s_t$ et on lui "commande" un retour élevé (ex: $R_{target} = \text{Max Score}$). Le modèle prédit l'action qui est la plus probable pour obtenir ce retour, basé sur les statistiques du dataset.

### Trajectory Transformer (TT)
Similaire au DT, mais discretise les états et les actions pour utiliser un modèle de langage standard. Il utilise la "Beam Search" pour planifier des trajectoires entières qui maximisent la probabilité d'atteindre un but ou une récompense élevée.

---

## 🤖 Applications et Workflow

### Exemples d'Applications
* **Manipulation Robotique (QT-Opt) :** Apprendre à saisir des objets à partir de mois de données collectées par plusieurs robots.
* **Navigation (BADGR / RECON) :** Apprendre à naviguer en tout-terrain en utilisant des données collectées "hors ligne" (ex: vidéos de conduite), en évitant les collisions et les terrains accidentés.

### Le Workflow de l'Offline RL
Contrairement au cycle classique "Entraîner-Tester-Entraîner", l'Offline RL propose un workflow plus proche du Supervised Learning :
1.  **Collecte :** Accumuler un large dataset $\mathcal{D}$ (via des politiques aléatoires, expertes, ou mixtes).
2.  **Entraînement :** Apprendre une politique $\pi$ sur $\mathcal{D}$ (via CQL, IQL, DT, etc.).
3.  **Évaluation (Le défi) :** Comment savoir si la politique est bonne sans la tester sur le robot ?
    * L'évaluation hors ligne (Off-Policy Evaluation - OPE) est difficile.
    * Souvent, on sélectionne les meilleurs modèles selon des métriques conservatrices (valeur Q moyenne pénalisée) avant de déployer le meilleur candidat.

---

## ✅ Résumé des Architectures Avancées

| Architecture | Principe | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- |
| **MOPO / MOREL** (Model-Based) | Pénaliser la récompense par l'incertitude d'un ensemble de modèles. | Généralise bien hors du dataset si la dynamique est apprenable. | Lourd (Ensemble), difficile pour les images complexes. |
| **Decision Transformer** (Sequence) | Conditionner l'action sur le retour désiré ($R_t$) via un Transformer. | Très stable (Supervised Learning), pas de problèmes de bootstrap/Q-values. | Ne peut pas "inventer" une stratégie meilleure que la meilleure trajectoire du dataset (pas de stitching optimal théorique). |
| **Conservative Q-Learning** (Model-Free) | Apprendre une Q-function pessimiste. | Souvent le plus performant pour "coudre" (stitch) des sous-trajectoires optimales. | Optimisation parfois instable, sensible aux hyperparamètres. |

---
*Source: CS 285 Lecture 16 Slides, Instructor: Sergey Levine, UC Berkeley.*