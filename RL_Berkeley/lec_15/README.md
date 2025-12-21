# CS 285 : Offline Reinforcement Learning (Lecture 15)

Ce document résume le cours sur l'**Offline RL** (aussi appelé Batch RL).
Contrairement au RL "Online" (qui collecte des données en apprenant) ou au RL "Off-policy" (qui utilise un buffer passé mais continue d'explorer), l'Offline RL doit apprendre une politique à partir d'un **dataset statique fixe** $\mathcal{D}$, sans jamais pouvoir interagir avec l'environnement pour tester ou corriger ses hypothèses.

## ⚠️ Le Problème Fondamental : Distribution Shift & OOD Actions

Si on applique un algorithme classique (DQN ou SAC) sur un dataset statique, il échoue catastrophiquement. Pourquoi ?

### 1. Counterfactual Queries (Requêtes Contrefactuelles)
L'algorithme de Q-Learning cherche à maximiser la valeur :
$$y = r + \gamma \max_{a'} Q(s', a')$$
Pour calculer la cible, l'algorithme interroge la Q-function sur des actions $a'$ qui maximisent $Q$. Or, ces actions $a'$ ne sont souvent **pas présentes dans le dataset** (elles sont "Out-Of-Distribution" ou OOD).

### 2. Overestimation & Exploitation
Comme la Q-function n'a jamais vu ces actions OOD lors de l'entraînement, elle prédit des valeurs arbitraires (souvent bruitées). L'opérateur $\max$ sélectionne systématiquement ces erreurs positives (hallucinations). L'agent pense avoir trouvé une stratégie miracle, alors qu'il exploite simplement les zones d'ombre du modèle.

---

## 🛠️ Solutions Algorithmiques

L'objectif de l'Offline RL est de rester "proche" des données pour éviter les zones inconnues, tout en essayant de faire mieux que la politique qui a généré les données ($\pi_\beta$).

### 1. Contraintes de Politique (Policy Constraints)
L'idée est de forcer la politique apprise $\pi_\theta$ à rester proche de la politique comportementale (Behavior Policy) $\pi_\beta$ (celle qui a généré le dataset).

**Formulation :**
$$\pi_\theta = \arg\max_\pi E_{(s,a) \sim \mathcal{D}} [Q_\phi(s, a)] \quad \text{s.t.} \quad D(\pi_\theta, \pi_\beta) \le \epsilon$$

* **Défis :** On ne connaît pas $\pi_\beta$ explicitement (on a juste des échantillons). Il faut souvent l'estimer (Behavior Cloning).
* **Algorithmes :**
    * **BCQ (Batch-Constrained Q-learning) :** Génère des actions candidates via un VAE (entraîné sur le dataset) et sélectionne la meilleure selon $Q$.
    * **BEAR (Bootstrapping Error Accumulation Reduction) :** Utilise le "Support Matching" (MMD) plutôt que la divergence KL. L'agent peut choisir n'importe quelle action tant qu'elle a une probabilité non-nulle dans le dataset.

### 2. Méthodes Conservatrices (Conservative Q-Learning - CQL)
Au lieu de contraindre l'acteur ($\pi$), on modifie le critique ($Q$) pour qu'il soit **pessimiste** sur les actions inconnues.

**Principe :**
On ajoute un terme de régularisation à la fonction de perte de Q-Learning pour **minimiser** la valeur des actions choisies par la politique actuelle, et **maximiser** la valeur des actions réelles du dataset.

$$\mathcal{L}(\theta) = \underbrace{\text{Standard Bellman Error}}_{\text{Fitting data}} + \alpha (\underbrace{E_{a \sim \pi}[Q(s,a)]}_{\text{Minimize policy actions}} - \underbrace{E_{a \sim \pi_\beta}[Q(s,a)]}_{\text{Maximize data actions}})$$

* **Résultat :** La Q-function apprend une **borne inférieure** (Lower Bound) de la vraie valeur. On est garanti de ne pas surestimer, ce qui rend l'optimisation sûre.

### 3. Model-Based Offline RL (MOPO / MOREL)
Si on apprend un modèle de la dynamique $T(s'|s,a)$ sur le dataset, on rencontre le même problème : le modèle va halluciner des états futurs optimistes pour les actions OOD.

**Solution :**
1.  Apprendre un modèle dynamique (souvent un ensemble pour estimer l'incertitude).
2.  Pénaliser la récompense par l'incertitude du modèle :
    $$r(s, a) = \hat{r}(s, a) - \lambda \cdot u(s, a)$$
    Où $u(s,a)$ est la variance des prédictions de l'ensemble.
3.  Planifier ou apprendre une politique dans ce MDP pénalisé (MDP Pessimiste).

---

## ⚖️ Comparaison des Approches

| Approche | Mécanisme | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- |
| **Importance Sampling** | Reweighting des retours | Théoriquement sans biais. | Variance exponentielle avec l'horizon (inutilisable en pratique pour l'entraînement). |
| **Policy Constraints (BCQ, BEAR)** | Restreindre $\pi$ à $\pi_\beta$ | Conceptuellement intuitif. | Nécessite d'estimer $\pi_\beta$ (Behavior Cloning), ce qui est difficile et source d'erreurs. |
| **Conservative Q (CQL)** | Apprendre une Q-function pessimiste | Très robuste, SOTA, pas besoin d'estimer $\pi_\beta$. | Peut être trop conservateur (sous-performance) si $\alpha$ est trop grand. |
| **Model-Based (MOPO)** | Pénaliser l'incertitude du modèle | Généralise mieux hors du dataset si la physique est simple. | Difficile si la dynamique est complexe (images). |

---

## 🔑 Résumé : Pourquoi l'Offline RL est dur ?
1.  **Pas de correction possible :** L'agent ne peut pas essayer une action pour voir "si ça marche vraiment".
2.  **Maximisation biaisée :** L'optimiseur cherche les erreurs du modèle (OOD) et les exploite.
3.  **Compromis Conservatisme/Performance :** Si on reste trop proche des données (Behavior Cloning), on ne s'améliore pas. Si on s'éloigne trop, on plante. L'art de l'Offline RL est de trouver la limite de généralisation sûre.

---
*Source: CS 285 Lecture 15 Slides, Instructor: Sergey Levine, UC Berkeley.*