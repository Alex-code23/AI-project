# CS 285 : Model-Based RL Part 2 - Policy Learning (Lecture 11)

Ce document résume la deuxième partie du cours sur le **Model-Based RL**. Alors que la partie 1 se concentrait sur la planification pure (Shooting/MPC) avec un modèle, cette partie traite de l'utilisation du modèle pour optimiser directement les paramètres $\theta$ d'une politique $\pi_\theta(a|s)$.

## 🎯 Objectif : Apprendre une Politique via le Modèle

Au lieu de replanifier à chaque pas de temps (ce qui est coûteux), nous voulons "distiller" la connaissance du modèle dans un réseau de neurones rapide et généralisable : la politique $\pi_\theta$.

Trois grandes approches sont abordées :
1.  **Backpropagation :** Différentier analytiquement à travers le modèle dynamique.
2.  **Model-Free avec Modèle (Dyna) :** Utiliser le modèle pour générer des données synthétiques.
3.  **Modèles Locaux & Guided Policy Search :** Utiliser des modèles simples (linéaires) localement pour guider une politique globale complexe.

---

## 1. Backpropagation à travers le Modèle

Puisque le modèle de dynamique $s_{t+1} = f_\phi(s_t, a_t)$ est souvent un réseau de neurones, il est **différentiable**.
On peut calculer le gradient de la somme des récompenses directement par rapport aux paramètres de la politique $\theta$ en utilisant la règle de la chaîne (Chain Rule) à travers le temps.

### Le Problème
Calculer $\frac{dJ}{d\theta}$ implique de multiplier des Jacobiennes à chaque pas de temps :
$$\frac{ds_{t+1}}{d\theta} = \frac{df}{ds_t} \frac{ds_t}{d\theta} + \frac{df}{da_t} \frac{da_t}{d\theta}$$
* **Gradients Explosifs/Disparaissants :** Comme pour les RNNs, multiplier de nombreuses matrices Jacobiennes sur un long horizon $T$ rend l'optimisation numériquement instable.
* **Sensibilité aux Paramètres :** Les méthodes de "Shooting" sont très sensibles aux petites erreurs de modèle qui s'amplifient exponentiellement.

### Solution : Collocation (Optimisation avec Contraintes)
Au lieu d'optimiser les actions séquentiellement (Shooting), on optimise tout la trajectoire $(s_1, a_1, \dots, s_T, a_T)$ simultanément en traitant la dynamique $s_{t+1} = f(s_t, a_t)$ comme une **contrainte d'égalité**. On utilise la méthode des Multiplicateurs de Lagrange (Dual Descent). C'est plus stable mais complexe à implémenter.

---

## 2. Approches "Dyna" (Model-Based pour accélérer Model-Free)

L'idée est d'utiliser le modèle appris comme un **simulateur** pour générer des données supplémentaires et entraîner un algorithme Model-Free (ex: TRPO, SAC, DQN).

### Algorithme Général (Style Dyna-Q)
1.  Collecter des données réelles $\mathcal{D}$.
2.  Apprendre le modèle $f_\phi$ sur $\mathcal{D}$.
3.  **Boucle Model-Free :**
    * Échantillonner un état $s$ (depuis $\mathcal{D}$).
    * Simuler une action et une transition avec le modèle : $s' = f_\phi(s, \pi(s))$.
    * Ajouter $(s, a, r, s')$ au buffer d'entraînement.
    * Mettre à jour $\pi$ avec ces données synthétiques.

### Model-Based Policy Optimization (MBPO)
Une innovation clé pour que cela fonctionne avec le Deep RL :
* Ne pas générer de longues trajectoires avec le modèle (l'erreur s'accumule trop vite).
* Générer des **rollouts très courts** ($k=1$ ou $k=2$) en partant d'états **réels** échantillonnés dans le replay buffer.
* Cela permet d'avoir des données très variées sans trop de biais de modèle.

---

## 3. Modèles Locaux et Guided Policy Search (GPS)

Il est très difficile d'apprendre un modèle global $f_\phi(s,a)$ précis partout. En revanche, il est facile d'apprendre des modèles **locaux linéaires** autour d'une trajectoire spécifique.

### Dynamique Linéaire Locale
Autour d'une trajectoire $(s_t, a_t)$, on approxime la dynamique par :
$$s_{t+1} \approx \mathbf{A}_t s_t + \mathbf{B}_t a_t + c_t$$
On peut apprendre ces matrices $\mathbf{A}_t, \mathbf{B}_t$ par régression linéaire simple sur quelques échantillons.

### Contrôle Optimal Local (iLQR)
Si la dynamique est linéaire et le coût quadratique, on peut résoudre le contrôle optimal exactement et efficacement avec **LQR** (Linear Quadratic Regulator).
Si le modèle n'est pas linéaire, on utilise **iLQR** (iterative LQR) pour ajuster itérativement la trajectoire.

### Algorithme GPS (Guided Policy Search)
GPS combine l'efficacité du contrôle optimal (iLQR) avec la généralisation des réseaux de neurones. C'est un algorithme de "Distillation".

1.  **Optimisation de Trajectoire (L'enseignant) :** Utiliser iLQR avec des modèles locaux pour trouver des trajectoires optimales et des contrôleurs locaux simples pour diverses conditions initiales.
2.  **Apprentissage Supervisé (L'élève) :** Entraîner une politique neuronale globale $\pi_\theta$ pour imiter les actions des contrôleurs locaux sur ces trajectoires.
    $$\min_\theta \sum_{t} D_{KL}(\pi_{\text{local}}(a_t|s_t) || \pi_\theta(a_t|s_t))$$
3.  **Adaptation :** La politique globale permet de généraliser à de nouveaux états, et sert à guider la collecte de nouvelles données pour raffiner les modèles locaux.

---

## 🔑 Résumé des Méthodes

| Méthode | Principe | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- |
| **Backprop through time** | Calculer $\nabla_\theta J$ via la chaîne de dérivées du modèle. | Conceptuellement simple. | Gradients instables (Vanishing/Exploding), très sensible au biais du modèle. |
| **Dyna / MBPO** | Utiliser le modèle pour générer des données d'entraînement pour un algo Model-Free. | Très efficace (Sample efficient), flexible. | Nécessite un calibrage fin de l'horizon de génération pour éviter le biais. |
| **Guided Policy Search** | Utiliser iLQR (modèles locaux) pour guider une politique globale. | Très stable, efficace pour la robotique complexe. | Complexe à implémenter, repose sur la linéarisation locale. |

---
*Source: CS 285 Lecture 11 Slides, Instructor: Sergey Levine, UC Berkeley.*