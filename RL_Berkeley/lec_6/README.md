# CS 285 : Actor-Critic Algorithms (Lecture 6)

Ce document résume le cours sur les algorithmes **Actor-Critic**. Ces méthodes visent à améliorer les **Policy Gradients** en réduisant la variance de l'estimation du gradient grâce à l'introduction d'une fonction de valeur apprise (le "Critic").

## 🎯 Objectif : Réduire la Variance du Policy Gradient

Le gradient de politique standard (REINFORCE) utilise la récompense cumulée réelle (Monte Carlo) pour estimer la qualité d'une action. Bien que sans biais, cette estimation a une **variance très élevée**.

L'idée principale est de remplacer la somme des récompenses réelles (Reward-to-go) par une estimation apprise :

* **Actor ($\pi_\theta$)** : La politique qui décide des actions.
* **Critic ($\hat{V}_\phi^\pi$)** : Une fonction de valeur qui estime les récompenses futures espérées.

---

## 🧮 Concepts Clés et Formules

### 1. L'amélioration du Gradient
Au lieu d'utiliser $Q(s,a)$ complet (Monte Carlo), on utilise l'estimation fournie par le critique. Le gradient devient :

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \hat{A}^\pi(s_{i,t}, a_{i,t})$$

Où $\hat{A}^\pi$ est la **Fonction d'Avantage**.

### 2. La Fonction d'Avantage (Advantage Function)
L'avantage mesure à quel point une action $a_t$ est meilleure que l'action moyenne dans l'état $s_t$.

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

En utilisant l'approximation $Q^\pi(s_t, a_t) \approx r(s_t, a_t) + \gamma V^\pi(s_{t+1})$, on peut calculer l'avantage en n'apprenant que la fonction de valeur $V$ :

$$\hat{A}^\pi(s_t, a_t) = r(s_t, a_t) + \gamma \hat{V}_\phi^\pi(s_{t+1}) - \hat{V}_\phi^\pi(s_t)$$

C'est ce qu'on appelle souvent l'erreur de différence temporelle (TD error).

### 3. Fitting du Critique (Policy Evaluation)
Le critique $\hat{V}_\phi^\pi$ est entraîné par régression supervisée pour minimiser l'erreur quadratique entre sa prédiction et une "target" (cible) :

$$\mathcal{L}(\phi) = \frac{1}{2} \sum_i || \hat{V}_\phi^\pi(s_i) - y_i ||^2$$

Les cibles $y_i$ peuvent être :
* **Monte Carlo :** $\sum r_{t'}$ (variance élevée, sans biais).
* **Bootstrapped (TD) :** $r_t + \gamma \hat{V}_\phi^\pi(s_{t+1})$ (variance faible, biaisé).

---

## 🤖 L'Algorithme Actor-Critic (Online)

1.  **Agir :** Prendre une action $a \sim \pi_\theta(a|s)$, observer $(s, a, s', r)$.
2.  **Mettre à jour le Critique :** Mettre à jour $\hat{V}_\phi^\pi$ en utilisant la cible $r + \gamma \hat{V}_\phi^\pi(s')$.
3.  **Evaluer l'Avantage :** Calculer $\hat{A}^\pi(s, a) = r + \gamma \hat{V}_\phi^\pi(s') - \hat{V}_\phi^\pi(s)$.
4.  **Mettre à jour l'Acteur :** Calculer le gradient $\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a|s) \hat{A}^\pi(s, a)$ et mettre à jour $\theta$.

*Note : Il existe aussi des versions "Batch" et asynchrones (A3C).*

---

## ⚖️ Le Compromis Biais-Variance (Bias-Variance Tradeoff)

L'utilisation d'un critique introduit un compromis fondamental :

* **Monte Carlo (REINFORCE)** : Aucune biais, mais variance très élevée.
* **Actor-Critic (avec Bootstrap)** : Variance faible (l'estimation est stable), mais **biaisé** (si le critique est faux, le gradient est faux).

### Generalized Advantage Estimation (GAE)
Pour équilibrer ce compromis, on utilise des **n-step returns** (regarder $n$ pas dans le futur avant d'utiliser le critique). GAE fait une moyenne pondérée exponentielle de tous les n-step returns possibles :

$$\hat{A}_{GAE}^\pi(s_t, a_t) = \sum_{t'=t}^\infty (\gamma \lambda)^{t'-t} \delta_{t'}$$

Où $\delta_{t'}$ est l'erreur TD standard.
* $\lambda = 1$ : Monte Carlo (haute variance).
* $\lambda = 0$ : Actor-Critic classique (haut biais).
* $0 < \lambda < 1$ : Compromis optimal.

---

## 🏗️ Architecture et Design

1.  **Réseaux partagés (Shared Network) :** Un seul réseau de neurones avec deux "têtes" (sorties) : une pour la politique $\pi$ (vecteur) et une pour la valeur $V$ (scalaire).
    * *Avantage :* Partage des features, efficace.
    * *Inconvénient :* L'optimisation est parfois instable car les gradients se mélangent.
2.  **Réseaux séparés :** Deux réseaux distincts.
    * *Avantage :* Plus simple à stabiliser.

---

## ✅ Avantages et ❌ Inconvénients

| Avantages | Inconvénients |
| :--- | :--- |
| **Variance Réduite :** L'apprentissage est beaucoup plus stable que le Policy Gradient pur. | **Biais :** L'utilisation du bootstrapping introduit un biais asymptotique si le critique ne converge pas parfaitement. |
| **Efficacité des Données :** Peut être mis à jour à chaque pas de temps (online), pas besoin d'attendre la fin de l'épisode. | **Complexité :** Plus difficile à implémenter (deux réseaux à gérer, hyperparamètres pour le critique). |
| **Modulaire :** Permet d'utiliser des astuces comme GAE pour affiner l'apprentissage. | **Sensibilité :** Si le critique apprend mal, l'acteur apprendra une mauvaise politique ("garbage in, garbage out"). |

---
*Source: CS 285 Lecture 6 Slides, Instructor: Sergey Levine, UC Berkeley.*