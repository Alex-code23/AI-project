# CS 285 : Policy Gradients (Lecture 5)

Ce document résume le cours sur les **Policy Gradients** (Gradients de Politique). L'objectif principal est d'apprendre une politique paramétrée $\pi_\theta(a|s)$ qui maximise la somme des récompenses espérées en optimisant directement les paramètres $\theta$ par descente de gradient.

## 🎯 Objectif du Reinforcement Learning

Le but est de maximiser l'espérance des récompenses cumulées sur une trajectoire $\tau$ :

$$\theta^* = \arg \max_\theta J(\theta)$$
$$J(\theta) = E_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(s_t, a_t) \right]$$

[cite_start]Où la probabilité d'une trajectoire $p_\theta(\tau)$ dépend de la politique et de la dynamique du système (bien que la dynamique s'annule dans le gradient final)[cite: 36, 54].

---

## 🧮 Dérivation du Gradient (The Log-Derivative Trick)

Pour calculer le gradient de l'espérance $\nabla_\theta J(\theta)$, on utilise l'identité $\nabla p(\tau) = p(\tau) \nabla \log p(\tau)$ :

$$\nabla_\theta J(\theta) = E_{\tau \sim p_\theta(\tau)} [\nabla_\theta \log p_\theta(\tau) r(\tau)]$$

En simplifiant grâce à la propriété de Markov (la dynamique $p(s_{t+1}|s_t, a_t)$ ne dépend pas de $\theta$), on obtient la formule standard :

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \right) \left( \sum_{t=1}^T r(s_{i,t}, a_{i,t}) \right)$$

**Intuition :** Cette formule formalise l'essai-erreur ("trial and error"). [cite_start]Elle augmente la probabilité des trajectoires ayant une récompense élevée et diminue celle des trajectoires à faible récompense[cite: 121, 122].

---

## 🤖 Algorithme REINFORCE

[cite_start]L'algorithme de base fonctionne comme suit[cite: 61]:

1.  **Échantillonner** $\{\tau^i\}$ à partir de la politique $\pi_\theta(a_t|s_t)$ (exécuter la politique sur le robot/environnement).
2.  **Estimer le gradient** $\nabla_\theta J(\theta)$ en utilisant les échantillons.
3.  **Mettre à jour les paramètres** : $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$.

---

## 📉 Réduction de la Variance

[cite_start]Le gradient de politique brut a une **variance très élevée**, ce qui rend l'apprentissage instable[cite: 156]. Deux techniques principales sont utilisées pour la réduire :

### 1. Causalité (Reward-to-go)
La politique au temps $t$ ne peut pas affecter les récompenses passées ($t' < t$). On remplace la somme totale des récompenses par la somme des récompenses futures (Reward-to-go $\hat{Q}_{i,t}$) :

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \underbrace{\sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})}_{\hat{Q}_{i,t}}$$

[cite_start]Cette modification est valide car le futur n'affecte pas le passé[cite: 176, 177].

### 2. Baselines (Lignes de base)
On peut soustraire une valeur constante ou dépendante de l'état (baseline $b$) à la récompense sans biaiser le gradient (car $E[\nabla \log p(\tau) \cdot b] = 0$) :

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau) [r(\tau) - b]$$

* **Pourquoi ?** Cela centre les retours. Si toutes les récompenses sont positives, sans baseline, on ne ferait qu'augmenter les probabilités de tout, juste à des vitesses différentes.
* [cite_start]**Baseline optimale :** La récompense moyenne pondérée par la magnitude du gradient[cite: 195].

---

## 🔄 Off-Policy Policy Gradients (Importance Sampling)

L'apprentissage "On-policy" est inefficace car chaque échantillon n'est utilisé qu'une fois. [cite_start]Pour utiliser des échantillons d'une ancienne politique $\bar{\pi}$, on utilise l'**Importance Sampling (IS)**[cite: 220, 225]:

$$J(\theta') = E_{\tau \sim \pi_\theta(\tau)} \left[ \frac{\pi_{\theta'}(\tau)}{\pi_\theta(\tau)} r(\tau) \right]$$

[cite_start]Cela permet de réutiliser les données passées, mais introduit un produit de ratios qui peut mener à une variance exponentielle en fonction de l'horizon $T$[cite: 252].

---

## 💻 Implémentation avec Différentiation Automatique

En pratique, on n'implémente pas la formule du gradient directement. [cite_start]On définit une "pseudo-loss" que l'on minimise avec un optimiseur standard (comme Adam)[cite: 270, 294]:

$$\tilde{J}(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot \hat{Q}_{i,t}$$

En TensorFlow/PyTorch : `loss = reduce_mean(cross_entropy * q_values)`.

---

## 🚀 Sujets Avancés : Natural Policy Gradient

Le gradient standard suit la direction la plus raide dans l'espace des paramètres (Euclidien), mais une petite modification de paramètres peut changer drastiquement la politique (probabilités).

[cite_start]**Solution :** Limiter le changement de la distribution de la politique (Divergence KL) plutôt que le changement des paramètres[cite: 352].

$$\theta \leftarrow \theta + \alpha F^{-1} \nabla_\theta J(\theta)$$

Où $F$ est la **matrice d'information de Fisher**. [cite_start]Cela mène à des algorithmes comme **TRPO** (Trust Region Policy Optimization) et **PPO**[cite: 367, 374].

---

## ✅ Avantages et ❌ Inconvénients

| Avantages | Inconvénients |
| :--- | :--- |
| **Direct :** Optimise directement l'objectif de RL. | [cite_start]**Haute Variance :** Le gradient est très bruité, nécessite de gros batchs[cite: 298]. |
| **Continu :** Gère facilement les espaces d'actions continus (ex: robots). | [cite_start]**Efficacité :** Souvent "On-policy", donc nécessite beaucoup d'échantillons (sample inefficient)[cite: 217]. |
| **Convergence :** Garanties de convergence locale. | **Optima Locaux :** Peut rester coincé dans des optimums locaux. |
| [cite_start]**Partiellement Observable :** Fonctionne sans modification si l'état n'est pas complet ($o_t$ vs $s_t$)[cite: 149]. | [cite_start]**Sensible :** Difficile à régler (learning rates instables sans méthodes avancées comme Adam ou Natural Gradient)[cite: 302]. |

---
*Source: CS 285 Lecture 5 Slides, Instructor: Sergey Levine, UC Berkeley.*