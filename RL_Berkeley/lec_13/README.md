# CS 285 : Exploration Part 1 - Bandits & Count-Based Exploration (Lecture 13)

Ce document résume la première partie du cours sur l'**Exploration**.
Dans les problèmes simples, l'exploration aléatoire ($\epsilon$-greedy) suffit. Mais dans des environnements complexes avec des récompenses éparses (ex: *Montezuma's Revenge*), l'agent peut ne jamais trouver la récompense par hasard. Ce cours introduit des méthodes pour une exploration **dirigée** et **intelligente**.

## ⚠️ Le Problème : Récompenses Éparses (Sparse Rewards)

Si la probabilité de trouver une récompense par hasard est exponentiellement faible par rapport à la longueur de l'épisode (horizon), les méthodes classiques (Policy Gradient, Q-Learning avec $\epsilon$-greedy) échouent.

L'objectif est de remplacer l'exploration non dirigée (bruit aléatoire sur les actions) par une exploration dirigée vers les zones inconnues de l'espace d'états.

---

## 🎰 Intuition : Les Bandits Manchots (Multi-Armed Bandits)

Avant de passer au Deep RL, on regarde comment le problème est résolu théoriquement dans le cas simple (1 état, $N$ actions).

### 1. Optimism in the Face of Uncertainty (UCB)
On ne choisit pas l'action avec la meilleure moyenne empirique, mais celle avec la **borne supérieure de confiance** la plus élevée.
$$a_t = \arg\max_a \left( \hat{\mu}(a) + \sqrt{\frac{2 \ln T}{N(a)}} \right)$$
* $\hat{\mu}(a)$ : Récompense moyenne estimée (Exploitation).
* $N(a)$ : Nombre de fois que l'action a été choisie.
* Le terme racine est un **bonus d'exploration** qui diminue quand $N(a)$ augmente.

### 2. Thompson Sampling (Posterior Sampling)
On maintient une distribution de probabilité sur les récompenses possibles $p(\theta | \mathcal{D})$.
* On échantillonne un modèle $\hat{\theta} \sim p(\theta | \mathcal{D})$.
* On agit de façon optimale selon $\hat{\theta}$.
* Cela permet une exploration proportionnelle à l'incertitude ("Probability Matching").

### 3. Information Gain
On choisit l'action qui maximise le gain d'information attendu sur la dynamique ou les récompenses (réduire l'entropie de notre croyance).

---

## 🧠 Deep RL : Count-Based Exploration

Dans un MDP (Markov Decision Process), l'analogue de l'UCB serait d'ajouter un **Bonus d'Exploration** intrinsèque à la récompense :

$$r^+(s, a) = r_{\text{env}}(s, a) + \mathcal{B}(N(s))$$

Où $\mathcal{B}(N(s))$ est souvent proportionnel à $\frac{1}{\sqrt{N(s)}}$.

### Le Défi des Espaces Continus
Dans des environnements complexes (images, robotique), on ne revisite **jamais** exactement le même état (pixels légèrement différents). Donc $N(s)$ est toujours égal à 0 ou 1, ce qui rend le comptage naïf inutile.

### Solution : Pseudo-Counts via Modèles de Densité
L'idée est d'utiliser un modèle génératif pour estimer la densité de probabilité $p_\theta(s)$ (la probabilité d'observer cet état selon nos données passées).
On peut relier la probabilité au comptage via :
$$N(s) \approx \frac{1}{\hat{p}(s)}$$
Si l'état a une probabilité faible (surprenant/nouveau), son pseudo-compte est faible, donc le bonus est élevé.

#### Algorithmes Concrets :
1.  **Modèles Génératifs (CTS / PixelCNN) :** (Bellemare et al. 2016)
    * Entraîner un modèle pour prédire la probabilité des pixels.
    * Utiliser la "probabilité d'enregistrement" (recording probability) pour dériver un pseudo-compte $\hat{N}$.
    * Ajouter un bonus $\frac{1}{\sqrt{\hat{N}}}$ à la récompense.

2.  **Hash-Based Counting (SimHash) :** (Tang et al. 2017)
    * Utiliser un Auto-Encodeur pour compresser l'image en un code latent $\phi(s)$.
    * Utiliser du **Locality-Sensitive Hashing (LSH)** pour discrétiser cet espace continu en "buckets" discrets.
    * Compter simplement les visites dans chaque bucket : $N(h(s))$.
    * C'est simple et très efficace.

3.  **Implicit Density (EX2) :** (Fu et al. 2017)
    * Entraîner un classifieur à distinguer les états visités de bruits aléatoires. La performance du classifieur donne une estimation de la densité.

---

## ⚖️ Résumé des Approches Count-Based

| Méthode | Principe | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- |
| **UCB (Tabulaire)** | Compter $N(s,a)$ dans un tableau. | Théoriquement optimal. | Impossible pour les grands espaces d'états. |
| **Density Models** | Estimer $p(s)$ avec un réseau (PixelCNN/VAE) pour dériver $\hat{N}$. | Gère les images directement. | Les modèles génératifs sont lourds et difficiles à entraîner. |
| **Hash-Based** | Discrétiser l'espace latent (SimHash) et compter. | Rapide, simple à implémenter. | Perd de l'information (aliasing) à cause du hachage. |

---

## 🔑 Points Clés à retenir
* L'exploration $\epsilon$-greedy est insuffisante pour les problèmes à récompenses éparses.
* Le principe d'**"Optimisme face à l'incertitude"** suggère d'ajouter un bonus aux états peu visités.
* En Deep RL, on ne peut pas compter les états. On utilise des **Pseudo-Comptes** dérivés de la densité de probabilité ($p(s)$) ou de la discrétisation (Hashing).
* L'objectif final est de modifier la fonction de récompense : $r_{total} = r_{externe} + \alpha \cdot r_{exploration}$.

---
*Source: CS 285 Lecture 13 Slides, Instructor: Sergey Levine, UC Berkeley.*