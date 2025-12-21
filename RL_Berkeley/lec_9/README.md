# CS 285 : Advanced Policy Gradients (Lecture 9)

Ce document résume le cours sur les **Advanced Policy Gradients**. Alors que le Policy Gradient standard (REINFORCE) est instable et sensible au pas d'apprentissage (step size), ce cours introduit des méthodes pour garantir une amélioration monotone de la politique et stabiliser l'apprentissage en utilisant la géométrie de l'espace des distributions (Gradient Naturel, TRPO).

## 🎯 Motivation : Policy Gradient comme Policy Iteration

L'objectif est de voir le Policy Gradient non plus comme une simple montée de gradient stochastique, mais comme une approximation d'une **Policy Iteration**.

On cherche une mise à jour $\pi'$ telle que $J(\pi') \ge J(\pi)$.
Pour cela, on utilise l'identité de l'avantage :
$$J(\pi') = J(\pi) + E_{\tau \sim \pi'} \left[ \sum_t \gamma^t A^\pi(s_t, a_t) \right]$$

Pour garantir une amélioration, il faut maximiser le second terme. Cependant, l'espérance dépend de $\pi'$ (la nouvelle politique) qu'on ne connaît pas encore.

---

## 🚧 Le Problème du "Distribution Mismatch"

Si $\pi'$ est proche de $\pi$, on peut approximer l'espérance sur $\pi'$ par une espérance sur $\pi$ (en ignorant le changement de distribution d'états) :

$$L_\pi(\pi') = J(\pi) + E_{s \sim \pi, a \sim \pi'} [ A^\pi(s, a) ]$$

Cependant, cette approximation introduit une erreur. La théorie (Schulman et al., TRPO) fournit une borne sur cette erreur en fonction de la divergence KL entre les politiques :

$$J(\pi') \ge L_\pi(\pi') - C \cdot \max_s D_{KL}(\pi(a|s) || \pi'(a|s))$$

* **Idée clé :** Si on maximise $L_\pi(\pi')$ tout en gardant la divergence KL petite (Trust Region), on garantit d'améliorer la vraie performance $J(\pi')$.

---

## 🧬 Natural Policy Gradient (NPG)

La montée de gradient standard (Vanilla Gradient Ascent) suit la direction la plus raide dans l'espace des **paramètres** (Euclidien). Or, une petite variation des paramètres $\theta$ peut entraîner un changement énorme de la distribution $\pi_\theta$ (la politique).

### 1. L'Objectif Contraint
On veut maximiser l'objectif sous une contrainte de changement de distribution :
$$\max_{\theta'} \nabla_\theta J(\theta)^T (\theta' - \theta)$$
$$\text{s.t. } D_{KL}(\pi_{\theta'} || \pi_\theta) \le \epsilon$$

### 2. Approximation Quadratique du KL
La divergence KL peut être approximée localement par la **Matrice d'Information de Fisher** ($F$) :
$$D_{KL}(\pi_{\theta'} || \pi_\theta) \approx \frac{1}{2} (\theta' - \theta)^T F (\theta' - \theta)$$
$$F = E_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T]$$

### 3. La Mise à jour (Natural Gradient Update)
La solution analytique de ce problème d'optimisation contraint donne la direction du gradient naturel :
$$\theta \leftarrow \theta + \alpha F^{-1} \nabla_\theta J(\theta)$$

Le pas $\alpha$ est choisi pour satisfaire la contrainte KL.

---

## 🚀 Algorithmes Pratiques

### TRPO (Trust Region Policy Optimization)
TRPO est une approximation pratique du NPG.
* Il utilise l'objectif "surrogate" (Importance Sampling) pour estimer $L_\pi(\pi')$.
* Il résout $F^{-1} g$ efficacement en utilisant la méthode du **Gradient Conjugué** (Conjugate Gradient) pour éviter d'inverser explicitement la matrice Hessienne/Fisher (très coûteux).
* Il impose une "Hard Constraint" sur le KL (Trust Region) via une recherche linéaire (Line Search).

### PPO (Proximal Policy Optimization)
Mentionné comme une simplification de TRPO.
* Au lieu d'une contrainte dure (Hard Constraint) difficile à optimiser, PPO utilise une **régularisation** (Clipping ou pénalité KL) directement dans la fonction objective.
* Beaucoup plus simple à implémenter (gradient descent standard de premier ordre).

---

## ✅ Avantages et ❌ Inconvénients

| Avantages | Inconvénients |
| :--- | :--- |
| **Stabilité :** Garantit une amélioration monotone (théoriquement) et évite les effondrements de performance ("policy collapse"). | **Complexité (TRPO/NPG) :** Nécessite le calcul (ou l'approximation) de la matrice de Fisher et l'algorithme du Gradient Conjugué. |
| **Indépendance de Paramétrage :** Le comportement de l'apprentissage dépend de la distribution, pas du choix arbitraire des paramètres du réseau. | **Coût de calcul :** Plus lourd qu'un simple gradient (REINFORCE/Adam). |
| **Pas d'apprentissage (Step Size) :** Plus robuste au choix du learning rate grâce à la région de confiance adaptive. | **Implémentation :** TRPO est notoirement difficile à implémenter correctement par rapport à PPO ou SAC. |

---
*Source: CS 285 Lecture 9 Slides, Instructor: Sergey Levine, UC Berkeley.*