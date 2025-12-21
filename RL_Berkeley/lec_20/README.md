# CS 285 : Inverse Reinforcement Learning (Lecture 20)

Ce document résume le cours sur l'**Inverse Reinforcement Learning (IRL)**.
Jusqu'à présent, nous avons supposé que la fonction de récompense $r(s,a)$ était donnée. Cependant, concevoir des récompenses est difficile et sujet à des effets secondaires indésirables ("reward hacking"). L'IRL vise à **apprendre la récompense** à partir de démonstrations d'un expert, pour ensuite utiliser le RL pour trouver une politique optimale.

## 🎯 Motivation : Pourquoi l'IRL ?

1.  **Imitation (Behavioral Cloning) :** Copier simplement les actions de l'expert (Apprentissage Supervisé) fonctionne mal à cause de l'accumulation d'erreurs (Distribution Shift). [cite_start]L'agent ne sait pas *pourquoi* l'expert agit ainsi[cite: 1].
2.  **Inférence de l'Intention :** L'IRL tente de comprendre le but sous-jacent (la récompense). [cite_start]Si on connaît la récompense, on peut trouver une politique qui généralise mieux et qui est robuste aux perturbations[cite: 1].

## 🧠 Le Principe du Maximum Entropy IRL

L'hypothèse centrale est que les démonstrations de l'expert sont des échantillons tirés d'une distribution optimale (ou sous-optimale Boltzmann). On utilise le modèle probabiliste vu au cours 19 :

$$p(\tau) \propto \exp(R(\tau))$$

Où $R(\tau) = \sum_t r(s_t, a_t)$. [cite_start]L'objectif est de trouver les paramètres $\psi$ de la récompense $r_\psi$ qui maximisent la vraisemblance des trajectoires de l'expert $\mathcal{D}_{demo} = \{\tau_i\}$[cite: 1].

### L'Objectif MaxEnt
$$\max_\psi \sum_{\tau \in \mathcal{D}_{demo}} \log p_{r_\psi}(\tau)$$
$$\log p_{r_\psi}(\tau) = R_\psi(\tau) - \log Z$$
Où $Z = \int \exp(R_\psi(\tau)) d\tau$ est la fonction de partition (très difficile à calculer).

### Feature Matching
Si la récompense est linéaire par rapport à des caractéristiques $\mathbf{f}(\tau)$ (soit $R(\tau) = \mathbf{w}^T \mathbf{f}(\tau)$), alors le gradient de la log-vraisemblance mène à une propriété élégante :
$$\nabla_\mathbf{w} \mathcal{L} = E_{\tau \sim \text{expert}} [\mathbf{f}(\tau)] - E_{\tau \sim \pi_{learned}} [\mathbf{f}(\tau)]$$
[cite_start]L'algorithme converge quand les **comptes de caractéristiques (feature counts)** de l'agent correspondent à ceux de l'expert[cite: 1].

---

## 🚀 Algorithmes Modernes & Deep IRL

Le calcul de la fonction de partition $Z$ nécessite de résoudre le problème de RL complet (Soft Value Iteration) à chaque étape d'optimisation de la récompense ("boucle interne"), ce qui est très coûteux.

### 1. Guided Cost Learning (GCL)
Pour passer à l'échelle avec des réseaux de neurones profonds :
* On utilise l'**Importance Sampling** pour estimer $Z$ sans tout ré-optimiser à chaque fois.
* On génère des échantillons avec la politique actuelle $q(\tau)$ pour estimer l'intégrale.
* [cite_start]Cela revient à entraîner la récompense pour donner un score élevé aux démos expertes et un score faible aux échantillons générés par la politique actuelle[cite: 1].

### 2. Generative Adversarial Imitation Learning (GAIL)
[cite_start]Il existe une connexion forte entre GCL et les **GANs** (Generative Adversarial Networks)[cite: 1].
* **Discriminateur ($D$) :** Essaie de distinguer les états/actions de l'expert (Vrai) de ceux de l'agent (Faux).
* **Générateur ($\pi$) :** L'agent (la politique) essaie de tromper le discriminateur.

Au lieu d'apprendre une fonction de récompense explicite $r_\psi$, on utilise le discriminateur comme récompense immédiate :
$$r(s,a) = \log D(s,a) - \log(1 - D(s,a))$$
L'agent RL maximise cette récompense, ce qui le force à imiter la distribution d'états de l'expert.

---

## 🏗️ Structure d'un Algorithme IRL Général

[cite_start]La plupart des algorithmes suivent cette boucle itérative[cite: 1]:

1.  **Collecte de Données :** L'agent exécute sa politique $\pi$ pour générer des trajectoires.
2.  **Mise à jour de la Récompense :** On ajuste $r_\psi$ pour qu'elle donne un score plus élevé aux démos de l'expert qu'aux trajectoires générées par l'agent.
    * *En MaxEnt IRL :* Monter le gradient de vraisemblance.
    * *En GAIL :* Mettre à jour le discriminateur.
3.  **Mise à jour de la Politique :** On utilise un algorithme de RL (ex: Policy Gradient, TRPO, SAC) pour maximiser la nouvelle récompense $r_\psi$.

---

## ✅ Avantages et ❌ Inconvénients

| Avantages | Inconvénients |
| :--- | :--- |
| **Généralisation :** Apprendre la récompense permet de s'adapter à de nouveaux environnements (Transfer Learning) mieux que le simple clonage. | **Ambiguïté :** Plusieurs fonctions de récompense peuvent expliquer le même comportement (ex: $R=0$ explique tout). Nécessite souvent de la régularisation (MaxEnt). |
| **Robustesse :** "Lisse" les erreurs de démonstration en cherchant l'intention optimale plutôt que de copier le bruit. | **Coût de Calcul :** Nécessite une boucle interne de RL. On résout un MDP à chaque étape d'apprentissage de la récompense. |
| **Moins de Données :** Souvent plus efficace en nombre de démos que le clonage comportemental pur. | **Instabilité :** Comme pour les GANs, l'entraînement adversarial (GAIL) peut être instable. |

---
*Source: CS 285 Lecture 20 Slides, Instructor: Sergey Levine, UC Berkeley.*