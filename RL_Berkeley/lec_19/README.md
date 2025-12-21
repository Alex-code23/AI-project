# CS 285 : Reframing Control as an Inference Problem (Lecture 19)

Ce document résume le cours sur la reformulation du contrôle et du RL comme un problème d'**inférence probabiliste**.
Au lieu de maximiser simplement une somme de récompenses, on modélise la tâche comme la génération d'une trajectoire conditionnée par l'observation d'une variable d'optimalité. Cela conduit naturellement à l'**entropie maximale** (Maximum Entropy RL) et à des politiques stochastiques robustes ("Soft Optimality").

## 🧠 L'Idée Fondamentale

Dans le RL classique, on cherche la politique optimale $\pi^*$ :
$$\pi^* = \arg\max_\pi \sum_t E[r(s_t, a_t)]$$

Dans l'approche **Inférence**, on introduit une variable binaire $\mathcal{O}_t$ (Optimalité) qui vaut 1 si l'agent est optimal au temps $t$. La probabilité d'être optimal est définie par la récompense :
$$p(\mathcal{O}_t | s_t, a_t) = \exp(r(s_t, a_t))$$

Le problème de RL devient alors : **Calculer la distribution a posteriori des trajectoires sachant qu'on est optimal tout le temps.**
$$p(\tau | \mathcal{O}_{1:T})$$

---

## 📉 Inférence Exacte et le Problème de l'Optimisme

Si on applique les algorithmes d'inférence classiques (type HMM Forward-Backward) à ce modèle graphique :
1.  **Messages Backward ($\beta_t$) :** Correspondent à la "Valeur" ("Reward-to-go").
    $$\beta_t(s_t) \approx \exp(V(s_t))$$
2.  **La Politique :**
    $$p(a_t | s_t, \mathcal{O}_{1:T}) \propto \exp(Q(s_t, a_t) - V(s_t))$$

### Le Problème (Optimisme)
Si on fait de l'inférence naïve pour trouver la trajectoire la plus probable, le modèle va "tricher". Il va supposer que la dynamique $p(s_{t+1}|s_t, a_t)$ va *aussi* changer pour nous aider à atteindre le but (ex: "J'ai gagné au loto, donc la probabilité de gagner devait être de 100%").
Mathématiquement : $p(s_{t+1} | s_t, a_t, \mathcal{O}_{1:T}) \neq p_{env}(s_{t+1} | s_t, a_t)$.

---

## 🛠️ Inférence Variationnelle (Variational Inference)

Pour résoudre ce problème, on fixe la dynamique (elle doit rester celle de l'environnement) et on cherche une distribution de trajectoire $q(\tau)$ qui soit proche du posterior optimal $p(\tau|\mathcal{O})$ tout en respectant la physique.

On minimise la divergence KL :
$$J(q) = D_{KL}(q(\tau) || p(\tau|\mathcal{O}_{1:T}))$$

Cela revient à maximiser la **Borne Inférieure Variationnelle (ELBO)** :
$$\sum_t E_{(s_t, a_t) \sim q} [r(s_t, a_t) + \mathcal{H}(q(a_t | s_t))]$$

**Résultat Clé :** Le RL probabiliste est équivalent à maximiser la récompense **PLUS** l'entropie de la politique ($\mathcal{H}$). C'est le fondement du **Maximum Entropy RL**.

---

## 🤖 Algorithmes "Soft"

Les équations de Bellman changent pour inclure cette "douceur" (Softness) due à l'entropie. Le `max` dur est remplacé par un `softmax` (LogSumExp).

### 1. Soft Value Iteration
Au lieu de $V(s) = \max_a Q(s,a)$, on a :
$$V_{soft}(s) = \log \int \exp(Q(s, a)) da \approx \text{soft\_max}_a Q(s,a)$$

### 2. Soft Q-Learning
L'algorithme modifie la cible de l'apprentissage Q (Target) :
$$y_i = r_i + \gamma V_{soft}(s_i') = r_i + \gamma \log \sum_{a'} \exp(Q_\phi(s_i', a'))$$
La politique induite est stochastique :
$$\pi(a|s) = \exp(Q_\phi(s, a) - V_{soft}(s))$$

### 3. Soft Actor-Critic (SAC)
C'est l'algorithme pratique le plus courant dérivé de cette théorie.
1.  **Critic :** Apprend $Q(s,a)$ en minimisant l'erreur de Bellman douce.
2.  **Actor :** Apprend une politique $\pi_\theta(a|s)$ pour minimiser la divergence KL avec la distribution exponentielle de Q (Projection d'information).
    $$J(\pi) = D_{KL} \left( \pi(\cdot|s) \Big|\Big| \frac{\exp(Q(s, \cdot))}{Z} \right)$$

---

## ✅ Pourquoi faire du Soft RL ? (Avantages)

1.  **Exploration :** L'agent cherche à maximiser l'entropie, ce qui l'incite naturellement à explorer des actions diverses et à ne pas converger prématurément vers une solution sous-optimale déterministe.
2.  **Robustesse :** La politique apprise est plus "large" (couvre plus d'états) et résiste mieux aux perturbations que les politiques "bang-bang" (tout ou rien) du RL standard.
3.  **Multimodalité :** Si deux actions sont aussi bonnes, Soft RL apprendra à jouer les deux avec probabilité égale (alors que Q-learning en choisirait une arbitrairement).
4.  **Pretraining & Transfert :** Les politiques à haute entropie sont d'excellents points de départ pour le finetuning sur des tâches plus spécifiques.

---

## 🔑 Résumé Mathématique

| Concept | Standard RL (Hard) | Inference RL (Soft) |
| :--- | :--- | :--- |
| **Objectif** | $\sum r_t$ | $\sum r_t + \alpha \mathcal{H}(\pi)$ |
| **Value Function** | $V(s) = \max_a Q(s,a)$ | $V(s) = \log \int \exp Q(s,a) da$ |
| **Politique** | Déterministe (Greedy) | Stochastique (Boltzmann/Energy-based) |
| ** Bellman Backup** | $r + \gamma \max Q'$ | $r + \gamma \text{softmax} Q'$ |

---
[cite_start]*Source: CS 285 Lecture 19 Slides[cite: 434, 435, 436], Instructor: Sergey Levine, UC Berkeley.*