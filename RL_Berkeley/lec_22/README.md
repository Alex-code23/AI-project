# CS 285 : Meta-Learning & Transfer Learning (Lecture 22)

Ce document résume le cours sur le **Meta-Learning** (apprendre à apprendre) et le **Transfer Learning**. Contrairement au RL classique qui apprend chaque tâche de zéro ("tabula rasa"), ces méthodes visent à utiliser l'expérience acquise sur des tâches passées pour apprendre de nouvelles tâches plus rapidement et plus efficacement.

## 🎯 Motivation : Au-delà du "Tabula Rasa"

Le Deep RL standard est inefficace en termes d'échantillons (sample inefficient). Pour résoudre une nouvelle tâche, il doit tout réapprendre.
* **Intuition humaine :** Si vous savez ouvrir une porte, vous savez probablement ouvrir un placard. Si vous savez jouer à Mario, vous apprendrez Sonic plus vite.
* **Le but :** Utiliser des connaissances préalables (Priors) structurelles ou dynamiques pour accélérer l'acquisition de nouvelles compétences.

---

## 🔄 Transfer Learning

L'objectif est d'utiliser l'expérience d'un **domaine source** pour être performant sur un **domaine cible**.

### Terminologie
* **0-shot :** Exécuter la politique entraînée directement sur la nouvelle tâche sans entraînement supplémentaire.
* **Few-shot :** L'agent a droit à quelques essais (épisodes) sur la nouvelle tâche pour s'adapter.
* **Fine-tuning :** Entraîner sur la source, puis continuer l'entraînement (avec un learning rate plus bas) sur la cible.

### Pourquoi le Fine-tuning échoue souvent en RL ?
Contrairement à la vision par ordinateur (ImageNet), le fine-tuning en RL est difficile :
1.  **Exploration :** Une politique optimale sur la tâche source est souvent **déterministe**. Elle a "oublié" comment explorer. Face à une nouvelle tâche, elle échoue à découvrir les nouvelles récompenses.
2.  **Spécialisation :** Les représentations apprises deviennent trop spécifiques à la dynamique de la tâche source.

---

## 🧠 Meta-Reinforcement Learning (Meta-RL)

Le Meta-RL formule le problème non pas comme "apprendre une tâche", mais comme **"apprendre un algorithme d'apprentissage"**.

### Formulation Mathématique
Si un algorithme d'apprentissage générique s'écrit $\phi = f_{learn}(\mathcal{D}^{tr})$, le Meta-Learning cherche à optimiser la fonction $f_\theta$ sur un ensemble de tâches :

$$\theta^* = \arg\max_\theta \sum_{i=1}^n E_{\pi_{\phi_i}(\tau)} [R(\tau)] \quad \text{où} \quad \phi_i = f_\theta(\mathcal{M}_i)$$

* $\mathcal{M}_i$ : Une tâche (MDP) échantillonnée depuis une distribution $p(\mathcal{M})$.
* $f_\theta$ : La procédure d'adaptation (le méta-modèle).
* $\phi_i$ : Les paramètres adaptés à la tâche $i$.

L'agent doit maximiser la récompense cumulée sur l'ensemble de l'expérience ("Meta-episode"), ce qui inclut les essais exploratoires et les essais finaux.

---

## 📐 Les 3 Perspectives du Meta-RL

Le cours classifie les algorithmes de Meta-RL en trois catégories principales, qui sont mathématiquement liées mais diffèrent par leur implémentation.

### 1. Perspective Récurrente (RNN / Black-Box)
On utilise un réseau de neurones récurrent (RNN, LSTM, Transformer) qui prend en entrée toute l'histoire des interactions (états, actions, récompenses).

* **Principe :** L'état caché $h_i$ du RNN sert de "mémoire" ou de "paramètres appris". Le RNN *apprend* à explorer et à adapter sa stratégie au fil des timesteps sans mise à jour explicite des poids (les poids $\theta$ du RNN sont fixes au test, c'est l'activité interne qui change).
* **Architecture :**
    $$\pi_\theta(a_t | s_t, h_t) \quad \text{où} \quad h_{t+1} = \text{RNN}(h_t, s_t, a_t, r_t)$$
    *Crucial :* L'état caché $h_t$ n'est **pas réinitialisé** entre les épisodes d'une même tâche.
* **Exemples :** RL2 (Duan et al.), Learning to Reinforcement Learn (Wang et al.).

### 2. Perspective Optimisation (Gradient-Based / MAML)
On force la procédure d'adaptation $f_\theta$ à être une étape de descente de gradient. On cherche des paramètres initiaux $\theta$ tels qu'un seul pas de gradient sur une nouvelle tâche mène à une politique performante.

* **Algorithme (MAML - Model-Agnostic Meta-Learning) :**
    $$J(\theta) = \sum_i J_i(\theta - \alpha \nabla_\theta J_i(\theta))$$
    On optimise $\theta$ pour que la performance *après* mise à jour soit maximale.
* **Avantage :** Modèle agnostique, garantit une convergence asymptotique (car c'est toujours du gradient descent).
* **Inconvénient :** Nécessite de calculer des dérivées secondes (Hessiennes) ou des approximations complexes.

### 3. Perspective Inférence Probabiliste (Task Inference / PEARL)
On considère que la tâche est définie par une variable latente cachée $z$ (ex: la vitesse cible, la gravité). Le problème devient un POMDP (Partially Observed MDP) où $z$ doit être inféré.

* **Principe :** Apprendre une politique conditionnée par le contexte $\pi_\theta(a|s, z)$ et un réseau d'inférence $q_\phi(z | \text{historique})$.
* **Posterior Sampling (Exploration) :**
    1.  On échantillonne une hypothèse $z \sim q_\phi(z|\text{context})$.
    2.  On agit selon cette hypothèse (exploration structurée).
    3.  On met à jour le contexte avec les nouvelles données.
* **Exemple :** PEARL (Probabilistic Embeddings for Actor-Critic RL). C'est souvent l'approche la plus efficace pour le RL off-policy.

---

## ⚖️ Comparaison des Architectures

| Perspective | Approche | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- |
| **RNN (RL2)** | "Just run an RNN" | Conceptuellement simple, facile à implémenter. | Difficile à optimiser sur de longues séquences, "Meta-Overfitting" fréquent. |
| **Gradient (MAML)** | Bi-level Optimization | Bonne extrapolation, structure inductive forte (le gradient est toujours bon). | Complexe à calculer (dérivées secondes), nécessite beaucoup de samples (On-policy). |
| **Inférence (PEARL)** | POMDP / Variable Latente | Exploration efficace (Posterior Sampling), permet l'Off-policy (Sample efficient). | Architecture plus complexe (Encoder + Policy), difficile à stabiliser. |

---

## 🧬 Phénomènes Émergents

Le Meta-RL est intéressant pour les neurosciences car il fait émerger des comportements complexes sans qu'ils soient explicitement programmés :
* **Apprentissage Épisodique :** Les réseaux récurrents apprennent à stocker des événements en mémoire pour les réutiliser.
* **Raisonnement Causal :** L'agent apprend à faire des expériences pour déduire les règles de l'environnement (inférence causale implicite).

---
*Source: CS 285 Lecture 22 Slides, Instructor: Sergey Levine, UC Berkeley.*