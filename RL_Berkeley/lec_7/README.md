# CS 285 : Value Function Methods (Lecture 7)

Ce document résume le cours sur les **Value Function Methods** (Méthodes basées sur la fonction de valeur). Contrairement aux méthodes précédentes qui optimisent directement une politique, ces algorithmes apprennent une fonction de valeur ($V$ ou $Q$) et définissent la politique comme étant celle qui maximise cette valeur (politique gloutonne/greedy).

## 🎯 Objectif : Omettre le Policy Gradient

L'objectif est de trouver la politique optimale sans la représenter explicitement par un réseau de neurones paramétré $\pi_\theta$. On apprend plutôt une fonction de valeur neuronale $V_\phi(s)$ ou $Q_\phi(s,a)$.

La politique devient implicite (Argmax policy) :
$$\pi(a_t|s_t) = \begin{cases} 1 & \text{si } a_t = \arg\max_a A^\pi(s_t, a) \\ 0 & \text{sinon} \end{cases}$$
Où $A^\pi(s,a)$ est la fonction d'avantage.

---

## 🔄 De l'Itération de Politique à l'Itération de Valeur

### 1. Policy Iteration (Itération de Politique)
L'algorithme alterne entre deux étapes jusqu'à convergence :
1.  **Policy Evaluation :** Calculer $A^\pi(s,a)$ pour la politique actuelle (souvent coûteux).
2.  **Policy Improvement :** Mettre à jour la politique $\pi \leftarrow \arg\max A^\pi$.

### 2. Value Iteration (Itération de Valeur)
On simplifie le processus en combinant les deux étapes. On met à jour directement la fonction de valeur optimale $V^*$ sans passer par une politique intermédiaire :

$$V(s) \leftarrow \max_a \sum_{s'} p(s'|s,a) [r(s,a) + \gamma V(s')]$$

---

## 🧠 Fitted Value Iteration & Q-Iteration

Pour les espaces d'états continus ou très grands, on ne peut pas utiliser de tableaux. On utilise un approximateur de fonction (Réseau de Neurones) avec paramètres $\phi$.

### Fitted Value Iteration
On apprend $V_\phi(s)$ en minimisant l'erreur quadratique par rapport à une cible $y_i$ :
$$y_i = \max_{a_i} (r(s_i, a_i) + \gamma E[V_\phi(s'_i)])$$
$$\mathcal{L}(\phi) = \frac{1}{2} \sum_i || V_\phi(s_i) - y_i ||^2$$
* **Limitation :** Pour calculer le $\max_a$ et l'espérance $E$, il faut connaître la dynamique $p(s'|s,a)$ (le modèle de transition).

### Fitted Q-Iteration (FQI)
Pour se passer de modèle (Model-Free), on apprend la fonction $Q_\phi(s,a)$.

**Algorithme complet :**
1.  **Collecte de données :** Obtenir un dataset $\mathcal{D} = \{(s_i, a_i, s'_i, r_i)\}$ en utilisant une politique d'exploration.
2.  **Calcul des cibles :** $y_i = r_i + \gamma \max_{a'} Q_\phi(s'_i, a')$.
3.  **Régression (Update) :** Entraîner $\phi$ pour minimiser $\sum (Q_\phi(s_i, a_i) - y_i)^2$.
4.  **Itération :** Répéter les étapes 2 et 3 $K$ fois.

C'est un algorithme **Off-Policy** : on peut utiliser des données collectées par n'importe quelle politique passée.

---

## 📉 Théorie et Convergence

Pourquoi ces méthodes fonctionnent-elles (ou échouent-elles) ?

### Cas Tabulaire (Tableau)
L'opérateur de Bellman $\mathcal{B}$ est une **contraction** pour la norme $\infty$ (max norm).
$$|| \mathcal{B}V - \mathcal{B}\bar{V} ||_\infty \le \gamma || V - \bar{V} ||_\infty$$
Cela garantit que *Value Iteration* converge toujours vers la solution unique $V^*$.

### Cas "Fitted" (Réseaux de Neurones)
L'algorithme alterne entre l'opérateur de Bellman $\mathcal{B}$ et une étape de projection $\Pi$ (la régression/minimisation de l'erreur).
* $\Pi$ est une contraction pour la norme $L_2$ (Euclidienne).
* $\mathcal{B}$ est une contraction pour la norme $L_\infty$.
* **Problème :** La composition $\Pi \mathcal{B}$ n'est **pas** une contraction.
* **Conséquence :** Fitted Q-Iteration n'est **pas garanti de converger** et peut osciller ou diverger avec des réseaux de neurones.

---

## 🔍 Exploration

Puisque la politique dérivée est déterministe ($a = \arg\max Q$), l'exploration explicite est cruciale.
* **Epsilon-Greedy :** Avec probabilité $\epsilon$, choisir une action au hasard ; sinon, choisir l'action optimale.
* **Boltzmann Exploration :** Choisir les actions proportionnellement à $\exp(Q(s,a))$.

---

## ✅ Avantages et ❌ Inconvénients

| Avantages | Inconvénients |
| :--- | :--- |
| **Sample Efficiency :** Méthodes **Off-policy**. Très efficaces car elles réutilisent les données passées. | **Convergence :** Aucune garantie théorique de convergence avec les réseaux de neurones. Risque de divergence. |
| **Simplicité :** Pas de gradient de politique à haute variance. | **Optimisation :** Calculer $\max_a Q(s,a)$ est facile pour les actions discrètes mais difficile pour les actions continues. |
| **Vitesse :** La régression supervisée (étape 3) est souvent plus stable que la montée de gradient sur une politique. | **Qualité :** La politique apprise peut être biaisée par les erreurs d'approximation de la fonction $Q$. |

---
*Source: CS 285 Lecture 7 Slides, Instructor: Sergey Levine, UC Berkeley.*