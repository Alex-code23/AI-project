# CS 285 : Deep RL with Q-Functions (Lecture 8)

Ce document résume le cours sur l'application des réseaux de neurones profonds au Q-Learning. Il explique pourquoi le Q-Learning "naïf" échoue avec les réseaux de neurones et introduit les algorithmes classiques comme **DQN** et ses améliorations pour les actions discrètes et continues.

## ⚠️ Les problèmes du Q-Learning avec Réseaux de Neurones

L'algorithme "online Q-learning" standard ressemble à une descente de gradient, mais il n'en est pas une vraie. Lorsqu'on utilise des approximateurs de fonction non-linéaires (réseaux de neurones), deux problèmes majeurs causent l'instabilité ou la divergence:

### 1. Échantillons Corrélés (Correlated Samples)
Dans l'apprentissage en ligne, les données arrivent séquentiellement $(s_t, a_t, r_t, s_{t+1})$. Ces échantillons sont fortement corrélés temporellement. La descente de gradient stochastique (SGD) suppose que les données sont i.i.d. (indépendantes et identiquement distribuées). Sans cela, le réseau sur-apprend sur les données récentes et oublie les anciennes.

### 2. Cibles Mouvantes (Moving Targets)
La cible de la régression $y_i = r + \gamma \max_{a'} Q_\phi(s', a')$ dépend des mêmes paramètres $\phi$ que ceux que l'on est en train d'optimiser.
Contrairement à la régression supervisée où la cible est fixe, ici la cible bouge à chaque mise à jour. Cela crée des boucles de rétroaction instables et des oscillations.

---

## 🎮 La Solution : DQN (Deep Q-Network)

L'algorithme DQN (Mnih et al., 2013/2015) introduit deux mécanismes pour stabiliser l'apprentissage sur les jeux Atari :

### 1. Replay Buffer (Tampon de Répétition)
Au lieu d'apprendre sur la dernière transition, on stocke les transitions $(s, a, r, s')$ dans un grand buffer $\mathcal{B}$. On échantillonne ensuite un **batch aléatoire** pour la mise à jour.
* **Avantage :** Brise la corrélation temporelle et rend les échantillons plus proches de l'i.i.d..

### 2. Target Network (Réseau Cible)
On utilise un second réseau $Q_{\phi'}$ (Target Network) pour calculer la cible, dont les paramètres $\phi'$ sont une copie retardée de $\phi$ (mise à jour périodique ou moyenne exponentielle/Polyak averaging).
* **Cible :** $y = r + \gamma \max_{a'} Q_{\phi'}(s', a')$
* **Avantage :** La cible reste stable pendant un certain temps, transformant le problème en une série de problèmes de régression supervisée plus stables.

---

## 📈 Améliorations de DQN

### Double Q-Learning
Le Q-Learning standard surestime systématiquement les valeurs Q car $E[\max(X)] \ge \max(E[X])$ (le bruit positif est amplifié par le max).
**Solution :** Découpler la *sélection* de l'action et son *évaluation*.
* Utiliser le réseau actuel $\phi$ pour choisir l'action.
* Utiliser le réseau cible $\phi'$ pour évaluer sa valeur.
$$y = r + \gamma Q_{\phi'}(s', \arg\max_{a'} Q_\phi(s', a'))$$.

### Multi-step Returns (N-step)
Au lieu d'utiliser un seul pas de récompense (Bellman pur), on utilise $N$ pas avant de bootstraper.
$$y_{i,t} = \sum_{k=0}^{N-1} \gamma^k r_{t+k} + \gamma^N \max_{a'} Q_{\phi'}(s_{i, t+N}, a')$$.
* **Trade-off :** Réduit le biais (moins de dépendance à l'estimation Q initiale) mais augmente la variance (plus de récompenses stochastiques accumulées). Souvent, $N$ entre 3 et 5 fonctionne bien.

---

## 🤖 Q-Learning pour Actions Continues

L'opération $\max_a Q(s,a)$ est difficile quand l'espace d'action est continu.

### 1. Optimisation Stochastique
Utiliser des méthodes comme CEM ou CMA-ES pour trouver le max, ou une simple descente de gradient sur l'input $a$. C'est souvent trop lent.

### 2. Normalized Advantage Functions (NAF)
On force l'architecture du réseau Q à être quadratique par rapport à l'action $a$, ce qui rend le maximum analytique et facile à calculer ($argmax$ est $\mu(s)$).
$$Q(s,a) = -\frac{1}{2}(a - \mu(s))^T P(s) (a - \mu(s)) + V(s)$$

### 3. DDPG (Deep Deterministic Policy Gradient)
On apprend un réseau "acteur" $\mu_\theta(s)$ qui prédit l'action maximisant $Q$.
* Le Critique apprend $Q_\phi(s,a)$ (similaire à DQN).
* L'Acteur apprend $\theta$ pour maximiser $Q_\phi(s, \mu_\theta(s))$ via la règle de la chaîne.
$$\frac{dQ}{d\theta} = \frac{dQ}{da} \frac{da}{d\theta}$$
C'est essentiellement du Q-Learning où le `max` est approximé par un réseau de neurones.

---

## 🛠️ Conseils Pratiques pour le Q-Learning

* **Fiabilité :** Le Q-Learning est moins stable que les Policy Gradients. Il nécessite beaucoup de réglages d'hyperparamètres.
* **Exploration :** Commence avec un $\epsilon$ élevé et diminue-le lentement.
* **Stabilité :**
    * Utiliser **Double Q-Learning** (aide presque toujours).
    * Utiliser des taux d'apprentissage (learning rates) bas.
    * Clipper les gradients ou utiliser la **Huber Loss** (pour éviter que les erreurs Bellman élevées ne déstabilisent tout).
* **Temps :** La convergence peut être très longue, ne pas arrêter l'entraînement trop tôt.

---

## ✅ Avantages et ❌ Inconvénients

| Avantages | Inconvénients |
| :--- | :--- |
| **Sample Efficiency :** Très efficace en données grâce au Replay Buffer (Off-policy). | **Convergence :** Pas de garantie de convergence avec l'approximation de fonction non-linéaire. |
| **Généralité :** Fonctionne bien sur des tâches complexes (Atari, Robotique continue avec DDPG/SAC). | **Instabilité :** Très sensible aux hyperparamètres. |
| **Pas de politique explicite :** (Pour DQN) Simplifie l'architecture (un seul réseau). | **Actions Continues :** Plus complexe à adapter (nécessite DDPG/NAF). |

---
*Source: CS 285 Lecture 8 Slides, Instructor: Sergey Levine, UC Berkeley.*