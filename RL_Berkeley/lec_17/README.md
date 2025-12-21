# CS 285 : Reinforcement Learning Theory Basics (Lecture 17)

Ce document résume le cours sur les **Fondements Théoriques du RL**.
Jusqu'à présent, nous avons vu des algorithmes qui fonctionnent "empiriquement" (Deep RL). Ici, on cherche à établir des bornes formelles sur la performance et la vitesse d'apprentissage, principalement dans le cadre tabulaire (états discrets finis) ou linéaire.

## 🎯 Les Questions Fondamentales

La théorie du RL cherche principalement à répondre à deux questions :

1.  **Sample Complexity (Complexité en Échantillons) :** Combien de pas de temps $N$ faut-il pour trouver une politique $\pi$ qui est $\epsilon$-proche de l'optimale ($\pi^*$) ?
    $$J(\pi^*) - J(\hat{\pi}) \le \epsilon \quad \text{avec probabilité } 1-\delta$$
2.  **Regret :** Quelle est la perte cumulée subie par l'agent pendant qu'il apprend (par rapport à un agent optimal) ?
    $$Reg(T) = \sum_{t=1}^T (J(\pi^*) - J(\pi_t))$$
    On cherche souvent un regret "sous-linéaire" (ex: $\sqrt{T}$), ce qui signifie que l'agent finit par converger vers l'optimal.

---

## 🏗️ 1. Model-Based RL (Analyse Tabulaire)

L'approche la plus simple à analyser est le Model-Based :
1.  Estimer le modèle de transition $\hat{T}(s'|s,a)$ et la récompense $\hat{r}(s,a)$ par comptage empirique.
2.  Planifier sur ce modèle estimé (ex: Value Iteration).

### Simulation Lemma (Le Lemme de Simulation)
Ce lemme fondamental relie l'erreur du modèle à l'erreur de valeur.
Si notre modèle a une erreur de prédiction $\epsilon_m$, l'erreur sur la valeur de la politique apprise est bornée par :
$$|V^\pi(s) - \hat{V}^\pi(s)| \le \frac{\gamma}{(1-\gamma)^2} \epsilon_m$$

* **Impact :** L'erreur est amplifiée quadratiquement par l'horizon effectif $\frac{1}{1-\gamma}$. Une petite erreur de modèle peut ruiner la politique à long terme.

### Exploration Optimiste (MBIE-EB / UCRL)
Pour garantir la convergence, il ne suffit pas d'apprendre un modèle moyen. Il faut être **optimiste**.
Au lieu d'utiliser le modèle moyen $\hat{T}$, on construit un ensemble de modèles plausibles (Confidence Set) et on choisit celui qui maximise la valeur.
En pratique, cela revient à ajouter un **bonus d'exploration** aux récompenses :
$$r^+(s,a) = \hat{r}(s,a) + \frac{C}{\sqrt{N(s,a)}}$$
Cela garantit (avec haute probabilité) que $Q^+(s,a) \ge Q^*(s,a)$.

---

## ⚡ 2. Model-Free RL (Q-Learning)

Peut-on avoir des garanties similaires sans apprendre de modèle ?
Oui, pour des algorithmes comme **Q-Learning avec UCB**.

### Lower Bounds (Bornes Inférieures)
On ne peut pas apprendre plus vite que la théorie de l'information ne le permet. Pour un MDP tabulaire, tout algorithme a besoin d'au moins $\Omega\left(\frac{|S||A|}{\epsilon^2 (1-\gamma)^3}\right)$ échantillons pour trouver une politique $\epsilon$-optimale.

### Upper Bounds (Bornes Supérieures)
Les algorithmes modernes (comme UCB-VI ou Q-learning optimiste) atteignent des performances proches de cette limite optimale.
* **Idée clé :** Ajouter un bonus $\frac{1}{\sqrt{N(s,a)}}$ directement dans la mise à jour de Q-Learning.

---

## 📉 3. Function Approximation & Offline RL

Quand on passe aux réseaux de neurones (Deep RL), les garanties deviennent plus floues.

### Approximation Linéaire
Si la Q-function est linéaire ($Q(s,a) = \theta^T \phi(s,a)$), on peut prouver la convergence si les données sont bien distribuées.

### Le Défi de l'Offline RL (Distribution Shift)
En Offline RL, la théorie se concentre sur le **Concentrability Coefficient** ($C$).
Il mesure le ratio de densité entre la politique que l'on veut apprendre ($\pi$) et la politique qui a généré les données ($\mu$).
$$C \approx \max_{s,a} \frac{d^\pi(s,a)}{d^\mu(s,a)}$$
* Si ce ratio est borné partout (nos données couvrent tout ce que $\pi$ pourrait visiter), on peut apprendre.
* Si ce ratio explose (il y a des états que $\pi$ visite mais que $\mu$ n'a jamais vus), l'erreur peut être arbitrairement grande. C'est la justification théorique des algorithmes conservateurs (CQL) vus au cours 15/16.

---

## ✅ Résumé des Concepts Théoriques

| Concept | Définition | Importance |
| :--- | :--- | :--- |
| **Simulation Lemma** | Relie l'erreur de modèle à l'erreur de valeur. | Montre pourquoi l'horizon long ($1/1-\gamma$) rend l'apprentissage difficile. |
| **Optimism (UCB)** | Agir comme si l'environnement était le "meilleur possible" compatible avec les données. | Indispensable pour une exploration provable (garantie). L'aléatoire ($\epsilon$-greedy) ne suffit pas. |
| **Sample Complexity** | Nombre d'échantillons nécessaires pour apprendre. | Généralement proportionnel au nombre d'états $|S|$ et d'actions $|A|$. |
| **Concentrability** | Ratio entre la distribution cible et la distribution des données. | Condition sine qua non pour la réussite de l'Offline RL. |

---
*Source: CS 285 Lecture 17 Slides, Instructor: Sergey Levine, UC Berkeley.*