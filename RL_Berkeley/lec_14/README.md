# CS 285 : Exploration Part 2 - Inference & Prediction (Lecture 14)

Ce document résume la deuxième partie du cours sur l'**Exploration**.
Alors que la partie 1 se concentrait sur le comptage des états (Count-Based), cette partie explore des méthodes plus générales basées sur l'**erreur de prédiction** (la curiosité) et la maximisation de l'**information mutuelle** (l'acquisition de compétences).

## 🔮 1. Erreur de Prédiction & Curiosité (Curiosity-Based Exploration)

L'intuition est simple : si le modèle est surpris par une transition, c'est que l'état est nouveau ou mal compris. On utilise l'erreur de prédiction comme signal de récompense intrinsèque.

### Le Problème du "TV Blanc" (The Noisy TV Problem)
Si on utilise l'erreur de prédiction brute sur les pixels ($||I_{t+1} - \hat{I}_{t+1}||^2$) comme récompense :
* L'agent sera attiré par le bruit stochastique imprévisible (ex: la neige sur un écran de télé, le mouvement des feuilles).
* Il restera bloqué à regarder ce bruit car l'erreur de prédiction restera toujours élevée, même s'il ne peut rien y apprendre ("procrastination").

### Solution : Intrinsic Curiosity Module (ICM)
Pour éviter ce piège, on ne prédit pas les pixels bruts, mais une représentation latente $\phi(s)$ qui ne contient que ce qui est **contrôlable** par l'agent.

L'architecture ICM (Pathak et al., 2017) comprend deux sous-modules :
1.  **Inverse Model (Modèle Inverse) :** Prédire l'action $a_t$ connaissant $s_t$ et $s_{t+1}$.
    * Cela force $\phi(s)$ à ne coder que les éléments de l'environnement sur lesquels l'agent peut agir. Le bruit de fond incontrôlable est ignoré.
2.  **Forward Model (Modèle Direct) :** Prédire $\phi(s_{t+1})$ connaissant $\phi(s_t)$ et $a_t$.
    * L'erreur de prédiction dans cet espace latent sert de récompense intrinsèque :
      $$r_i(s_t, a_t) = || \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) ||^2$$

---

## 🧠 2. Maximisation de l'Information (Information Gain)

On veut explorer pour réduire notre incertitude sur la dynamique de l'environnement $\theta$.
On cherche à maximiser le **Gain d'Information** (la réduction d'entropie de notre croyance sur $\theta$) :
$$IG(z, y) = H(\theta) - H(\theta | y)$$

### Variational Information Maximization (VIME)
Calculer le gain d'information exact est impossible. VIME (Houthooft et al., 2016) utilise une borne variationnelle :
* On apprend un modèle de dynamique Bayésien (BNN) $p_\theta(s_{t+1}|s_t, a_t)$.
* Le bonus d'exploration est la divergence KL entre la croyance *a posteriori* (après avoir vu la transition) et la croyance *a priori* :
  $$r_i(s_t, a_t) \approx D_{KL}(q_{\text{new}}(\theta) || q_{\text{old}}(\theta))$$

---

## 🎯 3. Exploration par Objectifs (Goal-Conditioned RL)

Au lieu d'explorer au hasard, l'agent peut se fixer ses propres objectifs.

### Apprendre DIADYN (DIAYN - Diversity Is All You Need)
On veut apprendre un ensemble de compétences (skills) distinctes sans récompense externe.
On maximise l'information mutuelle entre les états visités $S$ et une "compétence" latente $Z$ (un entier ou un vecteur one-hot choisi au début de l'épisode).
$$I(S; Z) = H(Z) - H(Z|S)$$

Cela se traduit par deux objectifs :
1.  **Discernabilité :** En voyant l'état $s$, on doit pouvoir deviner quelle compétence $z$ l'agent exécutait (via un discriminateur $q_\phi(z|s)$).
2.  **Diversité :** Les états visités doivent être aussi variés que possible (maximiser l'entropie des états).

La récompense intrinsèque devient : $r_i(s, a) = \log q_\phi(z|s) - \log p(z)$.

### GCRL (Goal-Conditioned RL)
On entraîne une politique $\pi(a|s, g)$ capable d'atteindre n'importe quel but $g$.
* **Hindsight Experience Replay (HER) :** Même si l'agent rate son but $g$, il a forcément atteint un autre état $s_{final}$. On ré-étiquette cette transition comme une réussite pour le but $g' = s_{final}$. "Je n'ai pas réussi ce que je voulais faire, mais j'ai réussi ce que j'ai fait".

---

## ✅ Résumé des Méthodes d'Exploration

| Méthode | Principe | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- |
| **ICM (Curiosité)** | Maximiser l'erreur de prédiction sur les features contrôlables (Inverse Model). | Filtre le bruit stochastique (TV Problem). | Peut rater des infos pertinentes mais non contrôlables. |
| **VIME (Info Gain)** | Maximiser la réduction d'incertitude du modèle (KL Divergence). | Théoriquement fondé. | Lourd (nécessite BNN), un peu daté. |
| **DIAYN (Skills)** | Maximiser l'Information Mutuelle entre État et Compétence. | Apprend des comportements utiles sans aucune récompense. | Difficile de transférer ces compétences vers une tâche précise ensuite. |
| **HER (Goals)** | Apprendre de ses échecs en changeant le but a posteriori. | Extrêmement efficace pour atteindre des états précis. | Suppose qu'on peut échantillonner des buts dans l'espace d'états. |

---
*Source: CS 285 Lecture 14 Slides, Instructor: Sergey Levine, UC Berkeley.*