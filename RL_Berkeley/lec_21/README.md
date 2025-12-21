# CS 285 : RL with Sequence Models (Lecture 21)

Ce document résume le cours sur l'utilisation des **Modèles de Séquence** en RL.
Jusqu'à présent, nous supposions souvent un MDP où l'état $s_t$ est entièrement observable. Ce cours traite des cas où l'agent ne voit qu'une observation partielle $o_t$ (POMDP) et doit mémoriser le passé. Il couvre également l'utilisation d'architectures comme les RNNs (LSTMs) et les Transformers pour résoudre ces problèmes, ainsi que l'application du RL au langage.

## 🌫️ 1. Au-delà des MDPs : POMDPs

Dans de nombreux problèmes réels (robotique, poker, dialogue), l'agent ne connaît pas l'état complet du monde.
* **Observation ($o_t$) :** Ce que l'agent perçoit (ex: image caméra).
* **État ($s_t$) :** La configuration réelle du monde.
* **Historique ($h_t$) :** La séquence des observations passées $(o_1, a_1, \dots, o_t)$.

Dans un POMDP (Partially Observed MDP), la politique doit dépendre de l'historique complet, pas juste de la dernière observation : $\pi(a_t | h_t)$.

### Solutions Architecturales
1.  **Windowing (Fenêtrage) :** Empiler les $k$ dernières images (ex: Atari DQN utilise 4 frames). [cite_start]Simple mais limité à une mémoire courte[cite: 5].
2.  **RNNs / LSTMs :** Maintenir un état caché récurrent $h_t = f(h_{t-1}, o_t)$ qui résume tout le passé. [cite_start]C'est l'approche standard pour les POMDPs complexes[cite: 5].

---

## 🔄 2. Entraîner des Politiques Récurrentes

[cite_start]L'utilisation de RNNs (Recurrent Neural Networks) en RL pose des défis techniques spécifiques, notamment pour le stockage et l'entraînement[cite: 14].

### Défis du Replay Buffer
* Pour entraîner un RNN, on a besoin de séquences temporelles, pas de transitions isolées $(s, a, r, s')$.
* **Problème :** Si on stocke des séquences entières, l'état caché initial $h_{init}$ du RNN stocké dans le buffer est "périmé" (il a été généré par une vieille version des poids).
* **Solutions :**
    1.  **Zero Start :** Toujours initialiser $h_0 = 0$ et ré-exécuter toute la séquence (coûteux).
    2.  **Burn-in :** Utiliser une partie de la séquence juste pour "chauffer" l'état caché avant de commencer à calculer les gradients.
    3.  [cite_start]**Stored State (R2D2) :** Stocker l'état caché $h_t$ dans le buffer, mais accepter qu'il soit légèrement incorrect (off-policyness de l'état caché)[cite: 16].

### Architectures Distribuées (IMPALA / R2D2)
Pour passer à l'échelle (ex: DOTA 2, StarCraft), on découple la collecte de données (Actors) de l'apprentissage (Learner).
* [cite_start]**IMPALA (V-trace) :** Corrige le décalage de politique (Lag) entre les acteurs et l'appreneur via des corrections d'Importance Sampling sophistiquées[cite: 16].

---

## 📜 3. RL comme Modélisation de Séquence (Transformers)

Plutôt que d'utiliser des RNNs avec Bellman/TD-learning, on peut utiliser des **Transformers** (comme GPT) pour modéliser la distribution des trajectoires complètes.

### Decision Transformer (DT)
[cite_start]On traite le RL comme un problème de prédiction de séquence supervisé (Autoregressive Modeling)[cite: 22].
La séquence d'entrée est :
$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots)$$
Où $\hat{R}_t$ est le **Return-to-go** (somme des récompenses futures désirées).

* **Entraînement :** Prédire l'action $a_t$ sachant le contexte passé et le retour cible $\hat{R}_t$ (Cross-entropy loss).
* **Inférence :** On donne à l'agent un retour cible élevé (ex: +1000) et il génère les actions qui mènent statistiquement à ce retour.
* **Avantage :** Pas de problèmes d'instabilité liés au bootstrapping ou aux Q-values surestimées. Stable comme du Supervised Learning.

### Meta-RL et In-Context Learning
Les modèles de séquence peuvent effectuer du "Meta-RL" implicite. [cite_start]En lisant l'historique de l'épisode courant (actions, récompenses), le Transformer "comprend" la tâche et adapte sa stratégie sans mettre à jour ses poids (In-Context Learning)[cite: 18].

---

## 🗣️ 4. RL pour le Langage (Language Models)

Le langage est le domaine par excellence des modèles de séquence. Le RL est utilisé pour finetuner les modèles de langage (LLMs) au-delà de la simple prédiction de mot suivant.

### Dialogue comme un POMDP
Une conversation est un processus séquentiel où l'état interne de l'interlocuteur est caché.
* **Action :** Un mot (Token) ou une phrase (Utterance).
* [cite_start]**Récompense :** Sentiment humain, succès de la négociation, clic, etc.[cite: 31].

### Offline RL pour le Langage
Souvent, on ne peut pas faire interagir un chatbot en live pour apprendre (risque de toxicité, lenteur). On utilise l'Offline RL sur des logs de conversations.
* **IQL (Implicit Q-Learning) pour le texte :** Apprend une Value Function sur le vocabulaire. [cite_start]Permet de filtrer les réponses toxiques ou de faible qualité tout en restant proche des données réalistes[cite: 32].
* **CHAI (Confidence-Harnessed Adversarial Imitation) :** Utilise des modèles pour distinguer le bon langage du mauvais et guider la génération.

---

## ✅ Résumé Technique

| Approche | Architecture | Gestion du passé | Avantages | Inconvénients |
| :--- | :--- | :--- | :--- | :--- |
| **Frame Stacking** | CNN / MLP | Fenêtre fixe (ex: 4 images) | Simple, compatible avec tout algo RL. | Mémoire très courte, rate les dépendances longues. |
| **Recurrent RL** | LSTM / GRU | État caché $h_t$ | Mémoire infinie (théorique), standard pour POMDP. | Difficile à entraîner (BPTT), gestion complexe du Replay Buffer. |
| **Decision Transformer** | Transformer | Attention sur toute la séquence | Très stable, gère les dépendances longues, pas de TD-error. | Ne peut pas "inventer" une stratégie meilleure que la meilleure démo (pas de stitching optimal). |
| **RLHF (Language)** | Transformer | Finetuning via Reward Model | Permet d'aligner les LLMs sur l'intention humaine. | Coûteux (collecte de préférences humaines). |

---
[cite_start]*Source: CS 285 Lecture 21 Slides, Instructor: Sergey Levine, UC Berkeley.* [cite: 1]