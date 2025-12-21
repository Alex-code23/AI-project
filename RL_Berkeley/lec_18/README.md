# CS 285 : Variational Inference & Generative Models (Lecture 18)

Ce document résume le cours sur l'**Inférence Variationnelle (VI)** et les **Modèles Génératifs**.
L'objectif est d'apprendre des modèles probabilistes complexes $p_\theta(x)$ capables de générer des données (images, trajectoires) ou de représenter des distributions multimodales, là où un simple réseau de neurones (MSE) échouerait à capturer la diversité.

## 🎯 Le Problème : Modèles à Variables Latentes

On suppose que nos données observables $x$ (ex: une image) sont générées par des variables cachées non-observées $z$ (ex: "un chat", "position", "couleur").
Le modèle probabiliste joint est :
$$p_\theta(x, z) = p_\theta(x|z) p(z)$$
* $p(z)$ : Le prior sur les variables latentes (souvent $\mathcal{N}(0, I)$).
* $p_\theta(x|z)$ : La vraisemblance (le "Décodeur"), souvent un réseau de neurones.

Pour entraîner ce modèle (trouver $\theta$), on veut maximiser la "log-vraisemblance marginale" des données :
$$\theta^* = \arg\max_\theta \sum_i \log p_\theta(x_i) = \arg\max_\theta \sum_i \log \int p_\theta(x_i|z) p(z) dz$$

**Problème :** L'intégrale $\int p_\theta(x|z) p(z) dz$ est **intractable** (impossible à calculer analytiquement) pour des réseaux de neurones complexes. On ne peut donc pas optimiser directement cette fonction.

---

## 🛠️ L'Inférence Variationnelle (Variational Inference)

Puisqu'on ne peut pas calculer $p(x)$, ni la vraie distribution *a posteriori* $p(z|x)$ (intractable aussi), on va l'**approximer**.
On introduit une distribution variationnelle $q_\phi(z|x)$ (l'"Encodeur") paramétrée par $\phi$, et on essaie de la rendre aussi proche que possible du vrai posterior $p(z|x)$.

### La Dérivation de l'ELBO
On utilise la divergence KL pour mesurer la distance entre notre approximation et la réalité :
$$D_{KL}(q_\phi(z|x) || p_\theta(z|x)) = E_{z \sim q} [\log q_\phi(z|x) - \log p_\theta(z|x)]$$

En réarrangeant les termes, on obtient l'identité fondamentale :
$$\log p_\theta(x) = D_{KL}(q_\phi(z|x) || p_\theta(z|x)) + \mathcal{L}(\theta, \phi)$$

Où $\mathcal{L}$ est la **Borne Inférieure de l'Évidence (ELBO - Evidence Lower Bound)** :
$$\mathcal{L}(\theta, \phi) = E_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$

* Puisque $D_{KL} \ge 0$, alors $\log p_\theta(x) \ge \mathcal{L}(\theta, \phi)$.
* Maximiser l'ELBO revient à maximiser la vraisemblance des données **ET** minimiser l'écart entre notre approximation $q$ et le vrai posterior $p(z|x)$.

---

## 🤖 Amortized Variational Inference & VAE

Au lieu d'optimiser une distribution $q_i$ différente pour chaque point de donnée $x_i$ (ce qui serait trop lent), on apprend un réseau de neurones **d'inférence** $q_\phi(z|x)$ qui prend $x$ en entrée et prédit les paramètres de la distribution de $z$ (ex: moyenne $\mu$ et variance $\sigma^2$). C'est l'inférence "amortie".

### Variational Auto-Encoder (VAE)
Le VAE est l'instanciation directe de ce principe avec des réseaux de neurones.

1.  **Encodeur ($q_\phi(z|x)$)** : Prédit $\mu_\phi(x)$ et $\sigma_\phi(x)$.
2.  **Décodeur ($p_\theta(x|z)$)** : Prend un $z$ échantillonné et reconstruit $x$.

**Fonction de Perte (Loss) = -ELBO :**
$$J(\theta, \phi) \approx \underbrace{- \log p_\theta(x|z)}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}(q_\phi(z|x) || p(z))}_{\text{Regularization Loss}}$$

* **Reconstruction :** Le modèle doit bien compresser/décompresser l'image (MSE ou Cross-Entropy).
* **Régularisation :** L'espace latent doit ressembler au prior (Gaussienne standard). Cela force l'espace à être lisse et continu (bon pour la génération).

### Le "Reparameterization Trick"
Pour entraîner tout cela par descente de gradient, il faut pouvoir backpropager à travers l'échantillonnage stochastique $z \sim q_\phi(z|x)$.
Si on échantillonne directement, le gradient est bloqué.
**Astuce :** On réécrit le bruit aléatoire comme une entrée externe.
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \text{où } \epsilon \sim \mathcal{N}(0, I)$$
Maintenant, $z$ est une fonction déterministe et différentiable de $\phi$ et d'une constante $\epsilon$. Le gradient peut passer !

---

## 📈 Liens avec le Reinforcement Learning

Pourquoi ce cours de VI en plein milieu du RL ?

1.  **Model-Based RL (Images) :** Comme vu au cours 12, les VAEs permettent d'apprendre des espaces d'états latents compacts pour planifier à partir d'images.
2.  **Exploration (Cours 14) :** VIME utilise l'inférence variationnelle pour estimer le gain d'information sur la dynamique.
3.  **Politiques Stochastiques Optimales :** Le "Soft Optimality" framework (Soft Q-Learning, SAC) peut être vu comme une forme d'inférence variationnelle où on infère la trajectoire optimale.
4.  **Offline RL (Cours 15) :** Les VAEs sont utilisés (ex: BCQ) pour modéliser la distribution des actions du dataset ($\pi_\beta$) et générer des actions valides.

---

## ✅ Résumé Technique

| Concept | Formule / Définition | Rôle |
| :--- | :--- | :--- |
| **Latent Variable Model** | $p(x) = \int p(x|z)p(z)dz$ | Capturer la structure cachée et multimodale des données. |
| **ELBO** | $E_q[\log p(x|z)] - D_{KL}(q||p)$ | Borne inférieure tractable de $\log p(x)$ qu'on maximise. |
| **Inference Network** | $q_\phi(z|x)$ | Réseau (Encoder) qui approxime le vrai posterior $p(z|x)$. |
| **Reparameterization** | $z = \mu + \sigma \epsilon$ | Permet la backpropagation à travers un nœud stochastique. |

---
[cite_start]*Source: CS 285 Lecture 18 Slides, Instructor: Sergey Levine, UC Berkeley.* [cite: 1, 2, 4]