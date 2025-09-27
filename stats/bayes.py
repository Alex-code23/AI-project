"""
Comparaison simple entre Gaussian Naive Bayes et d'autres classifieurs
- Dataset : breast_cancer de scikit-learn (problème de classification binaire)
- Modèles testés : GaussianNB, LogisticRegression, RandomForestClassifier, SVC, DummyClassifier
- Métrique : accuracy (validation croisée stratifiée 5 folds)

Exécution
> python compare_bayes_models.py

Prérequis
pip install scikit-learn matplotlib

"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt


def main():
    # Charger les données
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Définir les classifieurs 
    base_classifiers = {
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='rbf', probability=False, random_state=42),
        'Dummy (most_frequent)': DummyClassifier(strategy='most_frequent', random_state=42),
    }

    # On crée des pipelines pour appliquer un StandardScaler (utile pour LR et SVC)
    pipelines = {}
    for name, clf in base_classifiers.items():
        pipelines[name] = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    # Validation croiseée stratifiée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        results.append((name, scores.mean(), scores.std(), scores))

    # Trier par performance moyenne décroissante
    results.sort(key=lambda r: r[1], reverse=True)

    # Afficher les résultats
    print("\nAccuracy (5-fold CV) :\n")
    for name, mean_acc, std_acc, scores in results:
        print(f"{name:20s} -> mean = {mean_acc:.4f}, std = {std_acc:.4f}, folds = {np.round(scores,4)}")

    # Graphe simple
    names = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]

    plt.figure(figsize=(9,5))
    x = np.arange(len(names))
    plt.bar(x, means)
    # ajouter barres d'erreur (std)
    plt.errorbar(x, means, yerr=stds, fmt='none', capsize=5)
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylim(0.0, 1.05)
    plt.ylabel('Accuracy (moyenne CV)')
    plt.title('Comparaison de classifieurs - accuracy (5-fold CV)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
