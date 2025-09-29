from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=1_000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

classifiers = {
    "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),
    "Random Forest": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=0
    ),
    "Non-informative baseline": DummyClassifier(),
}

import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay

fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
ax_det.set_title("Detection Error Tradeoff (DET) curves")

ax_roc.grid(linestyle="--")
ax_det.grid(linestyle="--")

for name, clf in classifiers.items():
    (color, linestyle) = (
        ("black", "--") if name == "Non-informative baseline" else (None, None)
    )
    clf.fit(X_train, y_train)
    RocCurveDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        ax=ax_roc,
        name=name,
        curve_kwargs=dict(color=color, linestyle=linestyle),
    )
    DetCurveDisplay.from_estimator(
        clf, X_test, y_test, ax=ax_det, name=name, color=color, linestyle=linestyle
    )

plt.legend()
plt.show()