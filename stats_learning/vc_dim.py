# vc_explorer.py
# Outil d'exploration empirique de la VC-dimension pour quelques classes simples.
# Classes supportées:
#   - "threshold"      : demi-droites sur R (1D)
#   - "interval"       : intervalle sur R (1D)
#   - "rectangles_2d"  : rectangles axis-aligned (2D)
#   - "perceptron_2d"  : séparateurs linéaires en R^2
#
# Lancement conseillé dans un notebook (Jupyter) pour voir les graphiques inline.

import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd

rng = np.random.default_rng(12345)

# ---------- Witness / solveurs pour chaque classe ----------
def witness_threshold(points, labels):
    n = points.size
    order = np.argsort(points)
    lab_sorted = labels[order]
    for k in range(n+1):
        pattern = np.concatenate(( -np.ones(k), np.ones(n-k) ))
        if np.array_equal(lab_sorted, pattern):
            if k==0:
                thresh = points[order[0]] - 1e-6
            elif k==n:
                thresh = points[order[-1]] + 1e-6
            else:
                thresh = 0.5*(points[order[k-1]] + points[order[k]])
            return {"type":"threshold","threshold":thresh,"sign":+1,"order":order}
        if np.array_equal(lab_sorted, -pattern):
            if k==0:
                thresh = points[order[0]] - 1e-6
            elif k==n:
                thresh = points[order[-1]] + 1e-6
            else:
                thresh = 0.5*(points[order[k-1]] + points[order[k]])
            return {"type":"threshold","threshold":thresh,"sign":-1,"order":order}
    return None

def witness_interval(points, labels):
    n = points.size
    order = np.argsort(points)
    lab_sorted = labels[order]
    for i in range(n):
        for j in range(i, n):
            pattern = -np.ones(n)
            pattern[i:j+1] = 1
            if np.array_equal(lab_sorted, pattern):
                a = points[order[i]] - 1e-6
                b = points[order[j]] + 1e-6
                return {"type":"interval","a":a,"b":b,"sign":+1,"order":order}
            if np.array_equal(lab_sorted, -pattern):
                a = points[order[i]] - 1e-6
                b = points[order[j]] + 1e-6
                return {"type":"interval","a":a,"b":b,"sign":-1,"order":order}
    # trivial all +1 / all -1 cases
    if np.all(lab_sorted == 1):
        return {"type":"interval","a":points.min()-1e-6,"b":points.max()+1e-6,"sign":+1,"order":order}
    if np.all(lab_sorted == -1):
        return {"type":"interval","a":points.min()-1e-6,"b":points.min()-1e-7,"sign":+1,"order":order}
    return None

def witness_rectangles_2d(points, labels):
    pos = points[labels==1]
    neg = points[labels==-1]
    if pos.shape[0] == 0:
        xmin, xmax = points[:,0].min()-1.0, points[:,0].min()-0.5
        ymin, ymax = points[:,1].min()-1.0, points[:,1].min()-0.5
        return {"type":"rect","xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"sign":+1}
    xmin, xmax = pos[:,0].min(), pos[:,0].max()
    ymin, ymax = pos[:,1].min(), pos[:,1].max()
    eps = 1e-6
    xmin -= eps; xmax += eps; ymin -= eps; ymax += eps
    if neg.shape[0] == 0 or np.all((neg[:,0] < xmin) | (neg[:,0] > xmax) | (neg[:,1] < ymin) | (neg[:,1] > ymax)):
        return {"type":"rect","xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"sign":+1}
    # try complement (inside -> -1)
    pos2 = points[labels==-1]
    neg2 = points[labels==1]
    if pos2.shape[0] == 0:
        xmin, xmax = points[:,0].min()-1.0, points[:,0].min()-0.5
        ymin, ymax = points[:,1].min()-1.0, points[:,1].min()-0.5
        return {"type":"rect","xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"sign":-1}
    xmin2, xmax2 = pos2[:,0].min(), pos2[:,0].max()
    ymin2, ymax2 = pos2[:,1].min(), pos2[:,1].max()
    xmin2 -= 1e-6; xmax2 += 1e-6; ymin2 -= 1e-6; ymax2 += 1e-6
    if neg2.shape[0] == 0 or np.all((neg2[:,0] < xmin2) | (neg2[:,0] > xmax2) | (neg2[:,1] < ymin2) | (neg2[:,1] > ymax2)):
        return {"type":"rect","xmin":xmin2,"xmax":xmax2,"ymin":ymin2,"ymax":ymax2,"sign":-1}
    return None

def perceptron_find_separator(points, labels, max_iters=2000):
    n, d = points.shape
    X = np.hstack([points, np.ones((n,1))])
    y = labels.copy()
    w = np.zeros(d+1)
    for it in range(max_iters):
        errors = 0
        for i in range(n):
            if y[i] * (w.dot(X[i])) <= 0:
                w += y[i] * X[i]
                errors += 1
        if errors == 0:
            return {"type":"line","w":w[:-1],"b":w[-1]}
    return None

def witness_perceptron_2d(points, labels):
    sep = perceptron_find_separator(points, labels, max_iters=2000)
    if sep is not None:
        return sep
    for _ in range(5):
        sep = perceptron_find_separator(points, labels, max_iters=2000)
        if sep is not None:
            return sep
    return None

# ---------- test de shattering (énumère tout) ----------
def is_shattered(points, class_name):
    n = points.shape[0]
    for bits in itertools.product([1,-1], repeat=n):
        labels = np.array(bits)
        w = None
        if class_name == "threshold":
            w = witness_threshold(points.flatten(), labels)
        elif class_name == "interval":
            w = witness_interval(points.flatten(), labels)
        elif class_name == "rectangles_2d":
            w = witness_rectangles_2d(points, labels)
        elif class_name == "perceptron_2d":
            w = witness_perceptron_2d(points, labels)
        else:
            raise ValueError("Unknown class")
        if w is None:
            return False
    return True

def witness_for_some_labeling(points, class_name):
    n = points.shape[0]
    for bits in itertools.product([1,-1], repeat=n):
        labels = np.array(bits)
        if class_name == "threshold":
            w = witness_threshold(points.flatten(), labels)
        elif class_name == "interval":
            w = witness_interval(points.flatten(), labels)
        elif class_name == "rectangles_2d":
            w = witness_rectangles_2d(points, labels)
        elif class_name == "perceptron_2d":
            w = witness_perceptron_2d(points, labels)
        else:
            w = None
        if w is not None:
            return labels, w
    return None, None

# ---------- recherche empirique de VC ----------
def estimate_vc_dimension(class_name, max_n=7, trials_per_n=300, verbose=True):
    shatterable_ns = []
    shattered_examples = {}
    for n in range(1, max_n+1):
        if verbose: print(f"Searching n={n} ...")
        found = False
        for t in range(trials_per_n):
            if class_name in ("threshold", "interval"):
                pts = rng.random(n)
            else:
                pts = rng.random((n,2))
            if is_shattered(pts, class_name):
                found = True
                labels, witness = witness_for_some_labeling(pts, class_name)
                shattered_examples[n] = (pts, labels, witness)
                if verbose: print(f"  Found shattered set for n={n} on trial {t+1}")
                break
        if found:
            shatterable_ns.append(n)
        else:
            if verbose: print(f"  No shattered set found after {trials_per_n} trials for n={n}")
    vc_est = max(shatterable_ns) if shatterable_ns else 0
    return vc_est, shattered_examples

# ---------- fonctions de traçage (une figure par appel) ----------
def plot_witness(points, labels, witness, class_name, title_suffix=""):
    if class_name == "threshold":
        order = witness["order"]
        plt.figure(figsize=(6,2))
        plt.scatter(points, np.zeros_like(points), s=50)
        for i,x in enumerate(points):
            plt.text(x, 0.02, str(int(labels[i])), ha="center", va="bottom")
        plt.axvline(witness["threshold"], linestyle='--')
        plt.title(f"Threshold {title_suffix}")
        plt.yticks([])
        plt.tight_layout()
        plt.show()
    elif class_name == "interval":
        plt.figure(figsize=(6,2))
        plt.scatter(points, np.zeros_like(points), s=50)
        for i,x in enumerate(points):
            plt.text(x, 0.02, str(int(labels[i])), ha="center", va="bottom")
        plt.hlines(0, witness["a"], witness["b"])
        plt.title(f"Interval {title_suffix}")
        plt.yticks([])
        plt.tight_layout()
        plt.show()
    elif class_name == "rectangles_2d":
        plt.figure(figsize=(4,4))
        pos = points[labels==1]
        neg = points[labels==-1]
        if pos.size>0: plt.scatter(pos[:,0], pos[:,1], marker='o')
        if neg.size>0: plt.scatter(neg[:,0], neg[:,1], marker='x')
        xmin, xmax = witness["xmin"], witness["xmax"]
        ymin, ymax = witness["ymin"], witness["ymax"]
        plt.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin], linestyle='--')
        plt.title(f"Rectangle {title_suffix}")
        plt.tight_layout()
        plt.show()
    elif class_name == "perceptron_2d":
        plt.figure(figsize=(4,4))
        pos = points[labels==1]
        neg = points[labels==-1]
        if pos.size>0: plt.scatter(pos[:,0], pos[:,1], marker='o')
        if neg.size>0: plt.scatter(neg[:,0], neg[:,1], marker='x')
        w = witness["w"]; b = witness["b"]
        xs = np.array([points[:,0].min()-0.1, points[:,0].max()+0.1])
        if abs(w[1])>1e-9:
            ys = -(w[0]*xs + b)/w[1]
            plt.plot(xs, ys, linestyle='--')
        else:
            plt.axvline(-b/w[0], linestyle='--')
        plt.title(f"Perceptron {title_suffix}")
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("Unknown class")

# ---------- exemple de lancement ----------
if __name__ == "__main__":
    classes = ["threshold","interval","rectangles_2d","perceptron_2d"]
    results = []
    examples = {}
    for cls in classes:
        vc, ex = estimate_vc_dimension(cls, max_n=4, trials_per_n=100, verbose=True)
        results.append({"class":cls, "vc_estimate":vc})
        if ex:
            largest_n = max(ex.keys())
            examples[cls] = (largest_n, ex[largest_n])
    print("\nSummary:")
    print(pd.DataFrame(results))
    # plot one witness per class (si trouvé)
    for cls,(n,(pts,labels,wit)) in examples.items():
        plot_witness(pts, labels, wit, cls, title_suffix=f"(class={cls}, n={n})")
