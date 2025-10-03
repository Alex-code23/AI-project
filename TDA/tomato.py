"""
Étapes principales :
1. Estimation de densité locale par k-NN (densité proportionnelle à 1 / moyenne des distances aux k voisins).
2. Construction du graphe k-NN (symétrisé).
3. Pour chaque point, "pointer" vers le voisin de plus grande densité dans son voisinage immédiat.
   Iterer ce pointeur pour trouver le mode local (fixpoint).
4. Regrouper points par mode local (chaque mode a une "birth density").
5. Construire une hiérarchie de fusion des modes en triant les arêtes du graphe par "saddle height"
   (par ex. max minimal density le long de l'arête). Calculer la persistence = birth - death.
6. Conserver modes avec persistence >= seuil (ou top-k), recluster en attribuant chaque point
   au mode survivant correspondant.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, Counter
import heapq
import matplotlib.pyplot as plt

# ---------------------
# Utils
# ---------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, a):
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1
        return True

# ---------------------
# ToMATo-like implementation
# ---------------------
def local_density_knn(X, k=10, method='inverse_mean_dist'):
    """
    Estimation de densité locale basée sur k-NN.
    method:
      - 'inverse_mean_dist' : density = 1 / (mean distance to k neighbours)
      - 'inverse_volume' : density ~ k / volume (approx via distance to k-th neighbour)
    Retour: dens (n,)
    """
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)  # inclut self
    dists, idx = nbrs.kneighbors(X)
    # dists[:,0] == 0 (self-distance)
    knn_dists = dists[:, 1:]  # shape (n, k)
    if method == 'inverse_mean_dist':
        mean_dist = np.mean(knn_dists, axis=1)
        # éviter division par zéro
        mean_dist[mean_dist <= 1e-12] = 1e-12
        dens = 1.0 / mean_dist
    elif method == 'inverse_volume':
        # approx density ~ k / (sphere_volume(radius_k)) ; sphere_volume ~ r^D but D unknown; use 1/r_k
        r_k = knn_dists[:, -1]
        r_k[r_k <= 1e-12] = 1e-12
        dens = 1.0 / r_k
    else:
        raise ValueError("Unknown method")
    return dens, nbrs

def build_symmetric_knn_graph(nbrs, X, k):
    """
    Retourne adjacency list du graphe k-NN symétrisé et la matrice des distances correspondantes.
    renvoie edges : list de tuples (i,j,dist_ij) avec i<j (unique)
    """
    # use kneighbors to get neighbors for each point
    dists, idx = nbrs.kneighbors(X)
    n = X.shape[0]
    edges = {}
    for i in range(n):
        for j_idx, dist in zip(idx[i,1:], dists[i,1:]):  # skip self
            a,b = min(i,j_idx), max(i,j_idx)
            # garder la plus petite distance si doublon
            if (a,b) not in edges or dist < edges[(a,b)]:
                edges[(a,b)] = dist
    # convert to list
    edges_list = [(a,b,edges[(a,b)]) for (a,b) in edges]
    return edges_list

def find_modes_by_ascent(X, dens, nbrs):
    """
    Pour chaque point, pointer vers le voisin le plus dense (dans l'ensemble de ses voisins immédiats).
    Itérer jusqu'à ce que point soit un mode (aucun voisin immédiat n'a densité plus grande).
    Retour:
      - modes_index : list des indices qui sont des modes (points fixes)
      - labels_mode : array shape (n,) indiquant l'indice du mode final pour chaque point
      - mode_members : dict mode_idx -> list(points)
      - mode_birth_density : dict mode_idx -> density
    """
    n = X.shape[0]
    # récupérer voisins immédiats
    dists, idx = nbrs.kneighbors(X)
    neighbors = idx[:,1:]  # shape (n, k)
    # pointer vers le voisin à plus grande densité parmi voisins immédiats (sinon se pointer soi-même)
    pointer = np.arange(n)
    for i in range(n):
        neigh = neighbors[i]
        # s'il y a un voisin avec dens > dens[i], prendre le plus dense
        neigh_dens = dens[neigh]
        max_idx = np.argmax(neigh_dens)
        if neigh_dens[max_idx] > dens[i]:
            pointer[i] = neigh[max_idx]
        else:
            pointer[i] = i

    # itérer pour trouver le fixpoint (chemin discret vers mode)
    # appliquer "pointer jumping" / chemin jusqu'à point fixe
    def find_root(i):
        path = []
        while pointer[i] != i:
            path.append(i)
            i = pointer[i]
            # cycle prevention (should not happen if pointers strictly to higher density)
            if len(path) > n:
                break
        return i

    roots = np.array([find_root(i) for i in range(n)])
    unique_modes, inv = np.unique(roots, return_inverse=True)
    labels_mode = inv  # label per point as index in unique_modes
    mode_members = defaultdict(list)
    for pt_idx, mode_label in enumerate(labels_mode):
        mode_members[unique_modes[mode_label]].append(pt_idx)
    mode_birth_density = {m: dens[m] for m in unique_modes}
    return list(unique_modes), labels_mode, mode_members, mode_birth_density

def compute_persistence_and_merge(modes, mode_birth_density, edges):
    """
    Construire fusion hiérarchique entre modes en utilisant les arêtes du graphe.
    edges: list of (i,j,dist) where i<j (indices de points)
    Principe :
      - considérer uniquement arêtes reliant deux modes différentes (ou connectant régions de modes).
      - pour une arête (u,v), define saddle_height = max(min(dens[u], dens[v]), min_birth_of_modes_connected)
      - tri par saddle_height décroissant (fusionner d'abord les arêtes qui gardent haute densité)
    Simpler practical approach:
      - pour chaque arête (i,j) reliant points appartenant à modes a,b (a != b),
        define merge_level = max( dens[i], dens[j] )? Or min( max density along path )
      - we will use merge_level = max(min(dens[i], dens[j])) -> approximate saddle
    Retour:
      - for each mode m: persistence = birth - death (death = density at which it merged into a mode with higher birth)
      - merges : list of (mode_a, mode_b, merge_level)
    """
    # Gather edges between modes:
    # edges are (i,j,dist) ; need to map point->mode
    # We'll assume we have point_to_mode mapping global; but here caller will provide merged edges per-mode.
    # This function will be integrated in the main pipeline where mapping is available.
    raise NotImplementedError("Use compute_persistence_with_point_modes (below)")

def compute_persistence_with_point_modes(point_to_mode, mode_birth_density, edges):
    """
    edges : list (i,j,dist)
    point_to_mode : array shape (n,) giving mode index (point index of mode) for each point
    mode_birth_density : dict mode_idx -> birth density

    Retourne:
      - mode_death_density : dict mode_idx -> death density (>=0). If never merged -> death = 0
      - merges : list of tuples (mode_a, mode_b, merge_density)
    Algorithme (pratique et heuristique) :
      - Pour chaque arête (i,j), si mode(i) != mode(j), on calcule merge_density = max( min(dens[i], dens[j]) )
        (on peut aussi utiliser min(dens[i], dens[j])).
      - Trier arêtes par merge_density décroissant (fusionner d'abord celles avec plus haute selle)
      - Appliquer union-find sur modes ; quand on fusionne une composante A (ayant un leader mode with highest birth)
        dans une composante B dont leader a higher birth, on enregistre la death density de la mode qui perd = merge_density.
      - Au final persistence = birth - death. Si never merged death = 0 (ou min density).
    """
    n_edges = len(edges)
    # Precompute densities at points
    # We need dens[i] for every point; we can fetch them via mapping from mode_birth_density? No: mode_birth_density only for modes.
    # Therefore caller must also provide dens_points array (we will modify signature). To keep API consistent, change to:
    raise NotImplementedError("Use compute_persistence_full (below)")

def compute_persistence_full(point_to_mode, dens_points, mode_birth_density, edges):
    """
    Implementation working version.

    - point_to_mode: array shape (n_points,) giving the representative mode index (point index) for each point
    - dens_points: array shape (n_points,) density per point
    - mode_birth_density: dict mode_idx -> birth density (density at the mode point)
    - edges: list of (i,j,dist) for point graph (i<j)

    Returns:
      - mode_persistence: dict mode_idx -> persistence (birth - death)
      - mode_death: dict mode_idx -> death_density (0 if never merged)
      - merges: list of (mode_a, mode_b, merge_density) in the order they occur (descending merge_density)
    """
    # Build list of candidate merges between different modes, with merge_density = max(min(dens[i],dens[j]))
    merges = []
    for (i,j,dist) in edges:
        mi = point_to_mode[i]
        mj = point_to_mode[j]
        if mi == mj: 
            continue
        # merge level approximation: the highest density up to which the regions are connected is min(max of densities along edge endpoints)
        # use saddle = max(min(dens[i], dens[j])) -> here it's simply min(dens[i], dens[j])
        saddle = min(dens_points[i], dens_points[j])
        merges.append((mi, mj, saddle))
    # Sort merges by saddle desc (we fuse highest saddle first)
    merges.sort(key=lambda x: x[2], reverse=True)

    # Union-find over modes
    modes = list(mode_birth_density.keys())
    mode_to_idx = {m:i for i,m in enumerate(modes)}
    uf = UnionFind(len(modes))
    # track leader with highest birth in each uf-component, to know which mode "dominates"
    comp_leader = {i: modes[i] for i in range(len(modes))}  # comp index -> mode idx (point)
    # death densities init
    mode_death = {m: 0.0 for m in modes}
    # Process merges
    processed = []
    for a, b, saddle in merges:
        ia = mode_to_idx[a]
        ib = mode_to_idx[b]
        root_a = uf.find(ia)
        root_b = uf.find(ib)
        if root_a == root_b:
            continue
        # leaders (modes with largest birth density) of components
        leader_a = comp_leader[root_a]
        leader_b = comp_leader[root_b]
        birth_a = mode_birth_density[leader_a]
        birth_b = mode_birth_density[leader_b]
        # determine which leader has higher birth density
        if birth_a > birth_b:
            # b merges into a at height saddle -> b dies at saddle
            mode_death[leader_b] = saddle if mode_death[leader_b] == 0 else min(mode_death[leader_b], saddle)
            uf.union(root_a, root_b)
            newroot = uf.find(root_a)
            comp_leader[newroot] = leader_a  # leader stays the one with larger birth
        else:
            mode_death[leader_a] = saddle if mode_death[leader_a] == 0 else min(mode_death[leader_a], saddle)
            uf.union(root_a, root_b)
            newroot = uf.find(root_a)
            comp_leader[newroot] = leader_b
        processed.append((a,b,saddle))
    # compute persistence
    mode_persistence = {}
    for m in modes:
        birth = mode_birth_density[m]
        death = mode_death[m]
        # if never died, death = 0 => persistence = birth - 0 = birth
        mode_persistence[m] = birth - death
    return mode_persistence, mode_death, processed

def tomato_clustering(X, k=10, density_method='inverse_mean_dist', persistence_threshold=None, top_k_modes=None, verbose=True):
    """
    Pipeline ToMATo-like pour clustering.

    Paramètres :
      X: (n_samples, n_features)
      k: k pour k-NN (voisinage local)
      density_method: 'inverse_mean_dist' ou 'inverse_volume'
      persistence_threshold: si fourni, garder modes avec persistence >= threshold
      top_k_modes: si fourni (int), garder uniquement les top_k_modes modes par persistence
      (si les deux fournis, top_k_modes priorisé)
    Retourne :
      - labels_final: array shape (n_samples,) cluster label par point (-1 pour bruit si applicable)
      - info: dict avec info utiles (modes, persistence, graphs, etc.)
    """
    n = X.shape[0]
    dens_points, nbrs = local_density_knn(X, k=k, method=density_method)
    if verbose:
        print("[ToMATo] Estimation densité calculée. points:", n)

    # build symmetric knn graph edges
    edges = build_symmetric_knn_graph(nbrs, X, k)
    if verbose:
        print(f"[ToMATo] Graphe k-NN construit: {len(edges)} arêtes distinctes.")

    # find modes by discrete ascent
    modes, labels_mode_idx, mode_members, mode_birth_density = find_modes_by_ascent(X, dens_points, nbrs)
    if verbose:
        print(f"[ToMATo] Modes initiaux trouvés: {len(modes)}")

    # map points -> mode (use representative as the mode's point index)
    # labels_mode_idx currently gives index into unique_modes; reconstruct mapping to mode point index
    unique_modes = np.array(modes)
    point_to_mode = unique_modes[labels_mode_idx]  # each point mapped to mode (point index)

    # compute persistence via merges across edges
    mode_persistence, mode_death, merges = compute_persistence_full(point_to_mode, dens_points, mode_birth_density, edges)
    # sort modes by persistence desc
    sorted_modes = sorted(mode_persistence.items(), key=lambda x: x[1], reverse=True)
    if verbose:
        print("[ToMATo] Persistence (mode -> persistence) sample:", sorted_modes[:10])

    # decide which modes to keep
    if top_k_modes is not None:
        keep_modes = set([m for m,_ in sorted_modes[:top_k_modes]])
    elif persistence_threshold is not None:
        keep_modes = set([m for m,p in mode_persistence.items() if p >= persistence_threshold])
    else:
        # default: keep modes with persistence >= median persistence (heuristique), or keep all if small count
        pers_vals = np.array(list(mode_persistence.values()))
        if len(pers_vals) == 0:
            keep_modes = set()
        else:
            thr = np.median(pers_vals)
            keep_modes = set([m for m,p in mode_persistence.items() if p >= thr])
    if verbose:
        print(f"[ToMATo] Modes gardés: {len(keep_modes)} / {len(modes)}")

    # if no mode kept, fallback: keep the most persistent one
    if len(keep_modes) == 0 and len(sorted_modes) > 0:
        keep_modes = {sorted_modes[0][0]}

    # assign each surviving mode an index label 0..M-1
    keep_list = sorted(list(keep_modes))
    mode_to_cluster = {m:i for i,m in enumerate(keep_list)}

    # For points that map to a mode that was discarded, find nearest surviving mode via graph-based propagation:
    # We'll relabel by walking merges: for each point, climb pointers until you reach a surviving mode; if not, assign nearest via neighbors.
    labels_final = -np.ones(n, dtype=int)
    for i in range(n):
        mode = point_to_mode[i]
        # follow merges: climb by always moving to neighbor-mode with higher birth until reach surviving
        visited = set()
        cur_mode = mode
        while cur_mode not in keep_modes:
            if cur_mode in visited:
                break
            visited.add(cur_mode)
            # find merges where cur_mode was merged into another with higher birth
            # find in merges list a tuple where a==cur_mode or b==cur_mode and the other has higher birth
            next_mode = None
            for a,b,saddle in merges:
                if a == cur_mode and b != cur_mode:
                    # candidate: which one has higher birth? choose the other if it dominates
                    if mode_birth_density[b] >= mode_birth_density[a]:
                        next_mode = b; break
                if b == cur_mode and a != cur_mode:
                    if mode_birth_density[a] >= mode_birth_density[b]:
                        next_mode = a; break
            if next_mode is None:
                break
            cur_mode = next_mode
        if cur_mode in mode_to_cluster:
            labels_final[i] = mode_to_cluster[cur_mode]
    # for any remaining unlabeled, assign to nearest labeled neighbor (graph based)
    unlabeled = np.where(labels_final == -1)[0]
    if len(unlabeled) > 0:
        # build neighbor list from nbrs
        dists, idx = nbrs.kneighbors(X)
        for i in unlabeled:
            neigh = idx[i,1:]
            found = False
            for nb in neigh:
                if labels_final[nb] != -1:
                    labels_final[i] = labels_final[nb]
                    found = True
                    break
            if not found:
                # fallback: nearest surviving mode in Euclidean distance
                mode_points = np.array(list(keep_list))
                if len(mode_points) > 0:
                    d_to_modes = np.linalg.norm(X[i:i+1] - X[mode_points], axis=1)
                    labels_final[i] = mode_to_cluster[mode_points[np.argmin(d_to_modes)]]
                else:
                    labels_final[i] = -1

    info = {
        'dens_points': dens_points,
        'nbrs': nbrs,
        'edges': edges,
        'modes': modes,
        'mode_birth_density': mode_birth_density,
        'mode_persistence': mode_persistence,
        'mode_death': mode_death,
        'merges': merges,
        'keep_modes': keep_modes,
        'mode_to_cluster': mode_to_cluster
    }
    return labels_final, info

# ---------------------
# Exemple d'utilisation
# ---------------------
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=600, centers=10, cluster_std=2.0, random_state=0)
    labels, info = tomato_clustering(X, k=15, persistence_threshold=0.05, verbose=True)

    # affichage
    plt.figure(figsize=(8,5))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=12)
    # afficher modes retenus
    mode_pts = np.array(list(info['keep_modes']))
    if mode_pts.size > 0:
        plt.scatter(X[mode_pts,0], X[mode_pts,1], c='k', marker='x', s=100, label='modes kept')
    plt.title("ToMATo-like clustering (demo)")
    plt.legend()
    plt.show()
