from typing import Dict, Generic, Iterable, List, Optional, Sequence, TypeVar, DefaultDict
from collections import defaultdict

T = TypeVar("T")

class UnionFind(Generic[T]):
    """
    Union-Find (Disjoint Set Union) with path compression and union by size.
    Can be initialized with `n` (0..n-1) or with an iterable of elements.
    """
    def __init__(self, elements: Optional[Iterable[T]] = None, n: Optional[int] = None):
        self.parent: Dict[T, T] = {}
        self.size: Dict[T, int] = {}
        if elements is not None:
            for e in elements:
                self._make_set(e)
        elif n is not None:
            for i in range(n):
                self._make_set(i)

    def _make_set(self, x: T) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def make_set(self, x: T) -> None:
        """Public wrapper to guarantee element exists."""
        self._make_set(x)

    def find(self, x: T) -> T:
        """Find with path compression."""
        if x not in self.parent:
            raise KeyError(f"Element {x!r} not found. Call make_set({x}) first.")
        root = x        # begin at the bottom and climb
        while self.parent[root] != root:
            root = self.parent[root]
        # compress path
        while x != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: T, b: T) -> T:
        """
        Union by size. If elements are missing, they are created.
        Returns the representative after union.
        """
        if a not in self.parent:
            self._make_set(a)
        if b not in self.parent:
            self._make_set(b)

        r_a = self.find(a)
        r_b = self.find(b)
        if r_a == r_b:
            return r_a

        # attach smaller tree to larger
        if self.size[r_a] < self.size[r_b]:
            r_a, r_b = r_b, r_a
        # now r_a is the larger root
        self.parent[r_b] = r_a
        self.size[r_a] += self.size[r_b]
        del self.size[r_b]
        return r_a

    def connected(self, a: T, b: T) -> bool:
        if a not in self.parent or b not in self.parent:
            return False
        return self.find(a) == self.find(b)

    def component_size(self, x: T) -> int:
        r = self.find(x)
        return self.size[r]

    def components(self) -> Dict[T, List[T]]:
        """Return dict rep -> list of members."""
        groups: Dict[T, List[T]] = {}
        for x in list(self.parent.keys()):
            r = self.find(x)
            groups.setdefault(r, []).append(x)
        return groups

    def __len__(self) -> int:
        return len(self.parent)

    def __repr__(self) -> str:
        return f"UnionFind({len(self)} elements, {len(self.components())} components)"


def graph_based_hill_climbing(G: Sequence[Sequence[int]] or Dict[int, Sequence[int]], # type: ignore
                              f_hat: Sequence[float]):
    """
    Implémentation du graph-based hill climbing en intégrant UnionFind.

    Entrées
    - G : graphe de voisinage (liste d'adjacence ou dict). Sommets 0..n-1.
    - f_hat : vecteur de taille n avec les valeurs d'estimation de densité.

    Sorties
    - clusters_by_peak : dict peak_vertex -> list[members]
    - g : list[int] (g(i) = index du voisin max f parmi N, ou -1 si pic)
    - r_map : dict rep -> peak_vertex (r(e) dans le pseudo-code)
    """
    n = len(f_hat)
    # normaliser G
    if isinstance(G, dict):
        adj = [list(G.get(i, [])) for i in range(n)]
    else:
        adj = [list(nei) for nei in G]

    # tri décroissant selon f_hat
    idx_sorted = sorted(range(n), key=lambda i: -f_hat[i])
    pos = [0] * n
    for position, v in enumerate(idx_sorted):
        pos[v] = position

    # initialiser union-find (crée tous les éléments 0..n-1 pour simplicité)
    uf = UnionFind(n=n)

    g = [-1] * n
    r_map: Dict[int, int] = {}  # representative -> peak vertex

    for t in range(n):
        i = idx_sorted[t]
        # voisins traités (position plus petite dans l'ordre trié)
        N = [j for j in adj[i] if pos[j] < pos[i]]

        if len(N) == 0:
            # i est un pic : crée/assure l'entrée et associe le pic
            uf.make_set(i)          # déjà créé si uf initialisé avec n, mais safe
            rep = uf.find(i)
            r_map[rep] = i         # r(e) <- i
        else:
            # i n'est pas un pic : gradient approx = voisin traité avec f_hat maximal
            best_j = max(N, key=lambda j: f_hat[j])
            g[i] = best_j
            rep = uf.find(best_j)
            # attacher i à l'entrée rep ; union(rep, i) pour garder rep comme racine si possible
            uf.union(rep, i)

    # récupérer les composantes et regrouper par pic (peak)
    components = uf.components()  # rep -> members
    clusters_by_peak: Dict[int, List[int]] = {}
    for rep, members in components.items():
        # retrouver le pic associé à cette composante
        peak = r_map.get(rep)
        if peak is None:
            # défense : si jamais r_map manque, choisir le sommet du composant avec f_hat max
            peak = max(members, key=lambda v: f_hat[v])
        clusters_by_peak[peak] = members

    return clusters_by_peak, g, r_map


# ---------------------------
# Exemple d'utilisation
# ---------------------------
if __name__ == "__main__":
    # petit graphe (0-based)
    G = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 4],
        3: [1, 4],
        4: [2, 3],
        5: [6, 7],
        6: [5, 7],
        7: [5, 6],
    }
    f_hat = [0.1, 0.8, 0.5, 0.9, 0.2, 0.3, 0.3, 0.7]  

    clusters, g_vec, r_map = graph_based_hill_climbing(G, f_hat)

    print("Clusters (peak -> members):")
    for peak, members in clusters.items():
        print(f"  Peak {peak} (f={f_hat[peak]}): {sorted(members)}")
    print("g vector:", g_vec)
    print("representant map (rep -> peak):", r_map)
