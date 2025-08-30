#!/usr/bin/env python3
"""
attention_bench_torch.py

Adaptation PyTorch du script NumPy fourni par l'utilisateur.
But: remplace les projections Q/K/V par des modules/poids PyTorch (nn.Linear) afin
que vous puissiez mesurer le temps d'**apprentissage** (forward+backward+optim.step)
comparé au temps d'inférence (forward only).

Fonctions fournies :
 - attention_classique_torch
 - attention_optimisee_torch
 - attention_efficiente_torch (approx. pédagogique)
 - attention_super_torch
 - multi_head_attention_torch (wrapper)

Le script mesure pour différentes longueurs de séquence :
 - médiane du temps d'inférence (torch.no_grad)
 - médiane du temps d'apprentissage (forward+backward+step)

Usage exemple:
    python PayBetterAttention.py --seq 128 256 512 --d 64 --repeats 10 --train_steps 20 --device cpu
    # sur GPU (si disponible):
    python PayBetterAttention.py --seq 128 256 512 --d 64 --repeats 10 --train_steps 20 --device cuda

"""

import argparse
import time
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn


# ----------------------------- utilitaires -----------------------------

def sync_device(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def torch_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def make_qkv_torch(N: int, d: int, device: torch.device, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    return (torch.randn(N, d, device=device, dtype=torch.float32),
            torch.randn(N, d, device=device, dtype=torch.float32),
            torch.randn(N, d, device=device, dtype=torch.float32))


# ----------------------------- feature maps -----------------------------

def feature_map_torch(x: torch.Tensor, mode: Optional[str] = None, nb_features: Optional[int] = None, rng: Optional[np.random.RandomState] = None) -> torch.Tensor:
    """Transforme (N, d) en features pour attention linéaire (torch).

    Modes: 'relu1', 'elu1', 'favor' (approx Performer-like) ou None.
    """
    if x.ndim != 2:
        raise ValueError('feature_map_torch attend (N, d)')
    N, d = x.shape
    if mode == 'relu1':
        return torch.relu(x) + 1.0
    if mode == 'elu1':
        return torch.where(x > 0, x, x.exp() - 1.0) + 1.0
    if mode == 'favor':
        m = nb_features if nb_features is not None else d
        if rng is None:
            rng = np.random.RandomState()
        W = torch.from_numpy(rng.randn(d, m).astype(np.float32)).to(x.device)
        proj = x @ W
        x_norm2 = (x * x).sum(dim=1, keepdim=True)
        phi = torch.exp(proj - 0.5 * x_norm2)
        phi = phi / (m ** 0.5)
        return phi
    if mode is None:
        return x
    raise ValueError(f"mode inconnu: {mode}")


# ----------------------------- 4 attentions (torch) -----------------------------

def attention_classique_torch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                              Wq: Optional[nn.Linear] = None,
                              Wk: Optional[nn.Linear] = None,
                              Wv: Optional[nn.Linear] = None) -> torch.Tensor:
    """Scaled dot-product attention (avec nn.Linear optionnelles)
    Q/K/V : (N, d)
    Retour : (N, d_v) (ici d_v == d si Wv présent ou V.shape[1])
    """
    if Wq is not None:
        Q = Wq(Q)
    if Wk is not None:
        K = Wk(K)
    if Wv is not None:
        V = Wv(V)

    d_k = K.shape[1]
    scores = (Q @ K.t()) / (d_k ** 0.5)  # (N, N)
    A = torch_softmax(scores, dim=1)
    return A @ V


def attention_optimisee_torch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                              Wq: Optional[nn.Linear] = None,
                              Wk: Optional[nn.Linear] = None) -> torch.Tensor:
    # identique au classique mais garde la forme pour comparaison
    if Wq is not None:
        Q = Wq(Q)
    if Wk is not None:
        K = Wk(K)
    d_k = Q.shape[1]
    scores = (Q @ K.t()) / (d_k ** 0.5)
    A = torch_softmax(scores, dim=1)
    return A @ V


def attention_efficiente_torch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                               Wq: Optional[nn.Linear] = None,
                               feature: Optional[str] = 'relu1') -> torch.Tensor:
    """Approximation pédagogique : on montre l'usage de feature maps.
    Ici on renvoie une sortie similaire au dot-product (pour comparaison) :
    on calcule softmax(QK^T)V mais avec des features (non optimisé réel).
    """
    if Wq is not None:
        Q = Wq(Q)
    # Qf = feature_map_torch(Q, mode=feature)
    # Kf = feature_map_torch(K, mode=feature)

    d_k = Q.shape[1]
    scores = (Q @ K.t()) / (d_k ** 0.5)
    A = torch_softmax(scores, dim=1)
    return A @ V


def attention_super_torch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          Wq: Optional[nn.Linear] = None,
                          Wa: Optional[nn.Parameter] = None) -> torch.Tensor:
    """Attention locale / modifiée : applique Wa sur V si fourni.
    Wa est un paramètre (N, N) simulant un mixage d'attention appris.
    """
    if Wq is not None:
        Q = Wq(Q)
    if Wa is not None:
        # Wa: (N, N) appliqué à V (N, d) -> (N, d)
        V = Wa @ V
    d_k = Q.shape[1]
    scores = (Q @ K.t()) / (d_k ** 0.5)
    A = torch_softmax(scores, dim=1)
    return A @ V


# ----------------------------- multi-head wrapper (torch) -----------------------------

def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # x: (N, d) -> (N, num_heads, head_dim)
    N, d = x.shape
    assert d % num_heads == 0
    head_dim = d // num_heads
    return x.view(N, num_heads, head_dim)


def combine_heads(x: torch.Tensor) -> torch.Tensor:
    # x: (N, num_heads, head_dim) -> (N, d)
    N, num_heads, head_dim = x.shape
    return x.view(N, num_heads * head_dim)


def multi_head_attention_torch(attn_fn, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                               num_heads: int = 8,
                               Wo: Optional[nn.Linear] = None,
                               **attn_kwargs) -> torch.Tensor:
    N, d = Q.shape
    assert d % num_heads == 0, 'd must be divisible by num_heads'


    Qh = split_heads(Q, num_heads)  # (N, H, head_dim)
    Kh = split_heads(K, num_heads)
    Vh = split_heads(V, num_heads)

    heads = []
    for h in range(num_heads):
        q = Qh[:, h, :]
        k = Kh[:, h, :]
        v = Vh[:, h, :]
        out_h = attn_fn(q, k, v, **attn_kwargs)
        if out_h.ndim != 2 or out_h.shape[0] != N:
            raise ValueError(f"Unexpected head output shape: {out_h.shape}")
        heads.append(out_h.unsqueeze(1))  # (N,1,head_dim)

    stacked = torch.cat(heads, dim=1)  # (N, H, head_dim)
    concat = combine_heads(stacked)    # (N, d)
    return Wo(concat) if Wo is not None else concat


# ----------------------------- bench & main (torch) -----------------------------

def benchmark_torch(seq_lengths, d=64, repeats=10, seed=42, num_heads=4, device_str='cpu', train_steps=0):
    device = torch.device(device_str)
    results = []

    # modules / poids globaux (nn.Linear) : shape (d, d)
    torch.manual_seed(seed)
    Wq = nn.Linear(d // num_heads, d // num_heads, bias=False).to(device)
    Wk = nn.Linear(d // num_heads, d // num_heads, bias=False).to(device)
    Wv = nn.Linear(d // num_heads, d // num_heads, bias=False).to(device)
    Wo = nn.Linear(d , d , bias=False).to(device)
    # Wa utilisé dans attention_super : param (N,N) sera créé par taille N

    for N in seq_lengths:
        Q, K, V = make_qkv_torch(N, d, device=device, seed=seed)
        Wa_param = torch.randn(N, N, device=device, dtype=torch.float32)  # non-learned by default

        # -------- inference timings (no_grad) --------
        def time_fn_forward(fn, *args, repeats_local=repeats, **kwargs):
            """Mesure le temps d'une fonction de forward en passant *args et **kwargs.
            Utilise torch.no_grad() et synchronise le device avant/après pour des timings précis.
            """
            times = []
            for _ in range(repeats_local):
                sync_device(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = fn(*args, **kwargs)
                sync_device(device)
                times.append(time.perf_counter() - t0)
            return float(np.median(times))

        t_mha_classic_fwd = time_fn_forward(multi_head_attention_torch, attention_classique_torch, Q, K, V,
                                            num_heads=num_heads, Wo=Wo, Wq=Wq, Wk=Wk, Wv=Wv)

        t_mha_opt_fwd = time_fn_forward(multi_head_attention_torch, attention_optimisee_torch, Q, K, V,
                                        num_heads=num_heads, Wo=Wo, Wq=Wq, Wk=Wk)

        t_mha_eff_fwd = time_fn_forward(multi_head_attention_torch, attention_efficiente_torch, Q, K, V,
                                        num_heads=num_heads,  Wo=Wo, Wq=Wq)

        t_mha_super_fwd = time_fn_forward(multi_head_attention_torch, attention_super_torch, Q, K, V,
                                          num_heads=num_heads, Wo=Wo, Wq=Wq, Wa=Wa_param)

        # -------- training timings (forward+backward+step) --------
        # pour mesurer l'apprentissage, on crée un optimizer sur les paramètres (Wq,Wk,Wv,Wo)
        params = list(Wq.parameters()) + list(Wk.parameters()) + list(Wv.parameters()) + list(Wo.parameters())
        # ajouter Wa si on veut le considérer comme paramètre apprenable
        Wa_learn = nn.Parameter(Wa_param.clone(), requires_grad=True).to(device)
        params_with_Wa = params + [Wa_learn]

        opt = torch.optim.SGD(params, lr=1e-3)
        opt_wa = torch.optim.SGD(params_with_Wa, lr=1e-3)
        loss_fn = nn.MSELoss()

        # helper pour timing d'une étape d'apprentissage
        def time_training_step(fn, optimizer, include_Wa=False, steps_local=train_steps, **fn_kwargs):
            """Mesure le temps médian d'une étape d'entraînement : forward+backward+step.
            Appelle fn(Q, K, V, **fn_kwargs) à chaque itération.
            Gère aussi le cas spécial où `fn` est `multi_head_attention_torch` et
            attend `attn_fn` comme premier argument : si `attn_fn` est fourni dans
            `fn_kwargs`, on l'insère comme premier argument positionnel.
            """
            # vérification rapide : l'optimizer contient-il des paramètres entraînables ?
            opt_has_params = any(len(g.get('params', [])) > 0 for g in optimizer.param_groups)
            if not opt_has_params:
                raise RuntimeError('L\'optimizer ne contient pas de paramètres. Vérifie la liste passée au constructeur de l\'optimizer.')

            t0 = time.perf_counter()
            for _ in range(steps_local):
                optimizer.zero_grad()
                sync_device(device)
                

                # préparer les kwargs locaux pour éviter mutation
                local_kwargs = dict(fn_kwargs)

                # Forward: gérer fonctions qui attendent attn_fn en 1er argument (ex: multi_head_attention_torch)
                if 'attn_fn' in local_kwargs and fn.__name__ == 'multi_head_attention_torch':
                    attn = local_kwargs.pop('attn_fn')
                    out = fn(attn, Q, K, V, **local_kwargs)
                else:
                    out = fn(Q, K, V, **local_kwargs)

                # Target aléatoire de même shape
                target = torch.randn_like(out)
                loss = loss_fn(out, target)

                # Debugging: vérifier que loss/out nécessite un gradient
                if not loss.requires_grad or not out.requires_grad:
                    # construire message utile
                    params_requiring = [p for g in optimizer.param_groups for p in g.get('params', []) if getattr(p, 'requires_grad', False)]
                    msg = (
                        "Impossible d'appeler backward : 'out' ou 'loss' n\'a pas requires_grad."
                        f"out.requires_grad={out.requires_grad}, loss.requires_grad={loss.requires_grad}."
                        f"Nombre de paramètres dans l'optimizer: {sum(len(g.get('params', [])) for g in optimizer.param_groups)}."
                    )
                    if len(params_requiring) == 0:
                        msg += " Aucun paramètre avec requires_grad=True trouvé dans l'optimizer. Vérifie que tu as passé des modules nn.Parameter / nn.Linear aux optimizers."
                    msg += "Éventuelles causes : appel accidentel sous torch.no_grad(), passage des poids sous forme de tensors non liés, ou utilisation de lambdas qui détachent le graphe."
                    raise RuntimeError(msg)

                # Backward + step
                loss.backward()
                optimizer.step()
                sync_device(device)
            times = time.perf_counter() - t0
            return float(times) if times else 0.0

        # note: on mesure pour un petit nombre de steps (train_steps)
        if train_steps > 0:

            # multi-head training : on appelle directement multi_head_attention_torch
            t_mha_classic_train = time_training_step(multi_head_attention_torch, opt, steps_local=train_steps,
                                                    attn_fn=attention_classique_torch, num_heads=num_heads, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo)

            t_mha_opt_train = time_training_step(multi_head_attention_torch, opt, steps_local=train_steps,
                                                attn_fn=attention_optimisee_torch, num_heads=num_heads, Wq=Wq, Wk=Wk, Wo=Wo)

            t_mha_eff_train = time_training_step(multi_head_attention_torch, opt, steps_local=train_steps,
                                                attn_fn=attention_efficiente_torch, num_heads=num_heads, Wq=Wq, Wo=Wo, feature='relu1')

            t_mha_super_train = time_training_step(multi_head_attention_torch, opt_wa, include_Wa=True, steps_local=train_steps,
                                                  attn_fn=attention_super_torch, num_heads=num_heads, Wq=Wq, Wa=Wa_learn, Wo=Wo)
        else:
            t_mha_classic_train = t_mha_opt_train = t_mha_eff_train = t_mha_super_train = 0.0

        results.append({
            'N': N,
            'mha_classic_fwd': t_mha_classic_fwd, 'mha_optimisee_fwd': t_mha_opt_fwd, 'mha_efficiente_fwd': t_mha_eff_fwd, 'mha_super_fwd': t_mha_super_fwd,
            'mha_classic_train': t_mha_classic_train, 'mha_optimisee_train': t_mha_opt_train, 'mha_efficiente_train': t_mha_eff_train, 'mha_super_train': t_mha_super_train,
        })

        print(f"N={N:6d} | fwd classic={t_mha_classic_fwd:.6f}s opt={t_mha_opt_fwd:.6f}s eff={t_mha_eff_fwd:.6f}s super={t_mha_super_fwd:.6f}s | ")
        if train_steps > 0:
            print(f"       train classic={t_mha_classic_train:.6f}s opt={t_mha_opt_train:.6f}s eff={t_mha_eff_train:.6f}s super={t_mha_super_train:.6f}s")

    df = pd.DataFrame(results)
    return df


def plot_and_save(df: pd.DataFrame, out_png: str = 'attention_bench_torch.png') -> None:
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=False)
    ax = axs[0, 0]
    ax.plot(df['N'], df['mha_classic_fwd'], marker='o', label='mha_classic_fwd')
    ax.plot(df['N'], df['mha_optimisee_fwd'], marker='o', label='mha_optimisee_fwd')
    ax.plot(df['N'], df['mha_efficiente_fwd'], marker='o', label='mha_efficiente_fwd')
    ax.plot(df['N'], df['mha_super_fwd'], marker='o', label='mha_super_fwd')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sequence length N (log scale)')
    ax.set_ylabel('Time (s, médian) (log scale)')
    ax.set_title('MHA inference (torch)')
    ax.legend()
    ax.grid(True, which='both', ls='--', linewidth=0.5)

    ax2 = axs[0, 1]
    if 'mha_classic_fwd' in df.columns:
        ax2.plot(df['N'], df['mha_classic_train'], marker='x', linestyle='--', label='mha_classic_train')
        ax2.plot(df['N'], df['mha_optimisee_train'], marker='x', linestyle='--', label='mha_optimisee_train')
        ax2.plot(df['N'], df['mha_efficiente_train'], marker='x', linestyle='--', label='mha_efficiente_train')
        ax2.plot(df['N'], df['mha_super_train'], marker='x', linestyle='--', label='mha_super_train')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Sequence length N (log scale)')
        ax2.set_ylabel('Time (s, médian) (log scale)')
        ax2.set_title('MHA training (torch)')
        ax2.legend()
        ax2.grid(True, which='both', ls='--', linewidth=0.5)
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No multi-head results in dataframe', ha='center', va='center', fontsize=12)

    # --- Bottom-left : histogramme (moyenne inference single-head) ---
    ax3 = axs[1, 0]
    single_cols = ['mha_classic_fwd', 'mha_optimisee_fwd', 'mha_efficiente_fwd', 'mha_super_fwd']
    present_single = [c for c in single_cols if c in df.columns]
    if len(present_single) > 0:
        means = df[present_single].mean()
        # normaliser par classic_fwd
        if 'mha_classic_fwd' in means.index and means['mha_classic_fwd'] > 0:
            norm = (means / means['mha_classic_fwd']) * 100.0
        else:
            norm = means * 0.0
        labels = [s.replace('_fwd', '') for s in present_single]
        ax3.bar(labels, norm, color=['b','orange','g','r'])
        ax3.set_ylabel('Mean time compare to classic (classic = 100)')
        ax3.set_title('Mean time comparaison (inference)')
        for i, v in enumerate(norm.values):
            ax3.text(i, v + max(norm.values) * 0.02, f"{v:.1f}", ha='center')
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'No inference data', ha='center', va='center', fontsize=12)

    # --- Bottom-right : histogramme (moyenne training single-head) ---
    ax4 = axs[1, 1]
    train_cols = ['mha_classic_train', 'mha_optimisee_train', 'mha_efficiente_train', 'mha_super_train']
    present_train = [c for c in train_cols if c in df.columns]
    if len(present_train) > 0 and df[present_train].sum().sum() > 0:
        means_t = df[present_train].mean()
        if 'mha_classic_train' in means_t.index and means_t['mha_classic_train'] > 0:
            norm_t = (means_t / means_t['mha_classic_train']) * 100.0
        else:
            norm_t = means_t * 0.0
        labels_t = [s.replace('_train', '') for s in present_train]
        ax4.bar(labels_t, norm_t, color=['b','orange','g','r'])
        ax4.set_ylabel('Mean time compare to classic (classic = 100)')
        ax4.set_title('Mean time comparaison (training)')
        for i, v in enumerate(norm_t.values):
            ax4.text(i, v + max(norm_t.values) * 0.02, f"{v:.1f}", ha='center')
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'No single-head training data', ha='center', va='center', fontsize=12)

    fig.suptitle("Time attention comparison (PyTorch)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.show()
    print(f"Graphique sauvegardé : {out_png}")


def parse_args():
    p = argparse.ArgumentParser(description='Benchmark variants d\'attention (PyTorch)')
    p.add_argument('--seq', type=int, nargs='+', default=[32, 64, 128, 256, 512, 1024, 2048, 4096], help='tailles de séquence à tester')
    p.add_argument('--d', type=int, default=512, help='dimension des vecteurs Q/K/V')
    p.add_argument('--repeats', type=int, default=15, help='nombre de répétitions pour médiane (inference)')
    p.add_argument('--train_steps', type=int, default=50, help='nombre de pas d\'entrainement pour timing (0=none)')
    p.add_argument('--out', type=str, default='attention_bench_torch_results.csv', help='fichier CSV de sortie')
    p.add_argument('--png', type=str, default='attention_bench_torch.png', help='fichier PNG de sortie (graphique)')
    p.add_argument('--seed', type=int, default=42, help='seed pour reproductibilité (optionnel)')
    p.add_argument('--num_heads', type=int, default=16, help='nombre de têtes multi-head')
    p.add_argument('--device', type=str, default='cuda', help="device: 'cpu' ou 'cuda' (si disponible)")
    return p.parse_args()


def main():
    args = parse_args()
    df = benchmark_torch(args.seq, d=args.d, repeats=args.repeats, seed=args.seed, num_heads=args.num_heads,
                         device_str=args.device, train_steps=args.train_steps)
    df.to_csv(args.out, index=False)
    print(f"Résultats sauvegardés : {args.out}")
    plot_and_save(df, out_png=args.png)


if __name__ == '__main__':
    main()
