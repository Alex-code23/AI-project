import copy
import random
from multiprocessing import Process, Queue

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# Petit réseau
# ---------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, hidden=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Données synthétiques
# ---------------------------

def make_synthetic_data(n_samples, input_dim, n_classes, W_true, b_true, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    logits = X @ W_true + b_true + 0.1 * torch.randn(n_samples, n_classes)
    y = logits.argmax(dim=1)
    return X, y


# ---------------------------
# Normalisation (fit per dataset and apply)
# ---------------------------
def normalize_tensor(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(dim=0, keepdim=True)
    if std is None:
        std = X.std(dim=0, keepdim=True) + 1e-8
    return (X - mean) / std, mean, std


# ---------------------------
# Worker process
# ---------------------------
def worker_process(worker_id, to_worker_q: Queue, from_worker_q: Queue, local_data, input_dim, n_classes,
                   local_epochs=2, batch_size=32, lr=0.01, seed=0, mu=0.0, optimizer_name='sgd'):
    torch.manual_seed(seed + worker_id)
    model = SimpleNet(input_dim=input_dim, n_classes=n_classes)

    X_local, y_local = local_data
    dataset = TensorDataset(X_local, y_local)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    while True:
        msg = to_worker_q.get()
        if isinstance(msg, str) and msg == 'STOP':
            break

        # msg contient l'etat global et global params utilises pour FedProx
        global_state = msg['state']
        global_params = msg.get('params', None)  # optional flattened global params

        model.load_state_dict(global_state)

        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        model.train()
        total_loss = 0.0
        total_samples = 0
        for epoch in range(local_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)

                # FedProx proximal term
                if mu > 0.0 and global_params is not None:
                    prox = 0.0
                    for name, param in model.named_parameters():
                        prox = prox + (param - global_params[name].to(param.device)).pow(2).sum()
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)

        avg_loss = total_loss / max(1, total_samples)

        # calculer accuracy locale sur l'ensemble local
        model.eval()
        with torch.no_grad():
            preds = model(X_local).argmax(dim=1)
            acc_local = (preds == y_local).float().mean().item()

        state_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        from_worker_q.put({'worker_id': worker_id, 'state': state_cpu, 'n_samples': len(dataset),
                           'acc_local': acc_local, 'loss_local': avg_loss})


# ---------------------------
# FedAvg aggregation
# ---------------------------
def fedavg_aggregate(state_dicts, sizes):
    """
    This is a simple weighted average of the state_dicts according to sizes (eg number of samples per worker)
    """
    total = float(sum(sizes))
    agg_state = {}
    for k in state_dicts[0].keys():
        agg = None
        for sd, n in zip(state_dicts, sizes):
            v = sd[k].float() * (n / total)
            if agg is None:
                agg = v.clone()
            else:
                agg += v
        agg_state[k] = agg
    return agg_state


# ---------------------------
# Évaluation
# ---------------------------
def evaluate(model, Xtest, ytest, batch_size=256):
    model.eval()
    ds = TensorDataset(Xtest, ytest)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


# ---------------------------
# Entraînement centralisé (baseline), pour comparer et voir si le modèle peut apprendre la tâche
# ---------------------------
def train_centralized(X, y, input_dim, n_classes, epochs=10, batch_size=64, lr=0.01, optimizer_name='sgd'):
    model = SimpleNet(input_dim=input_dim, n_classes=n_classes)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for ep in range(epochs):
        total_loss = 0.0
        total = 0
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
        # print(f"Central epoch {ep+1}/{epochs}, loss={total_loss/total:.4f}")
    return model


# ---------------------------
# Main: serveur orchestration
# ---------------------------
if __name__ == '__main__':
    # Paramètres
    n_workers = 10
    input_dim = 20
    n_classes = 30
    samples_per_worker = 4000
    test_samples = 1000
    rounds = 20

    local_epochs = 2
    batch_size = 128
    lr = 0.01

    # Options avancées
    use_fedprox = True
    mu = 0.01  # proximal coefficient si FedProx activé
    server_momentum = 0.9  # si 0 -> pas de momentum serveur
    server_lr = 1.0  # scaling of aggregated update
    optimizer_name = 'sgd'  # 'sgd' or 'adam'

    # Création des données (non-iid via seeds)
    torch.manual_seed(0)
    W_true = torch.randn(input_dim, n_classes) * 2.0
    b_true = torch.randn(n_classes) * 0.5
    workers_data = []
    for i in range(n_workers):
        Xi, yi = make_synthetic_data(samples_per_worker, input_dim, n_classes, W_true=W_true, b_true=b_true, seed=42 + i * 7)
        # normaliser localement
        Xi, mean, std = normalize_tensor(Xi)
        workers_data.append((Xi, yi))

    Xtest, ytest = make_synthetic_data(test_samples, input_dim, n_classes, W_true=W_true, b_true=b_true, seed=999)
    Xtest, _, _ = normalize_tensor(Xtest)  # normaliser test avec ses propres stats

    # Afficher distribution des labels locaux
    for i, (Xi, yi) in enumerate(workers_data):
        vals, counts = torch.unique(yi, return_counts=True)
        print(f"Worker {i} label dist:", dict(zip([int(v) for v in vals.tolist()], [int(c) for c in counts.tolist()])))

    # Baseline centralisée (pour diagnoser si le modèle peut apprendre la tâche)
    X_all = torch.cat([d[0] for d in workers_data], dim=0)
    y_all = torch.cat([d[1] for d in workers_data], dim=0)
    central_model = train_centralized(X_all, y_all, input_dim, n_classes, epochs=20, batch_size=64, lr=0.01,
                                      optimizer_name=optimizer_name)
    central_acc = evaluate(central_model, Xtest, ytest)
    print(f"== Centralized baseline accuracy on test: {central_acc*100:.2f}% ==")

    # Queues
    to_worker_qs = [Queue() for _ in range(n_workers)]
    from_worker_qs = [Queue() for _ in range(n_workers)]

    # Démarrage des workers
    procs = []
    for i in range(n_workers):
        p = Process(target=worker_process, args=(i, to_worker_qs[i], from_worker_qs[i], workers_data[i],
                                                 input_dim, n_classes, local_epochs, batch_size, lr, 100, mu, optimizer_name))
        p.start()
        procs.append(p)

    # Modèle global initial
    global_model = SimpleNet(input_dim=input_dim, n_classes=n_classes)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

    # Pour FedProx on envoie aussi une copie des params parsés
    def get_params_dict(state_dict):
        return {k: v.clone() for k, v in state_dict.items()}

    # Server-side momentum velocity initial
    velocity = {k: torch.zeros_like(v) for k, v in global_state.items()}

    print("Starting federated training simulation with", n_workers, "workers")
    for r in range(1, rounds + 1):
        # 1) envoi aux workers
        send_msg = {'state': global_state, 'params': get_params_dict(global_state)} if use_fedprox else {'state': global_state}
        for q in to_worker_qs:
            q.put(send_msg)

        # 2) collecte
        collected_states = []
        collected_sizes = []
        per_worker_stats = []
        for q in from_worker_qs:
            res = q.get()
            collected_states.append(res['state'])
            collected_sizes.append(res['n_samples'])
            per_worker_stats.append((res['worker_id'], res['acc_local'], res['loss_local']))

        # afficher stats locales
        for wid, acc_local, loss_local in per_worker_stats:
            print(f"Round {r:02d} - Worker {wid} local acc: {acc_local*100:.2f}%, loss: {loss_local:.4f}")

        # 3) agrégation FedAvg
        new_state = fedavg_aggregate(collected_states, collected_sizes)

        # 4) server momentum (FedAvgM-like)
        if server_momentum > 0.0:
            for k in new_state.keys():
                update = new_state[k] - global_state[k]
                velocity[k] = server_momentum * velocity[k] + update
                global_state[k] = global_state[k] + server_lr * velocity[k]
        else:
            global_state = {k: v.clone() for k, v in new_state.items()}

        # charger modèle et évaluer
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, Xtest, ytest)
        print(f"Round {r:02d} - Test accuracy: {acc*100:.2f}%")

    # terminer
    for q in to_worker_qs:
        q.put('STOP')
    for p in procs:
        p.join()

    print("Training finished. Final test accuracy: {:.2f}%".format(100 * evaluate(global_model, Xtest, ytest)))

    # print("--- Résumé / next steps recommandés ---")
    # print("1) Si la baseline centralisée est basse, le problème vient probablement des données / modèle (augmentation de capacité ou +données nécessaire).")
    # print("2) Si la baseline centralisée est bonne mais fédéré échoue: tester FedProx (mu>0), baisser lr local, réduire local_epochs, augmenter batch_size.")
    # print("3) Afficher les acc locales pour détecter divergence entre workers.")
    # print("4) Vérifier normalisation et alignement des labels entre train/test.")
