import numpy as np
import torch
import torch.nn as nn
from tiny_ids import TinyIDS
import copy
import time

# ---- 1. Charger les donnees ----
print("Chargement des donnees...")
d = np.load("../data/cicids2017_processed.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
y_test  = torch.tensor(d["y_test"],  dtype=torch.long)

# ---- 2. Repartir les donnees entre 2 clients ----
# Chaque "switch" voit seulement SA moitie des donnees
torch.manual_seed(42)
n = X_train.shape[0]
perm = torch.randperm(n)
half = n // 2
idx_c1, idx_c2 = perm[:half], perm[half:]
clients_data = [(X_train[idx_c1], y_train[idx_c1]),
                (X_train[idx_c2], y_train[idx_c2])]
print(f"  Client 1 : {len(idx_c1)} flux | Client 2 : {len(idx_c2)} flux")

# ---- Fonctions utilitaires ----
def evaluate(model):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
        return (pred == y_test).float().mean().item()

def train_local(model, X, y, epochs=1, batch=4096):
    """Un client entraine le modele sur SES donnees. Retourne les poids appris."""
    model = copy.deepcopy(model)              # le client part du modele global
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    m = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(m)
        for i in range(0, m, batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return model.state_dict()                 # renvoie les poids

def fedavg(weights_list):
    """LE COEUR : moyenne les poids de tous les clients."""
    avg = copy.deepcopy(weights_list[0])
    for key in avg.keys():
        for w in weights_list[1:]:
            avg[key] += w[key]
        avg[key] = avg[key] / len(weights_list)
    return avg

# ---- 3. La boucle federee ----
ROUNDS = 5
global_model = TinyIDS()
print(f"\nFederated Learning : 2 clients, {ROUNDS} rounds")
print(f"  Accuracy initiale (avant entrainement) : {evaluate(global_model)*100:.2f}%")

for r in range(ROUNDS):
    t0 = time.time()
    # ① + ② chaque client entraine sur ses donnees, en partant du modele global
    client_weights = []
    for cx, cy in clients_data:
        w = train_local(global_model, cx, cy, epochs=1)
        client_weights.append(w)
    # ③ + ④ le serveur moyenne les poids (FedAvg) et met a jour le modele global
    global_model.load_state_dict(fedavg(client_weights))
    acc = evaluate(global_model)
    print(f"  Round {r+1}/{ROUNDS} | test acc {acc*100:.2f}% | {time.time()-t0:.1f}s")

print("\nFederated Learning termine.")
