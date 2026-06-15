import numpy as np
import torch
import torch.nn as nn
from split_model import SwitchPart, ServerPart
import copy, time

# ---- 1. Donnees ----
print("Chargement des donnees...")
d = np.load("../data/cicids2017_processed.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
y_test  = torch.tensor(d["y_test"],  dtype=torch.long)

# ---- 2. Repartir les donnees entre N switches ----
N_SWITCHES = 3
torch.manual_seed(42)
n = X_train.shape[0]
perm = torch.randperm(n)
parts = torch.chunk(perm, N_SWITCHES)
clients_data = [(X_train[p], y_train[p]) for p in parts]
print(f"  {N_SWITCHES} switches :", [len(p) for p in parts], "flux chacun")

loss_fn = nn.CrossEntropyLoss()

def evaluate(switch, server):
    switch.eval(); server.eval()
    with torch.no_grad():
        pred = server(switch(X_test)).argmax(dim=1)
        return (pred == y_test).float().mean().item()

def train_one_switch(switch_part, server, X, y, epochs=1, batch=4096):
    """Un switch entraine SA partie + collabore avec le serveur (split)."""
    switch_part = copy.deepcopy(switch_part)
    opt = torch.optim.Adam(
        list(switch_part.parameters()) + list(server.parameters()), lr=0.001
    )
    switch_part.train(); server.train()
    m = X.shape[0]
    for _ in range(epochs):
        pm = torch.randperm(m)
        for i in range(0, m, batch):
            idx = pm[i:i+batch]
            opt.zero_grad()
            h = switch_part(X[idx])      # forward : switch -> serveur
            out = server(h)
            loss = loss_fn(out, y[idx])
            loss.backward()              # backward : serveur -> switch
            opt.step()
    return switch_part.state_dict()

def fedavg(weights_list):
    """Moyenne les parties SWITCH des differents switches."""
    avg = copy.deepcopy(weights_list[0])
    for key in avg.keys():
        for w in weights_list[1:]:
            avg[key] += w[key]
        avg[key] = avg[key] / len(weights_list)
    return avg

# ---- 3. Boucle Split Federated ----
ROUNDS = 5
global_switch = SwitchPart()
server = ServerPart()                  # serveur unique, central (le controleur)

print(f"\nSplit Federated Learning : {N_SWITCHES} switches, {ROUNDS} rounds")
print(f"  Accuracy initiale : {evaluate(global_switch, server)*100:.2f}%")

for r in range(ROUNDS):
    t0 = time.time()
    switch_weights = []
    for cx, cy in clients_data:
        w = train_one_switch(global_switch, server, cx, cy, epochs=1)
        switch_weights.append(w)
    # FedAvg sur les parties switch
    global_switch.load_state_dict(fedavg(switch_weights))
    acc = evaluate(global_switch, server)
    print(f"  Round {r+1}/{ROUNDS} | test acc {acc*100:.2f}% | {time.time()-t0:.1f}s")

print("\nSplit + Federated combines : le modele est coupe ET appris en federe.")
