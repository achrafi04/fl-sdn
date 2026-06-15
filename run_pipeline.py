"""
Pipeline complete FL-SDN : enchaine toutes les etapes dans l'ordre.
Lance : python3 run_pipeline.py
"""
import numpy as np
import torch
import torch.nn as nn
import copy, time
import sys
sys.path.append("model")
from split_model import SwitchPart, ServerPart

def section(titre):
    print("\n" + "="*55)
    print(f"  {titre}")
    print("="*55)

# ============================================================
section("ETAPE 1 : Chargement des donnees")
# ============================================================
d = np.load("data/cicids2017_processed.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
y_test  = torch.tensor(d["y_test"],  dtype=torch.long)
print(f"Train : {X_train.shape[0]} flux | Test : {X_test.shape[0]} flux")

loss_fn = nn.CrossEntropyLoss()
def evaluate(sw, sv):
    sw.eval(); sv.eval()
    with torch.no_grad():
        return (sv(sw(X_test)).argmax(1) == y_test).float().mean().item()

# ============================================================
section("ETAPE 2 : Split Federated Learning (3 switches)")
# ============================================================
N_SWITCHES, ROUNDS = 3, 5
torch.manual_seed(42)
parts = torch.chunk(torch.randperm(X_train.shape[0]), N_SWITCHES)
clients = [(X_train[p], y_train[p]) for p in parts]

def train_one(sw, sv, X, y, epochs=1, batch=4096):
    sw = copy.deepcopy(sw)
    opt = torch.optim.Adam(list(sw.parameters())+list(sv.parameters()), lr=0.001)
    sw.train(); sv.train()
    for _ in range(epochs):
        pm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], batch):
            idx = pm[i:i+batch]
            opt.zero_grad()
            loss = loss_fn(sv(sw(X[idx])), y[idx])
            loss.backward(); opt.step()
    return sw.state_dict()

def fedavg(ws):
    avg = copy.deepcopy(ws[0])
    for k in avg.keys():
        for w in ws[1:]: avg[k] += w[k]
        avg[k] /= len(ws)
    return avg

global_switch = SwitchPart()
server = ServerPart()
print(f"Accuracy initiale : {evaluate(global_switch, server)*100:.2f}%")
for r in range(ROUNDS):
    weights = [train_one(global_switch, server, cx, cy) for cx, cy in clients]
    global_switch.load_state_dict(fedavg(weights))
    print(f"  Round {r+1}/{ROUNDS} | acc {evaluate(global_switch, server)*100:.2f}%")

# Sauvegarde du modele entraine
torch.save({"switch": global_switch.state_dict(), "server": server.state_dict()},
           "results/split_model_trained.pt")
print("Modele entraine sauvegarde.")

# ============================================================
section("ETAPE 3 : Switch federateur (division du trafic)")
# ============================================================
global_switch.eval(); server.eval()

def detect_solo(x):
    with torch.no_grad():
        return server(global_switch(x)).argmax(1)

def detect_federe(x, n):
    chunks = torch.chunk(x, n)
    return torch.cat([detect_solo(c) for c in chunks])

dec_solo = detect_solo(X_test)
dec_fed  = detect_federe(X_test, N_SWITCHES)
identique = torch.equal(dec_solo, dec_fed)
acc = (dec_fed == y_test).float().mean().item()
print(f"Resultats identiques (solo vs federe) : {identique}")
print(f"Accuracy via federateur : {acc*100:.2f}%")

# ============================================================
section("ETAPE 4 : Mesure de rapidite")
# ============================================================
_ = detect_solo(X_test[:1000])  # echauffement
print("Temps de detection selon le nombre de switches :")
for n in [1, 2, 3, 4, 5]:
    chunks = torch.chunk(X_test, n)
    times = []
    for c in chunks:
        t0 = time.time(); _ = detect_solo(c); times.append(time.time()-t0)
    print(f"  {n} switch(es) | {X_test.shape[0]//n:>7} flux/switch | {max(times)*1000:6.1f} ms")

section("PIPELINE TERMINEE")
print("Split + Federated + Federateur + Mesure : tout est passe.")
