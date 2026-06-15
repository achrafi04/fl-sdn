import numpy as np
import torch
import torch.nn as nn
from split_model import SwitchPart, ServerPart
import copy

print("Chargement...")
d = np.load("../data/cicids2017_processed.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
y_test  = torch.tensor(d["y_test"],  dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()

def evaluate(sw, sv):
    sw.eval(); sv.eval()
    with torch.no_grad():
        return (sv(sw(X_test)).argmax(1) == y_test).float().mean().item()

def train_one(sw, sv, X, y, epochs=1, batch=4096):
    sw = copy.deepcopy(sw)
    opt = torch.optim.Adam(list(sw.parameters())+list(sv.parameters()), lr=0.001)
    sw.train(); sv.train()
    for _ in range(epochs):
        pm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], batch):
            idx = pm[i:i+batch]
            opt.zero_grad()
            loss_fn(sv(sw(X[idx])), y[idx]).backward()
            opt.step()
    return sw.state_dict()

def fedavg(ws):
    avg = copy.deepcopy(ws[0])
    for k in avg.keys():
        for w in ws[1:]: avg[k] += w[k]
        avg[k] /= len(ws)
    return avg

def run_federated(n_switches, rounds=5):
    torch.manual_seed(42)
    parts = torch.chunk(torch.randperm(X_train.shape[0]), n_switches)
    clients = [(X_train[p], y_train[p]) for p in parts]
    gsw = SwitchPart(); sv = ServerPart()
    for r in range(rounds):
        weights = [train_one(gsw, sv, cx, cy) for cx, cy in clients]
        gsw.load_state_dict(fedavg(weights))
    return evaluate(gsw, sv)

print("\nEXPERIENCE 1 : impact du nombre de switches sur l'accuracy")
print("(plus de switches = moins de donnees par switch)\n")
print("  nb switches | flux/switch | accuracy finale")
for n in [2, 3, 5, 10, 20]:
    acc = run_federated(n)
    flux = X_train.shape[0] // n
    print(f"  {n:>11} | {flux:>11} | {acc*100:.2f}%")

print("\nRecommandation : voir jusqu'ou on peut fragmenter sans perdre.")

