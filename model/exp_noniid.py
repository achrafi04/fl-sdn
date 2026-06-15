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
        if X.shape[0] == 0: continue
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

def run_fed(clients, rounds=5):
    torch.manual_seed(42)
    gsw = SwitchPart(); sv = ServerPart()
    for r in range(rounds):
        weights = [train_one(gsw, sv, cx, cy) for cx, cy in clients if cx.shape[0] > 0]
        gsw.load_state_dict(fedavg(weights))
    return evaluate(gsw, sv)

# ---- Distribution IID : aleatoire (chaque switch un melange) ----
def split_iid(n):
    parts = torch.chunk(torch.randperm(X_train.shape[0]), n)
    return [(X_train[p], y_train[p]) for p in parts]

# ---- Distribution non-IID : par label (chaque switch surtout une classe) ----
def split_noniid(n):
    idx_normal = (y_train == 0).nonzero().squeeze()
    idx_attack = (y_train == 1).nonzero().squeeze()
    # switch 0 : surtout normal | switch 1 : surtout attaque | reste : melange
    clients = []
    # 80% du normal au switch 0, 80% des attaques au switch 1
    n_norm, n_atk = len(idx_normal), len(idx_attack)
    clients.append((X_train[idx_normal[:int(0.8*n_norm)]], y_train[idx_normal[:int(0.8*n_norm)]]))
    clients.append((X_train[idx_attack[:int(0.8*n_atk)]], y_train[idx_attack[:int(0.8*n_atk)]]))
    # le reste melange sur les autres switches
    rest = torch.cat([idx_normal[int(0.8*n_norm):], idx_attack[int(0.8*n_atk):]])
    rest = rest[torch.randperm(len(rest))]
    for p in torch.chunk(rest, max(1, n-2)):
        clients.append((X_train[p], y_train[p]))
    return clients

print("\nEXPERIENCE 3 : IID vs non-IID (3 switches, 5 rounds)\n")
torch.manual_seed(42)
acc_iid = run_fed(split_iid(3))
print(f"  IID (donnees melangees)      : {acc_iid*100:.2f}%")
acc_noniid = run_fed(split_noniid(3))
print(f"  non-IID (switches specialises): {acc_noniid*100:.2f}%")
print(f"\n  Ecart : {(acc_iid-acc_noniid)*100:.2f} points")
print("\nRecommandation : voir si FedAvg resiste au trafic desequilibre.")
