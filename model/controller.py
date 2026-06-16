import numpy as np
import torch
import torch.nn as nn
from split_model import SwitchPart, ServerPart
import copy

class Switch:
    """Un switch P4 dans un sous-reseau. A sa SwitchPart + ses donnees locales."""
    def __init__(self, switch_id, subnet, X_local, y_local):
        self.id = switch_id
        self.subnet = subnet           # ex: "10.0.1.0/24"
        self.X = X_local
        self.y = y_local
        self.switch_part = SwitchPart()  # sa moitie du modele

    def train_local(self, global_switch_state, server, epochs=1, batch=4096):
        """Entraine SA partie en collaborant avec le serveur (split)."""
        self.switch_part.load_state_dict(copy.deepcopy(global_switch_state))
        opt = torch.optim.Adam(
            list(self.switch_part.parameters()) + list(server.parameters()), lr=0.001
        )
        loss_fn = nn.CrossEntropyLoss()
        self.switch_part.train(); server.train()
        for _ in range(epochs):
            pm = torch.randperm(self.X.shape[0])
            for i in range(0, self.X.shape[0], batch):
                idx = pm[i:i+batch]
                opt.zero_grad()
                h = self.switch_part(self.X[idx])   # forward switch->serveur
                loss = loss_fn(server(h), self.y[idx])
                loss.backward()                      # backward serveur->switch
                opt.step()
        return self.switch_part.state_dict()


class Controller:
    """
    Le CONTROLEUR SDN = le FEDERATEUR (fusionnes, comme demande par Sebbar).
    - a la vue sur tout le reseau (tous les switches, tous sous-reseaux)
    - heberge la ServerPart (classification finale)
    - divise le trafic + agrege par FedAvg
    """
    def __init__(self):
        self.server = ServerPart()              # la partie serveur du modele
        self.global_switch = SwitchPart()        # le modele switch global (a distribuer)
        self.switches = []                       # tous les switches, tous sous-reseaux

    def register_switch(self, switch):
        self.switches.append(switch)

    def federated_round(self):
        """Un round : chaque switch entraine, le controleur agrege (FedAvg)."""
        # 1. chaque switch entraine SA partie en local
        switch_weights = []
        for sw in self.switches:
            w = sw.train_local(self.global_switch.state_dict(), self.server)
            switch_weights.append(w)
        # 2. FedAvg : moyenne des parties switch
        avg = copy.deepcopy(switch_weights[0])
        for key in avg.keys():
            for w in switch_weights[1:]:
                avg[key] += w[key]
            avg[key] /= len(switch_weights)
        # 3. mise a jour du modele global
        self.global_switch.load_state_dict(avg)

    def evaluate(self, X_test, y_test):
        self.global_switch.eval(); self.server.eval()
        with torch.no_grad():
            pred = self.server(self.global_switch(X_test)).argmax(1)
            return (pred == y_test).float().mean().item()
if __name__ == "__main__":
    print("Architecture Controleur=Federateur, 2 sous-reseaux NON-IID\n")
    d = np.load("../data/cicids2017_processed.npz")
    X_train = torch.tensor(d["X_train"], dtype=torch.float32)
    y_train = torch.tensor(d["y_train"], dtype=torch.long)
    X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
    y_test  = torch.tensor(d["y_test"],  dtype=torch.long)

    torch.manual_seed(42)
    # --- Repartition NON-IID par sous-reseau ---
    # Sous-reseau X (S1,S2) : surtout du trafic NORMAL
    # Sous-reseau Y (S3,S4) : beaucoup d'ATTAQUES
    idx_normal = (y_train == 0).nonzero().squeeze()
    idx_attack = (y_train == 1).nonzero().squeeze()
    idx_normal = idx_normal[torch.randperm(len(idx_normal))]
    idx_attack = idx_attack[torch.randperm(len(idx_attack))]

    n_norm, n_atk = len(idx_normal), len(idx_attack)
    # Sous-reseau X recoit 80% du normal + 20% des attaques
    x_data = torch.cat([idx_normal[:int(0.8*n_norm)], idx_attack[:int(0.2*n_atk)]])
    # Sous-reseau Y recoit 20% du normal + 80% des attaques
    y_data = torch.cat([idx_normal[int(0.8*n_norm):], idx_attack[int(0.2*n_atk):]])
    x_data = x_data[torch.randperm(len(x_data))]
    y_data = y_data[torch.randperm(len(y_data))]

    # 2 switches par sous-reseau
    x_parts = torch.chunk(x_data, 2)
    y_parts = torch.chunk(y_data, 2)

    ctrl = Controller()
    config = [("10.0.1.0/24", x_parts[0]), ("10.0.1.0/24", x_parts[1]),
              ("10.0.2.0/24", y_parts[0]), ("10.0.2.0/24", y_parts[1])]
    for i, (subnet, p) in enumerate(config):
        sw = Switch(i+1, subnet, X_train[p], y_train[p])
        ctrl.register_switch(sw)
        # afficher la composition du trafic de ce switch
        n_norm_sw = (y_train[p] == 0).sum().item()
        n_atk_sw  = (y_train[p] == 1).sum().item()
        pct_atk = 100 * n_atk_sw / len(p)
        print(f"  Switch {i+1} ({subnet}) : {len(p)} flux | {pct_atk:.0f}% attaques")

    print(f"\nAccuracy initiale : {ctrl.evaluate(X_test, y_test)*100:.2f}%")
    for r in range(5):
        ctrl.federated_round()
        print(f"  Round {r+1}/5 | acc {ctrl.evaluate(X_test, y_test)*100:.2f}%")
    print("\nFedAvg tient malgre des sous-reseaux aux trafics differents (non-IID).")
