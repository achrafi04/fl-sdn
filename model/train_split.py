import numpy as np
import torch
import torch.nn as nn
from split_model import SwitchPart, ServerPart
import time

# ---- 1. Charger les donnees ----
print("Chargement des donnees...")
d = np.load("../data/cicids2017_processed.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
y_test  = torch.tensor(d["y_test"],  dtype=torch.long)
print(f"  Train : {X_train.shape[0]} flux | Test : {X_test.shape[0]} flux")

# ---- 2. Creer les deux parties du modele splitte ----
torch.manual_seed(42)
switch = SwitchPart(n_features=37, hidden_size=32)
server = ServerPart(hidden_size=32, n_classes=2)

loss_fn = nn.CrossEntropyLoss()
# UN SEUL optimiseur pour les DEUX parties (elles apprennent ensemble)
optimizer = torch.optim.Adam(
    list(switch.parameters()) + list(server.parameters()),
    lr=0.001
)

# ---- 3. Entrainement : forward switch->serveur, backward serveur->switch ----
EPOCHS = 5
BATCH = 4096
n = X_train.shape[0]

print(f"\nEntrainement du modele SPLITTE sur {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    switch.train(); server.train()
    perm = torch.randperm(n)
    total_loss = 0.0
    t0 = time.time()
    for i in range(0, n, BATCH):
        idx = perm[i:i+BATCH]
        xb, yb = X_train[idx], y_train[idx]

        optimizer.zero_grad()
        h = switch(xb)              # 1. le SWITCH produit le hidden state
        out = server(h)             # 2. le SERVEUR classe
        loss = loss_fn(out, yb)     # 3. erreur cote serveur
        loss.backward()             # 4. gradient repasse serveur -> switch
        optimizer.step()            # 5. les DEUX parties s'ajustent
        total_loss += loss.item()

    # evaluation
    switch.eval(); server.eval()
    with torch.no_grad():
        pred = server(switch(X_test)).argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
    print(f"  Epoch {epoch+1}/{EPOCHS} | loss {total_loss:.1f} | "
          f"test acc {acc*100:.2f}% | {time.time()-t0:.1f}s")

print("\nLe modele splitte a appris (contrairement a la demo RNN figee).")
# Sauvegarder le modele entraine
torch.save({"switch": switch.state_dict(), "server": server.state_dict()},
           "../results/split_model_trained.pt")
print("Modele entraine sauvegarde dans results/split_model_trained.pt")
