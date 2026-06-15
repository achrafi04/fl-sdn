import numpy as np
import torch
import torch.nn as nn
from tiny_ids import TinyIDS
import time

# ---- 1. Charger les donnees ----
print("Chargement des donnees...")
d = np.load("../data/cicids2017_processed.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
y_test  = torch.tensor(d["y_test"],  dtype=torch.long)
print(f"  Train : {X_train.shape[0]} flux | Test : {X_test.shape[0]} flux")

# ---- 2. Creer le modele, la loss, l'optimiseur ----
torch.manual_seed(42)
model = TinyIDS(n_features=37, hidden1=32, hidden2=16, n_classes=2)
loss_fn = nn.CrossEntropyLoss()          # mesure l'erreur
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ajuste les poids

# ---- 3. La boucle d'entrainement ----
EPOCHS = 5
BATCH = 4096
n = X_train.shape[0]

print(f"\nEntrainement sur {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(n)             # melange les donnees
    total_loss = 0.0
    t0 = time.time()
    for i in range(0, n, BATCH):
        idx = perm[i:i+BATCH]
        xb, yb = X_train[idx], y_train[idx]

        optimizer.zero_grad()            # remet les gradients a zero
        out = model(xb)                  # 1. devine
        loss = loss_fn(out, yb)          # 2. mesure l'erreur
        loss.backward()                  # 3. calcule les gradients
        optimizer.step()                 # 4. ajuste les poids
        total_loss += loss.item()

    # ---- evaluation sur le test apres chaque epoch ----
    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
    print(f"  Epoch {epoch+1}/{EPOCHS} | loss {total_loss:.1f} | "
          f"test acc {acc*100:.2f}% | {time.time()-t0:.1f}s")

print("\nEntrainement termine.")
