import numpy as np
import torch
import torch.nn as nn
import time

print("Chargement...")
d = np.load("../data/cicids2017_processed.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_test  = torch.tensor(d["X_test"],  dtype=torch.float32)
y_test  = torch.tensor(d["y_test"],  dtype=torch.long)
loss_fn = nn.CrossEntropyLoss()

def make_model(n_hidden_layers, width=32):
    """Construit un MLP avec n hidden layers."""
    layers = [nn.Linear(37, width), nn.ReLU()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, 2)]
    return nn.Sequential(*layers)

def train_eval(model, epochs=5, batch=4096):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    n = X_train.shape[0]
    for _ in range(epochs):
        model.train()
        pm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = pm[i:i+batch]
            opt.zero_grad()
            loss_fn(model(X_train[idx]), y_train[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    return acc

print("\nEXPERIENCE 2 : impact de la profondeur du reseau")
print("(la contrainte de Sebbar est <= 2 hidden layers)\n")
print("  hidden layers | parametres | accuracy")
for n_layers in [1, 2, 3, 4]:
    torch.manual_seed(42)
    model = make_model(n_layers)
    n_params = sum(p.numel() for p in model.parameters())
    acc = train_eval(model)
    print(f"  {n_layers:>13} | {n_params:>10} | {acc*100:.2f}%")

print("\nRecommandation : voir si + de couches apporte vraiment qqch.")
