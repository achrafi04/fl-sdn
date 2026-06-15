import numpy as np
import torch
import time
from split_model import SwitchPart, ServerPart

# ---- Charger donnees + modele ----
print("Chargement...")
d = np.load("../data/cicids2017_processed.npz")
X_test = torch.tensor(d["X_test"], dtype=torch.float32)
n_total = X_test.shape[0]
print(f"  Trafic total a traiter : {n_total} flux\n")

torch.manual_seed(42)
switch = SwitchPart(); server = ServerPart()
switch.eval(); server.eval()

def detect(x):
    """Inference complete : switch -> serveur."""
    with torch.no_grad():
        return server(switch(x)).argmax(dim=1)

# Echauffement (le 1er passage est toujours plus lent, on l'ignore)
_ = detect(X_test[:1000])

print("Mesure du temps de detection selon le nombre de switches")
print("(chaque switch traite sa part du trafic)\n")

for n_switches in [1, 2, 3, 4, 5]:
    # on divise le trafic en n parts egales
    parts = torch.chunk(X_test, n_switches)

    # temps que met UN switch a traiter SA part (le plus charge)
    t_per_switch = []
    for p in parts:
        t0 = time.time()
        _ = detect(p)
        t_per_switch.append(time.time() - t0)

    # en parallele, le temps total = le switch le plus lent
    temps_parallele = max(t_per_switch)
    flux_par_switch = n_total // n_switches

    print(f"  {n_switches} switch(es) | {flux_par_switch:>7} flux/switch | "
          f"temps par switch : {temps_parallele*1000:6.1f} ms")

print("\nPlus on divise, moins chaque switch a de travail -> detection plus rapide.")
