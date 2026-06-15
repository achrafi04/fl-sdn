import numpy as np
import torch
import time
from split_model import SwitchPart, ServerPart

class WorkerSwitch:
    """Un switch travailleur : a le modele splitte, traite SON morceau de trafic."""
    def __init__(self, switch_part, server_part, switch_id):
        self.switch = switch_part
        self.server = server_part
        self.id = switch_id

    def process(self, x):
        """Detecte les anomalies sur le morceau recu."""
        with torch.no_grad():
            h = self.switch(x)          # partie switch
            out = self.server(h)        # partie serveur
            return out.argmax(dim=1)    # decisions


class FederatorSwitch:
    """
    Le switch FEDERATEUR (le chef).
    Recoit le trafic, le divise, distribue aux workers, rassemble.
    """
    def __init__(self, workers):
        self.workers = workers
        self.n = len(workers)

    def detect(self, traffic):
        # 2. DIVISER le trafic en N morceaux (un par worker)
        chunks = torch.chunk(traffic, self.n)

        # 3. DISTRIBUER : chaque worker traite son morceau
        results = []
        for worker, chunk in zip(self.workers, chunks):
            decisions = worker.process(chunk)
            results.append(decisions)

        # 4. RASSEMBLER les resultats dans l'ordre
        return torch.cat(results)


if __name__ == "__main__":
    # ---- Charger donnees ----
    print("Chargement...")
    d = np.load("../data/cicids2017_processed.npz")
    X_test = torch.tensor(d["X_test"], dtype=torch.float32)
    y_test = torch.tensor(d["y_test"], dtype=torch.long)

    # ---- Charger le modele ENTRAINE (sauvegarde par train_split.py) ----
    switch = SwitchPart(); server = ServerPart()
    ckpt = torch.load("../results/split_model_trained.pt")
    switch.load_state_dict(ckpt["switch"])
    server.load_state_dict(ckpt["server"])
    switch.eval(); server.eval()
    print("Modele ENTRAINE charge\n")

    # ---- Creer 3 workers (ils partagent le meme modele entraine) ----
    N = 3
    workers = [WorkerSwitch(switch, server, i) for i in range(N)]
    federator = FederatorSwitch(workers)
    print(f"Federateur cree avec {N} switches travailleurs\n")

    # ---- Detection via le federateur (trafic divise) ----
    t0 = time.time()
    decisions_divise = federator.detect(X_test)
    t_divise = time.time() - t0

    # ---- Comparaison : detection sans division (1 seul switch) ----
    t0 = time.time()
    with torch.no_grad():
        decisions_solo = server(switch(X_test)).argmax(dim=1)
    t_solo = time.time() - t0

    # ---- Verifications ----
    identique = torch.equal(decisions_divise, decisions_solo)
    acc = (decisions_divise == y_test).float().mean().item()

    print(f"Sans division (1 switch)      : {t_solo*1000:.1f} ms")
    print(f"Avec federateur ({N} switches) : {t_divise*1000:.1f} ms")
    print(f"\nResultats identiques ? {identique}")
    print(f"Accuracy globale : {acc*100:.2f}%")
    print("\nLe federateur divise le trafic SANS changer le resultat.")
