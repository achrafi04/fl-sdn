import torch
import torch.nn as nn

class SwitchPart(nn.Module):
    """
    Partie du modele qui tourne SUR LE SWITCH.
    Equivalent du RNN de Sebbar, mais entrainable.
    Prend les 37 features -> produit un hidden state.
    """
    def __init__(self, n_features=37, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)   # le "hidden state" envoye au serveur


class ServerPart(nn.Module):
    """
    Partie du modele qui tourne SUR LE CONTROLEUR.
    Prend le hidden state -> classe (normal/attaque).
    """
    def __init__(self, hidden_size=32, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )
    def forward(self, h):
        return self.net(h)


if __name__ == "__main__":
    # Test : on cree les deux parties et on fait passer un flux
    switch = SwitchPart()
    server = ServerPart()

    n_sw = sum(p.numel() for p in switch.parameters())
    n_sv = sum(p.numel() for p in server.parameters())
    print(f"Partie SWITCH  : {n_sw} parametres")
    print(f"Partie SERVEUR : {n_sv} parametres")
    print(f"Total          : {n_sw + n_sv} parametres")

    # Un flux test traverse le split : switch -> serveur
    x = torch.randn(1, 37)
    h = switch(x)            # le switch produit le hidden state
    out = server(h)          # le serveur classe
    print(f"\nFlux test : 37 features -> hidden {h.shape[1]}D -> {out.shape[1]} classes")
    print(f"Sortie : {out}")
    print("Le modele est SPLITTE : switch produit h, serveur classe h.")
