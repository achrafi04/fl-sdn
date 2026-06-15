import torch
import torch.nn as nn

class TinyIDS(nn.Module):
    """
    Petit reseau de neurones pour la detection d'intrusion.
    Architecture : 37 features -> hidden1 -> hidden2 -> 2 classes (normal/attaque)
    """
    def __init__(self, n_features=37, hidden1=32, hidden2=16, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),   # couche cachee 1
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),      # couche cachee 2
            nn.ReLU(),
            nn.Linear(hidden2, n_classes),    # couche de sortie
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # Test rapide : on cree le modele et on compte ses parametres
    model = TinyIDS()
    n_params = sum(p.numel() for p in model.parameters())
    print("Modele TinyIDS cree.")
    print("Architecture :", model.net)
    print("Nombre total de parametres :", n_params)

    # Test : un faux flux de 37 features passe dans le modele
    fake_flow = torch.randn(1, 37)
    output = model(fake_flow)
    print("Sortie pour un flux test :", output)
    print("  -> 2 nombres = scores pour [normal, attaque]")
