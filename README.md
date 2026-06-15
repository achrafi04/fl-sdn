# FL-SDN : Federated Learning dans une architecture SDN

Detection d'intrusions par apprentissage federe sur switches P4 (CICIDS2017).

## Structure
- `model/tiny_ids.py` : petit reseau de neurones (2 couches, 1778 params)
- `model/train_local.py` : entrainement centralise (reference, 97.84%)
- `model/federated.py` : federated learning + FedAvg (97.64%)
- `model/split_model.py` : modele splitte (partie switch / partie serveur)
- `model/train_split.py` : entrainement du split (97.84%)
- `model/split_federated.py` : split + federated combines (97.73%, 3 switches)
- `model/measure_speed.py` : mesure de la rapidite selon nb de switches
- `results/speed_results.txt` : resultats (optimum ~4 switches)

## Donnees
CICIDS2017 preprocesse (37 features, binaire). Non versionne (trop volumineux).

## Encadrement
Pr. Anass Sebbar (UIR), Pr. Zakaria Abou El Houda (INRS) — UIR TIClab 2026
