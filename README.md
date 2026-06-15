# FL-SDN : Federated Learning dans une architecture SDN

Detection d'intrusions par apprentissage federe et split sur switches P4 (CICIDS2017).
Chaque switch execute une partie d'un reseau de neurones, l'entraine localement,
et l'agregation FedAvg se fait au niveau du controleur SDN.

## Pipeline complete
- `run_pipeline.py` : enchaine tout (donnees -> split federated -> federateur -> mesure) en une commande

## Modeles et entrainement
- `model/tiny_ids.py` : petit reseau de neurones (2 couches, 1778 params)
- `model/train_local.py` : entrainement centralise (reference, 97.84%)
- `model/federated.py` : federated learning + FedAvg (97.64%)
- `model/split_model.py` : modele splitte (partie switch / partie serveur)
- `model/train_split.py` : entrainement du split (97.84%)
- `model/split_federated.py` : split + federated combines (97.73%, 3 switches)

## Switch federateur
- `model/federator.py` : divise le trafic entre switches, rassemble les resultats
  (resultats identiques au traitement solo, 97.84%)

## Experiences et recommandations
- `model/measure_speed.py` : temps de detection selon nb de switches
- `model/exp_nb_switches.py` : impact du nb de switches sur l'accuracy
- `model/exp_profondeur.py` : impact de la profondeur du reseau
- `model/exp_noniid.py` : robustesse au non-IID (trafic desequilibre)
- `results/speed_results.txt` : tous les resultats + recommandations

## Recommandations principales
1. Nombre de switches : accuracy stable jusqu'a ~5, puis chute -> optimum 4-5
2. Profondeur : 2 couches optimal, plateau au-dela -> valide la contrainte <= 2 couches
3. non-IID : FedAvg resiste mais coute ~1.3 pt -> piste : FedProx/FedNova

## Donnees
CICIDS2017 preprocesse (37 features, binaire). Non versionne (trop volumineux).
Regenerable depuis Kaggle (dataset ericanacletoribeiro/cicids2017-cleaned-and-preprocessed).

## Encadrement
Pr. Anass Sebbar (UIR), Pr. Zakaria Abou El Houda (INRS) — UIR TIClab 2026
