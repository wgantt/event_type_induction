## Results

This directory contains checkpoints and outputs for the models used in the paper.

- `checkpoints.zip`: Contains the final model checkpoints for the full clustering model (`full_model.pt`) and for the multiview mixture models used to initialize it. The names for these latter checkpoints have the form `mmm-{type}-{number}.pt`, where "mmm" denote "multiview mixture model," `{type}` indicates the ontology type (`event`, `participant`, `role`, or (event-event) `relation`), and `{number}` indicates the number of clusters learned. `full_model.pt` should be used to initialize an `EventTypeInductionModel` (see `event_type_induction/modules/induction.py`) and the `mmm-{type}-{number}.pt` checkpoints should be used to initialize a `MultiviewMixtureModel` (see `event_type_induction/scripts/cluster_init.py`). By default, the `train.json` training configuration file points to these checkpoints.
- `full-model-final-clustering.zip`: Contains the cluster centroids and the per-node/edge posterior distributions over types induced by the final model.
- `mmm-final-clustering.zip`: Contains the cluster centroids and the per-node/edge posterior distributions over types induced by each of the four multiview mixture models.
