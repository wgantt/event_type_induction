## Results

This directory contains checkpoints and outputs for the models used in the paper.

- `checkpoints.zip`: Contains the final model checkpoints for the full document-level model (After Stage 1 and Stage 2) and for the (sentence-level) multiview mixture models used to initialize that model. The names for these latter checkpoints have the form `mmm-{type}-{number}.pt`, where "mmm" denote "multiview mixture model," `{type}` indicates the ontology type (`event`, `participant`, `role`, or (event-event) `relation`), and `{number}` indicates the number of clusters learned.
- `mmm-final-clustering.zip`: Contains the cluster centroids and the per-node/edge posterior distributions over types induced by each of the four (sentence-level) multiview mixture models.
- `full-model-stage-1.zip`: Contains the cluster centroids and the per-node/edge posterior distributions over types induced by the final model at the end of Stage 1.
- `full-model-stage-2.zip`: Contains the cluster centroids and the per-node/edge posterior distributions over types induced by the final model at the end of Stage 2 (after applying ridit-scored annotator confidence weighting to the annotation likelihoods).
- `logs` contains the output logs for training each of these models.
