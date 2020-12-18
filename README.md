# UDS Event Type Induction

This package contains code for a generative event type induction model over
the Universal Decompositional Semantics corpus.

## Training

Currently, scripts are provided only for training the clustering model. To
run training, execute the following from the root package directory:

```python -m scripts.train [--parameters /path/to/parameters/file]```

If the --parameters argument is omitted, the script will use the provided
parameters file (`scripts/train.json`).
