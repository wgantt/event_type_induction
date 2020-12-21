# UDS Event Type Induction

This package contains code for a factor graph-based clustering model for event structural categories over the Universal Decompositional Semantics corpus.

## Setup

This code was developed using Python 3.7.9 and should work with Python >= 3.7. To install all dependencies, run the `setup.py` script:

```python setup.py install```

Note that this requires a version of the [decomp](https://github.com/decompositional-semantics-initiative/decomp) package (0.2.0a1) that is still under development.

## Training

Currently, scripts are provided only for training the clustering model. To run training, execute the following from the root package directory:

```python -m scripts.train [--parameters /path/to/parameters/file]```

If the --parameters argument is omitted, the script will use the provided parameters file (`scripts/train.json`).
