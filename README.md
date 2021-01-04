# UDS Event Type Induction

This package contains code for a factor graph-based clustering model for event structural categories over the Universal Decompositional Semantics corpus.

## Setup

This code was developed using Python 3.7.9 and should work with Python >= 3.7. To install all dependencies, run the `setup.py` script:

```python setup.py install```

Note that this requires a version of the [decomp](https://github.com/decompositional-semantics-initiative/decomp) package (0.2.0a1) that is still under development.

## Cluster initialization

Means and covariances are initialized per-property using a multiview mixture model. The script to determine this clustering is in `scripts/cluster_init.py` and may be run from the root project directory as follows:

```python -m scripts.cluster_init [TYPE_NAME] [MIN_COMPONENTS] [MAX_COMPONENTS]```

where `TYPE_NAME` is one of {`EVENT`, `PARTICIPANT`, `ROLE`, `RELATION`} and `MIN_COMPONENTS` and `MAX_COMPONENTS` are integers indicating the minimum and maximum number of clusters to try, respectively. Note that clustering on `RELATION` types is still under development and does not currently work.

## Training

Currently, scripts are provided only for training the clustering model. To run training, execute the following from the root package directory:

```python -m scripts.train [--parameters /path/to/parameters/file]```

If the --parameters argument is omitted, the script will use the provided parameters file (`scripts/train.json`).
