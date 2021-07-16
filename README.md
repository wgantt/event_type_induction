# UDS Event Type Induction

This package contains code for a factor graph-based clustering model for joint learning of event, entity, role, and event-event relation types over the Universal Decompositional Semantics (UDS) 2.0 corpus. If you use either the data or the code, we kindly ask that you cite the following paper:

> Gantt, William, Lelia Glass, and Aaron Steven White. 2021. [Decomposing and Recomposing Event Structure](https://arxiv.org/abs/2103.10387). arXiv:2103.10387 [cs.CL]

The GitHub repository for the Decomp Toolkit can be found [here](https://github.com/decompositional-semantics-initiative/decomp). More information about UDS, as well all UDS data is available at [decomp.io](http://decomp.io/).

## Setup

This code was developed using Python 3.7.10 and should work with Python >= 3.7, although we have not tested it with versions other than 3.7.10. If you are a Conda user, the easiest way to get started is by creating a new Conda environment using the provided `.yml` file:

```
conda env create --file event_type_induction.yml
conda activate event_type_induction
```

To install all dependencies, run the `setup.py` script:

```python setup.py install```

As this code relies heavily on the Decomp Toolkit (version 0.2.1), you will have to run through the installation and setup for that package first after running the `setup.py` script above (instructions [here](https://github.com/decompositional-semantics-initiative/decomp#installation)).

## Cluster initialization

Means and covariances are initialized per-property using a multiview mixture model. The script to determine this clustering is in `scripts/cluster_init.py` and may be run from the root project directory as follows:

```python -m scripts.cluster_init [TYPE_NAME] [MIN_COMPONENTS] [MAX_COMPONENTS]```

where `TYPE_NAME` is one of `EVENT`, `PARTICIPANT`, `ROLE`, `RELATION` and `MIN_COMPONENTS` and `MAX_COMPONENTS` are integers indicating the minimum and maximum number of clusters to try, respectively. The invocation above merely specifies the _required_ parameters for running the cluster initialization; the `cluster_init.py` script has several additional flags that may be set and whose description you can read about in the source. In our paper, we used the following invocations to generate the clusterings for the four ontologies:

```
events: python -m scripts.cluster_init EVENT 2 10 --model_name EVENT --dump_means --dump_posteriors --clip_min_ll --device cuda:0
roles: python -m scripts.cluster_init ROLE 2 10 --model_name ROLE--dump_means --dump_posteriors --clip_min_ll --device cuda:0
relations: python -m scripts.cluster_init RELATION 2 10 --model_name RELATION--dump_means --dump_posteriors --clip_min_ll --device cuda:0
participants: python -m scripts.cluster_init PARTICIPANT 2 10 --model_name PARTICIPANT--dump_means --dump_posteriors --clip_min_ll --device cuda:0
```

## Training

To run training, execute the following from the root package directory:

```python -m scripts.train [--parameters /path/to/parameters/file]```

If the --parameters argument is omitted, the script will default to the provided parameters file (`scripts/config/train.json`). If using the default multiview mixture model checkpoints pointed to by that file, please be sure to unzip `results/checkpoints.zip` first.
