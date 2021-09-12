# UDS Event Type Induction

This package contains code for a factor graph-based clustering model for joint learning of event, entity, role, and event-event relation types over the Universal Decompositional Semantics (UDS) 2.0 corpus. If you use either the data or the code, we kindly ask that you cite the following paper:

> Gantt, William, Lelia Glass, and Aaron Steven White. 2021. [Decomposing and Recomposing Event Structure](https://arxiv.org/abs/2103.10387). *Transactions of the Association for Computational Linguistics*.

The GitHub repository for the Decomp Toolkit can be found [here](https://github.com/decompositional-semantics-initiative/decomp). More information about UDS, as well all UDS data is available at [decomp.io](http://decomp.io/).

For all inquiries about the paper, the code, or the data, please contact Will Gantt (the first author) at wgantt.iv@gmail.com

## Setup

### Purpose

These setup instructions are aimed at enabling someone to reproduce results from the paper cited above. This code was developed while the Decomp Toolkit was undergoing substantial revisions, and, for this reason, unfortunately has subtle dependencies on an older version of the Toolkit, prior to the latest major release (0.2.0). If your primary interest is not in reproducing the results, but rather in using or building on the model for some other purpose, these instructions will likely be less helpful. In that case, please contact me (Will Gantt) at the email address above and I will be happy to help.

### Environment Setup

This code was developed using Python 3.7.10 and should work with Python >= 3.7, although we have not tested it with versions other than 3.7.10. If you are a Conda user, the easiest way to get started is by creating a new Conda environment using the provided `.yml` file:

```
conda env create --file event_type_induction.yml
conda activate event_type_induction
```

Next, from a different directory, clone the `decomp` repository:

```
git clone git://github.com/decompositional-semantics-initiative/decomp.git
```

As mentioned above, the model code depends on an earlier version of this repository, so before going through the setup instructions for it, you must reset to an older commit, as follows:

```
git checkout c3c2e8cae8c4e46382fd1d08aae700e529ea4cee
```

This will put the `decomp` repo in a "detached HEAD" state, which is fine. Now you are prepared to run the `decomp` installation. For our purposes, you will want to follow the installation instructions for *development* that use `setup.py`. So from the `decomp` repo, run:

```
pip install --user --no-cache-dir -r ./requirements.txt
python setup.py develop
```

This version of the Toolkit that you have now installed also includes an early version of the UDS-EventStructure data. Unfortunately, this data differs slightly in format from the data that was used to obtain the results in the paper, and which is included in this repository. To ensure that the correct version is loaded, you must remove the version in the Toolkit. From the root of the `decomp` repo, type:

```
rm decomp/data/2.0/raw/sentence/annotations/event_structure*
rm decomp/data/2.0/raw/document/annotations/event_structure_mereology.zip
```

This completes the environment setup. You should now return to the root of the `event_type_induction` repo.

## Models

As discussed in the paper, the modeling objective is fundamentally a clustering task: We aim to learn sets of clusters (types) for events, entities, (semantic) roles, and event-event relations, where each cluster is defined by a probability distribution over the relevant UDS properties. (Please see the paper for details on how these distributions are defined.) The clustering is essentially done in two steps:
1. A **sentence-level** clustering, in which cluster centroids are learned for each property without reference to the document-level structure (i.e. the cross-sentential relations between events). This is achieved with *multiview mixture models*.
2. A **document-level** clustering, where the centroids from (1) are used to initialize a document-level *generative model*, which then leverages the cross-sentential relations to update those centroids.

Thus, the first model must be run to generate the parameters used to initialize the second. Instructions for running each stage are discussed in greater detail below.
## Sentence-level models: cluster initialization with multiview mixture models (MMMs)

*Note: If you wish to run the document-level model directly, you may skip this part of the instructions and jump directly to the **Document-level model** section below.*

Means and covariances are initialized per-property using a multiview mixture model (MMM). If you are unfamiliar with MMMs, "Multiview" simply refers to the fact that different kinds of data are involved (categorical, ordinal, binary). "Mixture" refers to the fact that the likelihoods are expressed as weighted *mixtures* of per-cluster likelihood distributions. The script to determine this clustering is in `scripts/cluster_init.py` and may be run from the root project directory as follows:

```python -m scripts.cluster_init [TYPE_NAME] [MIN_COMPONENTS] [MAX_COMPONENTS]```

where `TYPE_NAME` is one of `EVENT`, `PARTICIPANT`, `ROLE`, `RELATION` and `MIN_COMPONENTS` and `MAX_COMPONENTS` are integers indicating the minimum and maximum number of clusters to try, respectively. The invocation above merely specifies the _required_ parameters for running the cluster initialization; the `cluster_init.py` script has several additional flags that may be set and whose description you can read about in the source. In our paper, we used the invocations below to generate the clusterings for the four ontologies. You obviously must specify the value of `<OUTPUT_DIR>`, and you may also change the `--model_name` and `--device` arguments as appropriate. See `scripts/cluster_init.py` for a full explanation of the arguments.

```
# Events
events: python -m scripts.cluster_init EVENT 2 10 --model_name EVENT --model_dir <OUTPUT_DIR> --dump_means --dump_posteriors --clip_min_ll --device cuda:0

# Roles
roles: python -m scripts.cluster_init ROLE 2 10 --model_name ROLE --model_dir <OUTPUT_DIR> --dump_means --dump_posteriors --clip_min_ll --device cuda:0

# Event-event relations
relations: python -m scripts.cluster_init RELATION 2 10 --model_name RELATION --model_dir <OUTPUT_DIR> --dump_means --dump_posteriors --clip_min_ll --device cuda:0

# Participants (AKA "entities")
participants: python -m scripts.cluster_init PARTICIPANT 2 10 --model_name --model_dir <OUTPUT_DIR> PARTICIPANT--dump_means --dump_posteriors --clip_min_ll --device cuda:0
```

### (Optional) Validating the number of Event, Role, Relation, and Participant types

The above runs actually generate *nine* clusterings for each type, ranging from 2 clusters to 10 clusters. However, the document-level model requires the number of clusters for each type to be fixed. As we explain in the paper, we select the appropriate number of clusters for each type by identifying the smallest number *N* such that there is no reliable increase in the evidence the model assigns to the dev data for any number of clusters *M > N*. To determine reliability, we compute the 95% CI for the dev evidence for each clustering using the provided `scripts/estimate_evidence_ci.py`. Although not necessary to running the document-level model, you can confirm our selection of 4 event types, 2 role types, 8 participant types, and 5 event-event relation types by running the script for each type as shown below. The first (and only) positional argument should match the value of `--model_name` for the corresponding run of `scripts/cluster_init.py` above, and the value of `--posteriors_dir` should match the value of `--model_dir`. For the remaining three arguments, `n_samples`, `min_types`, and `max_types`, the script defaults will be appropriate, assuming you have clusterings for all numbers of clusters from 2 to 10, as above. (Otherwise, you should adjust `min_types` and `max_types` as appropriate; `n_samples` should remain the same.)

```
# Events
python scripts/estimate_evidence_ci.py EVENT --posteriors_dir </PATH/TO/EVENT/POSTERIORS/DIR> --n_samples 999 --min_types 2 --max_types 10

# Roles
python scripts/estimate_evidence_ci.py ROLE --posteriors_dir </PATH/TO/ROLE/POSTERIORS/DIR> --n_samples 999 --min_types 2 --max_types 10

# Event-event relations
python scripts/estimate_evidence_ci.py RELATION --posteriors_dir </PATH/TO/RELATION/POSTERIORS/DIR> --n_samples 999 --min_types 2 --max_types 10

# Participants
python scripts/estimate_evidence_ci.py PARTICIPANT --posteriors_dir </PATH/TO/PARTICIPANT/POSTERIORS/DIR> --n_samples 999 --min_types 2 --max_types 10
```

The output will show confidence intervals for the dev evidence for each clustering from 2 to 10 clusters. Consistent with the methodology described above, the selected number of types corresponds to the smallest number for which `Median greater than all successive lower bounds?` is `True`.

## Document-level model

As mentioned above, training the document-level model requires fixing the number of Event, Role, Participant, and Event-Event Relation types. It also requires the *parameters* of the sentence-level models for that number of types, as these parameters are used to initialize the document-level model. If you followed the instructions in the previous section, you will have validated that the optimal number of types is 4 for Events, 2 for Roles, 8 for Participants, and 5 for Event-Event Relations, and will, moreover, have generated sentence-level model checkpoints for these clusterings (and for all clusterings from 2 to 10 clusters, for each type) in your `<OUTPUT_DIR>`. For convenience &mdash; and for those who *skipped* the above steps &mdash; we also provide the checkpoints for the optimal clusterings in the `results` directory.

Our training runs for the document-level model can be reproduced in two stages: an initial training stage (Stage 1) that does *not* include annotator confidence weighting in the model, and a fine-tuning stage (Stage 2) that *does*.

### Stage 1: Initial Training

To run the first stage of training (without confidence-weighting), execute the following:

```python -m scripts.train```

By default, this will read the training configuration from `scripts/config/train.json`. There are two things to note here:
1. The default configuration in `train.json` assumes that you have unzipped the `results/checkpoints.zip` directory and will read the sentence-level checkpoints for Events, Roles, Participants, and Relations from there, which are the ones we used for the paper. If you instead wish to use your own checkpoints, you will have to update the paths for the `*_mmm` parameters in the configuration to point to the appropriate checkpoints.
2. You may adjust the `ckpt_dir` and `ckpt_filename` as you see fit; the rest of the configuration should remain unchanged.

(Careful readers may find it curious that the device is specified as "cpu" here. Given that we're running (loopy) belief propagation under the hood here, and given the size of the tensors we're dealing with, there really is no speed-up from using a GPU. And indeed, we have not tested the latest version of the code on GPU, so we recommend sticking with CPU.)

The above procedure will likely take close to a day to complete and will save a checkpoint (named `<ckpt_filename>.pt`) to the specified `ckpt_dir`. We also include *our* checkpoint for this model in `results/checkpoints.zip` (with the file name `full_model_stage1.pt`). If you do not want to run Stage 1 training yourself, feel free to use this checkpoint.

### Stage 2: Confidence-Weighting

Here, we add ridit-scored annotator confidence values as weights on the annotation likelihoods. To run this part of training, execute the following (for `/PATH/TO/YOUR/STAGE1/MODEL`, you may specify either the checkpoint file you trained above or `results/checkpoints/full_model.pt`, which we provide):

```
python -m scripts.train --model_ckpt </PATH/TO/YOUR/STAGE1/MODEL> --parameters scripts/config/confidence-weight.json --model_overrides scripts/config/overrides.json
```

(Note: If, for some reason, you are using a clustering different than the optimal one discussed above, you will have to change the values of the `n_*_types` parameters in `overrides.json`.) Like Stage 1, Stage 2 may take up to a day to complete. Just as for Stage 1, if you do not wish to run this part of training, we have provided *our* checkpoint output from Stage 2 training in `results/checkpoints/full_model_stage2.pt`.

***

That's all! Once again, feel free to send any questions to me, Will Gantt, at wgantt.iv@gmail.com.
