import inspect
import numpy as np
import pandas as pd
import os
import torch

from collections import defaultdict
from decomp import UDSCorpus, RawUDSAnnotation
from decomp.semantics.uds import UDSSentenceGraph
from enum import Enum
from event_type_induction.constants import *
from glob import glob
from itertools import product
from pkg_resources import resource_filename
from typing import Any, Dict, Generator, Iterable, Iterator, List, Set, Tuple


class AllenRelation(Enum):
    E1_PRECEDES_E2 = 1
    E2_PRECEDES_E1 = 2
    E1_MEETS_E2 = 3
    E2_MEETS_E1 = 4
    E1_OVERLAPS_E2 = 5
    E2_OVERLAPS_E1 = 6
    E1_STARTS_E2 = 7
    E2_STARTS_E1 = 8
    E1_DURING_E2 = 9
    E2_DURING_E1 = 10
    E1_FINISHES_E2 = 11
    E2_FINISHES_E1 = 12
    E1_EQUALS_E2 = 13


def exp_normalize(t: torch.Tensor, dim=None) -> torch.Tensor:
    """Normalizes a tensor of log values using the exp-normalize trick:
       https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    """
    if dim is not None:
        t2 = torch.exp(t - t.max(dim).values)
        return torch.log(t2 / t2.sum(dim))
    else:
        t2 = torch.exp(t - t.max())
        return torch.log(t2 / t2.sum())


def load_pred_node_annotator_ids(uds: UDSCorpus) -> Set[str]:
    # predicate nodoe subspaces: time, genericity, factuality, event_structure
    pred_node_annotators = set()
    for subspace in PREDICATE_NODE_SUBSPACES:
        subspace_annotators = uds.metadata.sentence_metadata.annotators(subspace)
        if subspace == "genericity":
            # filter down to only predicate node genericity annotators
            subspace_annotators = {a for a in subspace_annotators if "pred" in a}
        pred_node_annotators = pred_node_annotators.union(subspace_annotators)
    return pred_node_annotators


def load_arg_node_annotator_ids(uds: UDSCorpus) -> Set[str]:
    # argument node subspaces: genericity, wordsense
    arg_node_annotators = set()
    for subspace in ARGUMENT_NODE_SUBSPACES:
        subspace_annotators = uds.metadata.sentence_metadata.annotators(subspace)
        if subspace == "genericity":
            # filter down to only argument node genericity annotators
            subspace_annotators = {a for a in subspace_annotators if "arg" in a}
        arg_node_annotators = arg_node_annotators.union(subspace_annotators)
    return arg_node_annotators


def load_sem_edge_annotator_ids(uds: UDSCorpus) -> Set[str]:
    # semantics edge subspaces: protoroles, distributivity
    sem_edge_annotators = set()
    for subspace in SEMANTICS_EDGE_SUBSPACES:
        subspace_annotators = uds.metadata.sentence_metadata.annotators(subspace)
        sem_edge_annotators = sem_edge_annotators.union(subspace_annotators)
    return sem_edge_annotators


def load_doc_edge_annotator_ids(uds: UDSCorpus) -> Set[str]:
    return uds.metadata.document_metadata.annotators()


def load_annotator_ids(uds: UDSCorpus) -> Tuple[Set[str]]:
    """Fetch all of the annotator IDs from an annotated UDS corpus

    Parameters
    ----------
    uds
        The UDSCorpus object from which annotator IDs are to
        be extracted
    """

    pred_node_annotators = load_pred_node_annotator_ids(uds)
    arg_node_annotators = load_arg_node_annotator_ids(uds)
    sem_edge_annotators = load_sem_edge_annotator_ids(uds)
    doc_edge_annotators = load_doc_edge_annotator_ids(uds)

    return (
        pred_node_annotators,
        arg_node_annotators,
        sem_edge_annotators,
        doc_edge_annotators,
    )


def get_documents_by_split(uds: UDSCorpus) -> Dict[str, Set[str]]:
    """Get sets of UDS document IDs based on their split

    This is really a utility UDS should provide on its own.

    Parameters
    ----------
    uds
        The UDSCorpus object from which document IDs are
        to be extracted
    """
    splits = defaultdict(set)
    for doc_id, doc in uds.documents.items():
        sample_sentence = list(doc.sentence_ids)[0]
        split = sample_sentence.split("-")[1]
        splits[split].add(doc_id)
    return splits


def get_item_iter(graph: UDSSentenceGraph, t: Type) -> Iterator:
    """Returns an iterator over sentence graph nodes or edges"""
    if t == Type.EVENT:
        return sorted(graph.predicate_nodes.items())
    elif t == Type.PARTICIPANT:
        return sorted(graph.argument_nodes.items())
    elif t == Type.ROLE:
        return sorted(graph.semantics_edges().items())
    else:
        raise ValueError(f"Unknown type {t}!")


def get_sentence_property_means(
    uds: UDSCorpus, data: List[str], t: Type, use_ordinal: bool = False
) -> Dict[str, np.ndarray]:
    """Get the mean annotation for all properties for a given type"""
    annotations_by_property = defaultdict(list)
    meta = uds.metadata.sentence_metadata

    if t == Type.EVENT:
        duration_to_category = {
            cat: idx
            for idx, cat in enumerate(meta["time"]["duration"].value.categories)
        }

    for sname in data:
        graph = uds[sname]
        for item, anno in get_item_iter(graph, t):
            for subspace in SUBSPACES_BY_TYPE[t]:
                for p in meta.properties(subspace):
                    if (t == Type.EVENT and "arg" in p) or (
                        t == Type.PARTICIPANT and "pred" in p
                    ):
                        continue
                    prop_dim = get_prop_dim(meta, subspace, p, use_ordinal)
                    if subspace in anno and p in anno[subspace]:
                        for a, value in anno[subspace][p]["value"].items():
                            vec = np.zeros(prop_dim)
                            conf = anno[subspace][p]["confidence"][a]
                            # None value --> "does not apply"
                            if value is None:
                                assert (
                                    p in CONDITIONAL_PROPERTIES
                                ), f"unexpected None value for property {p}"
                                if (
                                    use_ordinal
                                    and meta[subspace][p].value.is_ordered_categorical
                                ):
                                    # Maybe ignore these?
                                    n_categories = len(
                                        meta[subspace][p].value.categories
                                    )
                                    val = n_categories // 2
                                else:
                                    val = prop_dim - 1
                                # Only duration annotations are string-valued
                            elif isinstance(value, str):
                                assert (
                                    p == "duration"
                                ), f"unexpected string value for property {p}"
                                val = duration_to_category[value]
                            # Protoroles uses confidence to indicate whether the
                            # property applies or not
                            elif subspace == "protoroles":
                                if conf == 0:
                                    # n_categories // 2 == 2 for protoroles; Aaron suggested 3...
                                    # alternative is to exclude these cases from consideration
                                    # when computing means
                                    n_categories = len(
                                        meta[subspace][p].value.categories
                                    )
                                    val = (
                                        n_categories // 2
                                        if use_ordinal
                                        else prop_dim - 1
                                    )
                                else:
                                    val = value
                            else:
                                val = value

                            if prop_dim == 1:  # binary or ordinal
                                if not meta[subspace][p].value.is_ordered_categorical:
                                    assert (
                                        val == 0 or val == 1
                                    ), f"non-binary value for binary property {p}"
                                vec[0] = val
                            else:  # categorical
                                vec[val] = 1

                            annotations_by_property[p].append(vec)

    return {p: np.mean(v, axis=0) for p, v in annotations_by_property.items()}


def load_event_structure_annotations(uds: UDSCorpus) -> None:
    """Loads the UDS-EventStructure annotations

    These annotations are not included in v0.1.0 of UDS and
    must therefore be loaded after the corpus has been initialized

    Parameters
    ----------
    uds
        The UDSCorpus object into which the annotations will
        be loaded
    """
    data_dir = resource_filename("event_type_induction", "data")
    mereology = RawUDSAnnotation.from_json(os.path.join(data_dir, "mereology.json"))
    distributivity = RawUDSAnnotation.from_json(
        os.path.join(data_dir, "distributivity.json")
    )
    natural_parts = RawUDSAnnotation.from_json(
        os.path.join(data_dir, "natural_parts_and_telicity_corrected.json")
    )

    # arguments are (sentence_annotations, document_annotations)
    uds.add_annotation([natural_parts, distributivity], [mereology])


def get_allen_relation(
    e1_start: int, e1_end: int, e2_start: int, e2_end: int
) -> AllenRelation:
    """Determines an Allen relation given two event durations"""
    if e1_start == e2_start:
        if e1_end == e2_end:
            return AllenRelation.E1_EQUALS_E2
        elif e1_end > e2_end:
            return AllenRelation.E2_STARTS_E1
        else:
            return AllenRelation.E1_STARTS_E2
    elif e1_start < e2_start:
        if e1_end == e2_start:
            return AllenRelation.E1_MEETS_E2
        elif e1_end < e2_start:
            return AllenRelation.E1_PRECEDES_E2
        elif e1_end == e2_end:
            return AllenRelation.E2_FINISHES_E1
        elif e1_end > e2_end:
            return AllenRelation.E2_DURING_E1
        else:
            return AllenRelation.E1_OVERLAPS_E2
    else:  # e1_start > e2_start
        if e2_end == e2_start:
            return AllenRelation.E2_MEETS_E1
        elif e2_end < e2_start:
            return AllenRelation.E2_PRECEDES_E1
        elif e2_end == e1_end:
            return AllenRelation.E1_FINISHES_E2
        elif e2_end > e1_end:
            return AllenRelation.E1_DURING_E2
        else:
            return AllenRelation.E2_OVERLAPS_E1


def ridit_score_confidence(
    uds: UDSCorpus, sents: Set[str] = None, docs: Set[str] = None
) -> Dict[str, Dict[int, float]]:
    """Ridit score confidence values for each annotator

    TODO: generate ridit scores for all possible confidence values
          ensure correctness

    Parameters
    ----------
    uds
        The UDSCorpus
    sents
        The sentences to use for the ridit-scoring calculation
    docs
        The documents to use for the ridit-scoring calculation
    """

    def ridit(x: Iterable) -> Dict[int, float]:
        """ Apply ridit scoring

        Parameters
        ----------
        x
            The values to be ridit scored
        """
        x_vals = set(x)
        x_flat = np.array(x, dtype=int).flatten()
        x_shift = x_flat - x_flat.min()  # bincount requires nonnegative ints

        bincounts = np.bincount(x_shift)
        props = bincounts / bincounts.sum()

        cumdist = np.cumsum(props)
        cumdist[-1] = 0.0  # this looks odd but is right

        ridit_map = {
            val: cumdist[i - 1] + p / 2 for val, (i, p) in zip(x_vals, enumerate(props))
        }
        return ridit_map

    # def get_ridit_map(ridit_map: Dict[int, float]) -> Dict[int, float]:
    #     prev_val = 0.
    #     max_conf = max(ridit_map)
    #     for i in range(N_CONFIDENCE_SCORES):
    #         if i > max_conf:
    #             pass

    # Determine sentence- and document-level graphs for the split
    # (There should really be a UDS function to do this)
    if sents is None:
        split_sentence_graphs = uds.graphs
    else:
        split_sentence_graphs = {name: uds[name] for name in uds if name in sents}
    if docs is None:
        split_doc_graphs = uds.documents
    else:
        split_doc_graphs = {
            name: uds.documents[name] for name in uds.documents if name in docs
        }

    # Semantics node and edge properties
    annotator_confidences = defaultdict(list)
    for graphid, graph in split_sentence_graphs.items():
        for edge, edge_annotation in graph.semantics_edges().items():
            for subspace, properties in edge_annotation.items():
                if isinstance(properties, dict):
                    for prop in properties.keys():
                        for annotator, confidence in properties[prop][
                            "confidence"
                        ].items():
                            if confidence is not None:
                                annotator_confidences[annotator].append(confidence)
        for sem_node in edge:
            sem_node_annotation = graph.semantics_nodes[sem_node]
            for subspace, properties in sem_node_annotation.items():
                # Special case: wordsense has no confidence scores
                if subspace == "wordsense":
                    continue
                if isinstance(properties, dict):
                    for prop in properties.keys():
                        for annotator, confidence in properties[prop][
                            "confidence"
                        ].items():
                            if confidence is not None:
                                annotator_confidences[annotator].append(confidence)

    # Document edge properties
    for doc in split_doc_graphs.values():
        for doc_edge_annotation in doc.document_graph.edges.values():
            for subspace, properties in doc_edge_annotation.items():
                if isinstance(properties, dict):
                    for prop in properties.keys():
                        for annotator, confidence in properties[prop][
                            "confidence"
                        ].items():
                            if confidence is not None:
                                annotator_confidences[annotator].append(confidence)

    for annotator, confidences in annotator_confidences.items():
        try:
            ridit(confidences)
        except TypeError as te:
            print(f"error: {te}")
            print(f"annotator: {annotator}")
            print(f"confidences: {confidences}")
            return

    return {
        annotator: ridit(confidences)
        for annotator, confidences in annotator_confidences.items()
    }


def get_prop_dim(
    metadata: "UDSAnnotationMetadata",
    subspace: str,
    prop: str,
    use_ordinal: bool = False,
):
    """Determines the appropriate dimension for a UDS property"""
    prop_data = metadata.metadata[subspace][prop].value
    if prop_data.is_categorical:
        n_categories = len(prop_data.categories)
        # if ordinal values are *treated* ordinally (instead of
        # as categorical variables), the dimension is always 1
        if prop_data.is_ordered_categorical and use_ordinal:
            return 1
        # conditional categorical properties require an
        # additional dimension for the "does not apply" case
        elif prop in CONDITIONAL_PROPERTIES:
            return n_categories + 1
        # non-conditional, ordinal categorical properties
        elif prop_data.is_ordered_categorical:
            return n_categories
        # non-conditional, binary categorical properties
        else:
            return n_categories - 1
    # currently no non-categorical properties in UDS
    else:
        raise ValueError(
            f"Non-categorical property {property} found in subspace {subspace}"
        )


def parameter_grid(param_dict: Dict[str, Any]):
    """Generator for training hyperparameter grid

    Parameters
    ----------
    param_dict
        Dictionary containing the hyperparameters and their possible values
    """
    ks = list(param_dict.keys())
    vlists = []
    for k, v in param_dict.items():
        if isinstance(v, dict):
            vlists.append(parameter_grid(v))
        elif isinstance(v, list):
            vlists.append(v)
        else:
            errmsg = (
                "param_dict must be a dictionary contining lists or "
                "recursively other param_dicts"
            )
            raise ValueError(errmsg)
    for configuration in product(*vlists):
        yield dict(zip(ks, configuration))


def save_model(data_dict, ckpt_dir, file_name):
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, file_name)
    torch.save(data_dict, ckpt_path)
    return ckpt_path


def save_model_with_args(params, model, initargs, ckpt_dir, file_name):
    filtered_args = {}
    for p in inspect.signature(model.__class__.__init__).parameters:
        if p in initargs:
            filtered_args[p] = initargs[p]
    ckpt_dict = dict(
        params, **{"state_dict": model.state_dict(), "curr_hyper": filtered_args}
    )
    return save_model(ckpt_dict, ckpt_dir, file_name)


def load_model_with_args(cls, ckpt_path):
    ckpt_dict = torch.load(ckpt_path)
    hyper_params = ckpt_dict.get("curr_hyper", {})

    # Model has to be loaded with a UDSCorpus object
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    load_event_structure_annotations(uds)
    hyper_params["uds"] = uds

    model = cls(**hyper_params)
    if "state_dict" in ckpt_dict:
        model.load_state_dict(ckpt_dict["state_dict"])
    else:
        model.load_state_dict(ckpt_dict)
    return model, hyper_params


def dump_property_means(mus: torch.nn.ParameterDict, outfile: str) -> None:
    # create dataframe with components as rows and properties
    # as columns. Each dimension of a categorical property gets
    # its own column.
    df = pd.DataFrame()
    for prop, mean in mus.items():
        if mean.shape[-1] == 1:  # binary property
            df[prop] = torch.exp(mean[:, 0]).detach().cpu().numpy()
        else:
            for i in range(mean.shape[-1]):
                df[prop + f"-dim-{i+1}"] = torch.exp(mean[:, i]).detach().cpu().numpy()
    # write to file
    with open(outfile, "w") as f:
        df.to_csv(outfile, index=False)
