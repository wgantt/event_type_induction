# Package external imports
import argparse
from collections import defaultdict
from decomp import UDSCorpus
from decomp.semantics.uds import UDSSentenceGraph
import numpy as np
import random
from sklearn.mixture import GaussianMixture
import torch
from torch.nn import Parameter, ParameterDict, Module
from typing import List, Iterator

# Package internal imports
from event_type_induction.constants import *
from event_type_induction.modules.vectorized_likelihood import *
from scripts.setup_logging import setup_logging
from event_type_induction.utils import (
    load_pred_node_annotator_ids,
    load_arg_node_annotator_ids,
    load_sem_edge_annotator_ids,
    load_doc_edge_annotator_ids,
    load_event_structure_annotations,
    get_prop_dim,
    ridit_score_confidence,
    save_model,
)

LOG = setup_logging()
MODEL_DIR = "/data/wgantt/event_type_induction/"


class GMM:
    def __init__(
        self, uds: UDSCorpus, random_seed: int = 42,
    ):
        """Gaussian mixture model over UDS properties

        Parameters
        ----------
        uds
            the UDSCorpus
        random_seed
            optional random seed to use for the mixture model
        """
        self.uds = uds
        self.s_metadata = self.uds.metadata.sentence_metadata
        self.d_metadata = self.uds.metadata.document_metadata
        self.random_seed = random_seed
        self.str_to_category = {
            cat: idx
            for idx, cat in enumerate(
                self.uds.metadata.sentence_metadata["time"]["duration"].value.categories
            )
        }
        self.annotation_func_by_type = {
            Type.EVENT: self.get_event_annotations,
            Type.PARTICIPANT: self.get_participant_annotations,
            Type.ROLE: self.get_role_annotations,
            Type.RELATION: self.get_relation_annotations,
        }

    @staticmethod
    def get_type_iter(graph: UDSSentenceGraph, t: Type) -> Iterator:
        """Returns an iterator over sentence graph nodes or edges"""
        if t == Type.EVENT:
            return graph.predicate_nodes.values()
        elif t == Type.PARTICIPANT:
            return graph.argument_nodes.values()
        elif t == Type.ROLE:
            return graph.semantics_edges().values()
        else:
            raise ValueError(f"Unknown type {t}!")

    def get_annotations(
        self, t: Type, data: List[str], confidences: Dict[str, Dict[int, float]]
    ):
        all_annotations = []
        properties_to_indices = {}
        annotations_by_property = defaultdict(list)
        annotators_by_property = defaultdict(list)
        unique_annotators_by_property = defaultdict(set)
        confidences_by_property = defaultdict(list)
        anno_vec_len = 0
        for sname in data:
            graph = self.uds[sname]
            for anno in self.__class__.get_type_iter(graph, t):
                anno_vec = []
                for subspace in sorted(SUBSPACES_BY_TYPE[t]):
                    for p in sorted(self.s_metadata.properties(subspace)):
                        prop_dim = get_prop_dim(self.s_metadata, subspace, p)
                        vec = np.zeros(prop_dim)
                        if (t == Type.EVENT and "arg" in p) or (
                            t == Type.PARTICIPANT and "pred" in p
                        ):
                            continue  # hack
                        if p not in properties_to_indices:
                            properties_to_indices[p] = np.array(
                                [anno_vec_len, anno_vec_len + prop_dim]
                            )
                            anno_vec_len += prop_dim
                        if subspace in anno and p in anno[subspace]:
                            for a, value in anno[subspace][p]["value"].items():

                                # Get confidence for this annotation
                                # TODO: verify correctness
                                conf = anno[subspace][p]["confidence"][a]
                                ridit_conf = confidences[a]
                                if (
                                    ridit_conf is None
                                    or ridit_conf.get(conf) is None
                                    or ridit_conf[conf] < 0
                                ):
                                    ridit_conf = 1
                                else:
                                    ridit_conf = ridit_conf.get(conf, 1)

                                # Get the value for this annotation

                                # No value specified; assume this means "does not apply"
                                # and select last category
                                if value is None:
                                    assert (
                                        p in CONDITIONAL_PROPERTIES
                                    ), f"unexpected None value for property {p}"
                                    val = prop_dim - 1
                                # Only duration annotations are string-valued
                                elif isinstance(value, str):
                                    assert (
                                        p == "duration"
                                    ), f"unexpected string value for property {p}"
                                    val = self.str_to_category[value]
                                # Protoroles uses confidence to indicate whether the
                                # property applies or not
                                elif subspace == "protoroles":
                                    if conf == 0:
                                        val = prop_dim - 1
                                    else:
                                        val = value
                                    ridit_conf = (
                                        1  # No confidence values for protoroles
                                    )
                                else:
                                    val = value

                                if prop_dim == 1:  # binary
                                    assert (
                                        val == 0 or val == 1
                                    ), f"non-binary value for binary property {p}"
                                    vec[0] += val
                                else:  # categorical
                                    vec[val] += 1

                                # Raw annotations and confidences by property
                                annotations_by_property[p].append(val)
                                annotator_num = int(a.split("-")[-1])
                                annotators_by_property[p].append(annotator_num)
                                unique_annotators_by_property[p].add(annotator_num)
                                confidences_by_property[p].append(ridit_conf)

                        # Compute average annotation for this item; for all
                        # train data, there will be only one annotation
                        anno_vec.append(vec / max(vec.sum(), 1))

                # Append current annotation vector to list of all
                # annotation vectors
                all_annotations.append(np.concatenate(anno_vec))

        return (
            np.stack(all_annotations),
            properties_to_indices,
            {
                p: torch.FloatTensor(np.stack(v))
                for p, v in annotations_by_property.items()
            },
            {
                p: torch.LongTensor(np.array(v))
                for p, v in confidences_by_property.items()
            },
            {
                p: torch.LongTensor(np.array(v))
                for p, v in annotators_by_property.items()
            },
            unique_annotators_by_property,
        )

    def get_event_annotations(
        self, data: List[str], confidences: Dict[str, Dict[int, float]]
    ):
        return self.get_annotations(Type.EVENT, data, confidences)

    def get_participant_annotations(
        self, data: List[str], confidences: Dict[str, Dict[int, float]]
    ):
        return self.get_annotations(Type.PARTICIPANT, data, confidences)

    def get_role_annotations(
        self, data: List[str], confidences: Dict[str, Dict[int, float]]
    ):
        return self.get_annotations(Type.ROLE, data, confidences)

    def get_relation_annotations(
        self, data: List[str], confidences: Dict[str, Dict[int, float]]
    ):
        all_annotations = []
        annotations_by_property = defaultdict(list)
        confidences_by_property = defaultdict(list)
        properties_to_indices = {}
        anno_vec_len = 0
        for dname in data:
            graph = self.uds.documents[dname].document_graph
            for anno in graph.edges.values():
                anno_vec = []
                for subspace in sorted(SUBSPACES_BY_TYPE[Type.RELATION]):
                    for p in sorted(self.d_metadata.properties(subspace)):
                        vec = np.zeros(1)
                        n_annos = 0
                        if p not in properties_to_indices:
                            properties_to_indices[p] = np.array(
                                [anno_vec_len, anno_vec_len + 1]
                            )
                            anno_vec_len += 1
                        if subspace in anno:
                            for value in anno[subspace][p]["value"].values():
                                vec += value
                                n_annos += 1
                        anno_vec.append(vec / max(n_annos, 1))
                all_annotations.append(np.concatenate(anno_vec))
        return (
            np.stack(all_annotations),
            properties_to_indices,
            annotations_by_property,
            confidences_by_property,
        )

    def fit(
        self,
        data: List[str],
        t: Type,
        n_components: int,
        confidences: Dict[str, Dict[int, float]],
    ) -> GaussianMixture:
        gmm = GaussianMixture(n_components, random_state=self.random_seed)
        (
            average_annotations,
            properties_to_indices,
            annotations_by_property,
            confidences_by_property,
            annotators_by_property,
            unique_annotators_by_property,
        ) = self.annotation_func_by_type[t](data, confidences)
        return (
            gmm.fit(average_annotations),
            properties_to_indices,
            annotations_by_property,
            confidences_by_property,
            annotators_by_property,
            unique_annotators_by_property,
        )


class MultiviewMixtureModel(Module):
    def __init__(self, uds: UDSCorpus, random_seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.uds = uds
        self.random_seed = random_seed
        self.s_metadata = self.uds.metadata.sentence_metadata
        self.d_metadata = self.uds.metadata.document_metadata
        self.type_to_likelihood = {
            Type.EVENT: PredicateNodeAnnotationLikelihood,
            Type.PARTICIPANT: ArgumentNodeAnnotationLikelihood,
            Type.ROLE: SemanticsEdgeAnnotationLikelihood,
            Type.RELATION: DocumentEdgeAnnotationLikelihood,
        }
        self.type_to_annotator_ids = {
            Type.EVENT: load_pred_node_annotator_ids,
            Type.PARTICIPANT: load_arg_node_annotator_ids,
            Type.ROLE: load_sem_edge_annotator_ids,
            Type.RELATION: load_doc_edge_annotator_ids,
        }
        self.mus = None
        self.random_effects = None
        self.device = device

    def _data_iter(self, datum: str, t: Type) -> Iterator:
        if t == Type.RELATION:
            g = self.uds.documents[datum].document_graph
            for edge, anno in g.edges.items():
                yield edge, anno
        else:
            if t == Type.EVENT:
                for node, anno in self.uds[datum].predicate_nodes.items():
                    yield node, anno
            elif t == Type.PARTICIPANT:
                for node, anno in self.uds[datum].argument_nodes.items():
                    yield node, anno
            elif t == Type.ROLE:
                for edge, anno in self.uds[datum].semantics_edges().items():
                    yield edge, anno
            else:
                raise ValueError(f"Unrecognized type {t}!")

    def _init_mus(
        self, t: Type, gmm_means: np.ndarray, props_to_indices: Dict[str, np.ndarray]
    ) -> None:
        mu_dict = {}
        rel_start = np.inf
        rel_end = -np.inf
        for subspace in SUBSPACES_BY_TYPE[t]:
            if t == Type.RELATION:
                metadata = self.d_metadata
            else:
                metadata = self.s_metadata
            for p in metadata.properties(subspace):

                if (t == Type.EVENT and "arg" in p) or (
                    t == Type.PARTICIPANT and "pred" in p
                ):
                    continue

                if "rel-" in p:
                    indices = props_to_indices[p]
                    if indices[0] < rel_start:
                        rel_start = indices[0]
                    if indices[1] > rel_end:
                        rel_end = indices[1]
                    continue

                start, end = props_to_indices[p]
                mu = torch.log(torch.FloatTensor(gmm_means[:, start:end]))
                mu_dict[p.replace(".", "-")] = Parameter(mu)
        if t == Type.RELATION:
            time_mu = torch.log(torch.FloatTensor(gmm_means[:, rel_start:rel_end]))
            mu_dict["time"] = Parameter(time_mu)
        self.mus = ParameterDict(mu_dict).to(self.device)

    def _init_covs(
        self, t: Type, gmm_covs: np.ndarray, props_to_indices: Dict[str, np.ndarray]
    ) -> None:
        if t != Type.RELATION:
            raise ValueError(
                "Covariance matrices should be initialized only for relation types"
            )
        start = np.inf
        end = -np.inf
        for p, indices in props_to_indices.items():
            if "rel" in p:
                if indices[0] < start:
                    start = indices[0]
                if indices[1] > end:
                    end = indices[1]
        covs = torch.FloatTensor(gmm_covs[:, start:end, start:end])

        # Avoid singular matrices
        for i, c in enumerate(covs):
            if torch.matrix_rank(c).item() < len(c):
                covs[i] += torch.eye(len(c))

        self.covs = Parameter(covs).to(self.device)

    def _get_annotator_ridits(
        self, data: List[str], t: Type
    ) -> Dict[str, Dict[int, float]]:
        annotator_ids = self.type_to_annotator_ids[t](self.uds)
        if t == Type.RELATION:
            ridits = ridit_score_confidence(self.uds, sents=data)
        else:
            ridits = ridit_score_confidence(self.uds, docs=data)
        return {a: ridits.get(a) for a in annotator_ids}

    def fit(
        self,
        data: Dict[str, List[str]],
        t: Type,
        n_components: int,
        iterations: int = 2500,
        batch_size: int = 1,
        lr: float = 0.001,
    ) -> "MultiviewMixtureModel":
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        LOG.info(
            f"Fitting model on type {t.name} using {n_components} components on device {self.device}"
        )
        LOG.info("Fitting GMM...")
        train_confidences = self._get_annotator_ridits(data["train"], t)
        train_gmm = GMM(self.uds)
        (
            gmm,
            train_properties_to_indices,
            train_annotations_by_property,
            train_confidences_by_property,
            train_annotators_by_property,
            train_unique_annotators_by_property,
        ) = train_gmm.fit(data["train"], t, n_components, train_confidences)
        LOG.info("...GMM fitting complete")

        LOG.info("Loading dev data...")
        dev_confidences = self._get_annotator_ridits(data["dev"], t)
        (
            dev_avg_annotations,
            dev_properties_to_indices,
            dev_annotations_by_property,
            dev_confidences_by_property,
            dev_annotators_by_property,
            dev_unique_annotators_by_property,
        ) = train_gmm.annotation_func_by_type[t](data["dev"], dev_confidences)
        LOG.info("...Complete.")

        # TODO: make same assertion for dev
        assert (
            train_annotations_by_property.keys() == train_confidences_by_property.keys()
        )
        num_train_items = 0
        for p in train_annotations_by_property:
            anno_len = len(train_annotations_by_property[p])
            conf_len = len(train_confidences_by_property[p])
            num_annotators = len(train_annotators_by_property[p])
            assert (
                anno_len == conf_len
            ), f"mismatched annotation and confidence lengths ({anno_len} and {conf_len}) for property {p}"
            assert (
                num_annotators == anno_len
            ), f"mismatched annotation and annotator lengths ({anno_len} and {num_annotators})"
            num_train_items += anno_len

        # Determine which annotators in dev are also in train
        # This is necessary for determining the appropriate random
        # effects for each annotator
        dev_annotators_in_train = {}
        num_dev_items = 0
        for p, dev_annos in dev_annotators_by_property.items():
            new_dev = len(
                dev_unique_annotators_by_property[p]
                - train_unique_annotators_by_property[p]
            )
            train_annos = train_unique_annotators_by_property[p]
            dev_annotators_in_train[p] = torch.BoolTensor(
                [a.item() in train_annos for a in dev_annos]
            )
            train_unique_annotators_by_property[p] = torch.LongTensor(
                list(train_unique_annotators_by_property[p])
            )
            num_dev_items += len(dev_annos)
            # print(f"num new annotators in dev for property {p}: {new_dev}")

        # Get the right type of property metadata (document metadata for
        # relation types; sentence metadata for everything else)
        if t == Type.RELATION:
            metadata = self.d_metadata

            # Initialize covariance matrices
            self._init_covs(t, gmm.covariances_, train_properties_to_indices)
        else:
            metadata = self.s_metadata

        # Initialize Likelihood module, property means, and annotator random effects
        ll = self.type_to_likelihood[t](train_confidences, metadata, self.device)
        self._init_mus(t, gmm.means_, train_properties_to_indices)
        self.random_effects = ll.random_effects

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        prev_dev_ll = -np.inf
        LOG.info(f"Beginning training for {iterations} epochs")
        for i in range(iterations):

            # TODO: incorporate batch size ?
            # Training
            _, train_ll = ll(
                self.mus,
                train_annotations_by_property,
                train_annotators_by_property,
                train_confidences_by_property,
            )
            train_fixed_loss = -train_ll
            train_random_loss = ll.random_loss()
            train_loss = train_fixed_loss + train_random_loss
            train_loss.backward()
            optimizer.step()
            LOG.info(
                f"Epoch {i} train fixed loss: {np.round(train_fixed_loss.item() / num_train_items, 5)}"
            )

            # Eval
            # TODO: verify that dev and train annotators are using common indexing
            _, dev_ll = ll(
                self.mus,
                dev_annotations_by_property,
                dev_annotators_by_property,
                dev_confidences_by_property,
                train_unique_annotators_by_property,
                dev_annotators_in_train,
            )
            dev_fixed_loss = -dev_ll
            LOG.info(
                f"            dev fixed loss: {np.round(dev_fixed_loss.item() / num_dev_items, 5)}"
            )

            if i and dev_fixed_loss > prev_dev_fixed_loss:
                LOG.info("No improvement in dev LL. Stopping early.")
                return self.eval()
            prev_dev_fixed_loss = dev_fixed_loss

        # Max iterations reached
        return self.eval()


def main(args):
    # Load UDS and initialize the mixture model
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    load_event_structure_annotations(uds)
    mmm = MultiviewMixtureModel(uds)

    # Define train and dev splits
    train = [s for s in uds if "train" in s]
    dev = [s for s in uds if "dev" in s]
    data = {"train": train, "dev": dev}

    LOG.info(
        f"Fitting mixture model with all types in range {args.min_types} to {args.max_types}, inclusive"
    )
    t = STR_TO_TYPE[args.type.upper()]
    for n_components in range(args.min_types, args.max_types + 1):
        # Fit the model
        model_name = t.name + "-" + str(n_components) + ".pt"
        mmm = mmm.fit(data, t, n_components)
        # Save it
        save_model(mmm.state_dict(), MODEL_DIR, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, help="the type to cluster on")
    parser.add_argument(
        "min_types", type=int, help="minimum of range of numbers of types to try"
    )
    parser.add_argument(
        "max_types", type=int, help="maximum of range of numbers of types to try"
    )
    args = parser.parse_args()
    main(args)
