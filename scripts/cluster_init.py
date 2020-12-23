# Package external imports
import argparse
from collections import defaultdict
from decomp import UDSCorpus
from decomp.semantics.uds import UDSSentenceGraph
import numpy as np
from sklearn.mixture import GaussianMixture
import time
import torch
from torch.nn import Parameter, ParameterDict, Module
from typing import List, Iterator, Union

# Package internal imports
from event_type_induction.constants import *
from event_type_induction.modules.likelihood import *
from scripts.setup_logging import setup_logging
from event_type_induction.utils import (
    load_pred_node_annotator_ids,
    load_arg_node_annotator_ids,
    load_sem_edge_annotator_ids,
    load_doc_edge_annotator_ids,
    load_event_structure_annotations,
    get_prop_dim,
    ridit_score_confidence,
)

LOG = setup_logging()


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

    def get_average_annotations(self, t: Type, data: List[str]) -> np.ndarray:
        all_annotations = []
        properties_to_indices = {}
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
                                if value is None:
                                    val = prop_dim - 1
                                elif isinstance(value, str):
                                    val = self.str_to_category[value]
                                elif subspace == "protoroles":
                                    conf = anno[subspace][p]["confidence"][a]
                                    if conf == 0:
                                        val = prop_dim - 1
                                    else:
                                        val = value
                                else:
                                    val = value - 1  # is this right?
                                vec[val] += 1
                        anno_vec.append(vec / max(vec.sum(), 1))
                all_annotations.append(np.concatenate(anno_vec))
        return np.stack(all_annotations), properties_to_indices

    def get_event_annotations(self, data: List[str]) -> np.ndarray:
        return self.get_average_annotations(Type.EVENT, data)

    def get_participant_annotations(self, data: List[str]) -> np.ndarray:
        return self.get_average_annotations(Type.PARTICIPANT, data)

    def get_role_annotations(self, data: List[str]) -> np.ndarray:
        return self.get_average_annotations(Type.ROLE, data)

    def get_relation_annotations(self, data: List[str]) -> np.ndarray:
        all_annotations = []
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
        return np.stack(all_annotations), properties_to_indices

    def fit(self, data: List[str], t: Type, n_components: int) -> GaussianMixture:
        gmm = GaussianMixture(n_components, random_state=self.random_seed)
        average_annotations, properties_to_indices = self.annotation_func_by_type[t](
            data
        )
        return gmm.fit(average_annotations), properties_to_indices


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

    def _data_iter(self, datum: str, t: Type):
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
    ):
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

                if "rel" in p:
                    indices = props_to_indices[p]
                    if indices[0] < rel_start:
                        rel_start = indices[0]
                    if indices[1] > rel_end:
                        rel_end = indices[1]
                    continue

                start, end = props_to_indices[p]
                mu = torch.log(torch.FloatTensor(gmm_means[:, start:end]))
                mu_dict[p.replace(".", "-")] = Parameter(mu.squeeze())
        if t == Type.RELATION:
            time_mu = torch.log(torch.FloatTensor(gmm_means[:, rel_start:rel_end]))
            mu_dict["time"] = Parameter(time_mu)
        self.mus = ParameterDict(mu_dict).to(self.device)

    def _init_covs(
        self, t: Type, gmm_covs: np.ndarray, props_to_indices: Dict[str, np.ndarray]
    ):
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

    def _get_annotator_ridits(self, data: List[str], t: Type):
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
        iterations: int = 1000,
        batch_size: int = 1,
        lr: float = 0.001,
        tolerance: float = 0.0005,
        window_size: int = 2,
    ) -> "MultiviewMixtureModel":
        torch.manual_seed(self.random_seed)

        LOG.info(
            f"Fitting model on type {t.name} using {n_components} components on device {self.device}"
        )
        LOG.info("Fitting GMM...")
        gmm = GMM(self.uds)
        gmm, properties_to_indices = gmm.fit(data["train"], t, n_components)
        LOG.info("...GMM fitting complete")

        # Get the right type of property metadata (document metadata for
        # relation types; sentence metadata for everything else)
        if t == Type.RELATION:
            metadata = self.d_metadata

            # Initialize covariance matrices
            self._init_covs(t, gmm.covariances_, properties_to_indices)
        else:
            metadata = self.s_metadata

        # Initialize Likelihood module, property means, and annotator random effects
        ll = self.type_to_likelihood[t](
            self._get_annotator_ridits(data["train"], t), metadata, self.device
        )
        self._init_mus(t, gmm.means_, properties_to_indices)
        self.random_effects = ll.random_effects

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        prev_dev_ll = -np.inf
        dev_ll_deltas = []
        LOG.info(f"Beginning training for {iterations} epochs")
        for i in range(iterations):

            epoch_start = time.time()
            # Bookkeeping
            fixed_loss = torch.FloatTensor([0.0]).to(self.device)
            random_loss = torch.FloatTensor([0.0]).to(self.device)
            epoch_loss = torch.FloatTensor([0.0]).to(self.device)
            dev_ll_trace = []

            # Training
            LOG.info(f"beginning training for epoch {i}")
            for j, d in enumerate(data["train"]):
                for elem, anno in self._data_iter(d, t):
                    # Only bother computing likelihood for this node/edge
                    # if it's actually annotated for properties we care about
                    if SUBSPACES_BY_TYPE[t].intersection(anno.keys()):
                        _, total_ll = (
                            ll(self.mus, self.covs, anno)
                            if t == Type.RELATION
                            else ll(self.mus, anno)
                        )
                        fixed_loss -= total_ll
                        random_loss += ll.random_loss()

                # This can happen if no nodes or edges in the graph are
                # annotated for the relevant properties
                if fixed_loss.item() == 0:
                    continue

                # Parameter update
                if (j % batch_size) == 0:
                    epoch_loss += fixed_loss
                    loss = fixed_loss + random_loss
                    loss.backward()
                    optimizer.step()
                    fixed_loss = torch.FloatTensor([0.0]).to(self.device)
                    random_loss = torch.FloatTensor([0.0]).to(self.device)

            # Eval
            LOG.info(f"beginning dev eval for epoch {i}")
            epoch_dev_ll = torch.FloatTensor([0.0]).to(self.device)
            for j, d in enumerate(data["dev"]):
                for elem, anno in self._data_iter(d, t):
                    if SUBSPACES_BY_TYPE[t].intersection(anno.keys()):
                        _, dev_ll = (
                            ll(self.mus, self.covs, anno)
                            if t == Type.RELATION
                            else ll(self.mus, anno)
                        )
                        epoch_dev_ll -= dev_ll
            delta = (prev_dev_ll - epoch_dev_ll) / epoch_dev_ll
            dev_ll_deltas.append(delta.item())
            if len(dev_ll_deltas) > window_size:
                moving_avg_delta = np.mean(dev_ll_deltas[-window_size:])
                LOG.info(
                    f"Moving average % change in dev loss (window size={window_size}): {np.round(moving_avg_delta, 6)}"
                )
                if moving_avg_delta < tolerance:
                    LOG.info(f"Tolerance ({tolerance}) exceeded. Stopping early.")
                    return self
            prev_dev_ll = epoch_dev_ll
            epoch_end = time.time()

            LOG.info(
                f"Epoch {i} train fixed loss: {np.round(epoch_loss.item() / len(data['train']), 3)}"
            )
            LOG.info(
                f"          dev fixed loss: {np.round(epoch_dev_ll.item() / len(data['dev']), 3)}"
            )
            LOG.info(
                f"            time elapsed: {np.round(epoch_end - epoch_start,3)}s"
            )


def main():
    # Load UDS and initialize the mixture model
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    load_event_structure_annotations(uds)
    mmm = MultiviewMixtureModel(uds)

    # Define train and dev splits
    train = [s for s in uds if "train" in s][:100]
    dev = [s for s in uds if "dev" in s][:100]
    data = {"train": train, "dev": dev}

    # Fit the model
    mmm.fit(data, Type.EVENT, 2)


if __name__ == "__main__":
    main()
