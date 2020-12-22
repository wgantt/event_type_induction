# Package external imports
import numpy as np
import torch
from collections import defaultdict
from decomp import UDSCorpus
from decomp.semantics.uds import UDSSentenceGraph
from sklearn.mixture import GaussianMixture
from typing import List, Iterator, Union

# Package internal imports
from event_type_induction.constants import *
from event_type_induction.modules.likelihood import *
from event_type_induction.utils import (
    load_annotator_ids,
    load_event_structure_annotations,
    get_prop_dim,
    ridit_score_confidence,
)


class GMM:
    def __init__(
        self,
        uds: UDSCorpus,
        split_sents: List[str],
        split_docs: List[str],
        random_seed: int = 42,
    ):
        """Gaussian mixture model over UDS properties

        Parameters
        ----------
        uds
            the UDSCorpus
        split_sents
            the UDS sentence graph IDs to be used to fit the model for sentence-
            level properties
        split_docs
            the UDS document IDs to be used in fitting the model for document-
            level properties
        random_seed
            optional random seed to use for the mixture model
    """
        self.uds = uds
        self.s_metadata = self.uds.metadata.sentence_metadata
        self.d_metadata = self.uds.metadata.document_metadata
        self.split_sents = split_sents
        self.split_docs = split_docs
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

    def get_average_annotations(self, t: Type) -> np.ndarray:
        all_annotations = []
        properties_to_indices = {}
        anno_vec_len = 0
        for sname in self.split_sents:
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

    def get_event_annotations(self) -> np.ndarray:
        return self.get_average_annotations(Type.EVENT)

    def get_participant_annotations(self) -> np.ndarray:
        return self.get_average_annotations(Type.PARTICIPANT)

    def get_role_annotations(self) -> np.ndarray:
        return self.get_average_annotations(Type.ROLE)

    def get_relation_annotations(self) -> np.ndarray:
        all_annotations = []
        properties_to_indices = {}
        anno_vec_len = 0
        for dname in self.split_docs:
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

    def fit(self, t: Type, n_components: int) -> GaussianMixture:
        gmm = GaussianMixture(n_components, random_state=self.random_seed)
        average_annotations, properties_to_indices = self.annotation_func_by_type[t]()
        return gmm.fit(average_annotations), properties_to_indices


class MultiviewMixtureModel:
    def __init__(
        self,
        uds: UDSCorpus,
        random_seed: int = 42,
    ):
        self.uds = uds
        self.random_seed = random_seed
        self.s_metadata = self.uds.metadata.sentence_metadata
        self.d_metadata = self.uds.metadata.document_metadata
        self.type_to_likelihood = {
            Type.EVENT: PredicateNodeAnnotationLikelihood,
            Type.PARTICIPANT: ArgumentNodeAnnotationLikelihood,
            Type.ROLE: SemanticsEdgeLikelihood,
            Type.RELATION: DocumentEdgeAnnotationLikelihood,
        }

    def _data_iter(self, data: List[str], t: Type):
        if t == Type.RELATION:
            for doc in data:
                yield doc, self.uds.documents[doc].document_graph
        else:
            for sent in data:
                yield sent, self.uds[sent]

    def _init_mus(self, gmm_means: np.ndarray, props_to_indices: Dict[str, np.ndarray]):
        pass

    def _get_annotator_ridits(self):
        # Fetch annotator IDs from UDS, used by the likelihood
        # modules to initialize their random effects
        (
            pred_node_annotators,
            arg_node_annotators,
            sem_edge_annotators,
            doc_edge_annotators,
        ) = load_annotator_ids(self.uds)

        # Load ridit-scored annotator confidence values
        # TODO: fix so that you can specify a list of sentences or documents
        #       to base ridit scoring on
        ridits = ridit_score_confidence(self.uds)
        pred_node_annotator_confidence = {
            a: ridits.get(a) for a in pred_node_annotators
        }
        arg_node_annotator_confidence = {a: ridits.get(a) for a in arg_node_annotators}
        sem_edge_annotator_confidence = {a: ridits.get(a) for a in sem_edge_annotators}
        doc_edge_annotator_confidence = {a: ridits.get(a) for a in doc_edge_annotators}

    def fit(self, data: List[str], t: Type, n_components: int, iterations: int = 100, lr: float=0.001) -> MultiviewMixtureModel:
        torch.manual_seed(self.random_seed)

        gmm = GMM(self.uds, self.split_sents, self.split_docs)
        gmm, properties_to_indices = gmm.fit(t, n_components)

        # TODO: initialize mus based on GMM
        if t == Type.RELATION:
            metadata = self.d_metadata
        else:
            metadata = self.s_metadata

        ll = self.type_to_likelihood(t)(None,metadata)

        optimizer = torch.optim.Adam(lr=lr)
        data_iter = self._data_iter(data, t)
        for i in range(iterations):
            for d in data_iter:
                pass


def main():
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    load_event_structure_annotations(uds)
    gmm = GMM(
        uds,
        ["ewt-train-12"],
        ["weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000"],
    )
    gmm = gmm.fit(Type.ROLE, 4)
    print(gmm.means_)
    print(gmm.means_.shape)


if __name__ == "__main__":
    main()
