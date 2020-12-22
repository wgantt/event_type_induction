# Package external imports
from collections import defaultdict
from decomp import UDSCorpus
from decomp.semantics.uds import UDSSentenceGraph
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
from torch.nn import Parameter, ParameterDict
from typing import List, Iterator, Union

# Package internal imports
from event_type_induction.constants import *
from event_type_induction.modules.likelihood import *
from event_type_induction.utils import (
    load_pred_node_annotator_ids,
    load_arg_node_annotator_ids,
    load_sem_edge_annotator_ids,
    load_doc_edge_annotator_ids,
    load_event_structure_annotations,
    get_prop_dim,
    ridit_score_confidence,
)


class GMM:
    def __init__(
        self,
        uds: UDSCorpus,
        random_seed: int = 42,
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
        average_annotations, properties_to_indices = self.annotation_func_by_type[t](data)
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
            Type.ROLE: SemanticsEdgeAnnotationLikelihood,
            Type.RELATION: DocumentEdgeAnnotationLikelihood,
        }
        self.type_to_annotator_ids = {
            Type.EVENT: load_pred_node_annotator_ids,
            Type.PARTICIPANT: load_arg_node_annotator_ids,
            Type.ROLE: load_sem_edge_annotator_ids,
            Type.RELATION: load_doc_edge_annotator_ids,
        }

    def _data_iter(self, data: List[str], t: Type):
        if t == Type.RELATION:
            for doc in data:
                g = self.uds.documents[doc].document_graph
                for edge, anno in g.edges.items():
                    yield edge, anno
        else:
            for sent in data:
                if t == Type.EVENT:
                    for node, anno in self.uds[sent].predicate_nodes.items():
                        yield node, anno
                elif t == Type.PARTICIPANT:
                    for node, anno in self.uds[sent].argument_nodes.items():
                        yield node, anno
                elif t == Type.ROLE:
                    for edge, anno in self.uds[sent].semantics_edges().items():
                        yield edge, anno
                else:
                    raise ValueError(f"Unrecognized type {t}!")

    def _init_mus(self, t: Type, gmm_means: np.ndarray, props_to_indices: Dict[str, np.ndarray]):
        mu_dict = {}
        for subspace in SUBSPACES_BY_TYPE[t]:
            if t == Type.RELATION:
                metadata = self.d_metadata
            else:
                metadata = self.s_metadata
            for p in metadata.properties(subspace):
                start, end = props_to_indices[p]
                mu_dict[p.replace('.','-')] = torch.FloatTensor(gmm_means[:,start:end])
        return mu_dict

    def _get_annotator_ridits(self, data: List[str], t: Type):
        annotator_ids = self.type_to_annotator_ids[t](self.uds)
        if t == Type.RELATION:
            ridits = ridit_score_confidence(self.uds, sents=data)
        else:
            ridits = ridit_score_confidence(self.uds, docs=data)
        return {a: ridits.get(a) for a in annotator_ids}

    def fit(self, data: List[str], t: Type, n_components: int, iterations: int = 100, lr: float=0.001) -> "MultiviewMixtureModel":
        torch.manual_seed(self.random_seed)

        gmm = GMM(self.uds)
        gmm, properties_to_indices = gmm.fit(data, t, n_components)

        mus = self._init_mus(t, gmm.means_, properties_to_indices)
        metadata = self.d_metadata if t == Type.RELATION else self.s_metadata
        ll = self.type_to_likelihood[t](self._get_annotator_ridits(data,t),metadata)

        optimizer = torch.optim.Adam(mus, lr=lr)
        data_iter = self._data_iter(data, t)
        for i in range(iterations):
            for d, anno in data_iter:
                print(d, anno)


def main():
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    load_event_structure_annotations(uds)
    mmm = MultiviewMixtureModel(uds)
    mmm.fit(["ewt-train-12", "ewt-train-13", "ewt-train-14"], Type.ROLE, 2)


if __name__ == "__main__":
    main()
