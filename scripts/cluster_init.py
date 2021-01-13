# Package external imports
import argparse
from collections import defaultdict
from decomp import UDSCorpus
import numpy as np
import random
from sklearn.mixture import GaussianMixture
import torch
from torch.distributions import Dirichlet
from torch.nn import Parameter, ParameterDict, Module
from typing import List, Iterator

# Package internal imports
from event_type_induction.constants import *
from event_type_induction.modules.vectorized_likelihood import *
from scripts.setup_logging import setup_logging
from event_type_induction.utils import *


LOG = setup_logging()
MODEL_DIR = "/data/wgantt/event_type_induction/checkpoints"

# TODO: modularize


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

    def get_annotations(
        self,
        t: Type,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
    ):
        """Retrieves annotations from a UDSCorpus for a specified type

        Should definitely be factored out into utils or something; I just
        haven't taken the time yet.

        Parameters
        ----------
        t
            the type for which annotations should be retrieved
        data
            a list of identifiers for UDSSentenceGraphs, for which
            the annotations are to be retrieved
        confidences
            ridit-scored confidence values for each annotator, keyed on
            annotator name. Nested dict 
        property_means
            pre-computed mean values for each property; these are used to
            impute missing annotations for each item
        """
        # Averaged annotations for each item
        all_annotations = []

        # Maps properties to a range of indices in the annotation vector
        # for each item
        properties_to_indices = {}

        # All (raw) annotations, grouped by property
        annotations_by_property = defaultdict(list)

        # Annotators corresponding to each annotation in annotations_by_property
        annotators_by_property = defaultdict(list)

        # Confidence scores for each annotation in annotations_by_property
        confidences_by_property = defaultdict(list)

        # Unique annotators for each property
        unique_annotators_by_property = defaultdict(set)

        # The length of a full annotation vector
        anno_vec_len = 0

        # Total number of annotations
        total_annos = 0

        for sname in data:
            graph = self.uds[sname]
            for item, anno in get_item_iter(graph, t):
                anno_vec = []
                annotation_found = False

                if t == Type.ROLE and (
                    ("protoroles" not in anno) or ("distributivity" not in anno)
                ):
                    # We subset to only those edges annotated for both
                    # protoroles and distributivity to avoid having to
                    # do heavy imputation
                    continue

                for subspace in sorted(SUBSPACES_BY_TYPE[t]):
                    for p in sorted(self.s_metadata.properties(subspace)):
                        prop_dim = get_prop_dim(self.s_metadata, subspace, p)
                        vec = np.zeros(prop_dim)

                        # The genericity subspace includes properties associated
                        # with both events and participants. We need to mask the
                        # ones that aren't relevant in each case
                        if (t == Type.EVENT and "arg" in p) or (
                            t == Type.PARTICIPANT and "pred" in p
                        ):
                            continue

                        # Associate the current property with a range of indices
                        # in the annotation vector
                        if p not in properties_to_indices:
                            properties_to_indices[p] = np.array(
                                [anno_vec_len, anno_vec_len + prop_dim]
                            )
                            anno_vec_len += prop_dim

                        # Process annotations for this item only if they actually exist
                        if subspace in anno and p in anno[subspace]:
                            annotation_found = True
                            for a, value in anno[subspace][p]["value"].items():

                                # Get confidence for this annotation
                                conf = anno[subspace][p]["confidence"][a]
                                ridit_conf = confidences[a]
                                if (
                                    ridit_conf is None
                                    or ridit_conf.get(conf) is None
                                    or ridit_conf[conf] < 0
                                ):  # invalid confidence values; default to 1
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
                                total_annos += 1
                        else:
                            # No annotation for this property, so just use the mean
                            vec = property_means[p]

                        # Compute average annotation for this item; for all
                        # train data, there will be only one annotation
                        anno_vec.append(vec / max(vec.sum(), 1))

                # Append current annotation vector to list of all
                # annotation vectors (but only if we actually found
                # relevant annotations)
                if annotation_found:
                    all_annotations.append(np.concatenate(anno_vec))

        return (
            total_annos,
            np.stack(all_annotations),
            properties_to_indices,
            {
                p: torch.FloatTensor(np.stack(v))
                for p, v in annotations_by_property.items()
            },
            {
                p: torch.FloatTensor(np.array(v))
                for p, v in confidences_by_property.items()
            },
            {
                p: torch.LongTensor(np.array(v))
                for p, v in annotators_by_property.items()
            },
            unique_annotators_by_property,
        )

    def get_event_annotations(
        self,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
    ):
        return self.get_annotations(Type.EVENT, data, confidences, property_means)

    def get_participant_annotations(
        self,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
    ):
        return self.get_annotations(Type.PARTICIPANT, data, confidences, property_means)

    def get_role_annotations(
        self,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
    ):
        return self.get_annotations(Type.ROLE, data, confidences, property_means)

    def get_relation_annotations(
        self,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
    ):
        all_annotations = []
        annotations_by_property = defaultdict(list)
        annotators_by_property = defaultdict(list)
        unique_annotators_by_property = defaultdict(set)
        confidences_by_property = defaultdict(list)
        properties_to_indices = {}
        anno_vec_len = 0
        total_annos = 0

        for dname in data:
            graph = self.uds.documents[dname].document_graph
            for anno in graph.edges.values():
                anno_vec = []
                # assert "time" in anno, f"no time annotations found for document {dname}"
                # temporary while I don't have pred-arg document annotations
                if "mereology" in anno and "time" not in anno:
                    continue

                for subspace in sorted(SUBSPACES_BY_TYPE[Type.RELATION]):
                    for p in sorted(self.d_metadata.properties(subspace)):
                        vec = np.zeros(1)
                        n_annos = 0

                        # Associate this property with a range of indices in the type vector
                        if p not in properties_to_indices:
                            properties_to_indices[p] = np.array(
                                [anno_vec_len, anno_vec_len + 1]
                            )
                            anno_vec_len += 1

                        # Collect annotation and confidence
                        if subspace in anno and p in anno[subspace]:
                            for a, value in sorted(anno[subspace][p]["value"].items()):
                                # Get confidence for this annotation
                                conf = anno[subspace][p]["confidence"][a]
                                ridit_conf = confidences[a].get(conf, 1)

                                # Get value
                                vec += value

                                # Bookkeeping
                                n_annos += 1
                                total_annos += 1
                                annotations_by_property[p].append(value)
                                annotator_num = int(a.split("-")[-1])
                                annotators_by_property[p].append(annotator_num)
                                unique_annotators_by_property[p].add(annotator_num)
                                confidences_by_property[p].append(ridit_conf)
                        else:
                            # Since mereology annotations were conditioned on
                            # temporal containment, if they don't occur in a
                            # given annotation, there cannot be mereological
                            # containment, so we default to zero.
                            assert subspace == "mereology"
                            vec = torch.zeros(1)

                        # Average annotation for this item
                        anno_vec.append(vec / max(n_annos, 1))
                all_annotations.append(np.concatenate(anno_vec))

        return (
            total_annos,
            np.stack(all_annotations),
            properties_to_indices,
            {
                p: torch.FloatTensor(np.stack(v))
                for p, v in annotations_by_property.items()
            },
            {
                p: torch.FloatTensor(np.array(v))
                for p, v in confidences_by_property.items()
            },
            {
                p: torch.LongTensor(np.array(v))
                for p, v in annotators_by_property.items()
            },
            unique_annotators_by_property,
        )

    def fit(
        self,
        data: List[str],
        t: Type,
        n_components: int,
        confidences: Dict[str, Dict[int, float]],
    ) -> GaussianMixture:
        gmm = GaussianMixture(n_components, random_state=self.random_seed)
        if t == Type.RELATION:
            property_means = None
        else:
            property_means = get_sentence_property_means(self.uds, data, t)

        (
            total_annos,
            average_annotations,
            properties_to_indices,
            annotations_by_property,
            confidences_by_property,
            annotators_by_property,
            unique_annotators_by_property,
        ) = self.annotation_func_by_type[t](data, confidences, property_means)

        gmm = gmm.fit(average_annotations)
        LOG.info(
            f"GMM average train LL for {n_components} components: {gmm.score(average_annotations)}"
        )

        # Probably shouldn't be returning all these things from a call to
        # "fit", but didn't want to have to separately call the annotation
        # getter function again
        return (
            gmm,
            total_annos,
            average_annotations,
            properties_to_indices,
            annotations_by_property,
            confidences_by_property,
            annotators_by_property,
            unique_annotators_by_property,
        )


class MultiviewMixtureModel(Module):
    def __init__(self, uds: UDSCorpus, random_seed: int = 42, device: str = "cpu"):
        super(MultiviewMixtureModel, self).__init__()
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
        self.component_weights = None
        self.random_effects = None
        self.device = device

    def _init_mus(
        self,
        t: Type,
        gmm_means: np.ndarray,
        props_to_indices: Dict[str, np.ndarray],
        n_components: int,
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
                if "rel-" in p:
                    continue
                if (t == Type.EVENT and "arg" in p) or (
                    t == Type.PARTICIPANT and "pred" in p
                ):
                    continue

                start, end = props_to_indices[p]
                mu = torch.log(torch.FloatTensor(gmm_means[:, start:end]))
                mu_dict[p.replace(".", "-")] = Parameter(mu)
        if t == Type.RELATION:
            # time_mu = torch.log(torch.FloatTensor(gmm_means[:, rel_start:rel_end]))
            mu_dict["time-univariate_mu"] = Parameter(
                torch.cat([torch.FloatTensor([50]) for _ in range(n_components)])
            )
            mu_dict["time-bivariate_mu"] = Parameter(
                torch.stack([torch.FloatTensor([50, 50]) for _ in range(n_components)])
            )
        self.mus = ParameterDict(mu_dict).to(self.device)

    def _init_covs(self, n_components: int, sigma: int = 10) -> None:
        cov_dict = {}
        univariate_sigma = torch.ones(n_components) * sigma
        bivariate_sigma = torch.stack(
            [sigma * torch.eye(2) for _ in range(n_components)]
        )
        cov_dict["time-univariate_sigma"] = Parameter(univariate_sigma)
        cov_dict["time-bivariate_sigma"] = Parameter(bivariate_sigma)
        self.covs = ParameterDict(cov_dict).to(self.device)

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
        lr: float = 0.001,
        concentration: float = 2.5,
        verbosity: int = 10,
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
            train_total_annos,
            train_avg_annotations,
            train_properties_to_indices,
            train_annotations_by_property,
            train_confidences_by_property,
            train_annotators_by_property,
            train_unique_annotators_by_property,
        ) = train_gmm.fit(data["train"], t, n_components, train_confidences)
        LOG.info("...GMM fitting complete")

        LOG.info("Loading dev data...")
        dev_confidences = self._get_annotator_ridits(data["dev"], t)

        if t == Type.RELATION:
            dev_property_means = None
        else:
            dev_property_means = get_sentence_property_means(self.uds, data["dev"], t)

        (
            dev_total_annos,
            dev_avg_annotations,
            dev_properties_to_indices,
            dev_annotations_by_property,
            dev_confidences_by_property,
            dev_annotators_by_property,
            dev_unique_annotators_by_property,
        ) = train_gmm.annotation_func_by_type[t](
            data["dev"], dev_confidences, dev_property_means
        )
        LOG.info("...Complete.")
        LOG.info(f"total train annotations: {train_total_annos}")
        LOG.info(f"total dev annotations: {dev_total_annos}")

        # verify that annotations and confidence values are as expected
        assert (
            train_annotations_by_property.keys() == train_confidences_by_property.keys()
        )
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

        # Determine which annotators in dev are also in train
        # This is necessary for determining the appropriate random
        # effects for each annotator
        dev_annotators_in_train = {}
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

        # Get the right type of property metadata (document metadata for
        # relation types; sentence metadata for everything else)
        if t == Type.RELATION:
            metadata = self.d_metadata

            # Initialize covariance matrices
            self._init_covs(n_components)
        else:
            metadata = self.s_metadata
            self.covs = None

        # Initialize Likelihood module, property means,
        # annotator random effects, and component weights
        ll = self.type_to_likelihood[t](
            train_confidences, metadata, n_components, self.device
        )
        self._init_mus(t, gmm.means_, train_properties_to_indices, n_components)
        self.random_effects = ll.random_effects
        self.component_weights = Parameter(torch.log(torch.FloatTensor(gmm.weights_)))

        # Dirichlet hyperprior over component weights
        hyperprior = Dirichlet(torch.ones(n_components) * concentration)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        prev_dev_ll = -np.inf
        LOG.info(f"Beginning training for {iterations} epochs")
        for i in range(iterations):

            # training
            _, train_ll = ll(
                self.mus,
                train_annotations_by_property,
                train_annotators_by_property,
                train_confidences_by_property,
                covs=self.covs,
            )

            if t == Type.RELATION:
                train_fixed_loss = train_ll
            else:
                # sum over per-property annotations
                # TODO: why not do this inside the likelihood,
                #       as for relations?
                train_fixed_loss = torch.sum(train_ll, -1)

            # add in prior over components
            train_fixed_loss += self.component_weights

            # logsumexp over all components to get overall LL
            train_fixed_loss = -torch.logsumexp(train_fixed_loss, 0)

            # hyperprior loss over component weights
            train_hyperprior_loss = -hyperprior.log_prob(
                torch.exp(exp_normalize(self.component_weights))
            )

            # add in random loss, backprop, and take gradient step
            train_random_loss = ll.random_loss()
            train_loss = train_fixed_loss + train_random_loss + train_hyperprior_loss
            train_loss.backward()
            optimizer.step()

            # train logging
            if i % verbosity == 0:
                LOG.info(
                    f"raw train fixed loss: {np.round(train_fixed_loss.item(), 5)}"
                )
                LOG.info(
                    f"component weights: {torch.exp(exp_normalize(self.component_weights))}"
                )
                LOG.info(f"per-component loss: {-torch.sum(train_ll, -1)}")

                LOG.info(
                    f"Epoch {i} train fixed loss: {np.round(train_fixed_loss.item() / train_total_annos, 5)}"
                )
                LOG.info(
                    f"Epoch {i} train random loss: {np.round(train_random_loss.item(), 5)}"
                )
                LOG.info(
                    f"Epoch {i} hyperprior loss: {np.round(train_hyperprior_loss.item(), 5)}"
                )

            # eval
            with torch.no_grad():
                _, dev_ll = ll(
                    self.mus,
                    dev_annotations_by_property,
                    dev_annotators_by_property,
                    dev_confidences_by_property,
                    train_unique_annotators_by_property,
                    dev_annotators_in_train,
                    covs=self.covs,
                )

                if t == Type.RELATION:
                    dev_fixed_loss = dev_ll
                else:
                    dev_fixed_loss = torch.sum(dev_ll, -1)
                dev_fixed_loss += self.component_weights
                dev_fixed_loss = -torch.logsumexp(dev_fixed_loss, 0)

                if i % verbosity == 0:
                    LOG.info(
                        f"Epoch {i} dev fixed loss: {np.round(dev_fixed_loss.item() / dev_total_annos, 5)}"
                    )

                if i and dev_fixed_loss > prev_dev_fixed_loss:
                    LOG.info("No improvement in dev LL. Stopping early.")
                    LOG.info(
                        f"Final component weights (epoch {i}): {torch.exp(exp_normalize(self.component_weights))}"
                    )
                    LOG.info(
                        f"Final train fixed loss (epoch {i}): {np.round(train_fixed_loss.item() / train_total_annos, 5)}"
                    )
                    LOG.info(
                        f"Final dev fixed loss (epoch {i}): {np.round(dev_fixed_loss.item() / dev_total_annos, 5)}"
                    )
                    return self.eval()
                prev_dev_fixed_loss = dev_fixed_loss

        LOG.info(f"Max iterations reached")
        LOG.info(
            f"Final component weights (epoch {i}): {torch.exp(exp_normalize(self.component_weights))}"
        )
        LOG.info(
            f"Final train fixed loss (epoch {i}): {np.round(train_fixed_loss.item() / train_total_annos, 5)}"
        )
        LOG.info(
            f"Final dev fixed loss (epoch {i}): {np.round(dev_fixed_loss.item() / dev_total_annos, 5)}"
        )
        return self.eval()


def main(args):
    # Load UDS and initialize the mixture model
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    load_event_structure_annotations(uds)
    mmm = MultiviewMixtureModel(uds)

    # Define train and dev splits
    t = STR_TO_TYPE[args.type.upper()]

    if t == Type.RELATION:
        train = list(
            set([graph.document_id for name, graph in uds.items() if "train" in name])
        )
        dev = list(
            set([graph.document_id for name, graph in uds.items() if "dev" in name])
        )
    else:
        train = [s for s in uds if "train" in s]
        dev = [s for s in uds if "dev" in s]
    data = {"train": train, "dev": dev}

    LOG.info(
        f"Fitting mixture model with all types in range {args.min_types} to {args.max_types}, inclusive"
    )
    for n_components in range(args.min_types, args.max_types + 1):
        # Fit the model
        model_name = t.name + "-" + str(n_components) + ".pt"
        mmm = mmm.fit(data, t, n_components)
        # Save it
        save_model(mmm.state_dict(), MODEL_DIR, model_name)


if __name__ == "__main__":
    # TODO: add more arguments
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
