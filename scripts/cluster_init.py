# Package external imports
import argparse
from decomp import UDSCorpus
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
from event_type_induction.utils import *

LOG = setup_logging()


class GMM:
    def __init__(
        self,
        uds: UDSCorpus,
        random_seed: int = 42,
        use_ordinal: bool = False,
        device: str = "cpu",
    ):
        """Gaussian mixture model over UDS properties

        Parameters
        ----------
        uds
            the UDSCorpus
        random_seed
            optional random seed to use for the mixture model
        use_ordinal
            determines whether ordinal properties should actually be
            represented as scalar interval values or as categorical ones
        device
            the device on which data tensors are to be created
        """
        self.uds = uds
        self.s_metadata = self.uds.metadata.sentence_metadata
        self.d_metadata = self.uds.metadata.document_metadata
        self.random_seed = random_seed
        self.use_ordinal = use_ordinal
        self.device = device
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
        device: str = "cpu",
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
        device
            the device on which all the tensors are to be created
        """
        # Averaged annotations for each item
        all_annotations = []

        # Maps properties to a range of indices in the annotation vector
        # for each item
        properties_to_indices = {}

        # All (raw) annotations, grouped by property
        annotations_by_property = DefaultOrderedDict(list)

        # Annotators corresponding to each annotation in annotations_by_property
        annotators_by_property = DefaultOrderedDict(list)

        # Confidence scores for each annotation in annotations_by_property
        confidences_by_property = DefaultOrderedDict(list)

        # Item IDs corresponding to each annotation in annotations_by_property
        items_by_property = DefaultOrderedDict(list)

        # Maps integer IDs to UDS node/edge names
        idx_to_item = DefaultOrderedDict(str)

        # Unique annotators for each property
        unique_annotators_by_property = DefaultOrderedDict(set)

        # The length of a full annotation vector
        anno_vec_len = 0

        # Counts total nodes or edges
        item_ctr = 0

        for name in data:
            graph = self.uds[name]
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
                        prop_dim = get_prop_dim(
                            self.s_metadata, subspace, p, use_ordinal=self.use_ordinal
                        )
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
                        n_annos = 0  # number of annotations for this item
                        if subspace in anno and p in anno[subspace]:
                            annotation_found = True
                            for a, value in anno[subspace][p]["value"].items():

                                # Confidence for current annotation
                                # ---------------------------------
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

                                # Value for current annotation
                                # ----------------------------

                                # Special case 1: None values (i.e. property was annotated as "doesn't apply")
                                if value is None:
                                    # This should only be true of conditional properties
                                    assert (
                                        p in CONDITIONAL_PROPERTIES
                                    ), f"unexpected None value for property {p}"
                                    # If this is an ordinal property, and we're treating ordinal variables
                                    # as such, we set the "does not apply" value to the mean ordinal value
                                    if (
                                        self.use_ordinal
                                        and self.s_metadata[subspace][
                                            p
                                        ].value.is_ordered_categorical
                                    ):
                                        val = np.nan
                                    # Otherwise, the "does not apply" case corresponds to the last category
                                    # when treating ordinal variables nominally.
                                    else:
                                        val = prop_dim - 1

                                # Special case 2: String values (should only be duration annotations)
                                elif isinstance(value, str):
                                    assert (
                                        p == "duration"
                                    ), f"unexpected string value for property {p}"
                                    val = self.str_to_category[value]

                                # Special case 3: Protoroles properties
                                elif subspace == "protoroles":
                                    if conf == 0:
                                        if self.use_ordinal:
                                            val = np.nan
                                            ridit_conf = conf
                                        else:
                                            val = prop_dim - 1
                                            ridit_conf = 1
                                    else:
                                        val = value
                                        ridit_conf = 1

                                # Default case: all other properties
                                else:
                                    val = value

                                if prop_dim == 1:  # binary or ordinal
                                    if not self.s_metadata[subspace][
                                        p
                                    ].value.is_ordered_categorical:
                                        assert (
                                            val == 0 or val == 1
                                        ), f"non-binary value for binary property {p}"
                                    if not np.isnan(val):
                                        vec[0] += val
                                else:  # categorical
                                    vec[val] += 1

                                # Raw annotations and confidences by property
                                annotations_by_property[p].append(val)
                                items_by_property[p].append(item_ctr)
                                idx_to_item[item_ctr] = item
                                annotator_num = int(a.split("-")[-1])
                                annotators_by_property[p].append(annotator_num)
                                unique_annotators_by_property[p].add(annotator_num)
                                confidences_by_property[p].append(ridit_conf)
                                if not np.isnan(val):
                                    n_annos += 1
                        else:
                            # No annotation for this property, so just use the mean
                            vec = property_means[p]

                        # Compute average annotation for this item; for all
                        # train data, there will be only one annotation
                        """
                        assert (
                            n_annos <= 3
                        ), f"{n_annos} annotations found for property {p} on item {item}"
                        """
                        anno_vec.append(vec / max(n_annos, 1))

                # Append current annotation vector to list of all
                # annotation vectors (but only if we actually found
                # relevant annotations)
                if annotation_found:
                    all_annotations.append(np.concatenate(anno_vec))
                item_ctr += 1

        return (
            np.stack(all_annotations),
            properties_to_indices,
            {
                p: torch.FloatTensor(np.stack(v)).to(device)
                for p, v in annotations_by_property.items()
            },
            {
                p: torch.LongTensor(np.array(v)).to(device)
                for p, v in items_by_property.items()
            },
            idx_to_item,
            {
                p: torch.FloatTensor(np.array(v)).to(device)
                for p, v in confidences_by_property.items()
            },
            {
                p: torch.LongTensor(np.array(v)).to(device)
                for p, v in annotators_by_property.items()
            },
            unique_annotators_by_property,
        )

    def get_event_annotations(
        self,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
        device: str = "cpu",
    ):
        return self.get_annotations(
            Type.EVENT, data, confidences, property_means, device
        )

    def get_participant_annotations(
        self,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
        device: str = "cpu",
    ):
        return self.get_annotations(
            Type.PARTICIPANT, data, confidences, property_means, device
        )

    def get_role_annotations(
        self,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
        device: str = "cpu",
    ):
        return self.get_annotations(
            Type.ROLE, data, confidences, property_means, device
        )

    def get_relation_annotations(
        self,
        data: List[str],
        confidences: Dict[str, Dict[int, float]],
        property_means: Dict[str, np.ndarray],
        device: str = "cpu",
    ):
        all_annotations = []
        annotations_by_property = DefaultOrderedDict(list)
        items_by_property = DefaultOrderedDict(list)
        idx_to_item = DefaultOrderedDict(str)
        annotators_by_property = DefaultOrderedDict(list)
        unique_annotators_by_property = DefaultOrderedDict(set)
        confidences_by_property = DefaultOrderedDict(list)
        properties_to_indices = {}
        anno_vec_len = 0
        item_ctr = 0

        for dname in data:
            graph = self.uds.documents[dname].document_graph
            for edge, anno in sorted(graph.edges.items()):
                anno_vec = []
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
                                annotations_by_property[p].append(value)
                                items_by_property[p].append(item_ctr)
                                idx_to_item[item_ctr] = edge
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
                item_ctr += 1

        return (
            np.stack(all_annotations),
            properties_to_indices,
            {
                p: torch.FloatTensor(np.stack(v)).to(device)
                for p, v in annotations_by_property.items()
            },
            {
                p: torch.LongTensor(np.array(v)).to(device)
                for p, v in items_by_property.items()
            },
            idx_to_item,
            {
                p: torch.FloatTensor(np.array(v)).to(device)
                for p, v in confidences_by_property.items()
            },
            {
                p: torch.LongTensor(np.array(v)).to(device)
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
            property_means = get_sentence_property_means(
                self.uds, data, t, use_ordinal=self.use_ordinal
            )

        (
            average_annotations,
            properties_to_indices,
            annotations_by_property,
            items_by_property,
            idx_to_item,
            confidences_by_property,
            annotators_by_property,
            unique_annotators_by_property,
        ) = self.annotation_func_by_type[t](
            data, confidences, property_means, self.device
        )

        gmm = gmm.fit(average_annotations)
        LOG.info(
            f"GMM average train LL for {n_components} components: {gmm.score(average_annotations)}"
        )

        # Probably shouldn't be returning all these things from a call to
        # "fit", but didn't want to have to separately call the annotation
        # getter function again
        return (
            gmm,
            average_annotations,
            properties_to_indices,
            annotations_by_property,
            items_by_property,
            idx_to_item,
            confidences_by_property,
            annotators_by_property,
            unique_annotators_by_property,
        )


class MultiviewMixtureModel(Module):
    def __init__(
        self,
        uds: UDSCorpus,
        random_seed: int = 42,
        use_ordinal: bool = False,
        device: str = "cpu",
    ):
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
        self.final_train_posteriors = None
        self.final_dev_posteriors = None
        self.train_idx_to_item = None
        self.dev_idx_to_item = None
        self.use_ordinal = use_ordinal
        self.device = device

    def _init_mus(
        self,
        t: Type,
        gmm_means: np.ndarray,
        props_to_indices: Dict[str, np.ndarray],
        n_components: int,
    ) -> None:
        mu_dict = {}
        if t == Type.RELATION:
            metadata = self.d_metadata
        else:
            metadata = self.s_metadata
        for subspace in SUBSPACES_BY_TYPE[t]:
            for p in metadata.properties(subspace):
                # Random effects for relation types handled below
                if "rel-" in p:
                    continue
                # Restrict genericity to either argument or predicate
                # depending on the type we're clustering on
                if (t == Type.EVENT and "arg" in p) or (
                    t == Type.PARTICIPANT and "pred" in p
                ):
                    continue

                # Most means are set based on the GMM means
                start, end = props_to_indices[p]
                mu = torch.FloatTensor(gmm_means[:, start:end])
                min_mean = torch.ones(mu.shape) * MIN_MEAN
                mu = torch.log(torch.where(mu > MIN_MEAN, mu, min_mean))
                mu_dict[p.replace(".", "-")] = Parameter(mu)

                # For conditional (hurdle model) properties, the Bernoulli
                # that indicates whether the property applies or not is simply
                # initialized to 0.5, out of laziness
                is_ordinal = metadata[subspace][p].value.is_ordered_categorical
                if self.use_ordinal and is_ordinal and p in CONDITIONAL_PROPERTIES:
                    mu_applies = Parameter(torch.log(torch.ones(n_components) * 0.5))
                    mu_dict[p.replace(".", "-") + "-applies"] = mu_applies

        if t == Type.RELATION:
            # Randomly initialize probabilities that
            #   1. The events' start points are locked to 0.
            #   2. The events' end points are locked to 100.
            #   3. The events' midpoints are locked to each other.
            # Note: I tried an initialization based on the GMM clusters,
            # but this led to radical overfitting.

            # The three dimensions of these distributions correspond
            # to the probabilities that 1) both points are locked; 2) only
            # e1's point is locked; 3) only e2's point is locked
            mu_dict["time-lock_start_mu"] = Parameter(
                torch.log(torch.softmax(torch.randn((n_components, 3)), -1))
            )
            mu_dict["time-lock_end_mu"] = Parameter(
                torch.log(torch.softmax(torch.randn((n_components, 3)), -1))
            )

            # The three dimensions of these distributions correspond to
            # the probabailities that 1) e1's midpoint equals e2's midpoint;
            # 2) e1's midpoint comes before e2's; 3) e2's comes before e1's.
            mu_dict["time-lock_mid_mu"] = Parameter(
                torch.log(torch.softmax(torch.randn((n_components, 3)), -1))
            )

        self.mus = ParameterDict(mu_dict).to(self.device)

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
        iterations: int = 10000,
        lr: float = 0.001,
        clip_min_ll=False,
        confidence_weighting=False,
        patience: int = 1,
        verbosity: int = 10,
    ) -> "MultiviewMixtureModel":
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        LOG.info(
            f"Fitting model on type {t.name} using {n_components} components on device {self.device}"
        )
        LOG.info("Fitting GMM...")
        train_confidences = self._get_annotator_ridits(data["train"], t)
        train_gmm = GMM(self.uds, use_ordinal=self.use_ordinal, device=self.device)
        (
            gmm,
            train_avg_annotations,
            train_properties_to_indices,
            train_annotations_by_property,
            train_items_by_property,
            self.train_idx_to_item,
            train_confidences_by_property,
            train_annotators_by_property,
            train_unique_annotators_by_property,
        ) = train_gmm.fit(data["train"], t, n_components, train_confidences)
        LOG.info("...GMM fitting complete")

        LOG.info("Loading dev data...")
        dev_confidences = self._get_annotator_ridits(data["dev"], t)

        # Determine the total number of train items annotated across all properties
        train_items = set()
        for train_item in train_items_by_property.values():
            train_items |= set(train_item.tolist())
        total_train_items = len(train_items)
        train_items = torch.LongTensor(list(train_items)).to(self.device)

        if t == Type.RELATION:
            dev_property_means = None
        else:
            dev_property_means = get_sentence_property_means(
                self.uds, data["dev"], t, self.use_ordinal
            )

        (
            dev_avg_annotations,
            dev_properties_to_indices,
            dev_annotations_by_property,
            dev_items_by_property,
            self.dev_idx_to_item,
            dev_confidences_by_property,
            dev_annotators_by_property,
            dev_unique_annotators_by_property,
        ) = train_gmm.annotation_func_by_type[t](
            data["dev"], dev_confidences, dev_property_means, device=self.device
        )

        # Determine the total number of dev items annotated across all properties
        dev_items = set()
        for dev_item in dev_items_by_property.values():
            dev_items |= set(dev_item.tolist())
        total_dev_items = len(dev_items)
        dev_items = torch.LongTensor(list(dev_items)).to(self.device)

        LOG.info("...Complete.")
        LOG.info(f"total train items: {total_train_items}")
        LOG.info(f"total dev items: {total_dev_items}")

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
        else:
            metadata = self.s_metadata

        # Initialize Likelihood module, property means,
        # annotator random effects, and component weights
        ll = self.type_to_likelihood[t](
            train_confidences,
            metadata,
            n_components,
            use_ordinal=self.use_ordinal,
            clip_min_ll=clip_min_ll,
            confidence_weighting=confidence_weighting,
            device=self.device,
        )
        self._init_mus(
            t, gmm.means_, train_properties_to_indices, n_components
        )
        self.random_effects = ll.random_effects
        self.component_weights = Parameter(
            torch.log(torch.FloatTensor(gmm.weights_)).to(self.device)
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        min_train_fixed_loss = float("inf")
        min_dev_fixed_loss = float("inf")
        iters_without_improvement = 0
        LOG.info(f"Beginning training for {iterations} epochs")
        for i in range(iterations):

            # training
            _, train_ll = ll(
                self.mus,
                train_annotations_by_property,
                train_items_by_property,
                train_annotators_by_property,
                train_confidences_by_property,
            )

            # per-type likelihoods for all items
            train_fixed_loss = train_ll

            # add in prior over components
            prior = exp_normalize(self.component_weights)[:, None]
            train_posteriors = train_fixed_loss[:, train_items] + prior

            # logsumexp over all components to get log-evidence for each item,
            # then mean-reduce
            train_fixed_loss = (
                -torch.logsumexp(train_posteriors, 0).sum() / total_train_items
            )

            # add in random loss, backprop, and take gradient step
            train_random_loss = ll.random_loss()
            train_loss = train_fixed_loss + train_random_loss
            train_loss.backward()
            optimizer.step()

            # train logging
            if i % verbosity == 0:
                LOG.info(
                    f"component weights: {torch.exp(exp_normalize(self.component_weights))}"
                )
                LOG.info(f"Epoch {i} train log prior: {self.component_weights.data}")
                LOG.info(f"Epoch {i} train log likelihood: {train_ll.mean(-1)}")
                LOG.info(
                    f"Epoch {i} train fixed loss: {np.round(train_fixed_loss.item(), 5)}"
                )
                LOG.info(
                    f"Epoch {i} train random loss: {np.round(train_random_loss.item(), 5)}"
                )

            # eval
            with torch.no_grad():
                _, dev_ll = ll(
                    self.mus,
                    dev_annotations_by_property,
                    dev_items_by_property,
                    dev_annotators_by_property,
                    dev_confidences_by_property,
                    train_unique_annotators_by_property,
                    dev_annotators_in_train,
                )

                # dev fixed loss computed the same way as train
                dev_fixed_loss = dev_ll
                dev_posteriors = dev_fixed_loss[:, dev_items] + prior
                dev_fixed_loss = (
                    -torch.logsumexp(dev_posteriors, 0).sum() / total_dev_items
                )

                if i % verbosity == 0:
                    LOG.info(
                        f"Epoch {i} dev fixed loss: {np.round(dev_fixed_loss.item(), 5)}"
                    )

                # stop early if no improvement in dev
                if dev_fixed_loss < min_dev_fixed_loss:
                    min_dev_fixed_loss = dev_fixed_loss
                    iters_without_improvement = 0
                else:
                    iters_without_improvement += 1

                # since we're doing full GD, there's no real sense in setting
                # patience to anything other than 1
                if iters_without_improvement == patience:
                    self.final_train_posteriors = train_posteriors
                    self.final_dev_posteriors = dev_posteriors
                    LOG.info(
                        f"No improvement in dev LL for model with {n_components} components after {patience} iterations. Stopping early."
                    )
                    LOG.info(
                        f"Final component weights (epoch {i}): {torch.exp(exp_normalize(self.component_weights))}"
                    )
                    LOG.info(
                        f"Final train fixed loss (epoch {i}): {np.round(train_fixed_loss.item(), 5)}"
                    )
                    LOG.info(
                        f"Final dev fixed loss (epoch {i}): {np.round(dev_fixed_loss.item(), 5)}"
                    )
                    return self.eval()

        LOG.info(f"Max iterations reached")
        LOG.info(
            f"Final component weights (epoch {i}): {torch.exp(exp_normalize(self.component_weights))}"
        )
        LOG.info(
            f"Final train fixed loss (epoch {i}): {np.round(train_fixed_loss.item(), 5)}"
        )
        LOG.info(
            f"Final dev fixed loss (epoch {i}): {np.round(dev_fixed_loss.item(), 5)}"
        )
        self.final_train_posteriors = train_posteriors
        self.final_dev_posteriors = dev_posteriors
        return self.eval()


def main(args):
    # Load UDS and initialize the mixture model
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    load_event_structure_annotations(uds)

    # Define train and dev splits
    t = STR_TO_TYPE[args.type.upper()]
    if t == Type.RELATION:
        train = sorted(
            set([graph.document_id for name, graph in uds.items() if "train" in name])
        )
        dev = sorted(
            set([graph.document_id for name, graph in uds.items() if "dev" in name])
        )
    else:
        train = [s for s in uds if "train" in s]
        dev = [s for s in uds if "dev" in s]
    data = {"train": train, "dev": dev}

    LOG.info(
        f"Fitting mixture model with all types in range {args.min_types} to {args.max_types}, inclusive"
    )
    model_root = args.model_name if args.model_name is not None else t.name
    for n_components in range(args.min_types, args.max_types + 1):
        # Initialize and fit the model
        mmm = MultiviewMixtureModel(
            uds, use_ordinal=args.use_ordinal, device=args.device
        )
        model_name = model_root + "-" + str(n_components) + ".pt"
        mmm = mmm.fit(
            data,
            t,
            n_components,
            clip_min_ll=args.clip_min_ll,
            confidence_weighting=args.weight_by_confidence,
            patience=args.patience,
        )

        # Save it
        save_model(mmm.state_dict(), args.model_dir, model_name)

        # Dump property means to file
        if args.dump_means:
            means_file = "-".join([model_root, str(n_components), "means"]) + ".csv"
            means_file = os.path.join(args.model_dir, means_file)
            dump_params(means_file, mmm.mus)

        if args.dump_posteriors:
            posteriors_file = (
                "-".join([model_root, str(n_components), "posteriors"]) + ".csv"
            )
            posteriors_file = os.path.join(args.model_dir, posteriors_file)
            dump_posteriors(
                posteriors_file,
                mmm.final_train_posteriors,
                mmm.train_idx_to_item,
                mmm.final_dev_posteriors,
                mmm.dev_idx_to_item,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, help="the type to cluster on")
    parser.add_argument(
        "min_types", type=int, help="minimum of range of numbers of types to try"
    )
    parser.add_argument(
        "max_types", type=int, help="maximum of range of numbers of types to try"
    )
    parser.add_argument(
        "--model_name", type=str, help="name for model checkpoint files"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/wgantt/event_type_induction/checkpoints/",
        help="path to directory where checkpoint files are to be saved",
    )
    parser.add_argument(
        "--use_ordinal",
        action="store_true",
        help="indicates whether ordinal variables should be treated as ordinal or as categorical",
    )
    parser.add_argument(
        "--dump_means", action="store_true", help="dump MMM property means to file",
    )
    parser.add_argument(
        "--dump_posteriors",
        action="store_true",
        help="dump per-item MMM (log) posteriors to file",
    )
    parser.add_argument(
        "--clip_min_ll",
        action="store_true",
        help="clip all likelihoods to a minimum value (currently 10e-5)",
    )
    parser.add_argument(
        "--weight_by_confidence",
        action="store_true",
        help="weight likelihoods by annotator confidence",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=1,
        help="number of epochs tolerated without dev improvement",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)
