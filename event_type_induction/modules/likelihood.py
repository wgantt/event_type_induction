import numpy as np
import torch

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from event_type_induction.constants import *
from overrides import overrides
from torch import Tensor, randn
from torch.distributions import Bernoulli, Categorical, MultivariateNormal
from torch.nn import Module, ModuleDict, Parameter, ParameterDict, ParameterList
from typing import Any, Dict, Set, Tuple


class Likelihood(Module, metaclass=ABCMeta):
    def __init__(
        self,
        property_subspaces: Set[str],
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
    ):
        """ABC for Event Type Induction Module likelihood computations

        Parameters
        ----------
        property_subspaces
            the set of UDS subspaces associated with the likelihood
        annotator_confidences
            a dictionary mapping annotators to dictionaries that map
            raw confidence scores to ridit-scored ones for that annotator
        metadata
            the annotation metadata for all subspaces in property_subspaces
        """
        super().__init__()
        self.property_subspaces = property_subspaces
        self.annotator_confidences = annotator_confidences
        self.metadata = metadata

    def _initialize_random_effects(self) -> ModuleDict:
        """Initialize annotator random effects for each property"""
        random_effects = defaultdict(ParameterDict)
        for subspace in self.property_subspaces:
            for annotator in self.metadata.annotators(subspace):
                for p in self.metadata.properties(subspace):

                    # Determine property dimension
                    prop_dim = self._get_prop_dim(subspace, p)

                    # Single random intercept term per annotator per property
                    random_shift = Parameter(torch.randn(prop_dim))

                    # PyTorch doesn't allow periods in parameter names
                    prop_name = p.replace(".", "-")

                    # Store parameters both by annnotator and by property
                    # for easy
                    random_effects[prop_name][annotator] = random_shift

        self.random_effects = ModuleDict(random_effects)

    def _get_distribution(self, mu, random):
        """Generates an appropriate distribution given a mean and random effect"""
        if len(mu.shape) == 1:
            return Bernoulli(torch.exp(mu + random))
        else:
            return Categorical(torch.softmax(mu + random, -1))

    def _get_prop_dim(self, subspace, prop):
        """Determines the appropriate dimension for a UDS property"""
        prop_data = self.metadata.metadata[subspace][prop].value
        if prop_data.is_categorical:
            n_categories = len(prop_data.categories)
            # conditional categorical properties require an
            # additional dimension for the "does not apply" case
            if prop in CONDITIONAL_PROPERTIES:
                return n_categories + 1
            # non-conditional, ordinal categorical properties
            if prop_data.is_ordered_categorical:
                return n_categories
            # non-conditional, binary categorical properties
            else:
                return n_categories - 1
        # currently no non-categorical properties in UDS
        else:
            raise ValueError(
                f"Non-categorical property {property} found in subspace {subspace}"
            )

    def random_loss(self):
        """Computes log likelihood of annotator random effects

        Random effects terms are assumed to be distributed MV normal
        """
        loss = torch.FloatTensor([0.0])

        # Loss computed by property
        for prop, param_dict in self.random_effects.items():

            # All random shift parameters for a particular property
            random = torch.stack([p.data for p in param_dict.values()])

            # Mean subtract
            random -= random.mean(0)

            # Estimate covariance
            cov = torch.matmul(torch.transpose(random, 0, 1), random) / (
                len(random) - 1
            )

            # Compute loss
            loss += MultivariateNormal(random, cov).log_prob(random).mean(0)

        return loss

    @abstractmethod
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        """Compute the likelihood for a single annotation for a set of subspaces"""
        likelihoods = {}
        for subspace in self.property_subspaces:
            if subspace in annotation:
                for p in annotation[subspace]:
                    prop_name = p.replace(".", "-")

                    # The mean for the current property
                    mu = mus[prop_name]

                    # Compute log likelihood for each annotator in the annotation
                    # (for training, this should execute just once, as train data
                    # has only a single annotation per node)
                    for annotator, value in annotation[subspace][p]["value"].items():

                        # Annotator indicated the property doesn't apply;
                        # select last category in the categorical distribution
                        if value is None:
                            value = mu.shape[-1] - 1

                        # Obtain category index for string-valued annotations
                        elif isinstance(value, str):
                            value = self.str_to_category[value]

                        # Grab the random intercept for the current annotator
                        random = self.random_effects[prop_name][annotator]

                        # Get the appropriate distribution
                        dist = self._get_distribution(mu, random)

                        # Compute log likelihood
                        likelihood = dist.log_prob(torch.FloatTensor([value]))

                        # Get annotator confidence
                        conf = annotation[subspace][p]["confidence"][annotator]
                        ridit_conf = self.annotator_confidences[annotator]
                        if (
                            ridit_conf is None
                            or ridit_conf.get(conf) is None
                            or ridit_conf[conf] < 0
                        ):
                            ridit_conf = 1
                        else:
                            ridit_conf = ridit_conf.get(conf, 1)

                        # Add to likelihood-by-property
                        if p in likelihoods:
                            likelihoods[p] += ridit_conf * likelihood
                        else:
                            likelihoods[p] = ridit_conf * likelihood
        return likelihoods


class PredicateNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
    ):
        super().__init__(PREDICATE_NODE_SUBSPACES, annotator_confidences, metadata)
        self._initialize_random_effects()
        self.str_to_category = {
            cat: idx
            for idx, cat in enumerate(metadata["time"]["duration"].value.categories)
        }

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        return super().forward(mus, annotation)


class ArgumentNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
    ):
        super().__init__(ARGUMENT_NODE_SUBSPACES, annotator_confidences, metadata)
        self._initialize_random_effects()

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        return super().forward(mus, annotation)


class SemanticsEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
    ):
        super().__init__(SEMANTICS_EDGE_SUBSPACES, annotator_confidences, metadata)
        self._initialize_random_effects()

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        """Compute the likelihood for a single annotation for a set of subspaces"""
        likelihoods = {}
        for subspace in self.property_subspaces:
            if subspace in annotation:
                for p in annotation[subspace]:
                    prop_name = p.replace(".", "-")

                    # The mean for the current property
                    mu = mus[prop_name]

                    # Compute log likelihood for each annotator in the annotation
                    # (for training, this should execute just once, as train data
                    # has only a single annotation per node)
                    for annotator, value in annotation[subspace][p]["value"].items():

                        # Get annotator confidence
                        conf = annotation[subspace][p]["confidence"][annotator]

                        # Annotation value has to be specially determined for protoroles
                        if subspace == "protoroles":

                            # Annotator "confidence" actually determines whether the
                            # property applies or not
                            if conf == 0:
                                # Property doesn't apply; select last category
                                value = mu.shape[-1] - 1
                            # Otherwise, we use the value as given

                            # No confidence value for protoroles; default to 1
                            ridit_conf = 1

                        # All other properties (currently, just distributivity) use
                        # confidence as given
                        else:
                            ridit_conf = self.annotator_confidences[annotator].get(
                                conf, 1
                            )

                        # Grab the random intercept for the current annotator
                        random = self.random_effects[prop_name][annotator]

                        # Get the appropriate distribution
                        dist = self._get_distribution(mu, random)

                        # Compute log likelihood
                        likelihood = dist.log_prob(torch.FloatTensor([value]))

                        # Add to likelihood-by-property
                        if p in likelihoods:
                            likelihoods[p] += ridit_conf * likelihood
                        else:
                            likelihoods[p] = ridit_conf * likelihood
        return likelihoods


class DocumentEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
    ):
        super().__init__(DOCUMENT_EDGE_SUBSPACES, annotator_confidences, metadata)
        self._initialize_random_effects()

    @overrides
    def _initialize_random_effects(self) -> ModuleDict:
        random_effects = defaultdict(ParameterDict)
        for subspace in self.property_subspaces - {"time"}:
            for annotator in self.metadata.annotators(subspace):
                for p in self.metadata.properties(subspace):

                    # Determine property dimension
                    prop_dim = self._get_prop_dim(subspace, p)

                    # Single random intercept term per annotator per property
                    random_shift = Parameter(torch.randn(prop_dim))

                    # PyTorch doesn't allow periods in parameter names
                    prop_name = p.replace(".", "-")

                    # Store parameters both by annnotator and by property
                    # for easy
                    random_effects[prop_name][annotator] = random_shift

        # Each of the two start- and endpoints in temporal relations are
        # treated as separate properties, but we model them together as
        # a single 4-tuple property
        for annotator in self.metadata.annotators("time"):
            # TODO: maybe initialize with greater variance?
            random_effects["time"][annotator] = Parameter(torch.randn(4))

        self.random_effects = ModuleDict(random_effects)

    @overrides
    def forward(
        self, mus: ParameterDict, cov: Parameter, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        likelihoods = {}
        temp_rels = defaultdict(list)
        temp_rel_confs = defaultdict(int)
        for subspace in self.property_subspaces:
            if subspace in annotation:
                for p in annotation[subspace]:
                    prop_name = p.replace(".", "-")

                    # Compute log likelihood for each annotator in the annotation
                    # (for training, this should execute just once, as train data
                    # has only a single annotation per node)
                    for annotator, value in annotation[subspace][p]["value"].items():

                        if subspace == "time":
                            temp_rels[annotator].append(value)
                            conf = annotation[subspace][p]["confidence"][annotator]
                            temp_rel_confs[annotator] = self.annotator_confidences[
                                annotator
                            ].get(conf, 1)
                            continue

                        # The mean for the current property
                        mu = mus[prop_name]

                        # Grab the random intercept for the current annotator
                        random = self.random_effects[prop_name][annotator]

                        # Get the appropriate distribution
                        dist = self._get_distribution(mu, random)

                        # Compute log likelihood
                        likelihood = dist.log_prob(torch.FloatTensor([value]))

                        # Get annotator confidence
                        conf = annotation[subspace][p]["confidence"][annotator]
                        ridit_conf = self.annotator_confidences[annotator].get(conf, 1)

                        # Add to likelihood-by-property
                        if p in likelihoods:
                            likelihoods[p] += ridit_conf * likelihood
                        else:
                            likelihoods[p] = ridit_conf * likelihood

        # Can only compute the likelihood for temporal relations once
        # we have all four start- and endpoints for each annotator.
        # We assume the endpoints are distrubted MV normal.
        mu = mus["time"]
        likelihoods["time"] = torch.zeros(mu.shape[0])
        for a, rels in temp_rels.items():
            random = self.random_effects["time"][a]
            likelihood = MultivariateNormal(mu + random, cov).log_prob(torch.Tensor(rels))
            likelihoods["time"] += temp_rel_confs[a] * likelihood
        return likelihoods
