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
        """ABC for Event Type Induction Module likelihood computations"""
        super().__init__()
        self.property_subspaces = property_subspaces
        self.annotator_confidences = annotator_confidences
        self.metadata = metadata

    def _initialize_random_effects(self) -> ModuleDict:
        random_effects = defaultdict(ParameterDict)
        for subspace in self.property_subspaces:
            for annotator in self.metadata.annotators(subspace):
                for p in self.metadata.properties(subspace):

                    # Determine property dimension
                    prop_dim = self._get_prop_dim(subspace, p)

                    # Single random intercept term per annotator per property
                    random_shift = Parameter(torch.randn(prop_dim))

                    # PyTorch doesn't allow periods in parameter names
                    prop_name = "-".join([subspace, p]).replace(".", "-")

                    # Store parameters both by annnotator and by property
                    # for easy
                    random_effects[prop_name][annotator] = random_shift

        self.random_effects = ModuleDict(random_effects)

    def _get_distribution(self, mu, random):
        if len(mu) == 1:
            return Bernoulli(torch.sigmoid(mu + random))
        else:
            return Categorical(torch.softmax(mu + random, -1))

    def _get_prop_dim(self, subspace, prop):
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

        Random effects terms are assumed to be distributed multivariate normal
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
        likelihoods = {}
        for subspace, props in annotation.items():
            if subspace in self.property_subspaces:
                for p in props:
                    prop_name = "-".join([subspace, p]).replace(".", "-")

                    # The mean for the current property, for each event type
                    mu = mus[prop_name]

                    # Compute log likelihood for each annotator (for training,
                    # this should execute just once, as train data has only a
                    # single annotation)
                    for annotator, value in props[p]["value"].items():

                        # Ignores none-valued EventStructure annotations and
                        # list-valued Time annotations, handled by override
                        if value is None or isinstance(value, list):
                            continue

                        # Grab the random intercept for the current annotator
                        random = self.random_effects[annotator][prop_name]

                        # Get the appropriate distribution
                        dist = self._get_distribution(mu, random)

                        # Compute log likelihood
                        likelihood = dist.log_prob(torch.Tensor([value]))

                        # Add to likelihood-by-property
                        # TODO: key on prop_name instead of p?
                        if p in likelihoods:
                            likelihoods[p] += likelihood
                        else:
                            likelihoods[p] = likelihood
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
        # Likelihood computation is similarly straightforward
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
        return super().forward(mus, annotation)


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
    def forward(
        self, mus: ParameterDict, cov: ParameterList, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        pass
