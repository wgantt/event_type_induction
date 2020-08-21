import torch

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from event_type_induction.constants import *
from overrides import overrides
from torch import Tensor, randn
from torch.distributions import Bernoulli, Categorical
from torch.nn import Module, ModuleDict, Parameter, ParameterDict
from typing import Any, Dict, Set


class Likelihood(Module, metaclass=ABCMeta):
    """ABC for Event Type Induction Module likelihood computations"""

    def __init__(self, annotator_ids: Set[str], prop_attrs: Dict[str, Dict[str, int]]):
        super().__init__()
        self.annotator_ids = annotator_ids
        self.prop_attrs = prop_attrs
        self.prop_domains = prop_attrs.keys()
        self.random_effects = self._initialize_random_effects(annotator_ids)

    def _initialize_random_effects(self, annotator_ids: Set[str]) -> ModuleDict:
        random_effects = defaultdict(ParameterDict)
        for annotator in annotator_ids:
            for domain in self.prop_domains:
                for p in self.prop_attrs[domain].keys():
                    prop_name = "-".join([domain, p]).replace(".", "-")
                    prop_dim = self.prop_attrs[domain][p]["dim"]
                    random_effects[annotator][prop_name] = Parameter(
                        torch.randn(prop_dim)
                    )

        return ModuleDict(random_effects)

    def _get_distribution(self, mu, random, prop_type):
        if prop_type == BINARY:
            return Bernoulli(torch.sigmoid(mu + random))
        else:
            return Categorical(torch.softmax(mu + random, -1))

    @abstractmethod
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        likelihoods = {}
        for domain, props in annotation.items():
            if domain in self.prop_domains:
                for p in props:

                    # Determine the property type
                    prop_type = self.prop_attrs[domain][p]["type"]
                    prop_name = "-".join([domain, p]).replace(".", "-")

                    # The mean for the current property, for each event type
                    mu = mus[prop_name]

                    # Compute log likelihood for each annotator (for training,
                    # this should execute just once, as train data has only a
                    # single annotation)
                    for annotator, value in props[p]["value"].items():

                        # TODO: handle None-valued UDS-EventStructure
                        # annotations. I don't think there are other protocols
                        # liable to have None-valued annotations
                        if value is None:
                            continue

                        # Grab the random intercept for the current annotator
                        random = self.random_effects[annotator][prop_name]

                        # Determine the appropriate distribution for the
                        # current property type
                        dist = self._get_distribution(mu, random, prop_type)

                        # Compute log likelihood
                        likelihood = dist.log_prob(torch.Tensor([value])).squeeze()
                        if p in likelihoods:
                            likelihoods[p] += likelihood
                        else:
                            likelihoods[p] = likelihood
        return likelihoods


class PredicateNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(self, annotator_ids: Set[str]):
        super().__init__(annotator_ids, PREDICATE_ANNOTATION_ATTRIBUTES)

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        return super().forward(mus, annotation)


class ArgumentNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(self, annotator_ids: Set[str]):
        super().__init__(annotator_ids, ARGUMENT_ANNOTATION_ATTRIBUTES)

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        return super().forward(mus, annotation)


class SemanticsEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(self, annotator_ids: Set[str]):
        super().__init__(annotator_ids, SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES)

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        return super().forward(mus, annotation)


class DocumentEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(self, annotator_ids: Set[str]):
        super().__init__(annotator_ids, DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES)

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        pass
