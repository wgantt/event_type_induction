import torch

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from event_type_induction.constants import *
from overrides import overrides
from torch import Tensor, randn
from torch.distributions import Bernoulli, Categorical, MultivariateNormal
from torch.nn import Module, ModuleDict, Parameter, ParameterDict
from typing import Any, Dict, Set


class Likelihood(Module, metaclass=ABCMeta):
    """ABC for Event Type Induction Module likelihood computations"""

    def __init__(self, annotator_ids: Set[str], prop_attrs: Dict[str, Dict[str, int]]):
        super().__init__()
        self.annotator_ids = annotator_ids
        self.prop_attrs = prop_attrs
        self.prop_domains = prop_attrs.keys()
        (
            self.random_effects,
            self.random_effects_by_prop,
        ) = self._initialize_random_effects(annotator_ids)

    def _initialize_random_effects(self, annotator_ids: Dict[str, str]) -> ModuleDict:
        random_effects_by_annotator = defaultdict(ParameterDict)
        random_effects_by_prop = defaultdict(ParameterDict)
        for domain in self.prop_domains:
            for annotator in annotator_ids[domain]:
                for p in self.prop_attrs[domain].keys():
                    prop_name = "-".join([domain, p]).replace(".", "-")
                    prop_dim = self.prop_attrs[domain][p]["dim"]
                    random_shift = Parameter(torch.randn(prop_dim))
                    random_effects_by_annotator[annotator][prop_name] = random_shift
                    random_effects_by_prop[prop_name][annotator] = random_shift

        return (
            ModuleDict(random_effects_by_annotator),
            ModuleDict(random_effects_by_prop),
        )

    def _get_distribution(self, mu, random, prop_type):
        if prop_type == BINARY:
            return Bernoulli(torch.sigmoid(mu + random))
        else:
            return Categorical(torch.softmax(mu + random, -1))

    def compute_random_loss(self):
        """Computes log likelihood of annotator random effects

        Random effects terms are assumed to be distributed multivariate normal
        """
        loss = torch.FloatTensor([0.0])
        for prop, param_dict in self.random_effects_by_prop.items():
            # All random shift parameters for a particular property
            random = torch.stack([p.data for p in param_dict.values()])

            # Mean subtract
            random -= random.mean(0)

            # Compute covariance and inverse covariance
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

                        # TODO: handle None-valued UDS-EventStructure predicate
                        # annotations. I don't think there are other protocols
                        # liable to have None-valued annotations
                        if value is None:
                            continue

                        # TODO: Handle time document edge annotations, which are
                        # list-valued in the case of UDS-Time
                        elif isinstance(value, list):
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
        return super().forward(mus, annotation)
