import numpy as np
import torch

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from event_type_induction.constants import *
from overrides import overrides
from torch import Tensor, randn
from torch.distributions import Bernoulli, Categorical, MultivariateNormal
from torch.nn import Module, ModuleDict, Parameter, ParameterDict, ParameterList
from typing import Any, Dict, Tuple


class Likelihood(Module, metaclass=ABCMeta):
    def __init__(
        self,
        annotator_ids: Dict[str, str],
        annotator_conf: Dict[str, set],
        prop_attrs: Dict[str, Dict[str, int]],
    ):
        """ABC for Event Type Induction Module likelihood computations

        Not so sure this should be an ABC any more...

        TODO: replace annotator_ids with annotator confidences.
        """
        super().__init__()
        self.annotator_ids = annotator_ids
        self.annotator_conf = annotator_conf
        self.prop_attrs = prop_attrs
        self.prop_domains = prop_attrs.keys()

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

    def random_loss(self):
        """Computes log likelihood of annotator random effects

        Random effects terms are assumed to be distributed multivariate normal
        """
        loss = torch.FloatTensor([0.0])
        for prop, param_dict in self.random_effects_by_prop.items():

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
        # TODO: incorporate confidence
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

                        # Ignores none-valued EventStructure annotations and
                        # list-valued Time annotations, handled by override
                        if value is None or isinstance(value, list):
                            continue

                        # Grab the random intercept for the current annotator
                        random = self.random_effects[annotator][prop_name]

                        # Get the appropriate distribution
                        dist = self._get_distribution(mu, random, prop_type)

                        # Compute log likelihood
                        likelihood = dist.log_prob(torch.Tensor([value]))

                        # Add to likelihood-by-property
                        if p in likelihoods:
                            likelihoods[p] += likelihood
                        else:
                            likelihoods[p] = likelihood
        return likelihoods


class PredicateNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(self, annotator_ids: Dict[str, str], annotator_conf: Dict[str, set]):
        super().__init__(annotator_ids, annotator_conf, PREDICATE_ANNOTATION_ATTRIBUTES)

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        return super().forward(mus, annotation)


class ArgumentNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(self, annotator_ids: Dict[str, str], annotator_conf: Dict[str, set]):
        super().__init__(annotator_ids, annotator_conf, ARGUMENT_ANNOTATION_ATTRIBUTES)
        (
            self.random_effects,
            self.random_effects_by_prop,
        ) = self._initialize_random_effects(self.annotator_ids)

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        # Likelihood computation is similarly straightforward
        return super().forward(mus, annotation)


class SemanticsEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(self, annotator_ids: Dict[str, str], annotator_conf: Dict[str, set]):
        super().__init__(
            annotator_ids, annotator_conf, SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES
        )

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        return super().forward(mus, annotation)


class DocumentEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(self, annotator_ids: Dict[str, str], annotator_conf: Dict[str, set]):
        super().__init__(
            annotator_ids, annotator_conf, DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES
        )

    @overrides
    def forward(
        self, mus: ParameterDict, cov: ParameterList, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        pass

    @overrides
    def _initialize_random_effects(self, annotator_ids: Dict[str, str]):
        random_effects_by_annotator = defaultdict(ParameterDict)
        random_effects_by_prop = defaultdict(ParameterDict)
        for domain in self.prop_domains:
            for annotator in annotator_ids[domain]:
                for p in self.prop_attrs[domain].keys():
                    prop_name = "-".join([domain, p]).replace(".", "-")
                    prop_dim = self.prop_attrs[domain][p]["dim"]
                    if domain == "mereology":
                        # TODO: we may want a single random effect for both
                        # directions of mereological containment
                        random_shift = Parameter(torch.randn(prop_dim))
                    elif domain == "time":
                        # Separate per-annotator shift terms for each possible
                        # mereological relation
                        random_shift = Parameter(
                            torch.randn(N_MEREOLOGICAL_RELATIONS, prop_dim)
                        )
                    else:
                        raise ValueError(
                            "Unrecognized domain for document edge annotations!"
                        )
        return (
            ModuleDict(random_effects_by_annotator),
            ModuleDict(random_effects_by_prop),
        )
