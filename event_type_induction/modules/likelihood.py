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
    def __init__(self, annotator_ids: Set[str], prop_attrs: Dict[str, Dict[str, int]]):
        """ABC for Event Type Induction Module likelihood computations

        Not so sure this should be an ABC any more...
        """
        super().__init__()
        self.annotator_ids = annotator_ids
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

    def _initialize_ordinal_random_effects() -> Parameter:
        """Initialize a random effects parameter for ordinal values

        This is used for all binary properties, since we convert them to
        ordinal values. Ordinal random effects determine the cutpoints.
        """
        return Parameter(torch.randn(ORDINAL_RANDOM_EFFECTS_SIZE))

    def _get_ordinal_likelihood(
        self, binary_value, mu, random, confidence
    ) -> torch.FloatTensor:
        """Computes log probability of a given ordinal response

        TODO: verify correctness

        Parameters
        ----------
        binary_value
            The binary value whose log probability is to be returned
        mu
            The expected value of the per-property logistic distribution
        random
            Per-annotator random effects indicating the distances between
            cutpoints
        confidence
            The confidence score associated with the value
        """

        # Convert the binary value to an ordinal one (TODO: this is almost
        # certainly incorrect).
        ordinal_value = (
            np.power(-1, binary_value) * confidence
        ) + BINARY_TO_ORDINAL_SHIFT

        # The distances between cutpoints
        distances = torch.square(random)

        # The cutpoints, computed as the cumulative sum of the distances
        cutpoints = torch.cumsum(distances)

        # Probabilities that the ordinal response <= i
        probs = torch.sigmoid(cutpoints - mu)

        # Compute the probability that the response == i as the difference
        # between cumulative probabilities
        probs_high = torch.cat([probs, torch.Tensor([1.0])])
        probs_low = torch.cat([torch.Tensor([0.0]), probs])
        return Categorical(probs_high - probs_low).log_prob(
            torch.FloatTensor([ordinal_value])
        )

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

                        # Grab the random intercept for the current annotator
                        random = self.random_effects[annotator][prop_name]

                        # Compute log likelihood; binary properties are
                        # converted to ordinal values based on confidence
                        # scores
                        if prop_type == BINARY:
                            confidence = props[p]["confidence"][annotator]
                            likelihood = self._get_ordinal_likelihood(
                                value, mu, random, confidence
                            )
                        elif prop_type == NOMINAL:
                            likelihood = self._get_nominal_likelihood(
                                value, mu, random, confidence
                            )
                        else:
                            likelihood = 0

                        # Add to likelihood-by-property
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
        # No fancy things we need to do for argument annotation attributes,
        # as all properties are binary and none are contingent. Thus, we
        # can invoke default random effects method directly
        (
            self.random_effects,
            self.random_effects_by_prop,
        ) = self._initialize_random_effects(annotator_ids)

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        # Likelihood computation is similarly straightforward
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
        self, mus: ParameterDict, cov: ParameterList, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        mer_relation_type = MereologyRelation.UNRELATED
        # if "mereology" in annotation:
        #     p1_contains_p2 = annotation["mereology"]["containment.p1_contains_p2"]
        #     p2_contains_p1 = annotation["mereology"]["containment.p2_contains_p1"]
        #     mer_relation_type = MEREOLOGY_RELATION[p1_contains_p2][p2_contains_p1]

        #     # TODO: compute mereology likelihood
        # if "time" in annotation:
        #     pass

    @overrides
    def _initialize_random_effects(self, annotator_ids: Set[str]):
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
