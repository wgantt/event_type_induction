import torch
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from event_type_induction.constants import *
from event_type_induction.utils import exp_normalize
from overrides import overrides
from torch import Tensor, FloatTensor, randn, log
from torch.distributions import Bernoulli, Categorical, MultivariateNormal
from torch.nn import Module, ModuleDict, Parameter, ParameterDict, ParameterList
from typing import Any, Dict, Set, Tuple, Union


class Likelihood(Module, metaclass=ABCMeta):
    def __init__(
        self,
        property_subspaces: Set[str],
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        device: str = "cpu",
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
        self.device = torch.device(device)
        self.to(device)

    def _initialize_random_effects(self) -> ModuleDict:
        """Initialize annotator random effects for each property"""
        random_effects = defaultdict(ParameterDict)
        for subspace in sorted(self.property_subspaces):
            for annotator in sorted(self.metadata.annotators(subspace)):
                for p in sorted(self.metadata.properties(subspace)):

                    # Determine property dimension
                    prop_dim = self._get_prop_dim(subspace, p)

                    # Single random intercept term per annotator per property
                    random_shift = Parameter(torch.randn(prop_dim))

                    # PyTorch doesn't allow periods in parameter names
                    prop_name = p.replace(".", "-")

                    # Store parameters both by annnotator and by property
                    # for easy
                    random_effects[prop_name][annotator] = random_shift

        self.random_effects = ModuleDict(random_effects).to(self.device)

    def _get_distribution(self, mu, random):
        """Generates an appropriate distribution given a mean and random effect"""
        if len(mu.shape) == 1:
            return Bernoulli(torch.sigmoid(torch.exp(mu) + random))
        else:
            return Categorical(torch.softmax(torch.exp(mu) + random, -1))

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
        loss = FloatTensor([0.0]).to(self.device)

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
            invcov = torch.inverse(cov)

            # Compute loss
            loss += (
                torch.matmul(
                    torch.matmul(random.unsqueeze(1), invcov),
                    torch.transpose(random.unsqueeze(1), 1, 2),
                )
                .mean(0)
                .squeeze()
            )

        return loss

    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Compute the likelihood for a single annotation for a set of subspaces"""
        likelihoods = {}
        total_ll = torch.FloatTensor([0.0]).to(self.device)
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

                        # Compute log likelihood (clipping to prevent underflow)
                        dist = self._get_distribution(mu, random)
                        ll = dist.log_prob(FloatTensor([value]).to(self.device))
                        min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(
                            self.device
                        )
                        ll = torch.where(
                            ll > log(Tensor([MIN_LIKELIHOOD]).to(self.device)),
                            ll,
                            min_ll,
                        )

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
                            likelihoods[p] += ridit_conf * ll
                        else:
                            likelihoods[p] = ridit_conf * ll
                    total_ll += torch.logsumexp(likelihoods[p], 0)

        return likelihoods, total_ll


class PredicateNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        device: str = "cpu",
    ):
        super().__init__(
            PREDICATE_NODE_SUBSPACES, annotator_confidences, metadata, device=device
        )
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
        device: str = "cpu",
    ):
        super().__init__(
            ARGUMENT_NODE_SUBSPACES, annotator_confidences, metadata, device=device
        )
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
        device: str = "cpu",
    ):
        super().__init__(
            SEMANTICS_EDGE_SUBSPACES, annotator_confidences, metadata, device=device
        )
        self._initialize_random_effects()

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Compute the likelihood for a single annotation for a set of subspaces"""
        likelihoods = {}
        total_ll = torch.FloatTensor([0.0]).to(self.device)
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

                            # Annotator "confidence" for protoroles actually
                            # determines whether the property applies or not
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

                        # Compute log likelihood (clipping to prevent underflow)
                        dist = self._get_distribution(mu, random)
                        ll = dist.log_prob(FloatTensor([value]).to(self.device))
                        min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(
                            self.device
                        )
                        ll = torch.where(
                            ll > log(Tensor([MIN_LIKELIHOOD]).to(self.device)),
                            ll,
                            min_ll,
                        )

                        # Add to likelihood-by-property
                        if p in likelihoods:
                            likelihoods[p] += ridit_conf * ll
                        else:
                            likelihoods[p] = ridit_conf * ll
                    total_ll += torch.logsumexp(likelihoods[p], 0)
        return likelihoods, total_ll


class DocumentEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        device: str = "cpu",
    ):
        super().__init__(
            DOCUMENT_EDGE_SUBSPACES, annotator_confidences, metadata, device=device
        )
        self._initialize_random_effects()

    @overrides
    def _initialize_random_effects(self) -> ModuleDict:
        random_effects = defaultdict(ParameterDict)
        for subspace in sorted(self.property_subspaces - {"time"}):
            for annotator in sorted(self.metadata.annotators(subspace)):
                for p in sorted(self.metadata.properties(subspace)):

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
        for annotator in sorted(self.metadata.annotators("time")):
            # TODO: maybe initialize with greater variance?
            random_effects["time"][annotator] = Parameter(torch.randn(4))

        self.random_effects = ModuleDict(random_effects).to(self.device)

    @overrides
    def forward(
        self, mus: ParameterDict, cov: Parameter, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        likelihoods = {}
        total_ll = torch.FloatTensor([0.0])
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

                        # Compute log-likelihood (clipping to prevent underflow)
                        dist = self._get_distribution(mu, random)
                        ll = dist.log_prob(FloatTensor([value]).to(self.device))
                        min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(
                            self.device
                        )
                        ll = torch.where(
                            ll > log(Tensor([MIN_LIKELIHOOD]).to(self.device)),
                            ll,
                            min_ll,
                        )

                        # Get annotator confidence
                        conf = annotation[subspace][p]["confidence"][annotator]
                        ridit_conf = self.annotator_confidences[annotator].get(conf, 1)

                        # Add to likelihood-by-property
                        if p in likelihoods:
                            likelihoods[p] += ridit_conf * ll
                        else:
                            likelihoods[p] = ridit_conf * ll

                    if subspace != "time":
                        total_ll += torch.logsumexp(likelihoods[p], 0)

        # Can only compute the likelihood for temporal relations once
        # we have all four start- and endpoints for each annotator.
        # We assume the endpoints are distrubted MV normal.
        mu = mus["time"]
        likelihoods["time"] = torch.zeros(mu.shape[0]).to(self.device)
        for a, rels in temp_rels.items():
            # Handle annoying fact that some of the temporal relations
            # annotations have massively negative values (this needs to
            # be fixed in the annotations themselves)
            if any([r < 0 for r in rels]):
                continue

            # Manually compute unnormalized log likelihood.
            # Seemingly can't use MultivariateNormal.log_prob b/c
            # it leads to singular matrix errors during backprop
            random = self.random_effects["time"][a]
            invcov = torch.inverse(cov)
            x_minus_mu = FloatTensor(rels).to(self.device) - (mu + random)
            ll = -torch.matmul(
                torch.matmul(x_minus_mu.unsqueeze(1), invcov),
                torch.transpose(x_minus_mu.unsqueeze(1), 1, 2),
            ).squeeze()

            # Normalize and clip. This is currently causing NaNs in the
            # random loss for reasons I don't yet understand
            min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(self.device)
            ll = torch.where(
                ll > log(Tensor([MIN_LIKELIHOOD]).to(self.device)), ll, min_ll
            )
            likelihoods["time"] += temp_rel_confs[a] * ll
        # TODO: check whether this has the appropriate sign
        total_ll += torch.logsumexp(likelihoods["time"], 0)

        return likelihoods, total_ll
