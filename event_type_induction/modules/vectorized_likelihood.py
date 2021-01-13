import torch
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from event_type_induction.constants import *
from event_type_induction.utils import exp_normalize
from itertools import combinations
from overrides import overrides
from torch import Tensor, FloatTensor, randn, log
from torch.distributions import (
    Bernoulli,
    Categorical,
    Normal,
    MultivariateNormal,
    Uniform,
    Dirichlet,
)
from torch.nn import Module, Parameter, ParameterDict, ParameterList
from typing import Any, Dict, Set, Tuple, Union


class Likelihood(Module, metaclass=ABCMeta):
    def __init__(
        self,
        property_subspaces: Set[str],
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
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
        n_components
            the number of components (types) for this likelihood
        device
            the device to use
        """
        super().__init__()
        self.property_subspaces = property_subspaces
        self.annotator_confidences = annotator_confidences
        self.metadata = metadata
        self.n_components = n_components
        self.device = torch.device(device)
        self.to(device)

    def _initialize_random_effects(self) -> ParameterDict:
        """Initialize annotator random effects for each property"""
        random_effects = {}
        for subspace in sorted(self.property_subspaces):
            subspace_annotators = self.metadata.annotators(subspace)
            max_annotator_idx = max(
                [int(s.split("-")[-1]) for s in subspace_annotators]
            )
            num_annotators = max(len(subspace_annotators), max_annotator_idx + 1)
            for p in sorted(self.metadata.properties(subspace)):

                # Determine property dimension
                prop_dim = self._get_prop_dim(subspace, p)

                # Single random intercept term per annotator per property
                random_shift = Parameter(torch.randn(num_annotators, prop_dim))
                prop_name = p.replace(".", "-")
                random_effects[prop_name] = random_shift

        self.random_effects = ParameterDict(random_effects).to(self.device)

    def _get_distribution(self, mu, random):
        """Generates an appropriate distribution given a mean and random effect"""
        if mu.shape[-1] == 1:
            return Bernoulli(torch.sigmoid(torch.exp(mu) + random).squeeze())
        else:
            return Categorical(torch.softmax(torch.exp(mu) + random, -1).squeeze())

    def _get_prop_dim(self, subspace, prop):
        """Determines the appropriate dimension for a UDS property"""
        # TODO: replace with version in utils
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
        for prop, random in self.random_effects.items():

            normed = random - random.mean(0)
            # Estimate covariance
            cov = torch.matmul(torch.transpose(normed, 0, 1), normed) / (
                len(random) - 1
            )
            invcov = torch.inverse(cov)

            # Compute loss
            loss += (
                torch.matmul(
                    torch.matmul(normed.unsqueeze(1), invcov),
                    torch.transpose(normed.unsqueeze(1), 1, 2),
                )
                .mean(0)
                .squeeze()
            )

        return loss

    def forward(
        self,
        mus: ParameterDict,
        annotations: Dict[str, torch.Tensor],
        annotators: Dict[str, torch.Tensor],
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ):
        """Compute the likelihood for a single annotation for a set of subspaces"""
        likelihoods = {}
        for p, anno in annotations.items():
            prop_name = p.replace(".", "-")

            # The mean for the current property
            mu = mus[prop_name].unsqueeze(1)

            # The ridit-scored confidences for these annotations
            confidence = confidences[p]

            # Grab the random intercept for the current annotator
            if dev_annotators_in_train:
                # If evaluating on dev, we don't have tuned random effects for
                # annotators who weren't seen during training, so we use the
                # mean random effect across train set annotators.
                train_random_effects = self.random_effects[prop_name][
                    train_annotators[p], :
                ]
                mean_train_random_effect = train_random_effects.mean(0)
                random = self.random_effects[prop_name][annotators[p], :]
                random[~dev_annotators_in_train[p]] = mean_train_random_effect
            else:
                random = self.random_effects[prop_name][annotators[p], :].unsqueeze(0)

            # Compute log likelihood (clipping to prevent underflow)
            dist = self._get_distribution(mu, random)
            ll = dist.log_prob(anno).to(self.device)
            min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(self.device)
            ll = torch.where(ll > min_ll, ll, min_ll,)

            # Accumulate likelihood
            if p in likelihoods:
                likelihoods[p] += confidence * ll
                total_ll += likelihoods[p]
            else:
                likelihoods[p] = confidence * ll
                total_ll = likelihoods[p]

        return likelihoods, total_ll


class PredicateNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        device: str = "cpu",
    ):
        super().__init__(
            PREDICATE_NODE_SUBSPACES,
            annotator_confidences,
            metadata,
            n_components,
            device=device,
        )
        self._initialize_random_effects()
        self.str_to_category = {
            cat: idx
            for idx, cat in enumerate(metadata["time"]["duration"].value.categories)
        }

    @overrides
    def forward(
        self,
        mus: ParameterDict,
        annotations: Dict[str, torch.Tensor],
        annotators: torch.Tensor,
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ) -> Dict[str, Tensor]:
        return super().forward(
            mus,
            annotations,
            annotators,
            confidences,
            train_annotators,
            dev_annotators_in_train,
            covs,
        )


class ArgumentNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        device: str = "cpu",
    ):
        super().__init__(
            ARGUMENT_NODE_SUBSPACES,
            annotator_confidences,
            metadata,
            n_components,
            device=device,
        )
        self._initialize_random_effects()

    @overrides
    def forward(
        self,
        mus: ParameterDict,
        annotations: Dict[str, torch.Tensor],
        annotators: torch.Tensor,
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ) -> Dict[str, Tensor]:
        return super().forward(
            mus,
            annotations,
            annotators,
            confidences,
            train_annotators,
            dev_annotators_in_train,
            covs,
        )


class SemanticsEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        device: str = "cpu",
    ):
        super().__init__(
            SEMANTICS_EDGE_SUBSPACES,
            annotator_confidences,
            metadata,
            n_components,
            device=device,
        )
        self._initialize_random_effects()

    @overrides
    def forward(
        self,
        mus: ParameterDict,
        annotations: Dict[str, torch.Tensor],
        annotators: torch.Tensor,
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ) -> Union[Dict[str, Tensor], Tensor]:
        return super().forward(
            mus,
            annotations,
            annotators,
            confidences,
            train_annotators,
            dev_annotators_in_train,
            covs,
        )


class DocumentEdgeAnnotationLikelihood(Likelihood):

    # Temporal relations constants
    BOTH_LOCKED = 0
    E1_LOCKED = 1
    E2_LOCKED = 2

    DIFFERENT_MIDPOINTS = 0
    SAME_MIDPOINT = 1

    STARTPOINT_LOCK_VAL = 0
    ENDPOINT_LOCK_VAL = 100

    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        device: str = "cpu",
    ):
        super().__init__(
            DOCUMENT_EDGE_SUBSPACES,
            annotator_confidences,
            metadata,
            n_components,
            device=device,
        )
        self._initialize_random_effects()

        # Probabilities for determining whether event start and
        # endpoints are locked
        self.p_lock_start = Parameter(
            Dirichlet(torch.tensor([1.0, 1.0, 1.0])).sample((self.n_components,))
        )
        self.p_lock_end = Parameter(
            Dirichlet(torch.tensor([1.0, 1.0, 1.0])).sample((self.n_components,))
        )
        self.p_lock_mid = Parameter(Uniform(0, 1).sample((self.n_components,)))

    @overrides
    def _initialize_random_effects(self) -> None:
        # TODO: use parent implementation for mereology properties
        random_effects = {}
        for subspace in sorted(self.property_subspaces - {"time"}):
            subspace_annotators = self.metadata.annotators(subspace)
            max_annotator_idx = max(
                [int(s.split("-")[-1]) for s in subspace_annotators]
            )
            num_annotators = max(len(subspace_annotators), max_annotator_idx + 1)
            for p in sorted(self.metadata.properties(subspace)):

                # Determine property dimension
                prop_dim = self._get_prop_dim(subspace, p)

                # Single random intercept term per annotator per property
                random_shift = Parameter(torch.randn(num_annotators, prop_dim))
                prop_name = p.replace(".", "-")
                random_effects[prop_name] = random_shift

        # TODO: implement
        """
        time_annotators = self.metadata.annotators("time")
        max_annotator_idx = max(
            [int(s.split("-")[-1]) for s in subspace_annotators]
        )
        num_time_annotators = max(len(time_annotators), max_annotator_idx + 1)
        random_effects["time-lock_start"] = None
        random_effects["time-lock_end"] = None
        random_effects["time-lock_mid"] = None
        random_effects["time-same_midpoint"] = None
        random_effects["time-different_midpoints"] = None
        """

        self.random_effects = ParameterDict(random_effects).to(self.device)

    @overrides
    def forward(
        self,
        mus: ParameterDict,
        annotations: Dict[str, torch.Tensor],
        annotators: torch.Tensor,
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ) -> Dict[str, Tensor]:
        likelihoods = {}
        total_ll = torch.zeros(self.n_components)

        # First, we handle all annotations other than temporal relations.
        # Currently, this is just mereological containment
        for p, anno in annotations.items():
            if "rel-" in p:
                continue  # temporal relations annotations handled below

            prop_name = p.replace(".", "-")

            # The mean for the current property
            mu = mus[prop_name].unsqueeze(1)

            # The ridit-scored confidences for these annotations
            confidence = confidences[p]

            # Determine random intercepts
            # TODO: make sure these are initialized appropriately
            if dev_annotators_in_train:
                # If evaluating on dev, we don't have tuned random effects for
                # annotators who weren't seen during training, so we use the
                # mean random effect across train set annotators.
                train_random_effects = self.random_effects[prop_name][
                    train_annotators[p], :
                ]
                mean_train_random_effect = train_random_effects.mean(0)
                random = self.random_effects[prop_name][annotators[p], :]
                random[~dev_annotators_in_train[p]] = mean_train_random_effect
            else:
                random = self.random_effects[prop_name][annotators[p], :].unsqueeze(0)

            # Compute log likelihood (clipping to prevent underflow)
            dist = self._get_distribution(mu, random)
            ll = dist.log_prob(anno).to(self.device)
            min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(self.device)
            ll = torch.where(ll > min_ll, ll, min_ll,)

            # Accumulate likelihood
            if p in likelihoods:
                likelihoods[p] += confidence * ll
                total_ll += likelihoods[p]
            else:
                likelihoods[p] = confidence * ll
                total_ll = likelihoods[p]

        # Sum over individual annotations up to this point
        # Next, we handle temporal relations annotations
        total_ll = total_ll.sum(1)

        # Confidence values are the same for all four start-
        # and endpoints for a given annotation; arbitrarily
        # using property "rel-start1"
        confidence = confidences["rel-start1"]

        # At least one of the start points is always locked to
        # zero. Determine which.
        e1_start_locked = annotations["rel-start1"] == self.STARTPOINT_LOCK_VAL
        e2_start_locked = annotations["rel-start2"] == self.STARTPOINT_LOCK_VAL
        both_start_locked = e1_start_locked & e2_start_locked
        e1_start_only_locked = e1_start_locked & ~e2_start_locked
        e2_start_only_locked = ~e1_start_locked & e2_start_locked
        start_locked = torch.zeros(annotations["rel-start1"].shape)
        start_locked[both_start_locked] = self.BOTH_LOCKED
        start_locked[e1_start_only_locked] = self.E1_LOCKED
        start_locked[e2_start_only_locked] = self.E2_LOCKED

        # Similarly, at least one of the end points is always
        # locked to 100. Again, we determine which.
        e1_end_locked = annotations["rel-end1"] == self.ENDPOINT_LOCK_VAL
        e2_end_locked = annotations["rel-end2"] == self.ENDPOINT_LOCK_VAL
        both_end_locked = e1_end_locked & e2_end_locked
        e1_end_only_locked = e1_end_locked & ~e2_end_locked
        e2_end_only_locked = ~e1_end_locked & e2_end_locked
        end_locked = torch.zeros(annotations["rel-start1"].shape)
        end_locked[both_end_locked] = self.BOTH_LOCKED
        end_locked[e1_end_only_locked] = self.E1_LOCKED
        end_locked[e2_end_only_locked] = self.E2_LOCKED

        # Identify items where both the start- and end points are locked.
        start_and_end_locked = both_start_locked & both_end_locked

        # Now, we identify items where at least one start point and
        # one end point are free (i.e. not locked).
        e1_start_and_e1_end_free = ~(e1_start_locked | e1_end_locked)
        e2_start_and_e2_end_free = ~(e2_start_locked | e2_end_locked)
        e1_start_and_e2_end_free = ~(e1_start_locked | e2_end_locked)
        e2_start_and_e1_end_free = ~(e2_start_locked | e1_end_locked)

        # Verify that these items split into disjoint sets
        # Assertion was passing when last checked
        """
        free_endpoints = [
            e1_start_and_e1_end_free,
            e2_start_and_e2_end_free,
            e1_start_and_e2_end_free,
            e2_start_and_e1_end_free,
        ]
        for l1, l2 in combinations(free_endpoints, 2):
            assert not any(l1 & l2)
        """

        # The free start and endpoints may have either the same value
        # or different values and we treat these cases separately.

        # Same value
        e1_start_and_e1_end_same = e1_start_and_e1_end_free & (
            annotations["rel-start1"] == annotations["rel-end1"]
        )
        e2_start_and_e2_end_same = e2_start_and_e2_end_free & (
            annotations["rel-start2"] == annotations["rel-end2"]
        )
        e1_start_and_e2_end_same = e1_start_and_e2_end_free & (
            annotations["rel-start1"] == annotations["rel-end2"]
        )
        e2_start_and_e1_end_same = e2_start_and_e1_end_free & (
            annotations["rel-start2"] == annotations["rel-end1"]
        )
        same_midpoint = torch.cat(
            [
                annotations["rel-start1"][
                    e1_start_and_e1_end_same | e1_start_and_e2_end_same
                ],
                annotations["rel-start2"][
                    e2_start_and_e2_end_same | e2_start_and_e1_end_same
                ],
            ]
        )
        same_midpoint_conf = torch.cat(
            [
                confidences["rel-start1"][
                    e1_start_and_e1_end_same | e1_start_and_e2_end_same
                ],
                confidences["rel-start2"][
                    e2_start_and_e2_end_same | e2_start_and_e1_end_same
                ],
            ]
        )
        same_midpoint_start_locked = torch.cat(
            [
                start_locked[e1_start_and_e1_end_same | e1_start_and_e2_end_same],
                start_locked[e2_start_and_e2_end_same | e2_start_and_e1_end_same],
            ]
        )
        same_midpoint_end_locked = torch.cat(
            [
                end_locked[e1_start_and_e1_end_same | e1_start_and_e2_end_same],
                end_locked[e2_start_and_e2_end_same | e2_start_and_e1_end_same],
            ]
        )

        # Different value: There are four possible pairs of endpoints with differing
        # values and each must be separately extracted from the annotations
        e1_start_and_e1_end_diff = e1_start_and_e1_end_free & ~e1_start_and_e1_end_same
        e1_start_and_e2_end_diff = e1_start_and_e2_end_free & ~e1_start_and_e2_end_same
        e2_start_and_e1_end_diff = e2_start_and_e1_end_free & ~e2_start_and_e1_end_same
        e2_start_and_e2_end_diff = e2_start_and_e2_end_free & ~e2_start_and_e2_end_same

        # Assert the above four lists constitute disjoint sets of items
        # Assertion was passing when last checked
        """
        different_endpoints = [
            e1_start_and_e1_end_diff,
            e1_start_and_e2_end_diff,
            e2_start_and_e1_end_diff,
            e2_start_and_e2_end_diff,
        ]
        for l1, l2 in combinations(different_endpoints, 2):
            assert not any(l1 & l2)
        """

        # 1. Midpoints = e1_start, e1_end
        diff_midpoints1 = torch.stack(
            [
                annotations["rel-start1"][e1_start_and_e1_end_diff],
                annotations["rel-end1"][e1_start_and_e1_end_diff],
            ],
            dim=1,
        )
        diff_midpoints1_start_locked = start_locked[e1_start_and_e1_end_diff]
        diff_midpoints1_end_locked = end_locked[e1_start_and_e1_end_diff]
        diff_midpoints1_conf = confidences["rel-start1"][e1_start_and_e1_end_diff]

        # 2. Midpoints = e1_start, e2_end
        diff_midpoints2 = torch.stack(
            [
                annotations["rel-start1"][e1_start_and_e2_end_diff],
                annotations["rel-end2"][e1_start_and_e2_end_diff],
            ],
            dim=1,
        )
        diff_midpoints2_start_locked = start_locked[e1_start_and_e2_end_diff]
        diff_midpoints2_end_locked = end_locked[e1_start_and_e2_end_diff]
        diff_midpoints2_conf = confidences["rel-start1"][e1_start_and_e2_end_diff]

        # 3. Midpoints = e2_start, e1_end
        diff_midpoints3 = torch.stack(
            [
                annotations["rel-start2"][e2_start_and_e1_end_diff],
                annotations["rel-end1"][e2_start_and_e1_end_diff],
            ],
            dim=1,
        )
        diff_midpoints3_start_locked = start_locked[e2_start_and_e1_end_diff]
        diff_midpoints3_end_locked = end_locked[e2_start_and_e1_end_diff]
        diff_midpoints3_conf = confidences["rel-start2"][e2_start_and_e1_end_diff]

        # 4. Midpoints = e2_start, e2_end
        diff_midpoints4 = torch.stack(
            [
                annotations["rel-start2"][e2_start_and_e2_end_diff],
                annotations["rel-end2"][e2_start_and_e2_end_diff],
            ],
            dim=1,
        )
        diff_midpoints4_start_locked = start_locked[e2_start_and_e2_end_diff]
        diff_midpoints4_end_locked = end_locked[e2_start_and_e2_end_diff]
        diff_midpoints4_conf = confidences["rel-start2"][e2_start_and_e2_end_diff]

        # Combine
        diff_midpoints = torch.cat(
            [diff_midpoints1, diff_midpoints2, diff_midpoints3, diff_midpoints4]
        )
        diff_midpoints_start_locked = torch.cat(
            [
                diff_midpoints1_start_locked,
                diff_midpoints2_start_locked,
                diff_midpoints3_start_locked,
                diff_midpoints4_start_locked,
            ]
        )
        diff_midpoints_end_locked = torch.cat(
            [
                diff_midpoints1_end_locked,
                diff_midpoints2_end_locked,
                diff_midpoints3_end_locked,
                diff_midpoints4_end_locked,
            ]
        )
        diff_midpoints_conf = torch.cat(
            [
                diff_midpoints1_conf,
                diff_midpoints2_conf,
                diff_midpoints3_conf,
                diff_midpoints4_conf,
            ]
        )

        # Likelihood calculation
        # TODO: incorporate random effects

        # Case 1: both start and endpoints are locked. Likelihood is just the product
        #         of the probabilities that the start point is locked and that the
        #         end point is locked (weighted by confidence)
        start_and_end_locked_conf = confidences["rel-start1"][start_and_end_locked]
        start_and_end_locked_prob = Categorical(self.p_lock_start).log_prob(
            torch.tensor([self.BOTH_LOCKED])
        ) + Categorical(self.p_lock_end).log_prob(torch.tensor([self.BOTH_LOCKED]))
        start_and_end_locked_prob = (
            start_and_end_locked_conf[:, None] * start_and_end_locked_prob
        ).sum(0)

        # Case 2: One start- and one end point are free and these have the same value.
        #         We assume that value is normally distributed. We also incorporate the
        #         probability that the start- and end points are free to begin with.
        same_midpoint_start_prob = Categorical(self.p_lock_start).log_prob(
            same_midpoint_start_locked[:, None]
        )
        same_midpoint_end_prob = Categorical(self.p_lock_end).log_prob(
            same_midpoint_end_locked[:, None]
        )
        same_midpoint_mid_prob = Bernoulli(self.p_lock_mid).log_prob(self.SAME_MIDPOINT)
        same_midpoint_prob = (
            Normal(mus["time-univariate_mu"], covs["time-univariate_sigma"]).log_prob(
                same_midpoint[..., None]
            )
            + same_midpoint_start_prob
            + same_midpoint_end_prob
            + same_midpoint_mid_prob
        )
        same_midpoint_prob = (same_midpoint_conf[:, None] * same_midpoint_prob).sum(0)

        # Case 3: One start- and one end point are frree and have *different* values.
        #         We assume these two values are distributed multivariate normal. We
        #         also incorporate the probabilities that the start and end points are
        #         free to begin with.
        diff_midpoints_start_prob = Categorical(self.p_lock_start).log_prob(
            diff_midpoints_start_locked[:, None]
        )
        diff_midpoints_end_prob = Categorical(self.p_lock_end).log_prob(
            diff_midpoints_end_locked[:, None]
        )
        diff_midpoints_mid_prob = Bernoulli(self.p_lock_mid).log_prob(
            self.DIFFERENT_MIDPOINTS
        )
        diff_midpoints_prob = (
            MultivariateNormal(mus["time-bivariate_mu"], covs["time-bivariate_sigma"])
            .log_prob(diff_midpoints[:, None, None, :])
            .squeeze()
            + diff_midpoints_start_prob
            + diff_midpoints_end_prob
            + diff_midpoints_mid_prob
        )
        diff_midpoints_prob = (diff_midpoints_conf[:, None] * diff_midpoints_prob).sum(
            0
        )

        # Sum all three
        total_ll += same_midpoint_prob + diff_midpoints_prob + start_and_end_locked_prob
        return likelihoods, total_ll
