import torch
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


class Likelihood(Module):
    def __init__(
        self,
        property_subspaces: Set[str],
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        device: str = "cpu",
    ):
        """Base class for vectorized Event Type Induction Module likelihood computations

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

    def _get_distribution2(self, mu):
        """Generates an appropriate distribution given a mean and random effect"""
        if mu.shape[-1] == 1:
            return Bernoulli(torch.sigmoid(torch.exp(mu)).squeeze())
        else:
            return Categorical(torch.softmax(torch.exp(mu), -1).squeeze())

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
        items: Dict[str, torch.Tensor],
        annotators: Dict[str, torch.Tensor],
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ):
        likelihoods = {}
        num_items = max([i[-1].item() for i in items.values()]) + 1
        total_ll = torch.zeros(self.n_components, num_items).to(self.device)
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

            # Compute log likelihood (with clipping to prevent underflow)
            dist = self._get_distribution(mu, random)
            ll = dist.log_prob(anno)
            min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(self.device)
            ll = torch.where(ll > min_ll, ll, min_ll,)

            # Weight likelihoods by confidence, then accumulate
            # by item (node/edge)
            likelihoods[p] = ll * confidence
            total_ll.index_add_(1, items[p], ll * confidence)

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
        items: Dict[str, torch.Tensor],
        annotators: Dict[str, torch.Tensor],
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ) -> Dict[str, Tensor]:
        return super().forward(
            mus,
            annotations,
            items,
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
        items: Dict[str, torch.Tensor],
        annotators: Dict[str, torch.Tensor],
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ) -> Dict[str, Tensor]:
        return super().forward(
            mus,
            annotations,
            items,
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
        items: Dict[str, torch.Tensor],
        annotators: Dict[str, torch.Tensor],
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ) -> Union[Dict[str, Tensor], Tensor]:
        return super().forward(
            mus,
            annotations,
            items,
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
        self.both_locked = torch.tensor([self.BOTH_LOCKED]).to(self.device)

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

        # In-progress changes to random effects; commenting out for now
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
        items: Dict[str, torch.Tensor],
        annotators: torch.Tensor,
        confidences: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
        covs: ParameterDict = None,
    ) -> Dict[str, Tensor]:
        likelihoods = {}
        num_items = max([i[-1].item() for i in items.values()]) + 1
        total_ll = torch.zeros(self.n_components, num_items).to(self.device)

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
            """
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
            """
            random = self.random_effects[prop_name][annotators[p], :].unsqueeze(0)
            random = torch.zeros(random.shape).to(self.device)

            # Compute log likelihood (clipping to prevent underflow)
            dist = self._get_distribution(mu, random)
            ll = dist.log_prob(anno)
            min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(self.device)
            ll = torch.where(ll > min_ll, ll, min_ll,)

            likelihoods[p] = ll * confidence
            total_ll.index_add_(1, items[p], ll * confidence)

        # Confidence values are the same for all four start-
        # and endpoints for a given annotation, so I arbitrarily
        # am using the ones for "rel-start1"
        confidence = confidences["rel-start1"]

        # For time annotations, at least one of the start points
        # is always locked to zero. Here, we determine which.
        e1_start_locked = annotations["rel-start1"] == self.STARTPOINT_LOCK_VAL
        e2_start_locked = annotations["rel-start2"] == self.STARTPOINT_LOCK_VAL
        both_start_locked = e1_start_locked & e2_start_locked
        e1_start_only_locked = e1_start_locked & ~e2_start_locked
        e2_start_only_locked = ~e1_start_locked & e2_start_locked

        # For each annotation, start_locked indicates whether BOTH startpoints
        # are locked to zero, only e1's startpoint, or only e2's startpoint
        start_locked = torch.zeros(annotations["rel-start1"].shape).to(self.device)
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

        # end_locked serves the same function as start_locked but for endpoints
        end_locked = torch.zeros(annotations["rel-start1"].shape).to(self.device)
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

        # The free start and endpoints may have either the same value
        # or different values and we treat these cases separately.

        # SAME VALUE
        # There are four ways a start and endpoint could have
        # the same value.
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

        # Combine these annotations into a single tensor.
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

        # Combine the associated confidence scores into a single tensor
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

        # Combine the associated item (edge) IDs
        same_midpoint_items = torch.cat(
            [
                items["rel-start1"][
                    e1_start_and_e1_end_same | e1_start_and_e2_end_same
                ],
                items["rel-start2"][
                    e2_start_and_e2_end_same | e2_start_and_e1_end_same
                ],
            ]
        )

        # For the above items, determine which of the start- and endpoints are locked
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
        diff_midpoints1_items = items["rel-start1"][e1_start_and_e1_end_diff]

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
        diff_midpoints2_items = items["rel-start1"][e1_start_and_e2_end_diff]

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
        diff_midpoints3_items = items["rel-start2"][e2_start_and_e1_end_diff]

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
        diff_midpoints4_items = items["rel-start2"][e2_start_and_e2_end_diff]

        # Combine the four cases into a single tensor
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
        diff_midpoints_items = torch.cat(
            [
                diff_midpoints1_items,
                diff_midpoints2_items,
                diff_midpoints3_items,
                diff_midpoints4_items,
            ]
        )

        # Likelihood calculation
        # TODO: incorporate random effects

        # Case 1: both start and endpoints are locked. Likelihood is just the product
        #         of the probabilities that the start point is locked and that the
        #         end point is locked (weighted by confidence)
        start_and_end_locked_conf = confidences["rel-start1"][start_and_end_locked]
        start_and_end_locked_items = items["rel-start1"][start_and_end_locked]
        start_and_end_locked_prob = self._get_distribution2(
            mus["time-lock_start_mu"]
        ).log_prob(self.both_locked)
        start_and_end_locked_prob = (
            torch.ones(start_and_end_locked_conf.shape).to(self.device)[:, None]
            * start_and_end_locked_prob
        )

        # Case 2: One start- and one end point are free and these have the same value.
        #         We assume that value is normally distributed. We also incorporate the
        #         probability that the start- and end points are free to begin with.
        same_midpoint_start_prob = (
            self._get_distribution2(mus["time-lock_start_mu"])
            .log_prob(same_midpoint_start_locked[:, None])
            .to(self.device)
        )
        same_midpoint_end_prob = (
            self._get_distribution2(mus["time-lock_end_mu"])
            .log_prob(same_midpoint_end_locked[:, None])
            .to(self.device)
        )
        same_midpoint_mid_prob = (
            self._get_distribution2(mus["time-lock_mid_mu"])
            .log_prob(self.SAME_MIDPOINT)
            .to(self.device)
        )
        same_midpoint_prob = (
            Normal(mus["time-univariate_mu"], covs["time-univariate_sigma"])
            .log_prob(same_midpoint[..., None])
            .to(self.device)
            + same_midpoint_start_prob
            + same_midpoint_end_prob
            + same_midpoint_mid_prob
        )
        # same_midpoint_prob = (same_midpoint_conf[:, None] * same_midpoint_prob)
        same_midpoint_prob = (
            torch.ones(same_midpoint_conf.shape).to(self.device)[:, None]
            * same_midpoint_prob
        )

        # Case 3: One start- and one end point are free and have *different* values.
        #         We assume these two values are distributed multivariate normal. We
        #         also incorporate the probabilities that the start and end points are
        #         free to begin with.
        diff_midpoints_start_prob = (
            self._get_distribution2(mus["time-lock_start_mu"])
            .log_prob(diff_midpoints_start_locked[:, None])
            .to(self.device)
        )
        diff_midpoints_end_prob = (
            self._get_distribution2(mus["time-lock_end_mu"])
            .log_prob(diff_midpoints_end_locked[:, None])
            .to(self.device)
        )
        diff_midpoints_mid_prob = (
            self._get_distribution2(mus["time-lock_mid_mu"])
            .log_prob(self.DIFFERENT_MIDPOINTS)
            .to(self.device)
        )
        diff_midpoints_prob = (
            MultivariateNormal(mus["time-bivariate_mu"], covs["time-bivariate_sigma"])
            .log_prob(diff_midpoints[:, None, None, :])
            .squeeze()
            .to(self.device)
            + diff_midpoints_start_prob
            + diff_midpoints_end_prob
            + diff_midpoints_mid_prob
        )
        # diff_midpoints_prob = (diff_midpoints_conf[:, None] * diff_midpoints_prob)
        diff_midpoints_prob = (
            torch.ones(diff_midpoints_conf.shape).to(self.device)[:, None]
            * diff_midpoints_prob
        )

        # TODO: add time to per-property likelihoods ("likelihoods")
        total_ll.index_add_(1, start_and_end_locked_items, start_and_end_locked_prob.T)
        total_ll.index_add_(1, same_midpoint_items, same_midpoint_prob.T)
        total_ll.index_add_(1, diff_midpoints_items, diff_midpoints_prob.T)
        return likelihoods, total_ll
