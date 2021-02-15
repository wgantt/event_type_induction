import torch
from event_type_induction.constants import *
from event_type_induction.utils import exp_normalize, get_prop_dim
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
from torch.nn import Module, Parameter, ParameterDict
from typing import Dict, Set, Tuple, Union


class Likelihood(Module):
    def __init__(
        self,
        property_subspaces: Set[str],
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
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
        clip_min_ll
            whether to clip the likelihood to a minimum value
        confidence_weighting
            whether to weight the likelihood by annotator confidence
        device
            the device to use
        """
        super().__init__()
        self.property_subspaces = property_subspaces
        self.annotator_confidences = annotator_confidences
        self.metadata = metadata
        self.n_components = n_components
        self.use_ordinal = use_ordinal
        self.clip_min_ll = clip_min_ll
        self.confidence_weighting = confidence_weighting
        self.device = torch.device(device)
        self.to(device)

        # When using ordinal models, it's useful to have a quick
        # way of determining which properties are ordinal
        if self.use_ordinal:
            self.ordinal_properties = set()
            for subspace in self.metadata.subspaces:
                for prop, prop_meta in self.metadata[subspace].items():
                    if prop_meta.value.is_ordered_categorical:
                        self.ordinal_properties.add(prop)

    def _initialize_random_effects(self) -> ParameterDict:
        """Initialize annotator random effects for each property"""
        random_effects = {}
        for subspace in sorted(self.property_subspaces):
            subspace_annotators = self.metadata.annotators(subspace)
            if subspace == "genericity":
                if isinstance(self, PredicateNodeAnnotationLikelihood):
                    subspace_annotators = {
                        s for s in subspace_annotators if "pred" in s
                    }
                else:
                    subspace_annotators = {s for s in subspace_annotators if "arg" in s}
            max_annotator_idx = max(
                [int(s.split("-")[-1]) for s in subspace_annotators]
            )
            num_annotators = max(len(subspace_annotators), max_annotator_idx + 1)
            for p in sorted(self.metadata.properties(subspace)):

                prop_name = p.replace(".", "-")
                if self.use_ordinal and p in self.ordinal_properties:
                    # When using an ordinal model for ordinal properties, the random
                    # effects correspond to per-annotator cutpoints
                    n_categories = len(self.metadata[subspace][p].value.categories)
                    prop_dim = n_categories - 1
                    # TODO: determine how to set covariance for sampling
                    # TODO: should random loss be different for these?
                    ordinal_random_effect = Parameter(
                        MultivariateNormal(
                            torch.zeros(prop_dim), torch.eye(prop_dim) * 0.1
                        ).sample((num_annotators,))
                    )
                    random_effects[prop_name] = ordinal_random_effect

                    # If this is a conditional property (i.e. it may be annotated
                    # as "does not apply"), we include a Bernoulli to model this.
                    if p in CONDITIONAL_PROPERTIES:
                        random_effects[prop_name + "-applies"] = Parameter(
                            torch.randn(num_annotators) * 0.1
                        )
                else:
                    prop_dim = get_prop_dim(
                        self.metadata, subspace, p, use_ordinal=self.use_ordinal
                    )
                    random = Parameter(torch.randn(num_annotators, prop_dim))
                    random_effects[prop_name] = random

        self.random_effects = ParameterDict(random_effects).to(self.device)

    def _get_distribution(
        self, mu: torch.Tensor, random: torch.Tensor = None, prop_name: str = None,
    ):
        """Generates an appropriate distribution given means and random effects"""
        if self.use_ordinal and prop_name in self.ordinal_properties:
            cuts_annotator = torch.cumsum(torch.exp(random), axis=2)
            cuts_annotator -= torch.mean(cuts_annotator)
            cdf = torch.sigmoid(cuts_annotator - mu)
            cdf_high = torch.cat(
                [cdf, torch.ones(cdf.shape[0], cdf.shape[1], 1).to(self.device)], axis=2
            )
            cdf_low = torch.cat(
                [torch.zeros(cdf.shape[0], cdf.shape[1], 1).to(self.device), cdf],
                axis=2,
            )
            pmf = cdf_high - cdf_low
            return Categorical(pmf.squeeze())

        mean = mu if random is None else mu + random
        if mu.shape[-1] == 1:
            return Bernoulli(torch.sigmoid(mean).squeeze(dim=-1))
        else:
            return Categorical(torch.softmax(mean, -1).squeeze())

    def _get_random_effects(
        self,
        p: str,
        mu: torch.Tensor,
        annotators: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prop_name = p.replace(".", "-")
        if dev_annotators_in_train:
            # If evaluating on dev, we don't have tuned random effects for
            # annotators who weren't seen during training, so we use the
            # mean random effect across train set annotators.
            train_random_effects = self.random_effects[prop_name][
                train_annotators[p], :
            ]
            train_random_effects -= train_random_effects.mean()
            mean_train_random_effect = train_random_effects.mean(0)
            random = self.random_effects[prop_name][annotators[p], :]
            random[~dev_annotators_in_train[p]] = mean_train_random_effect
            random = (random - random.mean()).unsqueeze(0)

            # For conditional properties, there are additional random effects
            # for the Bernoulli that indicates whether the property applies or not
            if (
                self.use_ordinal
                and p in self.ordinal_properties
                and p in CONDITIONAL_PROPERTIES
            ):
                train_random_bernoulli = self.random_effects[prop_name + "-applies"][
                    train_annotators[p]
                ]
                train_random_bernoulli -= train_random_bernoulli.mean()
                mean_train_random_bernoulli = train_random_bernoulli.mean(0)
                random_bernoulli = self.random_effects[prop_name + "-applies"][
                    annotators[p]
                ]
                random_bernoulli[
                    ~dev_annotators_in_train[p]
                ] = mean_train_random_bernoulli
                random_bernoulli -= random_bernoulli.mean()
                random = (random_bernoulli, random)
        else:
            random = self.random_effects[prop_name][annotators[p], :].unsqueeze(0)
            random -= random.mean()
            if (
                self.use_ordinal
                and p in self.ordinal_properties
                and p in CONDITIONAL_PROPERTIES
            ):
                random_bernoulli = self.random_effects[prop_name + "-applies"][
                    annotators[p]
                ]
                random_bernoulli -= random.mean()
                random = (random_bernoulli, random)
        return random

    def random_loss(self):
        """Computes log likelihood of annotator random effects

           Random effects terms are assumed to be distributed MV normal
        """
        loss = FloatTensor([0.0]).to(self.device)

        # Loss computed by property
        for prop, random in self.random_effects.items():

            if len(random.shape) == 1:
                random = random[:, None]
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
            random = self._get_random_effects(
                p, mu, annotators, train_annotators, dev_annotators_in_train
            )

            # Handle hurdle model for ordinal properties
            if isinstance(random, tuple):
                property_applies_mu = mus[prop_name + "-applies"].unsqueeze(1)
                property_applies_shift, ordinal_shift = random
                property_applies_dist = Bernoulli(
                    torch.sigmoid(property_applies_mu + property_applies_shift)
                )
                ordinal_dist = self._get_distribution(mu, ordinal_shift, prop_name=p)
                bernoulli_ll = property_applies_dist.log_prob(confidence)

                nans = torch.isnan(anno)
                anno_no_nans = torch.clone(anno)
                anno_no_nans[nans] = 0
                ordinal_ll = ordinal_dist.log_prob(anno_no_nans)
                ordinal_ll[:, nans] = 0
                ll = ordinal_ll + bernoulli_ll
            else:
                ll = self._get_distribution(mu, random, prop_name=p).log_prob(anno)

            if self.clip_min_ll:
                min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(
                    self.device
                )
                ll = torch.where(ll > min_ll, ll, min_ll,)

            # Weight likelihoods by confidence, then accumulate
            # by item (node/edge)
            if self.confidence_weighting:
                ll *= confidence
            likelihoods[p] = ll
            total_ll.index_add_(1, items[p], ll)

        return likelihoods, total_ll


class PredicateNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            PREDICATE_NODE_SUBSPACES,
            annotator_confidences,
            metadata,
            n_components,
            use_ordinal,
            clip_min_ll,
            confidence_weighting,
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
    ):

        return super().forward(
            mus,
            annotations,
            items,
            annotators,
            confidences,
            train_annotators,
            dev_annotators_in_train,
        )


class ArgumentNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            ARGUMENT_NODE_SUBSPACES,
            annotator_confidences,
            metadata,
            n_components,
            use_ordinal,
            clip_min_ll,
            confidence_weighting,
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
    ) -> Dict[str, Tensor]:
        return super().forward(
            mus,
            annotations,
            items,
            annotators,
            confidences,
            train_annotators,
            dev_annotators_in_train,
        )


class SemanticsEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            SEMANTICS_EDGE_SUBSPACES,
            annotator_confidences,
            metadata,
            n_components,
            use_ordinal,
            clip_min_ll,
            confidence_weighting,
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
    ) -> Union[Dict[str, Tensor], Tensor]:
        return super().forward(
            mus,
            annotations,
            items,
            annotators,
            confidences,
            train_annotators,
            dev_annotators_in_train,
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
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
        use_random_effects: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            DOCUMENT_EDGE_SUBSPACES,
            annotator_confidences,
            metadata,
            n_components,
            use_ordinal,
            clip_min_ll,
            confidence_weighting,
            device=device,
        )
        # TODO: move into parent class
        self.use_random_effects = use_random_effects
        self._initialize_random_effects()
        self.e1_locked = torch.tensor([self.E1_LOCKED]).to(self.device)
        self.e2_locked = torch.tensor([self.E2_LOCKED]).to(self.device)
        self.both_locked = torch.tensor([self.BOTH_LOCKED]).to(self.device)

    @overrides
    def _initialize_random_effects(self) -> None:
        random_effects = {}
        for subspace in sorted(self.property_subspaces - {"time"}):
            subspace_annotators = self.metadata.annotators(subspace)
            max_annotator_idx = max(
                [int(s.split("-")[-1]) for s in subspace_annotators]
            )
            num_annotators = max(len(subspace_annotators), max_annotator_idx + 1)
            for p in sorted(self.metadata.properties(subspace)):

                # Determine property dimension
                prop_dim = get_prop_dim(self.metadata, subspace, p)

                # Single random intercept term per annotator per property
                random = Parameter(torch.randn(num_annotators, prop_dim))
                prop_name = p.replace(".", "-")
                random_effects[prop_name] = random

        # Unique likelihood model for UDS-Time annotations requires
        # special random effects here.
        time_annotators = self.metadata.annotators("time")
        max_annotator_idx = max([int(s.split("-")[-1]) for s in time_annotators])
        num_time_annotators = max(len(time_annotators), max_annotator_idx + 1)
        random_effects["time-lock_start"] = Parameter(
            torch.randn((num_time_annotators, 3))
        )
        random_effects["time-lock_end"] = Parameter(
            torch.randn((num_time_annotators, 3))
        )
        random_effects["time-lock_mid"] = Parameter(
            torch.randn((num_time_annotators, 1))
        )
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

            # Determine random intercepts
            if self.use_random_effects:
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
                    random = self.random_effects[prop_name][annotators[p], :].unsqueeze(
                        0
                    )
                dist = self._get_distribution(mu, random, prop_name)
            else:
                dist = self._get_distribution(mu)

            # min-clip LL, if applicable
            ll = dist.log_prob(anno)
            if self.clip_min_ll:
                min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(
                    self.device
                )
                ll = torch.where(ll > min_ll, ll, min_ll,)

            if self.confidence_weighting:
                ll *= confidences[p]
            likelihoods[p] = ll
            total_ll.index_add_(1, items[p], ll)

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

        # Identify items where exactly one start or endpoint is free
        only_e1_start_free = (
            ~e1_start_locked & e2_start_locked & e1_end_locked & e2_end_locked
        )
        only_e2_start_free = (
            e1_start_locked & ~e2_start_locked & e1_end_locked & e2_end_locked
        )
        only_e1_end_free = (
            e1_start_locked & e2_start_locked & ~e1_end_locked & e2_end_locked
        )
        only_e2_end_free = (
            e1_start_locked & e2_start_locked & e1_end_locked & ~e2_end_locked
        )

        # Now, we identify items where two start and/or endpoints are free
        # (b.c. of normalization, there will never be more than two)
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
        # TODO: incorporate random effects and confidence
        if self.use_random_effects:
            lock_start_dist = self._get_distribution(
                mus["time-lock_start_mu"], random["time-lock_start"], "time-lock_start",
            )
            lock_end_dist = self._get_distribution(
                mus["time-lock_end_mu"], random["time-lock_end"], "time-lock_end"
            )
            lock_mid_dist = self._get_distribution(
                mus["time-lock_mid_mu"], random["time-lock_mid"], "time-lock_mid"
            )
        else:
            lock_start_dist = self._get_distribution(mus["time-lock_start_mu"])
            lock_end_dist = self._get_distribution(mus["time-lock_end_mu"])
            lock_mid_dist = self._get_distribution(mus["time-lock_mid_mu"])

        # Case 1: both start and endpoints are locked. Likelihood is just the product
        #         of the probabilities that the start point is locked and that the
        #         end point is locked (weighted by confidence)
        start_and_end_locked_conf = confidences["rel-start1"][start_and_end_locked]
        start_and_end_locked_items = items["rel-start1"][start_and_end_locked]
        start_and_end_locked_prob = lock_start_dist.log_prob(
            self.both_locked
        ) + lock_end_dist.log_prob(self.both_locked)
        # TODO: add option for confidence weighting
        start_and_end_locked_prob = (
            torch.ones(start_and_end_locked_conf.shape).to(self.device)[:, None]
            * start_and_end_locked_prob
        )

        # Case 2: Exactly one start or end point is free
        # Likelihood is computed similarly to Case 1

        # e1 start
        only_e1_start_free_conf = confidences["rel-start1"][only_e1_start_free]
        only_e1_start_free_items = items["rel-start1"][only_e1_start_free]
        only_e1_start_free_prob = lock_start_dist.log_prob(
            self.e2_locked
        ) + lock_end_dist.log_prob(self.both_locked)
        # TODO: add option for confidence weighting
        only_e1_start_free_prob = (
            torch.ones(only_e1_start_free_conf.shape).to(self.device)[:, None]
            * only_e1_start_free_prob
        )

        # e2 start
        only_e2_start_free_conf = confidences["rel-start1"][only_e2_start_free]
        only_e2_start_free_items = items["rel-start1"][only_e2_start_free]
        only_e2_start_free_prob = lock_start_dist.log_prob(
            self.e1_locked
        ) + lock_end_dist.log_prob(self.both_locked)
        # TODO: add option for confidence weighting
        only_e2_start_free_prob = (
            torch.ones(only_e2_start_free_conf.shape).to(self.device)[:, None]
            * only_e2_start_free_prob
        )

        # e1 end
        only_e1_end_free_conf = confidences["rel-start1"][only_e1_end_free]
        only_e1_end_free_items = items["rel-start1"][only_e1_end_free]
        only_e1_end_free_prob = lock_start_dist.log_prob(
            self.both_locked
        ) + lock_end_dist.log_prob(self.e2_locked)
        # TODO: add option for confidence weighting
        only_e1_end_free_prob = (
            torch.ones(only_e1_end_free_conf.shape).to(self.device)[:, None]
            * only_e1_end_free_prob
        )

        # e2 end
        only_e2_end_free_conf = confidences["rel-start1"][only_e2_end_free]
        only_e2_end_free_items = items["rel-start1"][only_e2_end_free]
        only_e2_end_free_prob = lock_start_dist.log_prob(
            self.both_locked
        ) + lock_end_dist.log_prob(self.e1_locked)
        # TODO: add option for confidence weighting
        only_e2_end_free_prob = (
            torch.ones(only_e2_end_free_conf.shape).to(self.device)[:, None]
            * only_e2_end_free_prob
        )

        # Case 3: One start- and one end point are free and these have the same value.
        #         Here, we incorporate an additional probability for the midpoint being free.
        same_midpoint_start_prob = lock_start_dist.log_prob(
            same_midpoint_start_locked[:, None]
        ).to(self.device)
        same_midpoint_end_prob = lock_end_dist.log_prob(
            same_midpoint_end_locked[:, None]
        ).to(self.device)
        same_midpoint_mid_prob = lock_mid_dist.log_prob(self.SAME_MIDPOINT).to(
            self.device
        )
        same_midpoint_prob = (
            same_midpoint_start_prob + same_midpoint_end_prob + same_midpoint_mid_prob
        )
        # TODO: add option for confidence weighting
        # same_midpoint_prob = (same_midpoint_conf[:, None] * same_midpoint_prob)
        same_midpoint_prob = (
            torch.ones(same_midpoint_conf.shape).to(self.device)[:, None]
            * same_midpoint_prob
        )

        # Case 4: One start- and one end point are free and have *different* values.
        #         Likelihood is calculated the same as in Case 3
        diff_midpoints_start_prob = lock_start_dist.log_prob(
            diff_midpoints_start_locked[:, None]
        ).to(self.device)
        diff_midpoints_end_prob = lock_end_dist.log_prob(
            diff_midpoints_end_locked[:, None]
        ).to(self.device)
        diff_midpoints_mid_prob = lock_mid_dist.log_prob(self.DIFFERENT_MIDPOINTS).to(
            self.device
        )
        diff_midpoints_prob = (
            diff_midpoints_start_prob
            + diff_midpoints_end_prob
            + diff_midpoints_mid_prob
        )
        # TODO: add option for confidence weighting
        # diff_midpoints_prob = (diff_midpoints_conf[:, None] * diff_midpoints_prob)
        diff_midpoints_prob = (
            torch.ones(diff_midpoints_conf.shape).to(self.device)[:, None]
            * diff_midpoints_prob
        )

        # Accumulate likelihoods
        total_ll.index_add_(1, start_and_end_locked_items, start_and_end_locked_prob.T)
        total_ll.index_add_(1, only_e1_start_free_items, only_e1_start_free_prob.T)
        total_ll.index_add_(1, only_e2_start_free_items, only_e2_start_free_prob.T)
        total_ll.index_add_(1, only_e1_end_free_items, only_e1_end_free_prob.T)
        total_ll.index_add_(1, only_e2_end_free_items, only_e2_end_free_prob.T)
        total_ll.index_add_(1, same_midpoint_items, same_midpoint_prob.T)
        total_ll.index_add_(1, diff_midpoints_items, diff_midpoints_prob.T)
        # TODO: add time to per-property likelihoods ("likelihoods")
        return likelihoods, total_ll
