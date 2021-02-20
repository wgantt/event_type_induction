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
                if (
                    isinstance(self, PredicateNodeAnnotationLikelihood) and "arg" in p
                ) or (
                    isinstance(self, ArgumentNodeAnnotationLikelihood) and "pred" in p
                ):
                    continue

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
        annotators: Dict[str, torch.Tensor],
        train_annotators: Dict[str, torch.Tensor] = None,
        dev_annotators_in_train: Dict[str, torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prop_name = p.replace(".", "-")
        if "time" in p:
            p = "rel-start1"
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
                random_bernoulli -= random_bernoulli.mean()
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

    E1_EQUALS_E2 = 0
    E1_BEFORE_E2 = 1
    E2_BEFORE_E1 = 2

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
            torch.randn((num_time_annotators, 3))
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
            dist = self._get_distribution(mu, random, prop_name)

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

        # Confidence values and items are the same for all four start-
        # and endpoints for a given annotation, so I arbitrarily use
        # the ones for "rel-start1"
        all_confidences = confidences["rel-start1"]
        all_items = items["rel-start1"]

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
        # (b/c of normalization, there will never be more than two)
        e1_start_and_e1_end_free = ~(e1_start_locked | e1_end_locked)
        e2_start_and_e2_end_free = ~(e2_start_locked | e2_end_locked)
        e1_start_and_e2_end_free = ~(e1_start_locked | e2_end_locked)
        e2_start_and_e1_end_free = ~(e2_start_locked | e1_end_locked)

        # The free start and endpoints may come from the same event
        # or different events. The two cases are handled differently.

        # The two free midpoints are from the SAME EVENT
        same_event = e1_start_and_e1_end_free | e2_start_and_e2_end_free

        # The two free midpoints are from DIFFERENT EVENTS
        diff_events = e1_start_and_e2_end_free | e2_start_and_e1_end_free

        # For midpoints from DIFFERENT EVENTS, we furthermore
        # model the relationship between the two midpoints: whether
        # the one from e1 precedes e2 (or vice versa) or the two are equal

        # Midpoints = e1-start, e2-end
        e1_start_before_e2_end = e1_start_and_e2_end_free & (
            annotations["rel-start1"] < annotations["rel-end2"]
        )
        e1_start_after_e2_end = e1_start_and_e2_end_free & (
            annotations["rel-start1"] > annotations["rel-end2"]
        )

        # Midpoints = e2-start, e1-end
        e2_start_before_e1_end = e2_start_and_e1_end_free & (
            annotations["rel-start2"] < annotations["rel-end1"]
        )
        e2_start_after_e1_end = e2_start_and_e1_end_free & (
            annotations["rel-start2"] > annotations["rel-end1"]
        )

        # Midpoint probabilities for DIFFERENT EVENTS
        diff_events_midpoint_probs = torch.zeros(annotations["rel-start1"].shape).to(
            self.device
        )

        # Case 1: The two midpoints from different events are equivalent
        #         The default 0 value corresponds to this case.

        # Case 2: The midpoint from e1 precedes the midpoint from e2
        diff_events_midpoint_probs[
            e1_start_before_e2_end | e2_start_after_e1_end
        ] = self.E1_BEFORE_E2

        # Case 3: The midpoint from e2 precedes the midpoint from e1
        diff_events_midpoint_probs[
            e2_start_before_e1_end | e1_start_after_e2_end
        ] = self.E2_BEFORE_E1

        # Get mean
        lock_start_mean = mus["time-lock_start_mu"][:, None, :]
        lock_end_mean = mus["time-lock_end_mu"][:, None, :]
        lock_mid_mean = mus["time-lock_mid_mu"][:, None, :]

        # Get random effects
        # TODO
        lock_start_random = self._get_random_effects(
            "time-lock_start", annotators, train_annotators, dev_annotators_in_train
        )
        lock_end_random = self._get_random_effects(
            "time-lock_end", annotators, train_annotators, dev_annotators_in_train
        )
        lock_mid_random = self._get_random_effects(
            "time-lock_mid", annotators, train_annotators, dev_annotators_in_train
        )

        # Likelihood calculation
        lock_start_dist = self._get_distribution(lock_start_mean, lock_start_random)
        lock_end_dist = self._get_distribution(lock_end_mean, lock_end_random)
        lock_mid_dist = self._get_distribution(lock_mid_mean, lock_mid_random)

        # Case 1: both start and endpoints are locked. Likelihood is just the product
        #         of the probabilities that the start point is locked and that the
        #         end point is locked (weighted by confidence)
        start_and_end_locked_items = all_items[start_and_end_locked]
        start_and_end_locked_prob = (
            lock_start_dist.log_prob(self.both_locked)[:, start_and_end_locked]
            + lock_end_dist.log_prob(self.both_locked)[:, start_and_end_locked]
        )
        if self.confidence_weighting:
            start_and_end_locked_prob *= all_confidences[start_and_end_locked]

        # Case 2: Exactly one start or end point is free
        # Likelihood is computed similarly to Case 1

        # e1 start
        only_e1_start_free_items = all_items[only_e1_start_free]
        only_e1_start_free_prob = (
            lock_start_dist.log_prob(self.e2_locked)[:, only_e1_start_free]
            + lock_end_dist.log_prob(self.both_locked)[:, only_e1_start_free]
        )
        if self.confidence_weighting:
            only_e1_start_free_prob *= all_confidences[only_e1_start_free]

        # e2 start
        only_e2_start_free_items = all_items[only_e2_start_free]
        only_e2_start_free_prob = (
            lock_start_dist.log_prob(self.e1_locked)[:, only_e2_start_free]
            + lock_end_dist.log_prob(self.both_locked)[:, only_e2_start_free]
        )
        if self.confidence_weighting:
            only_e2_start_free_prob *= all_confidences[only_e2_start_free]

        # e1 end
        only_e1_end_free_items = all_items[only_e1_end_free]
        only_e1_end_free_prob = (
            lock_start_dist.log_prob(self.both_locked)[:, only_e1_end_free]
            + lock_end_dist.log_prob(self.e2_locked)[:, only_e1_end_free]
        )
        if self.confidence_weighting:
            only_e1_end_free_prob *= all_confidences[only_e1_end_free]

        # e2 end
        only_e2_end_free_items = all_items[only_e2_end_free]
        only_e2_end_free_prob = (
            lock_start_dist.log_prob(self.both_locked)[:, only_e2_end_free]
            + lock_end_dist.log_prob(self.e1_locked)[:, only_e2_end_free]
        )
        if self.confidence_weighting:
            only_e2_end_free_prob *= all_confidences[only_e2_end_free]

        # Case 3: One start- and one end point are free and they are from
        #         the SAME event. Here, we model only the probabilities of
        #         the start and endpoints being locked.
        same_event_items = all_items[same_event]
        same_event_start_prob = lock_start_dist.log_prob(start_locked)[
            :, same_event
        ].to(self.device)
        same_event_end_prob = lock_end_dist.log_prob(end_locked)[:, same_event].to(
            self.device
        )
        same_event_prob = same_event_start_prob + same_event_end_prob
        if self.confidence_weighting:
            same_event_prob *= all_confidences[same_event]

        # Case 4: One start- and one end point are free and they are from
        #         DIFFERENT events. Here, we model the probability that
        #         the one event comes before the other (in addition to the
        #         probabilities of the start and endpoints being locked).
        diff_events_items = all_items[diff_events]
        diff_events_start_prob = lock_start_dist.log_prob(start_locked)[
            :, diff_events
        ].to(self.device)
        diff_events_end_prob = lock_end_dist.log_prob(end_locked)[:, diff_events].to(
            self.device
        )
        diff_events_mid_prob = lock_mid_dist.log_prob(diff_events_midpoint_probs)[
            :, diff_events
        ].to(self.device)
        diff_events_prob = (
            diff_events_start_prob + diff_events_end_prob + diff_events_mid_prob
        )
        if self.confidence_weighting:
            diff_events_prob *= all_confidences[diff_events]

        # Accumulate likelihoods
        total_ll.index_add_(1, start_and_end_locked_items, start_and_end_locked_prob)
        total_ll.index_add_(1, only_e1_start_free_items, only_e1_start_free_prob)
        total_ll.index_add_(1, only_e2_start_free_items, only_e2_start_free_prob)
        total_ll.index_add_(1, only_e1_end_free_items, only_e1_end_free_prob)
        total_ll.index_add_(1, only_e2_end_free_items, only_e2_end_free_prob)
        total_ll.index_add_(1, same_event_items, same_event_prob)
        total_ll.index_add_(1, diff_events_items, diff_events_prob)
        # TODO: add time to per-property likelihoods ("likelihoods")

        if self.clip_min_ll:
            min_ll = torch.log(torch.ones(total_ll.shape) * MIN_LIKELIHOOD).to(
                self.device
            )
            total_ll = torch.where(total_ll > min_ll, total_ll, min_ll,)

        return likelihoods, total_ll
