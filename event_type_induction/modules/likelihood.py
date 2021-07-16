import torch
import numpy as np
from collections import defaultdict
from event_type_induction.constants import *
from event_type_induction.utils import exp_normalize, get_prop_dim
from overrides import overrides
from torch import Tensor, LongTensor, FloatTensor, randn, log
from torch.distributions import (
    Bernoulli,
    Categorical,
    MultivariateNormal,
)
from torch.nn import Module, Parameter, ParameterDict
from typing import Any, Dict, Set, Tuple, Union


class Likelihood(Module):
    def __init__(
        self,
        property_subspaces: Set[str],
        annotator_confidences: Dict[str, Dict[int, float]],
        train_annotators: Set[str],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = True,
        clip_min_ll: bool = True,
        confidence_weighting: bool = False,
        use_random_effects: bool = True,
        random_effects: Dict[str, FloatTensor] = None,
        device: str = "cpu",
    ):
        """Base class for (non-vectorized) Event Type Induction Module likelihood computations

        Parameters
        ----------
        property_subspaces
            the set of UDS subspaces associated with the likelihood
        annotator_confidences
            a dictionary mapping annotators to dictionaries that map
            raw confidence scores to ridit-scored ones for that annotator
        train_annotators
            all and only the annotators in the train split
        metadata
            the annotation metadata for all subspaces in property_subspaces
        n_components
            the number of components (types) for this likelihood
        clip_min_ll
            whether to clip the likelihood to a minimum value
        confidence_weighting
            whether to weight the likelihood by annotator confidence
        use_random_effects
            whether to use annotator random effects or not
        device
            the device to use
        """
        super().__init__()
        self.type = None  # Overridden in children
        self.property_subspaces = property_subspaces
        self.train_annotators = train_annotators
        self.annotator_confidences = annotator_confidences
        self.metadata = metadata
        self.n_components = n_components
        self.use_ordinal = use_ordinal
        self.clip_min_ll = clip_min_ll
        self.confidence_weighting = confidence_weighting
        self.use_random_effects = use_random_effects
        self.device = torch.device(device)

        # Pre-trained random effects may be provided from a
        # multiview mixture model
        if random_effects is not None:
            self.random_effects = random_effects.to(self.device)

        # When using ordinal models, it's useful to have a quick
        # way of determining which properties are ordinal
        if self.use_ordinal:
            self.ordinal_properties = set()
            for subspace in self.metadata.subspaces:
                for prop, prop_meta in self.metadata[subspace].items():
                    if prop_meta.value.is_ordered_categorical:
                        self.ordinal_properties.add(prop)

        self.to(device)

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
            cuts_annotator = torch.cumsum(torch.exp(random), axis=-1)
            cuts_annotator -= torch.mean(cuts_annotator)
            cdf = torch.sigmoid(cuts_annotator - mu)
            cdf_high = torch.cat(
                [cdf, torch.ones(cdf.shape[0], 1).to(self.device)], axis=-1
            )
            cdf_low = torch.cat(
                [torch.zeros(cdf.shape[0], 1).to(self.device), cdf], axis=-1,
            )
            pmf = cdf_high - cdf_low
            return Categorical(pmf.squeeze())

        mean = mu if random is None else mu + random
        if mu.shape[-1] == 1:
            return Bernoulli(torch.sigmoid(mean).squeeze())
        else:
            return Categorical(torch.softmax(mean, -1).squeeze())

    def _get_random_effects(
        self, p: str, annotator: str
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prop_name = p.replace(".", "-")
        annotator_idx = int(annotator.split("-")[-1])

        # The main random effects for this property
        random = self.random_effects[prop_name] - self.random_effects[prop_name].mean()
        # If this annotator appears in the train split, fetch
        # the random effect for that annotator
        if annotator in self.train_annotators:
            random = random[annotator_idx][None]
        # Otherwise, use the mean across annotators (effectively 0)
        else:
            random = random.mean(0)

        # If this is a conditional property, we also have a term
        # for the Bernoulli indicating whether it applies or not
        if (
            self.use_ordinal
            and p in self.ordinal_properties
            and p in CONDITIONAL_PROPERTIES
        ):
            random_bernoulli = (
                self.random_effects[prop_name + "-applies"]
                - self.random_effects[prop_name + "-applies"].mean()
            )
            # Same idea as above
            if annotator in self.train_annotators:
                random_bernoulli = random_bernoulli[annotator_idx]
            else:
                random_bernoulli = random_bernoulli.mean(0)
            random = (random_bernoulli, random)

        return random

    def _get_value_and_confidence(
        self,
        p: str,
        subspace: str,
        prop_dim: int,
        annotator: str,
        value: Union[int, str, None],
        conf: int,
    ) -> Tuple[int, float]:

        # Get raw and ridit-scored confidence
        # -----------------------------------
        ridit_conf = self.annotator_confidences[annotator]
        if (
            ridit_conf is None or ridit_conf.get(conf) is None or ridit_conf[conf] < 0
        ):  # invalid confidence values; default to 1
            ridit_conf = 1
        else:
            ridit_conf = ridit_conf.get(conf, 1)

        # Convert the annotation value to the appropriate form
        # ----------------------------------------------------
        # Special case 1: None values (i.e. property was annotated as "doesn't apply")
        if value is None:
            # This should only be true of conditional properties
            assert (
                p in CONDITIONAL_PROPERTIES
            ), f"unexpected None value for property {p}"
            # If this is an ordinal property, and we're treating ordinal variables
            # as such, we set to NaN; this will be handled elsewhere
            if (
                self.use_ordinal
                and self.metadata[subspace][p].value.is_ordered_categorical
            ):
                val = np.nan
            # Otherwise, the "does not apply" case corresponds to the last category
            # when treating ordinal variables nominally.
            else:
                val = prop_dim - 1

        # Special case 2: String values (should only be duration annotations)
        elif isinstance(value, str):
            assert p == "duration", f"unexpected string value for property {p}"
            val = self.str_to_category[value]

        # Special case 3: Protoroles properties
        elif subspace == "protoroles":
            if conf == 0:
                if self.use_ordinal:
                    val = np.nan
                    ridit_conf = conf
                else:
                    val = prop_dim - 1
                    ridit_conf = 1
            else:
                val = value
                ridit_conf = 1

        # Default case: all other properties
        else:
            val = value
        return FloatTensor([val]).to(self.device), ridit_conf

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
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Compute the likelihood for a single annotation for a set of subspaces"""
        likelihoods = {}
        total_ll = torch.zeros(self.n_components).to(self.device)
        for subspace in self.property_subspaces:
            if subspace in annotation:
                for p in annotation[subspace]:

                    # Property name and dimension
                    prop_name = p.replace(".", "-")
                    prop_dim = get_prop_dim(self.metadata, subspace, p)

                    # The mean for the current property
                    mu = mus[prop_name]

                    # Compute log likelihood for each annotator in the annotation
                    # (for training, this should execute just once, as train data
                    # has only a single annotation per node)
                    for annotator, value in annotation[subspace][p]["value"].items():

                        # If necessary, converts the value to an appropriate integer
                        # and also fetches the ridit-scored confidence
                        conf = annotation[subspace][p]["confidence"][annotator]
                        val, ridit_conf = self._get_value_and_confidence(
                            p, subspace, prop_dim, annotator, value, conf
                        )

                        # Grab the random intercept for the current annotator
                        random = self._get_random_effects(p, annotator)

                        # Handle hurdle model for ordinal properties
                        if isinstance(random, tuple):
                            # Does the property apply?
                            property_applies_mu = mus[prop_name + "-applies"][:, None]
                            property_applies_shift, ordinal_shift = random
                            property_applies_dist = Bernoulli(
                                torch.sigmoid(
                                    property_applies_mu + property_applies_shift
                                )
                            )
                            if subspace == "protoroles":
                                property_applies = conf
                            else:
                                property_applies = 0 if np.isnan(val.item()) else 1
                            ll = property_applies_dist.log_prob(
                                property_applies
                            ).squeeze()

                            # If so, what is its value?
                            if not np.isnan(val.item()):
                                ordinal_dist = self._get_distribution(
                                    mu, ordinal_shift, prop_name=p
                                )
                                ll += ordinal_dist.log_prob(val)

                        # All other properties
                        else:
                            ll = self._get_distribution(
                                mu, random, prop_name=p
                            ).log_prob(val)

                        # Clip likelihood to minimum value if too close to zero
                        if self.clip_min_ll:
                            min_ll = torch.log(
                                torch.ones(ll.shape) * MIN_LIKELIHOOD
                            ).to(self.device)
                            ll = torch.where(
                                ll > log(Tensor([MIN_LIKELIHOOD]).to(self.device)),
                                ll,
                                min_ll,
                            )

                        # Weight by ridit-scored confidence
                        if self.confidence_weighting:
                            ll *= ridit_conf

                        # Add to likelihood-by-property
                        if p in likelihoods:
                            likelihoods[p] += ll
                        else:
                            likelihoods[p] = ll
                        total_ll += ll

        return likelihoods, total_ll


class PredicateNodeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        train_annotators: Set[str],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
        use_random_effects: bool = True,
        random_effects: Dict[str, FloatTensor] = None,
        device: str = "cpu",
    ):
        super().__init__(
            PREDICATE_NODE_SUBSPACES,
            annotator_confidences,
            train_annotators,
            metadata,
            n_components,
            use_ordinal,
            clip_min_ll,
            confidence_weighting,
            use_random_effects,
            random_effects,
            device=device,
        )
        self.type = Type.EVENT
        self.str_to_category = {
            cat: idx
            for idx, cat in enumerate(metadata["time"]["duration"].value.categories)
        }
        if random_effects is None:
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
        train_annotators: Set[str],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
        use_random_effects: bool = True,
        random_effects: Dict[str, FloatTensor] = None,
        device: str = "cpu",
    ):
        super().__init__(
            ARGUMENT_NODE_SUBSPACES,
            annotator_confidences,
            train_annotators,
            metadata,
            n_components,
            use_ordinal,
            clip_min_ll,
            confidence_weighting,
            use_random_effects,
            random_effects,
            device=device,
        )
        self.type = Type.ROLE

        if random_effects is None:
            self._initialize_random_effects()

    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        return super().forward(mus, annotation)


class SemanticsEdgeAnnotationLikelihood(Likelihood):
    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        train_annotators: Set[str],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
        use_random_effects: bool = True,
        random_effects: Dict[str, FloatTensor] = None,
        device: str = "cpu",
    ):
        super().__init__(
            SEMANTICS_EDGE_SUBSPACES,
            annotator_confidences,
            train_annotators,
            metadata,
            n_components,
            use_ordinal,
            clip_min_ll,
            confidence_weighting,
            use_random_effects,
            random_effects,
            device=device,
        )
        self.type = Type.PARTICIPANT

        if random_effects is None:
            self._initialize_random_effects()

    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Union[Dict[str, Tensor], Tensor]:
        return super().forward(mus, annotation)


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

    TIME_PROPS = ["rel-start1", "rel-end1", "rel-start2", "rel-end2"]

    @overrides
    def __init__(
        self,
        annotator_confidences: Dict[str, Dict[int, float]],
        train_annotators: Set[str],
        metadata: "UDSAnnotationMetadata",
        n_components: int,
        use_ordinal: bool = False,
        clip_min_ll: bool = False,
        confidence_weighting: bool = False,
        use_random_effects: bool = True,
        random_effects: Dict[str, FloatTensor] = None,
        device: str = "cpu",
    ):
        super().__init__(
            DOCUMENT_EDGE_SUBSPACES,
            annotator_confidences,
            train_annotators,
            metadata,
            n_components,
            use_ordinal,
            clip_min_ll,
            confidence_weighting,
            use_random_effects,
            random_effects,
            device=device,
        )
        self.type = Type.RELATION
        self.e1_locked = torch.tensor([self.E1_LOCKED]).to(self.device)
        self.e2_locked = torch.tensor([self.E2_LOCKED]).to(self.device)
        self.both_locked = torch.tensor([self.BOTH_LOCKED]).to(self.device)

        if random_effects is None:
            self._initialize_random_effects()

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

    def _get_temporal_relation(
        self, temp_rels: Dict[str, Dict]
    ) -> Tuple[int, int, int]:
        e1_start_locked = temp_rels["rel-start1"] == self.STARTPOINT_LOCK_VAL
        e2_start_locked = temp_rels["rel-start2"] == self.STARTPOINT_LOCK_VAL
        e1_end_locked = temp_rels["rel-end1"] == self.ENDPOINT_LOCK_VAL
        e2_end_locked = temp_rels["rel-end2"] == self.ENDPOINT_LOCK_VAL

        e1_start_and_e2_end_free = (not e1_start_locked) and (not e2_end_locked)
        e2_start_and_e1_end_free = (not e2_start_locked) and (not e1_end_locked)

        # startpoints
        if e1_start_locked == e2_start_locked:
            start_locked = self.BOTH_LOCKED
        elif e1_start_locked:
            start_locked = self.E1_LOCKED
        else:
            start_locked = self.E2_LOCKED

        # endpoints
        if e1_end_locked == e2_end_locked:
            end_locked = self.BOTH_LOCKED
        elif e1_end_locked:
            end_locked = self.E1_LOCKED
        else:
            end_locked = self.E2_LOCKED

        # midpoints
        if e1_start_and_e2_end_free:
            if temp_rels["rel-start1"] == temp_rels["rel-end2"]:
                mid_locked = self.E1_EQUALS_E2
            elif temp_rels["rel-start1"] < temp_rels["rel-end2"]:
                mid_locked = self.E1_BEFORE_E2
            else:
                mid_locked = self.E2_BEFORE_E1
        elif e2_start_and_e1_end_free:
            if temp_rels["rel-start2"] == temp_rels["rel-end1"]:
                mid_locked = self.E1_EQUALS_E2
            elif temp_rels["rel-start2"] < temp_rels["rel-end1"]:
                mid_locked = self.E2_BEFORE_E1
            else:
                mid_locked = self.E1_BEFORE_E2
        else:
            mid_locked = None

        mid_locked = (
            LongTensor([mid_locked]).to(self.device) if mid_locked is not None else None
        )
        return (
            LongTensor([start_locked]).to(self.device),
            LongTensor([end_locked]).to(self.device),
            mid_locked,
        )

    @overrides
    def forward(
        self, mus: ParameterDict, annotation: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        likelihoods = {}
        total_ll = torch.zeros(self.n_components).to(self.device)
        temp_rels = defaultdict(dict)
        temp_rel_ridit_confs = defaultdict(int)
        for subspace in self.property_subspaces:
            if subspace in annotation:
                for p in annotation[subspace]:
                    prop_name = p.replace(".", "-")
                    for annotator, value in annotation[subspace][p]["value"].items():

                        # Temporal relations likelihoods handled separately below
                        if subspace == "time":
                            temp_rels[annotator][p] = value
                            conf = annotation[subspace][p]["confidence"][annotator]
                            temp_rel_ridit_confs[
                                annotator
                            ] = self.annotator_confidences[annotator].get(conf, 1)
                            continue

                        # The mean for the current property
                        mu = mus[prop_name]

                        # Grab the random intercept for the current annotator
                        random = self._get_random_effects(p, annotator)

                        # Compute log-likelihood (clipping to prevent underflow)
                        dist = self._get_distribution(mu, random)
                        ll = dist.log_prob(FloatTensor([value]).to(self.device))

                        if self.clip_min_ll:
                            min_ll = torch.log(
                                torch.ones(ll.shape) * MIN_LIKELIHOOD
                            ).to(self.device)
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
                        total_ll += likelihoods[p]

        # Can only compute the likelihood for temporal relations once
        # we have all four start- and endpoints for each annotator
        start_locked_mu, end_locked_mu, mid_locked_mu = (
            mus["time-lock_start_mu"],
            mus["time-lock_end_mu"],
            mus["time-lock_mid_mu"],
        )
        likelihoods["time"] = torch.zeros(self.n_components).to(self.device)
        for a, rels in temp_rels.items():

            # Determine whether the start-, end-, and midpoints are locked
            start_locked, end_locked, mid_locked = self._get_temporal_relation(rels)

            # Get random effects
            start_locked_random = self._get_random_effects("time-lock_start", a)
            end_locked_random = self._get_random_effects("time-lock_end", a)

            # Get categorical distributions for start-, endpoints
            start_locked_dist = self._get_distribution(
                start_locked_mu, start_locked_random
            )
            end_locked_dist = self._get_distribution(end_locked_mu, end_locked_random)

            # Calculate likelihood
            start_locked_prob = start_locked_dist.log_prob(start_locked)
            end_locked_prob = end_locked_dist.log_prob(end_locked)
            ll = start_locked_prob + end_locked_prob

            # If two midpoints are free, calculate midpoint likelihood
            if mid_locked is not None:
                mid_locked_random = self._get_random_effects("time-lock_mid", a)
                mid_locked_dist = self._get_distribution(
                    mid_locked_mu, mid_locked_random
                )
                mid_locked_prob = mid_locked_dist.log_prob(mid_locked)
                ll += mid_locked_prob

            # Normalize and clip
            if self.clip_min_ll:
                min_ll = torch.log(torch.ones(ll.shape) * MIN_LIKELIHOOD).to(
                    self.device
                )
                ll = torch.where(
                    ll > log(Tensor([MIN_LIKELIHOOD]).to(self.device)), ll, min_ll
                )

            # TODO: toggle confidence weight
            likelihoods["time"] += temp_rel_ridit_confs[a] * ll

        total_ll += likelihoods["time"]

        return likelihoods, total_ll
