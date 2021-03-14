# Package-internal imports
from event_type_induction.modules.factor_graph import (
    VariableNode,
    VariableType,
    LikelihoodFactorNode,
    PriorFactorNode,
    FactorGraph,
)
from event_type_induction.modules.likelihood import (
    ArgumentNodeAnnotationLikelihood,
    PredicateNodeAnnotationLikelihood,
    SemanticsEdgeAnnotationLikelihood,
    DocumentEdgeAnnotationLikelihood,
)
from event_type_induction.utils import *
from event_type_induction.constants import *
from scripts.cluster_init import MultiviewMixtureModel
from scripts.setup_logging import setup_logging

# Package-external imports
import time
import torch
import numpy as np
from decomp import UDSCorpus
from decomp.semantics.uds import UDSDocument, UDSDocumentGraph
from collections import defaultdict
import networkx as nx
from torch.nn import Module, Parameter, ParameterDict
from torch.nn.functional import softmax
from typing import Dict, Tuple, Union, Optional


LOG = setup_logging()


class EventTypeInductionModel(Module):
    def __init__(
        self,
        n_event_types: int,
        n_role_types: int,
        n_relation_types: int,
        n_participant_types: int,
        bp_iters: int,
        uds: "UDSCorpus" = None,
        use_ordinal: bool = True,
        clip_min_ll: bool = True,
        confidence_weighting: bool = True,
        use_random_effects: bool = True,
        mmm_ckpts=None,
        device: str = "cpu",
        random_seed: int = 42,
    ):
        """Base module for event type induction"""

        super().__init__()
        self.use_ordinal = use_ordinal
        self.clip_min_ll = clip_min_ll
        self.confidence_weighting = confidence_weighting
        self.use_random_effects = use_random_effects
        self.random_seed = random_seed
        self.device = torch.device(device)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        clz = self.__class__

        if uds is None:
            self.uds = UDSCorpus(version="2.0", annotation_format="raw")
            load_event_structure_annotations(self.uds)
        else:
            self.uds = uds

        # Number of iterations for which to run BP
        self.bp_iters = bp_iters

        # Initialize model params from MMM checkpoints
        if mmm_ckpts is not None:
            for t in Type:
                assert t in mmm_ckpts, f"Missing checkpoint for type {t}"

            # Load number of types, prior probabilties, means, and covariance
            # (for relation types) from the MMM checkpoints
            (
                self.n_event_types,
                event_probs,
                self.event_mus,
                event_random_effects,
            ) = self._load_mmm_params(mmm_ckpts[Type.EVENT], Type.EVENT)

            # events
            self.event_probs = Parameter(event_probs)
            (
                self.n_participant_types,
                participant_probs,
                self.participant_mus,
                participant_random_effects,
            ) = self._load_mmm_params(mmm_ckpts[Type.PARTICIPANT], Type.PARTICIPANT)

            # participants
            self.participant_probs = Parameter(participant_probs)

            # roles
            (
                self.n_role_types,
                role_probs,
                self.role_mus,
                role_random_effects,
            ) = self._load_mmm_params(mmm_ckpts[Type.ROLE], Type.ROLE)
            self.role_probs = Parameter(
                role_probs.repeat((self.n_event_types, self.n_participant_types, 1))
            )

            # relations
            (
                self.n_relation_types,
                relation_probs,
                self.relation_mus,
                relation_random_effects,
            ) = self._load_mmm_params(mmm_ckpts[Type.RELATION], Type.RELATION)

            # see comments below for more detail on these three parameters
            self.e_to_e_relation_probs = Parameter(
                relation_probs[None, None].repeat(
                    self.n_event_types, self.n_event_types, 1,
                )
            )
            self.e_to_p_relation_probs = Parameter(
                relation_probs[None, None].repeat(
                    self.n_event_types, self.n_participant_types, 1,
                )
            )
            self.p_to_p_relation_probs = Parameter(
                relation_probs[None, None].repeat(
                    self.n_participant_types, self.n_participant_types, 1,
                )
            )

        # No MMM params provided: randomly initialize model params
        else:
            # Initialize categorical distributions for the different types.
            # We do not place priors on any of these distributions,
            # initializing them randomly instead
            self.n_event_types = n_event_types
            self.n_role_types = n_role_types
            self.n_participant_types = n_participant_types
            self.n_relation_types = n_relation_types

            # Event types
            self.event_probs = self._initialize_log_prob((self.n_event_types,))

            # Entity-type participant probabilities (event-type participants
            # are selected from event_probs above)
            self.participant_probs = clz._initialize_log_prob(
                (self.n_participant_types,)
            )

            # Role types: separate distribution for each event type (of the
            # predicate) and participant type (of the argument)
            self.role_probs = clz._initialize_log_prob(
                (self.n_event_types, self.n_participant_types, self.n_role_types)
            )

            # Relation types: Separate parameters for each of the possible
            # pairs of items the relation may relate

            # 1. event-event relations
            self.e_to_e_relation_probs = clz._initialize_log_prob(
                (self.n_event_types, self.n_event_types, self.n_relation_types,),
            )

            # 2. event-participant relations
            self.e_to_p_relation_probs = clz._initialize_log_prob(
                (self.n_event_types, self.n_participant_types, self.n_relation_types,),
            )

            # 3. participant-participant relations
            self.p_to_p_relation_probs = clz._initialize_log_prob(
                (
                    self.n_participant_types,
                    self.n_participant_types,
                    self.n_relation_types,
                ),
            )

            # Initialize mus (expected annotations)
            self.event_mus = self._initialize_event_params()
            self.role_mus = self._initialize_role_params()
            self.participant_mus = self._initialize_participant_params()
            self.relation_mus, self._initialize_relation_params()

            # No random effects provided; they will be
            # randomly initialized in the likelihoods
            event_random_effects = None
            role_random_effects = None
            participant_random_effects = None
            relation_random_effects = None

        # Fetch annotator IDs from UDS, used by the likelihood
        # modules to initialize their random effects
        (
            pred_node_annotators,
            arg_node_annotators,
            sem_edge_annotators,
            doc_edge_annotators,
        ) = load_annotator_ids(self.uds)

        # Load the annotators that appear only in the train split
        train_annotators = load_train_annotators(self.uds)

        # Load ridit-scored annotator confidence values
        ridits = ridit_score_confidence(self.uds)
        pred_node_annotator_confidence = {
            a: ridits.get(a) for a in pred_node_annotators
        }
        arg_node_annotator_confidence = {a: ridits.get(a) for a in arg_node_annotators}
        sem_edge_annotator_confidence = {a: ridits.get(a) for a in sem_edge_annotators}
        doc_edge_annotator_confidence = {a: ridits.get(a) for a in doc_edge_annotators}

        # Modules for calculating likelihoods
        self.pred_node_likelihood = PredicateNodeAnnotationLikelihood(
            pred_node_annotator_confidence,
            train_annotators,
            self.uds.metadata.sentence_metadata,
            self.n_event_types,
            self.use_ordinal,
            self.clip_min_ll,
            self.confidence_weighting,
            self.use_random_effects,
            event_random_effects,
            self.device,
        )
        self.arg_node_likelihood = ArgumentNodeAnnotationLikelihood(
            arg_node_annotator_confidence,
            train_annotators,
            self.uds.metadata.sentence_metadata,
            self.n_participant_types,
            self.use_ordinal,
            self.clip_min_ll,
            self.confidence_weighting,
            self.use_random_effects,
            participant_random_effects,
            self.device,
        )
        self.semantics_edge_likelihood = SemanticsEdgeAnnotationLikelihood(
            sem_edge_annotator_confidence,
            train_annotators,
            self.uds.metadata.sentence_metadata,
            self.n_role_types,
            self.use_ordinal,
            self.clip_min_ll,
            self.confidence_weighting,
            self.use_random_effects,
            role_random_effects,
            self.device,
        )
        self.doc_edge_likelihood = DocumentEdgeAnnotationLikelihood(
            doc_edge_annotator_confidence,
            train_annotators,
            self.uds.metadata.document_metadata,
            self.n_relation_types,
            self.use_ordinal,
            self.clip_min_ll,
            self.confidence_weighting,
            self.use_random_effects,
            relation_random_effects,
            self.device,
        )

    def _load_mmm_params(
        self, ckpt_path: str, t: Type
    ) -> Tuple[int, Parameter, ParameterDict, Optional[ParameterDict]]:
        """Loads relevant parameters from a MultiviewMixtureModel"""
        ckpt_dict = torch.load(ckpt_path, map_location=self.device)
        n_types = ckpt_dict["component_weights"].shape[0]
        mmm_prior = exp_normalize(ckpt_dict["component_weights"])
        mmm_mus = ParameterDict(
            {k[4:]: Parameter(v) for k, v in ckpt_dict.items() if "mus" in k}
        )
        mmm_random_effects = ParameterDict(
            {
                k.split(".")[-1]: Parameter(v)
                for k, v in ckpt_dict.items()
                if "random_effects" in k
            }
        )
        return n_types, mmm_prior, mmm_mus, mmm_random_effects

    def _initialize_params(self, uds, n_types, subspaces) -> ParameterDict:
        """Initialize mu parameters for properties of a set of subspaces"""
        mu_dict = {}
        for subspace in sorted(subspaces):
            for prop, prop_metadata in sorted(
                uds.metadata.sentence_metadata.metadata[subspace].items()
            ):
                prop_dim = get_prop_dim(subspace, prop)
                mu_dict[prop.replace(".", "-")] = self.__class__._initialize_log_prob(
                    (n_types, prop_dim)
                )

        return ParameterDict(mu_dict)

    def _initialize_event_params(self) -> ParameterDict:
        """Initialize mu parameters for predicate node properties, for every cluster"""
        return self._initialize_params(self.uds, self.n_event_types, EVENT_SUBSPACES)

    def _initialize_role_params(self) -> ParameterDict:
        """Initialize parameters for semantics edge attributes"""
        return self._initialize_params(self.uds, self.n_role_types, ROLE_SUBSPACES)

    def _initialize_participant_params(self) -> ParameterDict:
        """Initialize mu parameters for argument node properties, for every cluster"""
        return self._initialize_params(
            self.uds, self.n_participant_types, PARTICIPANT_SUBSPACES
        )

    def _initialize_relation_params(self) -> Tuple[ParameterDict, Parameter]:
        """Initialize parameters for document edge attributes"""
        # TODO: update --- way out of date
        mu_dict = {}

        for subspace in sorted(RELATION_SUBSPACES):
            # temporal relations are handled specially below
            if subspace == "time":
                continue
            for prop, prop_metadata in sorted(
                self.uds.metadata.document_metadata.metadata[subspace].items()
            ):
                prop_dim = get_prop_dim(subspace, prop)
                mu_dict[prop.replace(".", "-")] = self.__class__._initialize_log_prob(
                    (self.n_relation_types, prop_dim)
                )

        # Initialize temporal relation mus based on centroids for
        # the most common Allen Relations in the data set
        temporal_relation_params = self._get_temporal_relations_params()
        sorted_means, sorted_covs = [], []
        for (_, mean, cov) in sorted(
            temporal_relation_params.values(), key=lambda x: x[0], reverse=True
        )[: self.n_relation_types]:
            sorted_means.append(mean)
            # hack to avoid initializing to singular matrices;
            # set zero entries to 1
            for i in range(4):
                if cov[i][i] == 0:
                    cov[i][i] = 1
            sorted_covs.append(cov)
        mu_dict["time"] = Parameter(torch.stack(sorted_means))
        covs = torch.stack(sorted_covs)

        return ParameterDict(mu_dict), Parameter(covs)

    def _get_temporal_relations_params(
        self,
    ) -> Dict[AllenRelation, Tuple[int, torch.Tensor]]:
        """Initializes means for temporal relation annotation likelihoods"""

        # Group all temporal relations annotations by Allen Relation
        allen_rels = defaultdict(list)
        for docid, doc in self.uds.documents.items():
            for edge, anno in doc.document_graph.edges().items():

                # Skip document edges not annotated for temporal relations
                if not "time" in anno:
                    continue

                # Get the four start- and endpoints for each annotator
                time = anno["time"]
                for annotator in time["rel-start1"]["value"]:
                    e1_start = time["rel-start1"]["value"][annotator]
                    e1_end = time["rel-end1"]["value"][annotator]
                    e2_start = time["rel-start2"]["value"][annotator]
                    e2_end = time["rel-end2"]["value"][annotator]

                    # This is a hack to avoid a select few annotations that
                    # currently have negative values, which they shouldn't
                    if any([t < 0 for t in [e1_start, e1_end, e2_start, e2_end]]):
                        continue

                    # Determine the Allen relation for this annotation, add
                    # it to the dictionary
                    allen_rels[
                        get_allen_relation(e1_start, e1_end, e2_start, e2_end)
                    ].append([e1_start, e1_end, e2_start, e2_end])

        # For each Allen relation, return the number of annotations that
        # realize that relation and the centroid of all such annotations
        return {
            k: (
                len(v),
                torch.FloatTensor(np.mean(np.array(v), axis=0)),
                torch.FloatTensor(np.cov(np.array(v), rowvar=False)),
            )
            for k, v in allen_rels.items()
        }

    @staticmethod
    def _initialize_log_prob(
        shape: Tuple[int], uniform=False, as_parameter=True
    ) -> Union[torch.Tensor, Parameter]:
        """Unit random normal-based initialization for model parameters

        The result is returned as log probabilitiess
        """
        if shape[-1] == 1:  # Bernoulli
            val = torch.sigmoid(torch.randn((shape[0],)))
        else:
            val = softmax(torch.randn(shape), -1)

        if as_parameter:
            return Parameter(torch.log(val))
        else:
            return torch.log(val)

    def compute_posteriors(
        self,
        fg: FactorGraph,
        beliefs: Dict[str, torch.Tensor],
        return_posteriors: bool = False,
    ) -> torch.FloatTensor:
        """Compute marginals for variable nodes in the factor graph

        Parameters
        ----------
        fg
            The factor graph used to compute the marginals
        beliefs
            A dictionary containing the beliefs (marginals) for each variable
            node
        """
        post = torch.FloatTensor([0]).to(self.device)
        per_item_posteriors = defaultdict(dict)
        for lf_node_name, lf_node in fg.likelihood_factor_nodes.items():
            # Only compute marginals over nodes that actually have
            # annotations (i.e. have likelihood factor nodes)
            if lf_node.per_type_likelihood is not None:
                # Get the variable node associated with this likelihood.
                # Each likelihood has exactly one edge, which connects it
                # to its variable node
                var_node = list(fg.edges(lf_node))[0][1]

                # Normalize beliefs
                prior = exp_normalize(var_node.belief(lf_node))
                likelihood = lf_node.per_type_likelihood
                post += torch.logsumexp(prior + likelihood, 0)
                if return_posteriors:
                    per_item_posteriors[var_node.label] = exp_normalize(
                        prior + likelihood
                    )

        return -post, per_item_posteriors

    def random_loss(self):
        """Compute the loss for the prior(s) over annotator random effects"""
        likelihoods = [
            self.pred_node_likelihood,
            self.arg_node_likelihood,
            self.semantics_edge_likelihood,
            self.doc_edge_likelihood,
        ]
        return torch.sum(torch.FloatTensor([ll.random_loss() for ll in likelihoods]))[
            None
        ].to(self.device)

    def construct_factor_graph(self, document: UDSDocumentGraph) -> FactorGraph:
        """Construct the factor graph for a document

        Parameters
        ----------
        document
            The UDSDocumentGraph for which to construct the factor graph
        """

        # Initialize the factor graph
        fg = FactorGraph(device=self.device)

        # Get the sentence graphs contained in the document
        sentences = list(document.sentence_ids)

        # Generate sentence-level factor graph structure first
        for s in sentences:

            # Retrieve the UDSSentenceGraph for the current sentence
            # in the document
            sentence = self.uds[s]

            # Process predicates and arguments via the semantics edges
            # that relate them
            for (v1, v2), sem_edge_anno in sentence.semantics_edges().items():
                # Determine which variable is the predicate and which the argument
                if "pred" in v1:
                    pred, arg = v1, v2
                else:
                    pred, arg = v2, v1

                # Optimization: If there are no annotations on the predicate
                # or argument nodes, or on the edge between them, we just omit
                # them all from the factor graph. This considerably speeds up BP.
                edge_subspaces = set(sem_edge_anno.keys()).intersection(
                    SEMANTICS_EDGE_SUBSPACES
                )
                pred_subspaces = set(
                    sentence.semantics_nodes[pred].keys()
                ).intersection(PREDICATE_NODE_SUBSPACES)
                arg_subspaces = set(sentence.semantics_nodes[arg].keys()).intersection(
                    ARGUMENT_NODE_SUBSPACES
                )
                if not (edge_subspaces or pred_subspaces or arg_subspaces):
                    continue

                # VARIABLE NODES ------------------------------------

                # Add a variable node for the predicate's event type,
                # but only if one is not already in the graph
                event_v_node_name = FactorGraph.get_node_name("v", pred, "event")
                if not event_v_node_name in fg.variable_nodes:
                    event_v_node = VariableNode(
                        event_v_node_name, VariableType.EVENT, self.n_event_types
                    )
                    fg.set_node(event_v_node)
                else:
                    event_v_node = fg.variable_nodes[event_v_node_name]

                # Similarly add a variable node for the argument's
                # type, but only if it isn't already in the graph
                participant_v_node_name = FactorGraph.get_node_name(
                    "v", arg, "participant"
                )
                if not participant_v_node_name in fg.variable_nodes:
                    participant_v_node = VariableNode(
                        participant_v_node_name,
                        VariableType.PARTICIPANT,
                        self.n_participant_types,
                    )
                    fg.set_node(participant_v_node)
                else:
                    participant_v_node = fg.variable_nodes[participant_v_node_name]

                # Add a variable node for the predicate's role type.
                # Roles are relational, so we add one for *each* semantics edge.
                role_v_node = VariableNode(
                    FactorGraph.get_node_name("v", v1, v2, "role"),
                    VariableType.ROLE,
                    self.n_role_types,
                )
                fg.set_node(role_v_node)

                # LIKELIHOOD FACTOR NODES ---------------------------

                # Add a likelihood factor for the predicate. This is a unary
                # factor that computes the likelihood of the node's annotations.
                pred_node_anno = sentence.predicate_nodes[pred]
                pred_lf_node_name = FactorGraph.get_node_name("lf", pred)
                if not pred_subspaces:
                    # Optimization: Do not add a likelihood
                    # factor for predicates without annotations
                    pass
                elif not pred_lf_node_name in fg.factor_nodes:
                    pred_lf_node = LikelihoodFactorNode(
                        pred_lf_node_name,
                        self.pred_node_likelihood,
                        self.event_mus,
                        pred_node_anno,
                    )
                    fg.set_node(pred_lf_node)
                    fg.set_edge(pred_lf_node, event_v_node, 0)
                else:
                    pred_lf_node = fg.factor_nodes[pred_lf_node_name]

                # Do the same for the argument.
                arg_node_anno = sentence.argument_nodes[arg]
                arg_lf_node_name = FactorGraph.get_node_name("lf", arg)
                if not arg_subspaces:
                    # Optimization: Do not add a likelihood
                    # factor for arguments without annotations
                    pass
                elif not arg_lf_node_name in fg.factor_nodes:
                    arg_lf_node = LikelihoodFactorNode(
                        arg_lf_node_name,
                        self.arg_node_likelihood,
                        self.participant_mus,
                        arg_node_anno,
                    )
                    fg.set_node(arg_lf_node)
                    fg.set_edge(arg_lf_node, participant_v_node, 0)
                else:
                    arg_lf_node = fg.factor_nodes[arg_lf_node_name]

                # We also add a likelihood factor node for the semantics
                # edge annotations (only if they exist). This is
                # conditioned only on the argument's participant type
                if edge_subspaces:
                    sem_edge_lf_node = LikelihoodFactorNode(
                        FactorGraph.get_node_name("lf", v1, v2),
                        self.semantics_edge_likelihood,
                        self.role_mus,
                        sem_edge_anno,
                    )
                    fg.set_node(sem_edge_lf_node)
                    fg.set_edge(sem_edge_lf_node, role_v_node, 0)

                # PRIOR FACTOR NODES --------------------------------

                # Create a prior factor for the predicate's event type,
                # which depends only on the predicate event type variable
                event_pf_node_name = FactorGraph.get_node_name("pf", pred, "event")
                if event_pf_node_name not in fg.factor_nodes:
                    event_pf_node = PriorFactorNode(
                        FactorGraph.get_node_name("pf", pred, "event"),
                        self.event_probs,
                        VariableType.EVENT,
                    )
                    fg.set_node(event_pf_node)
                    fg.set_edge(event_pf_node, event_v_node, 0)

                # Add a prior factor node for the participant type
                # (only one per argument). This depends only on the
                # argument's participant type variable.
                participant_type_pf_node_name = FactorGraph.get_node_name(
                    "pf", arg, "participant"
                )
                if participant_type_pf_node_name not in fg.factor_nodes:
                    participant_type_pf_node = PriorFactorNode(
                        participant_type_pf_node_name,
                        self.participant_probs,
                        VariableType.PARTICIPANT,
                    )
                    fg.set_node(participant_type_pf_node)
                    fg.set_edge(participant_type_pf_node, participant_v_node, 0)

                """
                Add a prior factor for the argument's role type
                (one per semantics edge). This depends not only
                on the argument role type variable, but also on
                the argument's participant type and the associated
                predicate's event type
                """
                role_pf_node = PriorFactorNode(
                    FactorGraph.get_node_name("pf", v1, v2, "role"),
                    self.role_probs,
                    VariableType.ROLE,
                )
                fg.set_node(role_pf_node)
                fg.set_edge(role_pf_node, event_v_node, 0)
                fg.set_edge(role_pf_node, participant_v_node, 1)
                fg.set_edge(role_pf_node, role_v_node, 2)

        # Generate document-level graph structure second (as it depends on the
        # sentence-level structure)
        for (v1, v2), doc_edge_anno in document.document_graph.edges.items():

            # VARIABLE NODES ------------------------------------

            # Create a variable node for the edge itself
            relation_v_node = VariableNode(
                FactorGraph.get_node_name("v", v1, v2, "relation"),
                VariableType.RELATION,
                self.n_relation_types,
            )
            fg.set_node(relation_v_node)

            # LIKELIHOOD FACTOR NODES ---------------------------

            # Create a likelihood factor node for its annotations
            doc_edge_lf_node = LikelihoodFactorNode(
                FactorGraph.get_node_name("lf", v1, v2),
                self.doc_edge_likelihood,
                self.relation_mus,
                doc_edge_anno,
            )
            fg.set_node(doc_edge_lf_node)
            fg.set_edge(relation_v_node, doc_edge_lf_node, 0)

            # PRIOR FACTOR NODES --------------------------------

            # Create a factor for the prior over the relation type
            # and connect it with the document edge variable node.
            # The factor we use for the prior depends on whether
            # the items related are predicate or argument nodes.
            factor_dim = {}
            if ("pred" in v1) and ("pred" in v2):
                relation_probs = self.e_to_e_relation_probs
                factor_dim[v1], factor_dim[v2] = (0, 1)
            elif ("arg" in v1) and ("arg" in v2):
                relation_probs = self.p_to_p_relation_probs
                factor_dim[v1], factor_dim[v2] = (0, 1)
            else:
                relation_probs = self.e_to_p_relation_probs
                factor_dim[v1], factor_dim[v2] = (0, 1) if "pred" in v1 else (1, 0)

            relation_pf_node = PriorFactorNode(
                FactorGraph.get_node_name("pf", v1, v2, "relation"),
                relation_probs,
                VariableType.RELATION,
            )
            fg.set_node(relation_pf_node)
            fg.set_edge(relation_v_node, relation_pf_node, 2)

            # We also connect this factor node to the variable nodes for
            # the predicates or arguments it relates.
            for var_node_name in [v1, v2]:

                # Identify the variable node for the predicate/argument
                # (It must have been created in the sentence-level loop)
                sem_node_name = var_node_name.replace("document", "semantics")
                if "pred" in var_node_name:
                    # Fetch the variable node for the predicate's event type
                    fg_node_name = FactorGraph.get_node_name(
                        "v", sem_node_name, "event"
                    )
                else:
                    # Fetch the variable node for the argument's
                    # participant type
                    fg_node_name = FactorGraph.get_node_name(
                        "v", sem_node_name, "participant"
                    )
                var_node = fg.variable_nodes.get(fg_node_name)

                # Verify that the variable node was created in
                # the sentence-level loop
                if var_node is None:
                    LOG.warn(
                        f"Variable node {var_node_name} not found in factor graph; annotations: {doc_edge_anno}"
                    )
                else:
                    # Connect the variable node to the prior factor node for the
                    # document edge
                    fg.set_edge(var_node, relation_pf_node, factor_dim[var_node_name])

        assert len(fg._variable_nodes) == len(
            fg._prior_factor_nodes
        ), f"Some variable node is missing a prior for document {document.name}"
        assert len(fg._likelihood_factor_nodes) <= len(
            fg._variable_nodes
        ), f"More likelihoods than expected for document {document.name}"
        return fg

    def forward(
        self, document: UDSDocument, time_run=False, save_posteriors=False
    ) -> torch.FloatTensor:
        LOG.info(
            f"processing document {document.name} which has {len(document.sentence_ids)} sentences"
        )
        if time_run:
            LOG.info(f"Runtime analysis for {document.name}:")
            start_time = time.time()
            fg = self.construct_factor_graph(document)
            fg_construction_time = time.time()
            ncc = nx.number_connected_components
            LOG.info(
                f"Factor graph for document {document.name} has {ncc} connected components"
            )
            LOG.info(
                f"  Factor graph construction: {np.round(fg_construction_time - start_time, 3)}"
            )
            beliefs = fg.loopy_sum_product(self.bp_iters, fg.variable_nodes.values())
            bp_time = time.time()
            LOG.info(f"  BP: {np.round(bp_time - fg_construction_time, 3)}")
            fixed_loss, posteriors = self.compute_posteriors(
                fg, beliefs, save_posteriors
            )
            random_loss = self.random_loss()
            loss_calc_time = time.time()
            LOG.info(f"  Loss calculation: {np.round(loss_calc_time - bp_time, 3)}")
        else:
            fg = self.construct_factor_graph(document)
            ncc = nx.number_connected_components(fg)
            LOG.info(
                f"Factor graph for document {document.name} has {ncc} connected components"
            )
            beliefs = fg.loopy_sum_product(self.bp_iters, fg.variable_nodes.values())
            fixed_loss, posteriors = self.compute_posteriors(
                fg, beliefs, save_posteriors
            )
            random_loss = self.random_loss()
        return fixed_loss, random_loss, posteriors
