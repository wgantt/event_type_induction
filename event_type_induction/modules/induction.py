# Package-internal imports
from event_type_induction.modules.factor_graph import (
    VariableNode,
    VariableType,
    LikelihoodFactorNode,
    PriorFactorNode,
    FactorGraph,
)
from event_type_induction.constants import *
from event_type_induction.modules.likelihood import (
    ArgumentNodeAnnotationLikelihood,
    PredicateNodeAnnotationLikelihood,
    SemanticsEdgeAnnotationLikelihood,
    DocumentEdgeAnnotationLikelihood,
)
from event_type_induction.modules.freezable_module import FreezableModule
from event_type_induction.utils import load_annotator_ids

# Package-external imports
import torch
import decomp
from decomp.semantics.uds import UDSDocumentGraph
from collections import defaultdict
from torch.nn import Parameter, ParameterDict, ModuleDict
from torch.nn.functional import softmax
from torch.distributions import Categorical, Normal, Bernoulli, MultivariateNormal
from typing import Dict, Tuple


class EventTypeInductionModel(FreezableModule):
    """Base module for event type induction

    TODO
        - Implement forward function
        - Reduce graph size by filtering out semantics edges that don't
          have annotations
    """

    def __init__(
        self,
        n_event_types: int,
        n_role_types: int,
        n_relation_types: int,
        n_entity_types: int,
        bp_iters: int,
        uds: "UDSCorpus",
        device: str = "cpu",
        random_seed: int = 42,
    ):
        super().__init__()

        # Utilities
        self.uds = uds
        self.random_seed = random_seed
        self.device = torch.device(device)

        # Number of iterations for which to run message passing
        self.bp_iters = bp_iters

        # Initialize categorical distributions for the different types.
        # We do not place priors on any of these distributions,
        # initializing them randomly instead
        self.n_event_types = n_event_types
        self.n_role_types = n_role_types
        self.n_entity_types = n_entity_types

        # +1 for the "no-relation" relation
        self.n_relation_types = n_relation_types + 1

        # Participants may be either events or entities
        self.n_participant_types = n_event_types + n_entity_types

        # We initialize all probabilities using a softmax-transformed
        # unit normal, and store them as log probabilities
        clz = self.__class__

        # Event types
        self.event_probs = clz._initialize_log_prob((self.n_event_types))

        # Participants are either events or entities, which is
        # determined by a Bernoulli
        self.participant_domain_prob = clz._initialize_log_prob((1))

        # Entity-type participant probabilities (event-type participants
        # are selected from event_probs above)
        self.entity_probs = clz._initialize_log_prob((self.n_entity_types))

        # Role types: separate distribution for each event type (of the
        # predicate) and participant type (of the argument)
        self.role_probs = clz._initialize_log_prob(
            (self.n_event_types, self.n_participant_types, self.n_role_types)
        )

        # Relation types: separate distribution for each possible pairing
        # of participant types
        self.relation_probs = clz._initialize_log_prob(
            (self.n_participant_types, self.n_participant_types, self.n_relation_types)
        )

        # We enforce that the only possible relation type for a relation
        # involving an entity participant is "no relation." This is
        # because a relation can obtain only between two *events*
        self.relation_probs[self.n_entity_types :, :, :-1] = NEG_INF  # = log(0)
        self.relation_probs[self.n_entity_types :, :, -1] = 0  # = log(1)
        self.relation_probs[:, self.n_entity_types :, :-1] = NEG_INF
        self.relation_probs[:, self.n_entity_types :, -1] = 0

        # Initialize mus (expected annotations)
        self.event_mus = clz._initialize_mus(
            PREDICATE_ANNOTATION_ATTRIBUTES, self.n_event_types
        )
        self.role_mus = clz._initialize_mus(
            SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES, self.n_role_types
        )
        self.participant_mus = clz._initialize_mus(
            ARGUMENT_ANNOTATION_ATTRIBUTES, self.n_participant_types
        )
        self.relation_mus = clz._initialize_mus(
            DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES, self.n_relation_types
        )

        # Fetch annotator IDs from UDS, used by the likelihood
        # modules to initialize their random effects
        (
            pred_node_annotators,
            arg_node_annotators,
            sem_edge_annotators,
            doc_edge_annotators,
        ) = load_annotator_ids(uds)

        # Modules for calculating likelihoods
        self.pred_node_likelihood = PredicateNodeAnnotationLikelihood(
            pred_node_annotators
        )
        self.arg_node_likelihood = ArgumentNodeAnnotationLikelihood(arg_node_annotators)
        self.semantics_edge_likelihood = SemanticsEdgeAnnotationLikelihood(
            sem_edge_annotators
        )
        self.doc_edge_likelihood = DocumentEdgeAnnotationLikelihood(doc_edge_annotators)

    @classmethod
    def _initialize_mus(cls, attribute_dict, n_types) -> ParameterDict:
        """Initialize mu parameters for every UDS property, for every cluster"""

        # One mu per event type (cluster)
        mu_dict = {}
        for domain, props in attribute_dict.items():
            for prop, prop_features in props.items():
                # Dictionary keys are of the form 'domain-prop'
                prop_name = "-".join([domain, prop]).replace(".", "-")
                mu_dict[prop_name] = cls._initialize_log_prob(
                    (n_types, prop_features["dim"])
                )
        return ParameterDict(mu_dict)

    @staticmethod
    def _initialize_log_prob(shape: Tuple[int]) -> Parameter:
        """Unit random normal-based initialization for model parameters

        The result is returned as log probabilities
        """
        return Parameter(torch.log(softmax(torch.randn(shape), -1)))

    def compute_annotation_likelihood(
        self, fg: FactorGraph, beliefs: Dict[str, torch.Tensor]
    ) -> float:
        """Compute likelihood of UDS annotations

        Parameters
        ----------
        fg
            The factor graph used to compute the likelihood
        beliefs
            A dictionary containing the beliefs (marginals) for each variable
            node
        """
        ll = torch.FloatTensor([0])
        for lf_node_name, lf_node in fg.likelihood_factor_nodes.items():
            # Only compute likelihoods over nodes that actually have
            # annotations
            if lf_node.per_type_likelihood is not None:
                # Get the variable node associated with this likelihood.
                # Each likelihood has exactly one edge, which connects it
                # to its variable node
                var_node = list(fg.edges(lf_node))[0][1]

                # Normalize beliefs + likelihood
                belief = beliefs[var_node.label]
                likelihood = lf_node.per_type_likelihood
                belief_norm = torch.log(
                    torch.exp(belief) / torch.sum(torch.exp(belief))
                )
                likelihood_norm = torch.log(
                    torch.exp(likelihood) / torch.sum(torch.exp(likelihood))
                )
                ll += torch.logsumexp(belief_norm + likelihood_norm, 0)

        # TODO: Fix! This is currently always NaN.
        return ll

    def compute_random_loss(self):
        """Compute the loss for the prior(s) over annotator random effects"""
        likelihoods = [
            self.pred_node_likelihood,
            self.arg_node_likelihood,
            self.semantics_edge_likelihood,
            self.doc_edge_likelihood,
        ]
        return torch.sum([ll.compute_random_loss() for ll in likelihoods])

    def construct_factor_graph(self, document: UDSDocumentGraph) -> FactorGraph:
        """Construct the factor graph for a document

        Parameters
        ----------
        document
            The UDSDocumentGraph for which to construct the factor graph

        TODO:
            - Verify that messages are initialized according to the uniform
              distribution
            - Verify that variable nodes are added only for nodes or edges
              that are actually annotated
        """

        # Initialize the factor graph
        fg = FactorGraph()

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
                if not pred_lf_node_name in fg.factor_nodes:
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
                if not arg_lf_node_name in fg.factor_nodes:
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
                # edge annotations. This is conditioned only on the
                # argument's participant type
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
                    participant_probs = torch.cat(
                        [
                            self.event_probs + self.participant_domain_prob,
                            self.entity_probs - self.participant_domain_prob,
                        ]
                    )
                    participant_type_pf_node = PriorFactorNode(
                        participant_type_pf_node_name,
                        participant_probs,
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
            # and connect it with the document edge variable node
            relation_pf_node = PriorFactorNode(
                FactorGraph.get_node_name("pf", v1, v2, "relation"),
                self.relation_probs,
                VariableType.RELATION,
            )
            fg.set_node(relation_pf_node)
            fg.set_edge(relation_v_node, relation_pf_node, 2)

            # We also connect this factor node to the variable nodes for
            # the predicates or arguments it relates.
            for factor_dim, var_node_name in enumerate([v1, v2]):

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

                # Verify that the variable node was in fact created in
                # the sentence-level loop
                assert (
                    var_node is not None
                ), f"Variable node {var_node_name} not found in factor graph; annotations: {doc_edge_anno}"

                # Connect the variable node to the prior factor node for the
                # document edge
                fg.set_edge(var_node, relation_pf_node, factor_dim)

        return fg

    def forward(self, document: decomp.semantics.uds.UDSDocument) -> torch.FloatTensor:
        fg = self.construct_factor_graph(document)
        beliefs = fg.loopy_sum_product(self.bp_iters, fg.variable_nodes.values())
        fixed_loss = torch.FloatTensor(self.compute_annotation_likelihood(fg, beliefs))
        random_loss = torch.FloatTensor(self.compute_random_loss())
        return fixed_loss, random_loss
