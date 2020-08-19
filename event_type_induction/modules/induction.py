# Package-internal imports
from event_type_induction.modules.factor_graph import (
	VariableNode,
	LikelihoodFactorNode,
	PriorFactorNode,
	FactorGraph,
	)
from event_type_induction.constants import *
from event_type_induction.modules.likelihood import (
	ArgumentNodeAnnotationLikelihood,
	PredicateNodeAnnotationLikelihood,
	SemanticsEdgeAnnotationLikelihood,
	DocumentEdgeAnnotationLikelihood
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
from torch.distributions import Categorical, Normal, Bernoulli
from typing import Tuple

class EventTypeInductionModel(FreezableModule):
	"""Base module for event type induction

	TODO
		- Implement forward function
	"""
	def __init__(self, n_event_types: int, n_role_types: int,
					   n_relation_types: int, n_entity_types: int,
					   uds: 'UDSCorpus', device: str = 'cpu',
					   random_seed: int = 42):
		super().__init__()

		# Utilities
		self.uds = uds
		self.random_seed = random_seed
		self.device = torch.device(device)

		"""
		Initialize categorical distributions for the different types.
		We do not place priors on any of these distributions, instead
		initializing them randomly
		"""
		self.n_event_types = n_event_types
		self.n_role_types = n_role_types
		self.n_relation_types = n_relation_types
		self.n_entity_types = n_entity_types
		self.n_participant_types = n_event_types + n_entity_types

		# Participants are always either events or entities
		self.n_participant_domain_types = 2

		# We initialize all probabilities using a softmax-transformed
		# unit normal, and store them as log probabilities
		clz = self.__class__

		# Event types
		self.event_probs = clz._initialize_log_prob((self.n_event_types))

		# Role types: separate distribution for each event type
		self.role_probs = clz._initialize_log_prob((self.n_event_types,
														self.n_role_types))

		# Relation types: separate distribution for each event type-role
		# pair
		self.relation_probs = clz._initialize_log_prob((self.n_event_types,
													self.n_event_types,
													self.n_relation_types))

		"""
		Participant types: Participants can include events as well as things,
		so we require separate distributions for these. Both are conditioned
		on the event and the role. We stochastically choose between these
		domains on the basis of a Bernoullil, also conditioned on event 
		and role.
		"""
		self.participant_domain_probs = clz._initialize_log_prob((self.n_event_types,
															  self.n_role_types,
															  self.n_participant_domain_types))
		self.event_participant_probs = clz._initialize_log_prob((self.n_event_types,
															self.n_role_types,
															self.n_event_types))
		self.entity_participant_probs = clz._initialize_log_prob((self.n_event_types,
															 self.n_role_types,
															 self.n_entity_types))

		# Initialize mus (expected annotations)
		self.event_mus = clz._initialize_mus(PREDICATE_ANNOTATION_ATTRIBUTES, self.n_event_types)
		self.role_mus = clz._initialize_mus(SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES, self.n_role_types)
		self.participant_mus = clz._initialize_mus(ARGUMENT_ANNOTATION_ATTRIBUTES, self.n_participant_types)
		self.relation_mus = clz._initialize_mus(DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES, self.n_relation_types)

		# Fetch annotator IDs from UDS, used by the likelihood
		# modules to initialize their random effects
		pred_node_annotators, arg_node_annotators,\
		   sem_edge_annotators, doc_edge_annotators = load_annotator_ids(uds)

		# Modules for calculating likelihoods
		self.pred_node_likelihood = PredicateNodeAnnotationLikelihood(pred_node_annotators)
		self.arg_node_likelihood = ArgumentNodeAnnotationLikelihood(arg_node_annotators)
		self.semantics_edge_likelihood = SemanticsEdgeAnnotationLikelihood(sem_edge_annotators)
		self.doc_edge_likelihood = DocumentEdgeAnnotationLikelihood(doc_edge_annotators)

	@classmethod
	def _initialize_mus(cls, attribute_dict, n_types) -> ParameterDict:
		"""Initialize mu parameters for every UDS property, for every cluster"""

		# One mu per event type (cluster)
		mu_dict = {}
		for domain, props in attribute_dict.items():
			for prop, prop_features in props.items():
				# Dictionary keys are of the form 'domain-prop'
				prop_name = '-'.join([domain, prop]).replace('.', '-')
				mu_dict[prop_name] = cls._initialize_log_prob((n_types, prop_features['dim']))
		return ParameterDict(mu_dict)

	@staticmethod
	def _initialize_log_prob(shape: Tuple[int]) -> Parameter:
		"""Unit random normal-based initialization for model parameters

		The result is returned as log probabilities
		"""
		return Parameter(torch.log(softmax(torch.randn(shape), -1)))

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
			- Decide how best to handle participant domain
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
				if 'pred' in v1:
					pred, arg = v1, v2
				else:
					pred, arg = v2, v1

				# Add both as variable nodes
				pred_v_node = VariableNode(FactorGraph.get_node_name('v', pred), self.n_event_types)
				arg_v_node = VariableNode(FactorGraph.get_node_name('v', arg), self.n_role_types)
				fg.set_node(pred_v_node)
				fg.set_node(arg_v_node)

				# Add a likelihood factor for both the predicate and the
				# argument. These are unary factors that compute the likelihood
				# of the annotations on each
				pred_node_anno = sentence.predicate_nodes[pred]
				arg_node_anno = sentence.argument_nodes[arg]
				pred_lf_node = LikelihoodFactorNode(FactorGraph.get_node_name('lf', pred),
								self.pred_node_likelihood, self.event_mus, pred_node_anno)
				arg_lf_node = LikelihoodFactorNode(FactorGraph.get_node_name('lf', arg),
								self.arg_node_likelihood, self.role_mus, arg_node_anno)
				fg.set_node(pred_lf_node)
				fg.set_node(arg_lf_node)

				# Connect the likelihood factor nodes to their variable nodes
				fg.set_edge(pred_lf_node, pred_v_node, 0)
				fg.set_edge(arg_lf_node, arg_v_node, 0)

				# Also create a prior factor for each
				pred_pf_node = PriorFactorNode(FactorGraph.get_node_name('pf', pred),
											   self.event_probs)
				arg_pf_node = PriorFactorNode(FactorGraph.get_node_name('pf', arg),
											   self.role_probs)
				fg.set_node(pred_pf_node)
				fg.set_node(arg_pf_node)

				# The prior factor for the predicate (over event types)
				# is unary, but the one for the predicate (over role types)
				# additionally depends on the (event type of the) predicate.
				fg.set_edge(pred_pf_node, pred_v_node, 0)
				fg.set_edge(arg_pf_node, pred_v_node, 0)
				fg.set_edge(arg_pf_node, arg_v_node, 1)

				# Add a variable node for the semantics edge itself
				sem_edge_v_node = VariableNode(FactorGraph.get_node_name('v', v1, v2), self.n_participant_types)
				fg.set_node(sem_edge_v_node)

				# Add a likelihood factor node for the semantics edge
				# annotations
				sem_edge_lf_node = LikelihoodFactorNode(
									FactorGraph.get_node_name('lf', v1, v2),
									self.semantics_edge_likelihood,
									self.participant_mus, sem_edge_anno)
				fg.set_node(sem_edge_lf_node)

				# Connect the semantics edge likelihood factor and variable nodes
				fg.set_edge(sem_edge_v_node, sem_edge_lf_node)

				# Add a variable node for the participant domain. Participants
				# may be either events or entities.
				participant_domain_v_node = VariableNode(FactorGraph.get_node_name('v', v1, v2, 'domain'),
															self.n_participant_domain_types)
				fg.set_node(participant_domain_v_node)

				# Add a prior factor node for the Bernoulli prior over the
				# participant domain
				participant_domain_pf_node = PriorFactorNode(FactorGraph.get_node_name('pf', v1, v2, 'domain'),
																self.participant_domain_probs)
				fg.set_node(participant_domain_pf_node)

				# The above factor depends on the event type of the associated
				# predicate and the role type of the associated argument (in
				# addition to the participant domain itself)
				fg.set_edge(participant_domain_pf_node, pred_v_node, 0)
				fg.set_edge(participant_domain_pf_node, arg_v_node, 1)
				fg.set_edge(participant_domain_pf_node, participant_domain_v_node, 2)

				"""
				Add a prior factor node for the participant type itself
				The associated tensor contains the probabilities for both
				event and entity participants.
				
				TODO: This may not be the right way of doing things; may
				need to change this.
				"""
				participant_probs = torch.cat([self.event_participant_probs,
											   self.entity_participant_probs], dim=2)
				participant_type_pf_node = PriorFactorNode(FactorGraph.get_node_name('pf', v1, v2),
												participant_probs)
				fg.set_node(participant_type_pf_node)

				# This factor depends on the event type of the predicate,
				# the role type of the argument, and the participant domain,
				# in addition to the participant type
				fg.set_edge(participant_type_pf_node, pred_v_node, 0)
				fg.set_edge(participant_type_pf_node, arg_v_node, 1)
				fg.set_edge(participant_type_pf_node, sem_edge_v_node, 2)
				fg.set_edge(participant_type_pf_node, participant_domain_v_node, 3)

		# Generate document-level graph structure second (as it depends on the
		# sentence-level structure)
		for (v1, v2), doc_edge_anno in document.document_graph.edges.items():
			
			# Create a variable node for the edge itself
			doc_edge_v_node = VariableNode(FactorGraph.get_node_name('v', v1, v2),
								self.n_relation_types)
			fg.set_node(doc_edge_v_node)

			# Create a likelihood factor node for its annotations
			doc_edge_lf_node = LikelihoodFactorNode(FactorGraph.get_node_name('lf', v1, v2),
								self.doc_edge_likelihood, self.relation_mus, doc_edge_anno)
			fg.set_node(doc_edge_lf_node)
			fg.set_edge(doc_edge_v_node, doc_edge_lf_node)

			# Create a factor for the prior over the relation type
			# and connect it with the document edge variable node
			doc_edge_pf_node = PriorFactorNode(FactorGraph.get_node_name('lf', v1, v2),
								self.relation_probs)
			fg.set_node(doc_edge_pf_node)
			fg.set_edge(doc_edge_v_node, doc_edge_pf_node)

			"""
			We also need to connect this factor node to the variable nodes for
			the predicate(s) it relates. First, we have to identify the semantics
			nodes corresponding to the document node(s) and retrieve (or create)
			the associated variable nodes.
			"""
			factor_dim = 0
			for var_node_name in [v1, v2]:
				# Fetch the variable node for the semantics node of the
				# predicate or argument
				sem_node_name = var_node_name.replace('document', 'semantics')
				fg_node_name = FactorGraph.get_node_name('v', sem_node_name)
				var_node = fg.variable_nodes.get(fg_node_name)

				# Verify that the variable node for the argument or predicate
				# was in fact created in the sentence-level loop above
				assert var_node is not None,\
					f"Variable node {var_node} not found in factor graph"

				# Connect the variable node to the prior factor node for the
				# document edge
				fg.set_edge(var_node, doc_edge_pf_node, factor_dim)
				factor_dim += 1

		return fg

	def forward(self, document: decomp.semantics.uds.UDSDocument) -> torch.FloatTensor:
		# TODO
		pass