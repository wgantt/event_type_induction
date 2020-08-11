import torch
import decomp

from constants import *
from modules.likelihood import (
	ArgumentNodeAnnotationLikelihood,
	PredicateNodeAnnotationLikelihood,
	SemanticsEdgeAnnotationLikelihood,
	DocumentEdgeAnnotationLikelihood
	)
from modules.freezable_module import FreezableModule

from collections import defaultdict
from torch.nn import Parameter, ParameterDict, ModuleDict
from torch.nn.functional import softmax
from torch.distributions import Categorical, Normal, Bernoulli
from typing import Tuple

class EventTypeInductionModel(FreezableModule):
	"""Base module for event type induction
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
															  self.n_role_types))
		self.event_participant_probs = clz._initialize_log_prob((self.n_event_types,
										  					self.n_role_types,
										  					self.n_event_types))
		self.entity_participant_probs = clz._initialize_log_prob((self.n_event_types,
															 self.n_role_types,
															 self.n_entity_types))

		# Initialize mus (expected annotations)
		self.event_mus = clz._initialize_mus(PREDICATE_ANNOTATION_ATTRIBUTES, self.n_event_types)
		self.role_mus = clz._initialize_mus(SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES, self.n_role_types)
		self.participant_mus = clz._initialize_mus(ARGUMENT_ANNOTATION_ATTRIBUTES, self.n_event_types + self.n_entity_types)
		self.relation_mus = clz._initialize_mus(DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES, self.n_relation_types)

		# Random effects: nested ModuleDict with annotator at first level,
		# property at the second level
		self.pred_random_effects, self.arg_random_effects,\
			self.sem_edge_random_effects, self.doc_edge_random_effects = clz._initialize_random_effects(uds)

		# Modules for calculating likelihoods
		self.pred_node_likelihood = PredicateNodeAnnotationLikelihood(self.pred_random_effects)
		self.arg_node_likelihood = ArgumentNodeAnnotationLikelihood(self.arg_random_effects)
		self.semantics_edge_likelihood = SemanticsEdgeAnnotationLikelihood(self.sem_edge_random_effects)
		self.doc_edge_likelihood = DocumentEdgeAnnotationLikelihood(self.doc_edge_random_effects)

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

	@staticmethod
	def _initialize_random_effects(uds: decomp.UDSCorpus) -> ModuleDict:
		"""Initialize annotator random intercepts"""
		def random_effects_helper(items, prop_attrs, random_effects):
			# Iterate over annotation keys, not prop_attrs keys
			prop_domains = prop_attrs.keys()
			for item, annotation in items:
				for domain, props in annotation.items():
					if domain in prop_domains:
						for p in annotation[domain].keys():
							prop_dim = prop_attrs[domain][p]['dim']
							# PyTorch forbids ModuleDict keys to contain '.'
							prop_name = '-'.join([domain, p]).replace('.', '-')
							annotator_ids = props[p]['value'].keys()
							for annotator in annotator_ids:
								random_effects[annotator][prop_name] =\
									Parameter(torch.randn(prop_dim))

		# Random effects dictionaries by domain for all annotators.
		# We break them out this way so that they may be passed to the
		# appropriate Likelihood subclass constructor.
		pred_random_effects = defaultdict(ParameterDict)
		arg_random_effects = defaultdict(ParameterDict)
		sem_edge_random_effects = defaultdict(ParameterDict)
		doc_edge_random_effects = defaultdict(ParameterDict)

		# Random effects for sentence-level annotations
		for graph in uds:
			pred_items = uds[graph].predicate_nodes.items()
			arg_items = uds[graph].argument_nodes.items()
			sem_edge_items = uds[graph].semantics_edges().items()

			# Initialize random effects for each annotator and each property
			random_effects_helper(pred_items, PREDICATE_ANNOTATION_ATTRIBUTES, pred_random_effects)
			random_effects_helper(arg_items, ARGUMENT_ANNOTATION_ATTRIBUTES, arg_random_effects)
			random_effects_helper(sem_edge_items, SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES, sem_edge_random_effects)

		# Random effects for document-level annotations
		for doc in uds.documents.values():
			doc_edge_items = doc.document_graph.edges.items()
			random_effects_helper(doc_edge_items, DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES, doc_edge_random_effects)

		# Separate ModuleDicts for each annotation domain
		return (ModuleDict(pred_random_effects), ModuleDict(arg_random_effects),
				ModuleDict(sem_edge_random_effects), ModuleDict(doc_edge_random_effects))

	def forward(self, document: decomp.semantics.uds.UDSDocument) -> torch.FloatTensor:
		
		# Sentence-level annotations
		ll = torch.FloatTensor([0.])
		for s in document.sentence_ids:
			
			# Process predicate annotations
			for pred, pred_anno in self.uds[s].predicate_nodes.items():

				# Compute likelihood of the predicate node annotations for each event type.
				# This returns a dictionary of n_event_types-length log likelihood vectors
				# for each property.
				pred_node_likelihoods = self.pred_node_likelihood(self.event_mus, pred_anno)

				# Sum log likelihoods of individual properties to get overall predicate
				# annotation likelihoods for each event type, adding in the log prior over
				# event types. Some predicates will not have any annotations, in which case
				# we just use the prior.
				if pred_node_likelihoods:
					pred_likelihood =\
						self.event_probs + torch.sum(torch.stack(list(pred_node_likelihoods.values())), dim=0)
				else:
					pred_likelihood = self.event_probs

				# Process semantics edge and argument annotations
				for sem_edge, sem_edge_anno in self.uds[s].semantics_edges(pred).items():

					# Compute likelihood of the semantics edge annotations for each role type
					sem_edge_likelihoods = self.semantics_edge_likelihood(self.role_mus, sem_edge_anno)

					# Similar procedure for semantics edges as for predicates, only
					# here we add in the prior over role probabilities, relying on
					# Tensor broadcasting (as we do below as well)
					if sem_edge_likelihoods:
						sem_edge_likelihood =\
							self.role_probs + torch.sum(torch.stack(list(sem_edge_likelihoods.values())), dim=0)
					else:
						sem_edge_likelihood = self.role_probs

					# Add in predicate log likelihoods. At this point, we have an
					# n_event_types x n_role_types tensor of log probabilities of the predicate
					# and semantics edge annotations
					sem_edge_likelihood = sem_edge_likelihood + pred_likelihood.unsqueeze(1).repeat(1,self.n_role_types)

					# Fetch the argument associated with this edge, along with its annotations
					arg = sem_edge[0] if 'arg' in sem_edge[0] else sem_edge[1]
					arg_anno = self.uds[s].argument_nodes[arg]

					# Compute likelihood of the argument node annotations for each event type.
					# This gives us likelihoods for both event-type and entity-type arguments,
					# which must be split out
					arg_node_likelihoods = self.arg_node_likelihood(self.participant_mus, arg_anno)

					# Fork on participant domain. A Bernoulli indicates which domain to choose
					# from, with entity as the positive class.
					entity_participant_likelihood = sem_edge_likelihood + self.participant_domain_probs
					event_participant_likelihood = sem_edge_likelihood - self.participant_domain_probs

					# Break out arg node likelihoods based on event- and entity-type participants
					# (event types come first in the vector)
					if arg_node_likelihoods:

						# Sum over individual property likelihoods
						arg_node_likelihood_by_participant_type =\
							torch.sum(torch.stack(list(arg_node_likelihoods.values())), dim=0)

						# Separate into event and entity tensors
						arg_node_likelihood_event =\
							arg_node_likelihood_by_participant_type[:self.n_event_types]
						arg_node_likelihood_entity =\
							arg_node_likelihood_by_participant_type[self.n_event_types:]

						# Compute separate likelihoods for each
						arg_node_likelihood_event = self.event_participant_probs + arg_node_likelihood_event
						arg_node_likelihood_entity = self.entity_participant_probs + arg_node_likelihood_entity

						# Add in likelihoods for semantics edges
						arg_node_likelihood_event = arg_node_likelihood_event +\
							sem_edge_likelihood.unsqueeze(2).repeat(1,1,self.n_event_types)
						arg_node_likelihood_entity = arg_node_likelihood_entity +\
							sem_edge_likelihood.unsqueeze(2).repeat(1,1,self.n_entity_types)

					# No argument node annotations
					else:
						arg_node_likelihood_event = self.event_participant_probs +\
							sem_edge_likelihood.unsqueeze(2).repeat(1,1,self.n_event_types)
						arg_node_likelihood_entity = self.entity_participant_probs +\
							sem_edge_likelihood.unsqueeze(2).repeat(1,1,self.n_entity_types)


					"""
					Each element of arg_node_likelihood_entity now contains the joint log
					probability of the annotations assuming the corresponding
					(event_type, role_type, entity_type) triple; the same goes for
					arg_node_likelihood_event, but with event types instead of entity types.
					We take the log of the summed exponentials of each element across
					both tensors to get the overall likelihood of the annotation
					"""
					ll += torch.logsumexp(torch.cat([torch.flatten(arg_node_likelihood_event),
													torch.flatten(arg_node_likelihood_entity)]),0)


			# TODO: document edges
			# for doc_edge, doc_edge_anno in document.document_graph.edges.items():
				# doc_edge_likelihoods = self.doc_edge_likelihood(self.relation_mus, doc_edge_anno)

		return ll