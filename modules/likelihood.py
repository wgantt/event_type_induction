import torch

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from constants import *
from overrides import overrides
from torch import Tensor
from torch.distributions import Bernoulli, Categorical
from torch.nn import Module, ModuleDict, Parameter, ParameterDict
from typing import Any, Dict

class Likelihood(Module, metaclass=ABCMeta):
	"""ABC for Event Type Induction Module likelihood computations"""
	def __init__(self, random_effects: ModuleDict, prop_attrs: Dict[str, Dict[str, int]]):
		super().__init__()
		self.random_effects = random_effects
		self.prop_attrs = prop_attrs
		self.prop_domains = prop_attrs.keys()

	def _get_distribution(self, mu, random, prop_type):
		if prop_type == BINARY:
			return Bernoulli(torch.sigmoid(mu + random))
		else:
			return Categorical(torch.softmax(mu + random, 0))

	def _scalar_to_vector(value, dim):
		"""Returns a one-hot vector for representing nominal annotations"""
		vec = torch.zeros(dim)
		vec[value - 1] = 1
		return vec

	@abstractmethod
	def forward(self, mus: ParameterDict, annotation: Dict[str, Any]) -> Dict[str, Tensor]:
		likelihoods = {}
		for domain, props in annotation.items():
			# Time is a special; skipping it now for debugging purposes
			# General TODO: ensure this works with nominal properties
			if 'time' in domain:
				continue
			if domain in self.prop_domains:
				for p in props:

					# Determine the property type
					prop_type = self.prop_attrs[domain][p]['type']
					prop_name = '-'.join([domain, p]).replace('.', '-')

					# The mean for the current property, for each event type
					mu = mus[prop_name]

					# Compute log likelihood for each annotator (for training,
					# this should execute just once, as train data has only a
					# single annotation)
					for annotator, value in props[p]['value'].items():

						# Grab the random intercept for the current annotator
						random = self.random_effects[annotator][prop_name]

						# Determine the appropriate distribution for the
						# current property type
						dist = self._get_distribution(mu, random, prop_type)

						# Compute log likelihood
						likelihood = dist.log_prob(torch.Tensor([value])).squeeze()
						if p in likelihoods:
							likelihoods[p] += likelihood
						else:
							likelihoods[p] = likelihood
		return likelihoods

class PredicateNodeAnnotationLikelihood(Likelihood):

	@overrides
	def __init__(self, random_effects: ModuleDict):
		super().__init__(random_effects, PREDICATE_ANNOTATION_ATTRIBUTES)

	@overrides
	def forward(self, mus: ParameterDict, annotation: Dict[str, Any]) -> Dict[str, Tensor]:
		return super().forward(mus, annotation)


class ArgumentNodeAnnotationLikelihood(Likelihood):

	@overrides
	def __init__(self, random_effects: ModuleDict):
		super().__init__(random_effects, ARGUMENT_ANNOTATION_ATTRIBUTES)

	@overrides
	def forward(self, mus: ParameterDict, annotation: Dict[str, Any]) -> Dict[str, Tensor]:
		return super().forward(mus, annotation)


class SemanticsEdgeAnnotationLikelihood(Likelihood):

	@overrides
	def __init__(self, random_effects: ModuleDict):
		super().__init__(random_effects, SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES)

	@overrides
	def forward(self, mus: ParameterDict, annotation: Dict[str, Any]) -> Dict[str, Tensor]:
		return super().forward(mus, annotation)

class DocumentEdgeAnnotationLikelihood(Likelihood):

	@overrides
	def __init__(self, random_effects: ModuleDict):
		super().__init__(random_effects, DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES)

	@overrides
	def forward(self, mus: ParameterDict, annotation: Dict[str, Any]) -> Dict[str, Tensor]:
		pass