import networkx as nx

from abc import ABC, abstractmethod, abstractproperty
from decomp.semantics.uds import UDSSentenceGraph, UDSDocumentGraph
from enum import Enum
from event_type_induction.modules.likelihood import Likelihood
from overrides import overrides
from torch import Tensor, logsumexp
from torch.nn import ParameterDict
from typing import List, Optional, Tuple, Any, Dict

"""
Classes for representing a factor graph and its components. These
borrow heavily from danbar's Python factor graph library fglib
(https://github.com/danbar/fglib)

TODOs:
	- Ensure tensors are sent to appropriate device
	- Implement loopy max-product
"""
class NodeType(Enum):
	"""Enumeration for node types."""
	VARIABLE = 0
	FACTOR = 1


class Node(ABC):

	"""Abstract base class for all nodes."""
	def __init__(self, label: str):
		"""Create a node with an associated label."""
		self._label = str(label)

		# The graph is set on a node when it is added
		# to a FactorGraph object via the set_node or
		# set_nodes methods
		self._graph = None

	def __str__(self):
		"""Return string representation."""
		return self._label

	@abstractproperty
	def type(self):
		"""Specify the NodeType."""

	@property
	def graph(self):
		return self._graph

	@graph.setter
	def graph(self, graph):
		self._graph = graph

	@property
	def label(self):
		return self._label
	
	@label.setter
	def label(self, label):
		self._label = label

	def neighbors(self, exclusion: Optional['Node'] = None):
		"""Retrieve all but an excluded set of this node's neighbors

		exclusion
			The neighbors to exclude (should only ever be a single
			node --- the target)
		"""

		if exclusion is None:
			return nx.all_neighbors(self.graph, self)
		else:
			# Build iterator set
			iterator = (exclusion,) \
				if not isinstance(exclusion, list) else exclusion

			# Return neighbors excluding iterator set
			return (n for n in nx.all_neighbors(self.graph, self)
					if n not in iterator)

	@abstractmethod
	def sum_product(self, target_node: 'VariableNode'):
		"""Message computation for the sum-product algorithm"""
		raise NotImplementedError()

	@abstractmethod
	def max_product(self, target_node: 'VariableNode'):
		"""Message computation for the max-product algorithm"""
		raise NotImplementedError()


class VariableNode(Node):

	def __init__(self, label: str, observed: bool = False):
		"""Create a variable node."""
		super().__init__(label)
		# Not sure whether we'll actually need this (if we do,
		# it would just be for variable nodes corresponding to
		# particular annotations), but I'm leaving it in just in case.
		self.observed = observed

	@property
	def type(self):
		return NodeType.VARIABLE

	@property
	def init(self):
		return self._init

	@init.setter
	def init(self, init: Tensor):
		self._init = init

	def belief(self) -> Tensor:
		"""Return the belief of the variable node

		This method assumes that:

		1. There are no self-loops (this node is not its own neighbor)
		2. All neighbors are factor nodes
		3. Messages are stored as log probabilities, and hence
		   are added, not multiplied.
		"""

		# Fetch the neighbors
		neighbors = self.graph.neighbors(self)

		# Fetch the first neighbor
		n = next(neighbors)

		# Initialize this node's belief with the message of the
		# first incoming factor
		belief = self.graph[n][self]['object'].get_message(n, self)

		# Multiply (=add log) values of all other incoming messages
		for n in neighbors:
			belief += self.graph[n][self]['object'].get_message(n, self)

		# TODO: add option for normalizing
		return belief

	@overrides
	def sum_product(self, target_node: 'FactorNode') -> Tensor:
		"""Sum-product (belief-propagation) message passing"""

		msg = self.init

		# Unless this node is observed, we have to multiply
		# in (= add the log of) all incoming messages
		if not self.observed:
			# 'self.neighbors(target_node)' returns an
			# iterator over all neighbors *except* target_node
			for n in self.neighbors(target_node):
				msg += self.graph[n][self]['object'].get_message(n, self)

		return msg

	@overrides
	def max_product(self, target_node: 'FactorNode'):
		"""Max-product message passing"""

		# Message passing equation for variables to factors does
		# not change from the sum-product case
		return self.sum_product(target_node)


class FactorNode(Node):

	def __init__(self, label: str, factor: Any):
		"""Base class for all factor nodes

		label
			The name of the node
		factor
			Any object (tensor, module, etc.) used to compute the message
			for this node
		"""
		super().__init__(label)
		self._factor = factor

		# Used to track optimal values for max-product
		self._record = None

	@property
	def type(self):
		return NodeType.FACTOR

	@property
	def factor(self):
		return self._factor

	@property
	def record(self):
		return self._record
	

class LikelihoodFactorNode(FactorNode):

	def __init__(self, label: str, factor: Likelihood,
					   mus: ParameterDict, annotation: Dict[str, Any]):
		"""Unary leaf factors to compute annotation likelihoods

		Parameters
		----------
		factor
			A likelihood module used to obtain per-property likelihoods for
			the provided annotation
		mus
			The mean values for each property
		annotation
			The annotation whose likelihood is to be computed
		"""
		super().__init__(label, factor)
		self.mus = mus
		self.annotation = annotation

	@overrides
	def sum_product(self, target_node: VariableNode) -> Tensor:
		# Compute per-property log likelihoods for each of the
		# relevant event types
		likelihoods = self.factor(mus, annotation)

		# Sum the per-property log likelihoods to obtain overall,
		# per-type likelihoods (with dimension equal to the number
		# of possible values of the target node)
		per_type_likelihood = torch.sum(torch.stack(list(likelihoods.values())), dim=0)

		# Likelihood nodes are leaf factors, so no integration
		# over other variable nodes is necessary here. Return
		# the per-type log likelihood as is.
		return per_type_likelihood

	@overrides
	def max_product(self, target_node: VariableNode) -> Tensor:
		# A likelihood factor node's behavior is the same in the
		# max-product case as in the max-sum case; the only difference
		# is that we track the argmax for each type
		per_type_likelihood = sum_product(target_node)
		self.record[target_node] = per_type_likelihood.argmax()
		return per_type_likelihood


class PriorFactorNode(FactorNode):

	def __init__(self, label: str, prior: Tensor):
		"""Factors for priors on types

		Parameters
		----------
		label
			The name of the node
		prior
			A tensor containing the prior for the type of the
			target variable node, indexed by the types of the
			incoming variable nodes
		"""
		super().__init__(label, prior)

	@overrides
	def sum_product(self, target_node: VariableNode) -> Tensor:
		"""Sum-product message passing to a variable node

		In this implementation of sum-product, "product" is actually
		log addition and "sum" is logsumexp

		Parameters
		----------
		target_node
			The variable node to which the message is to be passed
		"""
		# Initialize the message as the factor's tensor
		outgoing_msg = self.factor

		# For each incoming message (from a variable node),
		# add the message to the tensor along the appropriate
		# dimension
		for n in self.neighbors(target_node):
			# Fetch the edge between this factor node and
			# the current neighbor
			edge = self.graph[n][self]['object']

			# Get the incoming message
			incoming_msg = edge.get_message(n, self)

			# Reshape appropriately so that it can be
			# added to the outgoing message. May want to
			# move this inside the get_message call itself.
			broadcast_shape = [1] * len(self.factor.shape)
			broadcast_shape[edge.factor_dim] = len(incoming_message)

			# Use tensor broadcasting to add in the
			# incoming message to the outgoing one
			outgoing_msg += incoming_msg.view(broadcast_shape) 

		# The last dimension of the factor tensor is the only
		# one not marginalized out
		marginalize_dims = list(range(len(self.factor.shape)))[:-1]

		# The actual passing of the message is performed by the
		# graph-level sum_product call
		return logsumexp(outgoing_msg, marginalize_dims)

	@overrides
	def max_product(self, target_node: VariableNode) -> Tensor:
		"""Max-product message passing to a variable node"""
		outgoing_msg = self.factor

		# Loop over neighbors is the same as in sum-product
		for n in self.neighbors(target_node):

			# Get message
			edge = self.graph[n][self]['object']
			incoming_msg = edge.get_message(n, self)

			# Reshape
			broadcast_shape = [1] * len(self.factor.shape)
			broadcast_shape[edge.factor_dim] = len(incoming_message)

			# Sum
			outgoing_msg += incoming_msg.view(broadcast_shape)

		# TODO: track max value for each non-target variable node

		# Take the max over all dimensions except the last.
		# This is a fancy workaround to do that, given that torch.max
		# does not support max'ing over multiple dimensions
		return outgoing_msg.view(outgoing_msg.size(-1), -1).max(dim=-1).values


class Edge:
	def __init__(self, source_node: Node, target_node: Node,
				factor_dim=0, msg_init=0):
		"""Base class for edges

		Messages are tracked are separately for each direction (obviously)
		"""
		
		# Array Index
		self.index = {source_node: 0, target_node: 1}

		# Set the message
		self.message = [[None, msg_init],
						[msg_init, None]]

		# The dimension of the factor node's tensor with which
		# the variable node is associated (only really relevant
		# for prior factors, not likelihood factors)
		self.factor_dim = factor_dim

	def __str__(self):
		"""Return string representation."""
		return str(self.message)

	def set_message(self, source_node, target_node, msg) -> None:
		"""Set value of message from source node to target node."""
		self.message[self.index[source_node]][self.index[target_node]] = msg

	def get_message(self, source_node, target_node) -> None:
		"""Return value of message from source node to target node."""
		return self.message[self.index[source_node]][self.index[target_node]]


class FactorGraph(nx.Graph):
	"""Class for factor graphs"""

	def __init__(self):
		"""Initialize a factor graph."""
		super().__init__(self, name="Factor Graph")
		self._variable_nodes = {}
		self._factor_nodes = {}

	@staticmethod
	def get_node_name(ntype: str, *args: Tuple[str]) -> str:
		"""Generates a factor graph node name

		The name consists of the values in args separated by '-',
		and postfixed with the node type. By convetion, use 'v'
		for variable nodes, 'lf' for likelihood factor nodes,
		and 'pf' for prior factor nodes

		Parameters
		----------
		ntype
			The type of node
		args
			Other components of the name
		"""
		return '-'.join([*args, ntype])

	def set_node(self, node: Node) -> None:
		"""Add a single node to the factor graph.

		Parameters
		----------
		node
			The node to be added
		"""
		node.graph = self
		if node.type == NodeType.VARIABLE:
			self.variable_nodes[node.label] = node
		elif node.type == NodeType.FACTOR:
			self.factor_nodes[node.label] = node
		else:
			# TODO: replace with warning when logging is set up
			print(f'Invalid node type {node.type}!')

		self.add_node(node, type=node.type)

	def set_edge(self, node1: Node, node2: Node,
					factor_dim=0, init_msg=None) -> None:
		"""Add a single edge to the factor graph

		all edges are undirected and connect a variable node
		to a factor node

		Parameters
		----------
		node1
			the first node incident on the edge
		node2
			the second node incident on the edge
		factor_dim
			the dimension of the factor node's tensor with which
			the variable node is associated
		init_msg
			the value with which to initialize the message(s)
			for this edge
		"""
		# Verify that each edge connects a variable node to a factor node,
		# then identify which is which
		if node1.type == node2.type:
			raise ValueError(f"Attempted to create edge between {node1.label} "
				f"and {node2.label}, but they have the same type ({node1.type}!")
		self.add_edge(node1, node2, object=Edge(node1, node2, factor_dim, init_msg))

	@property
	def variable_nodes(self):
		return self._variable_nodes

	@property
	def factor_nodes(self):
		return self._factor_nodes

	def loopy_sum_product(self, n_iters: int,
						  query_nodes: Optional[List[VariableNode]] = None,
						  order: Optional[List[VariableNode]] = None) -> Dict[str, float]:
		"""Loopy belief propagation

		Parameters
		----------
		n_iters
			The number of iterations for which to run message passing
		query_nodes
			The nodes whose beliefs are to be tracked
		order
			The order in which to visit nodes in the graph. For loopy BP,
			any order is technically permissible
		"""

		# If no order is specified, visit factor nodes first, followed by
		# variable nodes
		if order is None:
			order = list(self._factor_nodes.items()) + list(self._variable_nodes.items())

		# Track the beliefs of each query node
		beliefs = {n.label: 0 for n in query_nodes}

		for _ in range(n_iters):
			for node in order:
				for neighbor in node.neighbors():
					msg = node.sum_product(neighbor)
					self[node][neighbor]['object'].set_message(node, neighbor)

		# Return final beliefs for query nodes
		for n in query_node:
			beliefs[n.label] = n.belief()

		return beliefs

	def loopy_max_product(self, n_iters: int, order=None):
		# TODO
		pass