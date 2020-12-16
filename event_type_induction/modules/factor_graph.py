import networkx as nx
import numpy as np
import torch

from abc import ABC, abstractmethod, abstractproperty
from decomp.semantics.uds import UDSSentenceGraph, UDSDocumentGraph
from enum import Enum
from event_type_induction.modules.likelihood import Likelihood
from event_type_induction.constants import NEG_INF
from overrides import overrides
from torch import Tensor, logsumexp
from torch.nn import Parameter, ParameterDict
from typing import List, Optional, Tuple, Any, Dict

"""
Classes for representing a factor graph and its components. These
borrow heavily from danbar's Python factor graph library fglib
(https://github.com/danbar/fglib)

TODOs:
    - Ensure tensors are sent to appropriate device
    - Rename 'source' and 'target' node variables where these
      distinctions are irrelevant
    - Refactor factor graph components into separate module
"""


def normalize_message(msg: Tensor):
    """Normalizes a message of log values using the exp-normalize trick:
       https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    """
    b = msg.max()
    y = torch.exp(msg - b)
    return torch.log(y / y.sum())


class NodeType(Enum):
    """Enumeration for node types."""

    VARIABLE = 0
    FACTOR = 1


class VariableType(Enum):
    """Enumeration for variable types."""

    EVENT = 0
    PARTICIPANT = 1
    ROLE = 2
    RELATION = 3


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

    def neighbors(self, exclusion: Optional["Node"] = None):
        """Retrieve all but an excluded set of this node's neighbors

        exclusion
            The neighbors to exclude (should only ever be a single
            node --- the target)
        """

        if exclusion is None:
            return nx.all_neighbors(self.graph, self)
        else:
            # Build iterator set
            iterator = (exclusion,) if not isinstance(exclusion, list) else exclusion

            # Return neighbors excluding iterator set
            return (n for n in nx.all_neighbors(self.graph, self) if n not in iterator)

    @abstractmethod
    def sum_product(self, target_node: "VariableNode"):
        """Message computation for the sum-product algorithm"""
        raise NotImplementedError()

    @abstractmethod
    def max_product(self, target_node: "VariableNode"):
        """Message computation for the max-product algorithm"""
        raise NotImplementedError()


class VariableNode(Node):
    def __init__(
        self, label: str, vtype: VariableType, ntypes: int, observed: bool = False
    ):
        """Representation of a variable node in the factor graph

        Parameters
        ----------
        label
            The name of this variable node
        vtype
            The type of variable node this is
        ntypes
            The number of types associated with this variable node
        observed
            Whether or not this variable is observed (currently not
            really used)
        """
        super().__init__(label)
        self.observed = observed
        self.vtype = vtype
        self.ntypes = ntypes

        # Initialize the message for this variable node. We
        # initialize with zeros (log(1)), as the actual probabilities
        # for each type will come from the likelihood factors
        self.init = torch.zeros(self.ntypes)

    @property
    def type(self):
        return NodeType.VARIABLE

    def belief(self, normalize=False) -> Tensor:
        """Return the belief of the variable node

        As throughout this class, this method assumes that messages
        are stored as log probabilities.
        """

        # Fetch the neighbors
        neighbors = self.graph.neighbors(self)

        # Fetch the first neighbor
        n = next(neighbors)

        # Initialize this node's belief with the message of the
        # first incoming factor
        belief = self.graph[n][self]["object"].get_message(n, self)

        # Multiply (=add log) values of all other incoming messages
        for n in neighbors:
            belief += self.graph[n][self]["object"].get_message(n, self)

        if normalize:
            return normalize_message(belief)
        else:
            return belief

    @overrides
    def sum_product(self, target_node: "FactorNode") -> Tensor:
        """Sum-product (belief-propagation) message passing"""

        msg = torch.zeros(self.ntypes)

        # Unless this node is observed, we have to multiply
        # in (= add the log of) all incoming messages
        if not self.observed:
            # 'self.neighbors(target_node)' returns an
            # iterator over all neighbors *except* target_node
            for n in self.neighbors(target_node):
                msg += self.graph[n][self]["object"].get_message(n, self)
        return normalize_message(msg)

    @overrides
    def max_product(self, target_node: "FactorNode"):
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
        self._record = {}

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
    def __init__(
        self,
        label: str,
        factor: Likelihood,
        mus: ParameterDict,
        annotation: Dict[str, Any],
        cov: Parameter = None,
    ):
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
        self.per_type_likelihood = None
        self.cov = cov

    @overrides
    def sum_product(self, target_node: VariableNode) -> Tensor:
        # Compute per-property log likelihoods for each of the
        # relevant event types
        if self.cov is not None:
            likelihoods = self.factor(self.mus, self.cov, self.annotation)
        else:
            likelihoods = self.factor(self.mus, self.annotation)

        # If the annotation does not include properties of interest,
        # the returned likelihoods will be empty. The likelihood in
        # this case is 1, so we return a tensor of log(1) == 0 values.
        if len(likelihoods) == 0:
            self.per_type_likelihood = None
            return torch.zeros(target_node.ntypes)

        # Sum the per-property log likelihoods to obtain overall,
        # per-type likelihoods (with dimension equal to the number
        # of possible values of the target node)
        per_type_likelihood = torch.sum(torch.stack(list(likelihoods.values())), dim=0)

        # Likelihood nodes are leaf factors, so no integration
        # over other variable nodes is necessary here. Return
        # the per-type log likelihood as is.
        self.per_type_likelihood = per_type_likelihood
        return per_type_likelihood

    @overrides
    def max_product(self, target_node: VariableNode) -> Tensor:
        # A likelihood factor node's behavior is the same in the
        # max-product case as in the max-sum case; the only difference
        # is that we track the argmax for each type
        per_type_likelihood = self.sum_product(target_node)
        self.record[target_node] = per_type_likelihood.argmax()
        self.per_type_likelihood = per_type_likelihood
        return per_type_likelihood


class PriorFactorNode(FactorNode):
    def __init__(self, label: str, prior: Tensor, vtype: VariableType):
        """Factors for priors on types

        Parameters
        ----------
        label
            The name of the node
        prior
            A tensor containing the prior for the type of the
            target variable node, indexed by the types of the
            incoming variable nodes
        vtype
            The type of variable for which this node represents
            the prior distribution.
        """
        super().__init__(label, prior)
        self.vtype = vtype

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
        # Initialize the message as the factor's tensor (have to add
        # tensor of zeros to avoid in-place operations with a leaf variable)
        outgoing_msg = torch.zeros(self.factor.shape) + self.factor

        # For each incoming message (from a variable node),
        # add the message to the tensor along the appropriate
        # dimension
        for n in self.neighbors(target_node):
            # Fetch the edge between this factor node and
            # the current neighbor
            edge = self.graph[n][self]["object"]

            # Get the incoming message
            incoming_msg = edge.get_message(n, self)

            # Reshape appropriately so that it can be
            # added to the outgoing message. May want to
            # move this inside the get_message call itself.
            broadcast_shape = [1] * len(self.factor.shape)
            broadcast_shape[edge.factor_dim] = len(incoming_msg)

            # Use tensor broadcasting to add in the
            # incoming message to the outgoing one
            outgoing_msg += incoming_msg.view(broadcast_shape)

        """
        Marginalize over all dimensions except the one associated
        with the target node, and return the result---but obviously
        only if there are other dimensions to marginalize. The actual
        passing of the message is performed by the graph-level
        sum_product call
        """
        if len(self.factor.shape) > 1:
            target_dim = self.graph[self][target_node]["object"].factor_dim
            marginalize_dims = [
                i for i in range(len(self.factor.shape)) if i != target_dim
            ]
            msg = logsumexp(outgoing_msg, marginalize_dims)
        else:
            msg = logsumexp(outgoing_msg, 0)

        return normalize_message(msg)

    @overrides
    def max_product(self, target_node: VariableNode) -> Tensor:
        """Max-product message passing to a variable node"""

        # Record for storing optimal assignments for
        # each incoming variable node
        self.record[target_node] = {}

        # Initialize the outgoing message
        outgoing_msg = torch.zeros(self.factor.shape) + self.factor

        # Loop over incoming neighbors is the same as in sum-product
        for n in self.neighbors(target_node):

            # Get message
            edge = self.graph[n][self]["object"]
            incoming_msg = edge.get_message(n, self)

            # Reshape
            broadcast_shape = [1] * len(self.factor.shape)
            broadcast_shape[edge.factor_dim] = len(incoming_msg)

            # Sum
            outgoing_msg += incoming_msg.view(broadcast_shape)

        """
        Determine the maximum value for each possible assignment to the
        target variable. The below is a fancy way to compute this that
        gets around torch.max's lack of support for max'ing over multiple
        dimensions simultaneously.
        """
        target_dim = self.graph[self][target_node]["object"].factor_dim
        maxes = outgoing_msg.view(outgoing_msg.size(target_dim), -1).max(dim=-1)

        # If there *are* incoming variable nodes, we record the best assignment
        # for each one, for each possible value of the target variable node.
        if len(self.factor.shape) > 1:
            # The dimensions of the factor tensor associated with the
            # incoming variables
            non_target_dimensions = {
                i: dim for i, dim in enumerate(self.factor.shape) if i != target_dim
            }

            # The indices in the message tensor corresponding to the
            # maximum values
            max_indices = np.unravel_index(
                maxes.indices, tuple(non_target_dimensions.values())
            )

            # Associate each index with the appropriate incoming variable
            max_indices_by_factor_dim = dict(
                zip(non_target_dimensions.keys(), max_indices)
            )

            # Record the best possible assignment for each incoming variable
            for n in self.neighbors(target_node):
                factor_dim = self.graph[n][self]["object"].factor_dim
                self.record[target_node][n] = max_indices_by_factor_dim[factor_dim]

        # Return the max values
        return maxes.values


class Edge:
    def __init__(self, source_node: Node, target_node: Node, factor_dim=0):
        """Base class for edges

        Messages are stored on the edges, one for each direction

        Parameters
        ----------
        source_node
            The first node incident on this edge
        target_node
            The second node incident on this edge
        factor_dim
            The dimension of the factor tensor with which the variable
            is associated
        """

        # Array Index
        self.index = {source_node: 0, target_node: 1}

        # Initialize the messages
        if isinstance(source_node, VariableNode):
            msg_dim = source_node.ntypes
        elif isinstance(target_node, VariableNode):
            msg_dim = target_node.ntypes
        else:
            raise ValueError(
                f"Edge from {source_node.label} to {target_node.label} connects two factor nodes!"
            )
        msg_init = torch.zeros(msg_dim)

        # Set the message
        self.message = [[None, msg_init], [msg_init, None]]

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


class DimensionMismatchEdge(Edge):
    """Class for handling dimension mismatch between variables and factors"""

    @overrides
    def __init__(self, source_node, target_node, factor_dim=0) -> None:
        self.index = {source_node: 0, target_node: 1}
        self.factor_dim = factor_dim

        # Determine which variable is source and which target
        if source_node.type == NodeType.VARIABLE:
            var_node, factor_node = source_node, target_node
        else:
            var_node, factor_node = target_node, source_node

        # Initialize the messages. Owing to the dimension mismatch,
        # the two messages must have different dimensions.
        var_to_factor_msg_dim = factor_node.factor.shape[self.factor_dim]
        var_to_factor_msg_init = torch.ones(var_to_factor_msg_dim) * NEG_INF
        var_to_factor_msg_init[: var_node.ntypes] = 0
        factor_to_var_msg_dim = var_node.ntypes
        factor_to_var_msg_init = torch.zeros(factor_to_var_msg_dim)

        self.message = [[None, None], [None, None]]
        self.message[self.index[var_node]][
            self.index[factor_node]
        ] = var_to_factor_msg_init
        self.message[self.index[factor_node]][
            self.index[var_node]
        ] = factor_to_var_msg_init

    @overrides
    def set_message(self, source_node, target_node, msg) -> None:
        new_msg = self._correct_dimension_mismatch(source_node, target_node, msg)
        super().set_message(source_node, target_node, new_msg)

    def _correct_dimension_mismatch(
        self, source_node, target_node, msg
    ) -> torch.Tensor:
        """Correct dimension mismatch between factors and variables

        This should only ever occur for edges between an event type
        variable node and a relation type prior node, where the number
        of relation types is strictly greater than the number of
        event types.
        """

        # variable --> factor: expand message dimension
        if source_node.type == NodeType.VARIABLE:
            var_node, factor_node = source_node, target_node
            desired_msg_length = factor_node.factor.shape[self.factor_dim]
            new_msg = torch.ones(desired_msg_length) * NEG_INF
            new_msg[: len(msg)] = msg
        # factor --> variable: contract message dimension
        else:
            var_node, factor_node = target_node, source_node
            desired_msg_length = var_node.ntypes
            new_msg = msg[:desired_msg_length]
        return new_msg


class FactorGraph(nx.Graph):
    """Class for factor graphs"""

    def __init__(self):
        """Initialize a factor graph."""
        super().__init__(self, name="Factor Graph")
        self._variable_nodes = {}
        self._factor_nodes = {}
        self._likelihood_factor_nodes = {}
        self._prior_factor_nodes = {}

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
        return "-".join([*args, ntype])

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
            if isinstance(node, LikelihoodFactorNode):
                self.likelihood_factor_nodes[node.label] = node
            elif isinstance(node, PriorFactorNode):
                self.prior_factor_nodes[node.label] = node
            else:
                raise ValueError(f"Invalid factor node type {node.type}!")
        else:
            # TODO: replace with warning when logging is set up
            print(f"Invalid node type {node.type}!")

        self.add_node(node, type=node.type)

    def set_edge(self, node1: Node, node2: Node, factor_dim=0) -> None:
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
        """
        # Verify that each edge connects a variable node to a factor node,
        # then identify which is which
        if node1.type == node2.type:
            raise ValueError(
                f"Attempted to create edge between {node1.label} "
                f"and {node2.label}, but they have the same type ({node1.type}!"
            )

        # Default edge
        edge = Edge(node1, node2, factor_dim)

        # Special case: dimension mismatch between variable and factor nodes.
        # This should occur only between event type variable nodes and
        # relation type factor nodes.
        if isinstance(node1, PriorFactorNode):
            if (
                node1.vtype == VariableType.RELATION
                and node2.vtype == VariableType.EVENT
            ):
                edge = DimensionMismatchEdge(node1, node2, factor_dim)
        elif isinstance(node2, PriorFactorNode):
            if (
                node2.vtype == VariableType.RELATION
                and node1.vtype == VariableType.EVENT
            ):
                edge = DimensionMismatchEdge(node1, node2, factor_dim)

        # Add the edge to the NetworkX graph
        self.add_edge(node1, node2, object=edge)

    @property
    def variable_nodes(self):
        return self._variable_nodes

    @property
    def factor_nodes(self):
        return self._factor_nodes

    @property
    def likelihood_factor_nodes(self):
        return self._likelihood_factor_nodes

    @property
    def prior_factor_nodes(self):
        return self._prior_factor_nodes

    def loopy_sum_product(
        self,
        n_iters: int,
        query_nodes: Optional[List[VariableNode]] = [],
        order: Optional[List[Node]] = None,
        exclusions: Optional[Dict[Node, Optional[Node]]] = {},
    ) -> Dict[str, float]:
        return self.schedule(n_iters, "sum_product", query_nodes, order, exclusions)

    def loopy_max_product(
        self,
        n_iters: int,
        query_nodes: Optional[List[VariableNode]] = [],
        order: Optional[List[Node]] = None,
        exclusions: Optional[Dict[Node, Optional[Node]]] = {},
    ) -> Dict[str, int]:
        """Loopy max-product"""
        return self.schedule(n_iters, "max_product", query_nodes, order, exclusions)

    def schedule(
        self,
        n_iters: int,
        semiring: str,
        query_nodes: Optional[List[VariableNode]] = [],
        order: Optional[List[Node]] = None,
        exclusions: Optional[Dict[Node, Optional[Node]]] = {},
    ) -> Dict[str, Any]:
        """Runs message passing on graphs with cycles

        If no order is specified, messages are updated for all factor
        nodes first, followed by all variable nodes. This is the default
        order specified in fglib, and I have preserved it here.

        Parameters
        ----------
        n_iters
            The number of iterations for which to run message passing
        semiring
            The type of message passing to use (should be one of
            "sum_product" or "max_product"
        query_nodes
            The nodes whose beliefs are to be tracked
        order
            The order in which to visit nodes in the graph. For loopy BP,
            any order is technically permissible
        exclusions
            For each node in the order, specifies the edges the exclude
            when performing message updates. Useful for forward-backward.
        """

        # Default order
        if order is None:
            order = list(self._factor_nodes.values()) + list(
                self._variable_nodes.values()
            )

        if not exclusions:
            exclusions = {node: None for node in order}

        # Track the beliefs of each query node
        beliefs = {n.label: 0 for n in query_nodes}

        # Do message passing for the specified number of iterations
        for i in range(n_iters):
            for node in order:
                for neighbor in node.neighbors(exclusions[node]):
                    msg = getattr(node, semiring)(neighbor)
                    self[node][neighbor]["object"].set_message(node, neighbor, msg)

        # Return final beliefs for query nodes
        for n in query_nodes:
            beliefs[n.label] = n.belief()

        return beliefs
