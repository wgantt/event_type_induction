import networkx as nx

from abc import ABC, abstractmethod, abstractproperty
from decomp.semantics.uds import UDSSentenceGraph, UDSDocumentGraph
from enum import Enum
from torch import Tensor

"""
Classes for representing a factor graph, borrowed with modest modification
from danbar's factor graph library fglib (https://github.com/danbar/fglib)

TODOs:
	- Add methods for handling max-product and sum-product
	- Add method for factor graph construction from UDS graph
	- Determine whether it's necessary to add distinct 'Edge' class
"""
class NodeType(Enum):
    """Enumeration for node types."""
    VARIABLE = 0
    FACTOR = 1


class RVType(Enum):
	"""Enumeration for random variable types"""
	BERNOULLI = 0
	CATEGORICAL = 1


class Node(ABC):

    """Abstract base class for all nodes."""
    def __init__(self, label: str):
        """Create a node with an associated label."""
        self.__label = str(label)
        self.__graph = None

    def __str__(self):
        """Return string representation."""
        return self.__label

    @abstractproperty
    def type(self):
        """Specify the NodeType."""

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph):
        self.__graph = graph

    def neighbors(self, exclusion=None):
        """Get all neighbors with a given exclusion.
        Return iterator over all neighboring nodes
        without the given exclusion node.
        Positional arguments:
        exclusion -- the exclusion node
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


class VariableNode(Node):

    def __init__(self, label, rv_type):
        """Create a variable node."""
        super().__init__(label)
        self.init = rv_type.unity(self)

    @property
    def type(self):
        return NodeType.VARIABLE

    @property
    def init(self):
        return self.__init

    @init.setter
    def init(self, init):
        self.__init = init

    # TODO: add methods for sum-product, max-product


class FactorNode(Node):

    def __init__(self, label, factor=None):
        """Create a factor node."""
        super().__init__(label)
        self.factor = factor
        self.record = {}

    @property
    def type(self):
        return NodeType.FACTOR

    @property
    def factor(self):
        return self.__factor

    @factor.setter
    def factor(self, factor):
        self.__factor = factor

    # TODO: add methods for sum-product, max-product

class FactorGraph(nx.Graph):
    """Class for factor graphs"""

    def __init__(self):
        """Initialize a factor graph."""
        super().__init__(self, name="Factor Graph")

    # TODO: add from_uds method

    def set_node(self, node):
        """Add a single node to the factor graph.
        A single node is added to the factor graph.
        Optional attributes can be added to the single node by using keyword
        arguments.
        Args:
            node: A single node
        """
        node.graph = self
        self.add_node(node, type=node.type)

    def set_nodes(self, nodes):
        """Add multiple nodes to the factor graph.
        Multiple nodes are added to the factor graph.
        Args:
            nodes: A list of multiple nodes
        """
        for n in nodes:
            self.set_node(n)

    def set_edge(self, snode, tnode, init=None):
        """Add a single edge to the factor graph.
        A single edge is added to the factor graph.
        It can be initialized with a given random variable.
        Args:
            snode: Source node for edge
            tnode: Target node for edge
            init: Initial message for edge
        """
        self.add_edge(snode, tnode)

    def set_edges(self, edges):
        """Add multiple edges to the factor graph.
        Multiple edges are added to the factor graph.
        Args:
            edges: A list of multiple edges
        """
        for (snode, tnode) in edges:
            self.set_edge(snode, tnode)

    def get_vnodes(self):
        """Return variable nodes of the factor graph.
        Returns:
            A list of all variable nodes.
        """
        return [n for (n, d) in self.nodes(data=True)
                if d['type'] == NodeType.VARIABLE]

    def get_fnodes(self):
        """Return factor nodes of the factor graph.
        Returns:
            A list of all factor nodes.
        """
        return [n for (n, d) in self.nodes(data=True)
                if d['type'] == NodeType.FACTOR]

    # TODO: add methods for sum-product, max-product