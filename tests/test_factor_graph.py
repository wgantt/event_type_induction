import torch
import unittest

from event_type_induction.modules.factor_graph import *

class TestFactorGraph(unittest.TestCase):

	def test_sum_product(self):
		fg = FactorGraph()

		# Add a single variable node (the variable type is unimportant
		# here, as are the number of types).
		v1 = VariableNode('v1', VariableType.EVENT, 4)
		fg.set_node(v1)

		# Create a prior factor node, and connect it to the variable
		# node above
		pf1_factor = torch.log(torch.FloatTensor([0.1, 0.2, 0.3, 0.4]))
		pf1 = PriorFactorNode('pf1', pf1_factor, VariableType.EVENT)
		fg.set_node(pf1)
		fg.set_edge(v1, pf1)

		# Run sum-product from the prior to the variable. The message
		# should simply be the log of summed exponentials of the
		# elements in the prior factor.
		actual_message = pf1.sum_product(v1)
		expected_message = torch.logsumexp(pf1_factor, 0)
		assert actual_message == expected_message

		# Try sum-product in the other direction. Since the messages at
		# the variable nodes are initialized with zeros, a zero tensor
		# should be passed to the prior node.
		actual_message = v1.sum_product(pf1)
		expected_message = torch.zeros(4)
		assert torch.equal(actual_message, expected_message)

		# Now test graph-level sum-product. This will actually store
		# the messages on the edges
		beliefs = fg.loopy_sum_product(1, [v1], [pf1, v1])
		assert len(beliefs) == 1
		assert beliefs['v1'] == torch.logsumexp(pf1_factor, 0)