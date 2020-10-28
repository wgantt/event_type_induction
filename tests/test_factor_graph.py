import torch
import networkx as nx
import unittest

from event_type_induction.modules.factor_graph import *


class TestFactorGraph(unittest.TestCase):
    def one_variable_setup(self):
        fg = FactorGraph()

        # Add a single variable node (the variable type is unimportant
        # here, as are the number of types).
        v1 = VariableNode("v1", VariableType.EVENT, 4)
        fg.set_node(v1)

        # Create a prior factor node, and connect it to the variable
        # node above
        pf1_factor = torch.log(torch.FloatTensor([0.1, 0.2, 0.3, 0.4]))
        pf1 = PriorFactorNode("pf1", pf1_factor, VariableType.EVENT)
        fg.set_node(pf1)
        fg.set_edge(v1, pf1)

        return fg, v1, pf1

    def two_variable_setup(self):
        """Two-variable linear chain (no likelihood factors)"""
        fg = FactorGraph()

        # Add variable nodes. Variable types have no
        # significance here.
        v1 = VariableNode("v1", VariableType.EVENT, 2)
        v2 = VariableNode("v2", VariableType.ROLE, 3)
        fg.set_node(v1)
        fg.set_node(v2)

        # Add prior factor node to connect the two
        pf_factor = torch.log(torch.FloatTensor([[0.1, 0.5], [0.2, 0.3], [0.3, 0.4]]))
        pf1 = PriorFactorNode("pf1", pf_factor, VariableType.EVENT)
        fg.set_node(pf1)
        fg.set_edge(v1, pf1, 1)
        fg.set_edge(pf1, v2, 0)

        return fg, v1, v2, pf1

    def four_variable_setup(self):
        """Four-variable linear chain (no likelihood factors)"""
        fg = FactorGraph()

        v1 = VariableNode("v1", VariableType.EVENT, 2)
        v2 = VariableNode("v2", VariableType.EVENT, 2)
        v3 = VariableNode("v3", VariableType.EVENT, 2)
        v4 = VariableNode("v4", VariableType.EVENT, 2)
        fg.set_node(v1)
        fg.set_node(v2)
        fg.set_node(v3)
        fg.set_node(v4)

        dist_fa = torch.log(torch.Tensor([[0.3, 0.4], [0.3, 0.0]]))
        pf1 = PriorFactorNode("pf1", dist_fa, VariableType.EVENT)
        fg.set_node(pf1)
        fg.set_edge(v1, pf1, 0)
        fg.set_edge(pf1, v2, 1)

        dist_fb = torch.log(torch.Tensor([[0.3, 0.4], [0.3, 0.0]]))
        pf2 = PriorFactorNode("pf2", dist_fb, VariableType.EVENT)
        fg.set_node(pf2)
        fg.set_edge(v2, pf2, 0)
        fg.set_edge(pf2, v3, 1)

        dist_fc = torch.log(torch.Tensor([[0.3, 0.4], [0.3, 0.0]]))
        pf3 = PriorFactorNode("pf3", dist_fc, VariableType.EVENT)
        fg.set_node(pf3)
        fg.set_edge(v3, pf3, 0)
        fg.set_edge(pf3, v4, 1)

        return fg, v1, v2, v3, v4, pf1, pf2, pf3

    @unittest.skip("blah")
    def test_sum_product_one_variable_node(self):

        fg, v1, pf1 = self.one_variable_setup()

        # Run sum-product from the prior to the variable. The message
        # should simply be the log of summed exponentials of the
        # elements in the prior factor.
        actual_message = pf1.sum_product(v1)
        expected_message = normalize_message(torch.logsumexp(pf1.factor, 0))
        assert actual_message == expected_message

        # Try sum-product in the other direction. Since the messages at
        # the variable nodes are initialized with zeros, a zero tensor
        # should be passed to the prior node.
        actual_message = v1.sum_product(pf1)
        expected_message = normalize_message(torch.zeros(4))
        assert torch.equal(actual_message, expected_message)

        # Now test graph-level sum-product. This will actually store
        # the messages on the edges
        beliefs = fg.loopy_sum_product(1, [v1], [pf1, v1])
        assert len(beliefs) == 1
        assert beliefs["v1"] == normalize_message(torch.logsumexp(pf1.factor, 0))

        # Run for several iterations; results should not change
        beliefs = fg.loopy_sum_product(5, [v1], [pf1, v1])
        assert len(beliefs) == 1
        assert beliefs["v1"] == normalize_message(torch.logsumexp(pf1.factor, 0))

        # Switching the schedule shouldn't change the result
        belifs = fg.loopy_sum_product(5, [v1], [v1, pf1])
        assert len(beliefs) == 1
        assert beliefs["v1"] == normalize_message(torch.logsumexp(pf1.factor, 0))

    @unittest.skip("blah")
    def test_sum_product_two_variable_nodes(self):
        fg, v1, v2, pf1 = self.two_variable_setup()

        # Compute beliefs for both variable nodes
        beliefs = fg.loopy_sum_product(5, [v1, v2], [v1, pf1, v2])
        assert len(beliefs) == 2

        # Validate against expected beliefs
        expected_v1_beliefs = normalize_message(
            torch.log(torch.FloatTensor([0.6, 1.2]))
        )
        expected_v2_beliefs = normalize_message(
            torch.log(torch.FloatTensor([0.6, 0.5, 0.7]))
        )
        assert torch.allclose(
            beliefs["v1"], expected_v1_beliefs
        ), f"expected {expected_v1_beliefs} but got {beliefs['v1']}"
        assert torch.allclose(
            beliefs["v2"], expected_v2_beliefs
        ), f"expected {expected_v2_beliefs} but got {beliefs['v2']}"

    def test_sum_product_four_variable_nodes(self):
        fg, v1, v2, v3, v4, pf1, pf2, pf3 = self.four_variable_setup()

        # Mimic non-loopy belief propagation by forcing a
        # forward/backward schedule
        query_nodes = v1, v2, v3, v4
        forward_schedule = [v1, pf1, v2, pf2, v3, pf3, v4]
        forward_exclusions = {
            v1: None,
            pf1: v1,
            v2: pf1,
            pf2: v2,
            v3: pf2,
            pf3: v3,
            v4: pf3,
        }
        backward_schedule = list(reversed(forward_schedule))
        backward_exclusions = {
            v4: None,
            pf3: v4,
            v3: pf3,
            pf2: v3,
            v2: pf2,
            pf1: v2,
            v1: pf1,
        }

        beliefs = fg.loopy_sum_product(
            1, query_nodes, forward_schedule, forward_exclusions
        )
        beliefs = fg.loopy_sum_product(
            1, query_nodes, backward_schedule, backward_exclusions
        )
        assert len(beliefs) == 4
        print(
            [
                torch.exp(beliefs[v]) / torch.sum(torch.exp(beliefs[v]))
                for v in ["v1", "v2", "v3", "v4"]
            ]
        )
