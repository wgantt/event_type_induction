import torch
import networkx as nx
import unittest

from event_type_induction.modules.factor_graph import *


class TestFactorGraph(unittest.TestCase):
    def fglib_test_setup(self):
        fg = FactorGraph()

        x1 = VariableNode("x1", VariableType.EVENT, 2)
        x2 = VariableNode("x2", VariableType.EVENT, 2)
        x3 = VariableNode("x3", VariableType.EVENT, 2)
        x4 = VariableNode("x4", VariableType.EVENT, 2)
        fg.set_node(x1)
        fg.set_node(x2)
        fg.set_node(x3)
        fg.set_node(x4)

        dist_fa = torch.Tensor(np.array([[0.3, 0.4], [0.3, 0.0]]))
        dist_fa = torch.log(dist_fa)
        pfa = PriorFactorNode("f-x1-x2", dist_fa, VariableType.EVENT)
        fg.set_node(pfa)
        fg.set_edge(x1, pfa, 0)
        fg.set_edge(pfa, x2, 1)

        dist_fb = torch.Tensor(np.array([[0.3, 0.4], [0.3, 0.0]]))
        dist_fb = torch.log(dist_fb)
        pfb = PriorFactorNode("f-x2-x3", dist_fb, VariableType.EVENT)
        fg.set_node(pfb)
        fg.set_edge(x2, pfb, 0)
        fg.set_edge(pfb, x3, 1)

        dist_fc = torch.Tensor(np.array([[0.3, 0.4],[0.3, 0.0]]))
        dist_fc = torch.log(dist_fc)
        pfc = PriorFactorNode("f-x3-x4", dist_fc, VariableType.EVENT)
        fg.set_node(pfc)
        fg.set_edge(x3, pfc, 0)
        fg.set_edge(pfc, x4, 1)

        return fg

    def two_variable_setup(self):
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

    def test_sum_product_one_variable_node(self):
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
        assert beliefs["v1"] == torch.logsumexp(pf1_factor, 0)

        # Run for several iterations; results should not change
        beliefs = fg.loopy_sum_product(5, [v1], [pf1, v1])
        assert len(beliefs) == 1
        assert beliefs["v1"] == torch.logsumexp(pf1_factor, 0)

        # Switching the schedule shouldn't change the result
        belifs = fg.loopy_sum_product(5, [v1], [v1, pf1])
        assert len(beliefs) == 1
        assert beliefs["v1"] == torch.logsumexp(pf1_factor, 0)

    def test_sum_product_two_variable_nodes(self):
        fg, v1, v2, pf1 = self.two_variable_setup()

        # Compute beliefs for both variable nodes
        beliefs = fg.loopy_sum_product(5, [v1, v2], [v1, pf1, v2])
        assert len(beliefs) == 2

        # Validate against expected beliefs
        expected_v1_beliefs = torch.FloatTensor([0.6, 1.2])
        expected_v2_beliefs = torch.FloatTensor([0.6, 0.5, 0.7])
        assert torch.allclose(
            torch.exp(beliefs["v1"]), expected_v1_beliefs
        ), f"expected {expected_v1_beliefs} but got {torch.exp(beliefs['v1'])}"
        assert torch.allclose(
            torch.exp(beliefs["v2"]), expected_v2_beliefs
        ), f"expected {expected_v2_beliefs} but got {torch.exp(beliefs['v2'])}"

    def test_run_fglib_test(self):
        fg = self.fglib_test_setup()

        # Mimic non-loopy belief propagation by forcing a
        # forward/backward schedule
        query_nodes = fg.variable_nodes["x4"]
        forward_schedule = [edge[0] for edge in nx.dfs_edges(fg, query_nodes)] + [
            list(nx.dfs_edges(fg, query_nodes))[-1][1]
        ]
        backward_schedule = list(reversed(forward_schedule))
        beliefs = fg.loopy_sum_product(1, [query_nodes], forward_schedule)
        beliefs = fg.loopy_sum_product(1, [query_nodes], backward_schedule)
        assert len(beliefs) == 1
        print(torch.exp(beliefs["x4"]) / torch.sum(torch.exp(beliefs["x4"])))

    # def test_max_product_two_variable_nodes(self):
    #     fg, v1, v2, pf1 = self.two_variable_setup()

    #     beliefs = fg.loopy_max_product(5, [v1, v2], [v1, pf1, v2])
    #     assert len(beliefs) == 2

    #     expected_v1_beliefs = torch.FloatTensor([0.5, 0.3])
    #     expected_v2_beliefs = torch.FloatTensor([0.5, 0.3, 0.4])
    #     assert torch.allclose(
    #         torch.exp(beliefs["v1"]), expected_v1_beliefs
    #     ), f"expected {expected_v1_beliefs} but got {torch.exp(beliefs['v1'])}"
    #     assert torch.allclose(
    #         torch.exp(beliefs["v2"]), expected_v2_beliefs
    #     ), f"expected {expected_v2_beliefs} but got {torch.exp(beliefs['v2'])}"
