import torch
import networkx as nx
import unittest

from event_type_induction.modules.factor_graph import *
from event_type_induction.utils import exp_normalize


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

    def three_variable_dimension_mismatch_setup(self):
        fg = FactorGraph()

        # Initialize variable nodes
        e1 = VariableNode("e1", VariableType.EVENT, 2)
        e2 = VariableNode("e2", VariableType.EVENT, 2)
        r1 = VariableNode("r1", VariableType.RELATION, 3)
        fg.set_node(e1)
        fg.set_node(e2)
        fg.set_node(r1)

        # Initialize single prior factor node
        pf_factor = -torch.FloatTensor(
            [i + j + k for i in range(4) for j in range(4) for k in range(3)]
        )
        pf_factor = pf_factor.reshape((4, 4, 3))
        pf = PriorFactorNode("pf", pf_factor, VariableType.RELATION)
        fg.set_node(pf)
        fg.set_edge(e1, pf, 0)
        fg.set_edge(e2, pf, 1)
        fg.set_edge(r1, pf, 2)

        # Verify that appropriate edge types are created
        assert isinstance(fg[e1][pf]["object"], DimensionMismatchEdge)
        assert isinstance(fg[e2][pf]["object"], DimensionMismatchEdge)
        assert not isinstance(fg[r1][pf]["object"], DimensionMismatchEdge)

        # Verify that they are initialized with the appropriate messages
        e1_pf = fg[e1][pf]["object"]
        e2_pf = fg[e2][pf]["object"]
        contracted_msg_init = torch.zeros(2)
        expanded_msg_init = torch.zeros(4)
        expanded_msg_init[2:] = NEG_INF
        assert torch.equal(e1_pf.get_message(pf, e1), contracted_msg_init)
        assert torch.equal(e2_pf.get_message(pf, e2), contracted_msg_init)
        assert torch.equal(e1_pf.get_message(e1, pf), expanded_msg_init)
        assert torch.equal(e2_pf.get_message(e2, pf), expanded_msg_init)

        return fg, e1, e2, r1, pf

    def four_variable_setup(self, cyclic=False):
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

        # Connect V1 and V4 to create a cycle
        if cyclic:
            dist_fd = torch.log(torch.Tensor([[0.3, 0.4], [0.3, 0.0]]))
            pf4 = PriorFactorNode("pf4", dist_fd, VariableType.EVENT)
            fg.set_node(pf4)
            fg.set_edge(v4, pf4, 0)
            fg.set_edge(pf4, v1, 1)
            return fg, v1, v2, v3, v4, pf1, pf2, pf3, pf4
        else:
            return fg, v1, v2, v3, v4, pf1, pf2, pf3

    def test_sum_product_one_variable_node(self):

        fg, v1, pf1 = self.one_variable_setup()

        # Run sum-product from the prior to the variable. The message
        # should simply be the log of summed exponentials of the
        # elements in the prior factor.
        actual_message = pf1.sum_product(v1)
        expected_message = pf1.factor
        assert torch.equal(actual_message, expected_message)

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
        assert torch.equal(beliefs["v1"], pf1.factor)

        # Run for several iterations; results should not change
        beliefs = fg.loopy_sum_product(5, [v1], [pf1, v1])
        assert len(beliefs) == 1
        assert torch.equal(beliefs["v1"], pf1.factor)

        # Switching the schedule shouldn't change the result
        belifs = fg.loopy_sum_product(5, [v1], [v1, pf1])
        assert len(beliefs) == 1
        assert torch.equal(beliefs["v1"], pf1.factor)

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
        fg, v1, v2, v3, v4, pf1, pf2, pf3 = self.four_variable_setup(cyclic=False)

        query_nodes = v1, v2, v3, v4
        forward_schedule = [v1, pf1, v2, pf2, v3, pf3, v4]

        # Even though this is a linear chain, we still run "loopy" BP
        # just to test convergence.
        beliefs = fg.loopy_sum_product(20, query_nodes, forward_schedule)

        assert len(beliefs) == 4
        actual_beliefs = [
            torch.exp(beliefs[v]) / torch.sum(torch.exp(beliefs[v]))
            for v in ["v1", "v2", "v3", "v4"]
        ]

        # These are the marginals output in the terminal after 20 iterations
        # of loopy BP using the schedule specified above, and are very close
        # to the exact marginals.
        expected_beliefs = [
            torch.FloatTensor([0.6489, 0.3511]),
            torch.FloatTensor([0.7021, 0.2979]),
            torch.FloatTensor([0.7447, 0.2553]),
            torch.FloatTensor([0.5745, 0.4255]),
        ]
        for a, e in zip(actual_beliefs, expected_beliefs):
            assert torch.allclose(a, e, atol=1e-04), f"expected {e} but got {a}"

    def test_sum_product_four_variable_nodes_with_cycle(self):
        fg, v1, v2, v3, v4, pf1, pf2, pf3, pf4 = self.four_variable_setup(cyclic=True)

        query_nodes = v1, v2, v3, v4
        schedule = [v1, pf1, v2, pf2, v3, pf3, v4, pf4]

        actual_beliefs = fg.loopy_sum_product(20, query_nodes, schedule)
        actual_beliefs = [torch.exp(exp_normalize(b)) for b in actual_beliefs.values()]

        # Since all factors are identical and since the graph is a ring,
        # all the the beliefs are the same
        expected_beliefs = [
            torch.Tensor([0.6987, 0.3013]) for _ in range(len(actual_beliefs))
        ]
        for a, e in zip(actual_beliefs, expected_beliefs):
            assert torch.allclose(a, e, atol=1e-04), f"expected {e} but got {a}"

    def test_dimension_mismatch_edge(self):
        fg, e1, e2, r1, pf = self.three_variable_dimension_mismatch_setup()

        query_nodes = e1, e2, r1
        schedule = [e1, e2, r1, pf]

        actual_beliefs = fg.loopy_sum_product(20, query_nodes, schedule)
        actual_beliefs = [torch.exp(exp_normalize(b)) for b in actual_beliefs.values()]
        e1_expected = torch.Tensor([0.7311, 0.2689])
        e2_expected = torch.Tensor([0.7311, 0.2689])
        r1_expected = torch.Tensor([0.6652, 0.2447, 0.0900])
        expected_beliefs = [e1_expected, e2_expected, r1_expected]
        for a, e in zip(actual_beliefs, expected_beliefs):
            assert torch.allclose(a, e, atol=1e-04), f"expected {e} but got {a}"
