import unittest

from decomp import UDSCorpus
from event_type_induction.modules.induction import EventTypeInductionModel, FactorGraph
from event_type_induction.utils import load_event_structure_annotations


class TestEventTypeInductionModel(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        n_event_types = 2
        n_role_types = 3
        n_relation_types = 4
        n_entity_types = 5

        # Load UDS with raw annotations
        uds = UDSCorpus(split="train", annotation_format="raw")
        load_event_structure_annotations(uds)

        cls.uds = uds
        cls.model = EventTypeInductionModel(
            n_event_types, n_role_types, n_relation_types, n_entity_types, cls.uds
        )

    def test_factor_graph_construction(self):
        uds = self.__class__.uds
        model = self.__class__.model

        # A test document
        test_doc_id = uds["ewt-train-1"].document_id
        test_doc = uds.documents[test_doc_id]

        # Construct the factor graph
        fg = model.construct_factor_graph(test_doc)

        # Verify that all appropriate factor graph nodes and edges
        # have been constructed for the sentence level
        for graphid, graph in test_doc.sentence_graphs.items():

            # Predicates
            for pred in graph.predicate_nodes:

                # Verify nodes
                pred_event_v_node = FactorGraph.get_node_name("v", pred, "event")
                pred_event_lf_node = FactorGraph.get_node_name("lf", pred, "event")
                pred_event_pf_node = FactorGraph.get_node_name("pf", pred, "event")
                assert (
                    pred_event_v_node in fg.variable_nodes
                ), f"{pred} has no event variable node"
                assert (
                    pred_event_lf_node in fg.factor_nodes
                ), f"{pred} has no likelihood factor node"
                assert (
                    pred_event_lf_node in fg.factor_nodes
                ), f"{pred} has no prior factor node"

                # Verify edges
                pred_event_v_node = fg.variable_nodes[pred_event_v_node]
                pred_event_lf_node = fg.factor_nodes[pred_event_lf_node]
                pred_event_pf_node = fg.factor_nodes[pred_event_pf_node]
                assert fg.has_edge(
                    pred_event_v_node, pred_event_lf_node
                ), f"{pred} event has no edge between variable and likelihood factor nodes"
                assert fg.has_edge(
                    pred_event_v_node, pred_event_pf_node
                ), f"{pred} event has no edge between variable and prior factor nodes"

            # Arguments
            for arg in graph.argument_nodes:

                # Verify nodes
                arg_lf_node = FactorGraph.get_node_name("lf", arg, "role")
                arg_participant_v_node = FactorGraph.get_node_name(
                    "v", arg, "participant"
                )
                arg_participant_pf_node = FactorGraph.get_node_name(
                    "pf", arg, "participant"
                )
                assert (
                    arg_lf_node in fg.factor_nodes
                ), f"{arg} has no role likelihood factor node"
                assert (
                    arg_participant_v_node in fg.variable_nodes
                ), f"{arg} has no participant variable node"
                assert (
                    arg_participant_pf_node in fg.factor_nodes
                ), f"{arg} has no participant prior factor node"

                # Verify edges
                arg_lf_node = fg.factor_nodes[arg_lf_node]
                arg_participant_v_node = fg.variable_nodes[arg_participant_v_node]
                arg_participant_pf_node = fg.factor_nodes[arg_participant_pf_node]
                assert fg.has_edge(
                    arg_participant_v_node, arg_participant_pf_node
                ), f"{arg} participant has no edge between variable and prior factor nodes"

                # Semantics edges
                for v1, v2 in graph.semantics_edges(arg):

                    # Verify nodes
                    arg_role_v_node = FactorGraph.get_node_name("v", v1, v2, "role")
                    arg_role_pf_node = FactorGraph.get_node_name("pf", v1, v2, "role")
                    sem_edge_lf_node = FactorGraph.get_node_name(
                        "lf", v1, v2, "participant"
                    )
                    assert (
                        arg_role_pf_node in fg.factor_nodes
                    ), f"{arg} has no role prior factor node"
                    assert (
                        arg_role_v_node in fg.variable_nodes
                    ), f"{arg} has no role variable node"
                    assert (
                        sem_edge_lf_node in fg.factor_nodes
                    ), f"{(v1, v2)} has no likelihood factor node"

                    # Verify edges
                    pred = v1 if "pred" in v1 else v2
                    pred_event_v_node = FactorGraph.get_node_name("v", pred, "event")
                    pred_event_v_node = fg.variable_nodes[pred_event_v_node]
                    arg_role_v_node = fg.variable_nodes[arg_role_v_node]
                    arg_role_pf_node = fg.factor_nodes[arg_role_pf_node]
                    sem_edge_lf_node = fg.factor_nodes[sem_edge_lf_node]
                    assert fg.has_edge(
                        arg_role_v_node, arg_lf_node
                    ), f"{arg} role has no edge between variable and likelihood factor nodes"
                    assert fg.has_edge(
                        arg_role_v_node, arg_role_pf_node
                    ), f"{arg} role has no edge between variable and prior factor nodes"
                    assert fg.has_edge(
                        pred_event_v_node, arg_role_pf_node
                    ), f"{arg} role has no edge between prior factor node and pred event variable node"
                    assert fg.has_edge(
                        arg_participant_v_node, arg_role_pf_node
                    ), f"{arg} role has no edge between prior factor node and participant variable node"
                    assert fg.has_edge(
                        arg_participant_v_node, sem_edge_lf_node
                    ), f"{(v1, v2)} has no edge between arg variable and edge likelihood factor nodes"

        # Do the same for the document level
        for v1, v2 in test_doc.document_graph.edges:

            # Verify nodes
            relation_v_node = FactorGraph.get_node_name("v", v1, v2, "relation")
            relation_lf_node = FactorGraph.get_node_name("lf", v1, v2, "relation")
            relation_pf_node = FactorGraph.get_node_name("pf", v1, v2, "relation")
            assert (
                relation_v_node in fg.variable_nodes
            ), f"{(v1, v2)} has no variable node"
            assert (
                relation_lf_node in fg.factor_nodes
            ), f"{(v1, v2)} has no likelihood factor node"
            assert (
                relation_pf_node in fg.factor_nodes
            ), f"{(v1, v2)} has no prior factor node"

            # Verify edges
            relation_v_node = fg.variable_nodes[relation_v_node]
            relation_lf_node = fg.factor_nodes[relation_lf_node]
            relation_pf_node = fg.factor_nodes[relation_pf_node]
            assert fg.has_edge(
                relation_v_node, relation_lf_node
            ), f"{(v1, v2)} has no edge between variable and likelihood factor nodes"
            assert fg.has_edge(
                relation_v_node, relation_pf_node
            ), f"{(v1, v2)} has no edge between variable and prior factor nodes"

            for factor_dim, var_node_name in enumerate([v1, v2]):

                sem_node_name = var_node_name.replace("document", "semantics")
                if "pred" in var_node_name:
                    fg_node_name = FactorGraph.get_node_name(
                        "v", sem_node_name, "event"
                    )
                else:
                    fg_node_name = FactorGraph.get_node_name(
                        "v", sem_node_name, "participant"
                    )
                var_node = fg.variable_nodes.get(fg_node_name)
                assert fg.has_edge(
                    var_node, relation_pf_node
                ), f"{(v1, v2)} has no edge between {var_node} and the prior factor node"
