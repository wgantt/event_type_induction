import torch
import unittest
from decomp import UDSCorpus

from event_type_induction.modules.induction import (
    EventTypeInductionModel,
    FactorGraph,
    LikelihoodFactorNode,
    PriorFactorNode,
    VariableNode,
    VariableType,
)
from event_type_induction.modules.likelihood import (
    Likelihood,
    PredicateNodeAnnotationLikelihood,
    ArgumentNodeAnnotationLikelihood,
    SemanticsEdgeAnnotationLikelihood,
    DocumentEdgeAnnotationLikelihood,
)
from event_type_induction.utils import (
    load_event_structure_annotations,
    load_annotator_ids,
    ridit_score_confidence,
)
from event_type_induction.trainers.induction_trainer import EventTypeInductionTrainer


class TestEventTypeInductionModel(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.n_event_types = 2
        cls.n_role_types = 3
        cls.n_relation_types = 4
        cls.n_entity_types = 5

        cls.bp_iters = 1

        # Load UDS with raw annotations
        uds = UDSCorpus(split="train", annotation_format="raw")
        load_event_structure_annotations(uds)

        cls.uds = uds

        # Construct the model
        cls.model = EventTypeInductionModel(
            cls.n_event_types,
            cls.n_role_types,
            cls.n_relation_types,
            cls.n_entity_types,
            cls.bp_iters,
            cls.uds,
        )

        # Construct a test factor graph
        cls.test_doc_id = cls.uds["ewt-train-1"].document_id
        cls.test_doc = cls.uds.documents[cls.test_doc_id]
        cls.test_fg = cls.model.construct_factor_graph(cls.test_doc)

        # All annotators
        (
            pred_node_annotators,
            arg_node_annotators,
            sem_edge_annotators,
            doc_edge_annotators,
        ) = load_annotator_ids(uds)

        # Load ridit-scored annotator confidence values
        ridits = ridit_score_confidence(uds)
        cls.pred_node_annotator_confidence = {
            a: ridits.get(a) for a in pred_node_annotators
        }
        cls.arg_node_annotator_confidence = {
            a: ridits.get(a) for a in arg_node_annotators
        }
        cls.sem_edge_annotator_confidence = {
            a: ridits.get(a) for a in sem_edge_annotators
        }
        cls.doc_edge_annotator_confidence = {
            a: ridits.get(a) for a in doc_edge_annotators
        }

        # Trainer
        cls.trainer = EventTypeInductionTrainer(
            cls.n_event_types,
            cls.n_role_types,
            cls.n_relation_types,
            cls.n_entity_types,
            cls.bp_iters,
            cls.uds,
            model=cls.model,
        )

    @unittest.skip("Faster iteration on other tests")
    def test_factor_graph_construction(self):
        """Verify correct graph structure"""
        uds = self.__class__.uds
        model = self.__class__.model

        # A test document
        test_doc_id = self.__class__.test_doc_id
        test_doc = self.__class__.test_doc

        # Get the factor graph
        fg = self.__class__.test_fg

        # Verify that all appropriate factor graph nodes and edges
        # have been constructed for the sentence level
        for graphid, graph in test_doc.sentence_graphs.items():

            # Predicates
            for pred in graph.predicate_nodes:

                # Verify nodes
                event_v_node = FactorGraph.get_node_name("v", pred, "event")
                pred_lf_node = FactorGraph.get_node_name("lf", pred)
                event_pf_node = FactorGraph.get_node_name("pf", pred, "event")
                assert (
                    event_v_node in fg.variable_nodes
                ), f"{pred} has no event variable node"
                assert (
                    pred_lf_node in fg.factor_nodes
                ), f"{pred} has no likelihood factor node"
                assert (
                    pred_lf_node in fg.factor_nodes
                ), f"{pred} has no prior factor node"

                # Verify edges
                event_v_node = fg.variable_nodes[event_v_node]
                pred_lf_node = fg.factor_nodes[pred_lf_node]
                event_pf_node = fg.factor_nodes[event_pf_node]
                assert fg.has_edge(
                    event_v_node, pred_lf_node
                ), f"{pred} event has no edge between variable and likelihood factor nodes"
                assert fg.has_edge(
                    event_v_node, event_pf_node
                ), f"{pred} event has no edge between variable and prior factor nodes"

            # Arguments
            for arg in graph.argument_nodes:

                # Verify nodes
                arg_lf_node = FactorGraph.get_node_name("lf", arg)
                participant_v_node = FactorGraph.get_node_name("v", arg, "participant")
                participant_pf_node = FactorGraph.get_node_name(
                    "pf", arg, "participant"
                )
                assert (
                    arg_lf_node in fg.factor_nodes
                ), f"{arg} has no role likelihood factor node"
                assert (
                    participant_v_node in fg.variable_nodes
                ), f"{arg} has no participant variable node"
                assert (
                    participant_pf_node in fg.factor_nodes
                ), f"{arg} has no participant prior factor node"

                # Verify edges
                arg_lf_node = fg.factor_nodes[arg_lf_node]
                participant_v_node = fg.variable_nodes[participant_v_node]
                participant_pf_node = fg.factor_nodes[participant_pf_node]
                assert fg.has_edge(
                    participant_v_node, participant_pf_node
                ), f"{arg} participant has no edge between variable and prior factor nodes"

                # Semantics edges
                for v1, v2 in graph.semantics_edges(arg):

                    # Verify nodes
                    role_v_node = FactorGraph.get_node_name("v", v1, v2, "role")
                    role_pf_node = FactorGraph.get_node_name("pf", v1, v2, "role")
                    sem_edge_lf_node = FactorGraph.get_node_name("lf", v1, v2)
                    assert (
                        role_pf_node in fg.factor_nodes
                    ), f"{arg} has no role prior factor node"
                    assert (
                        role_v_node in fg.variable_nodes
                    ), f"{arg} has no role variable node"
                    assert (
                        sem_edge_lf_node in fg.factor_nodes
                    ), f"{(v1, v2)} has no likelihood factor node"

                    # Verify edges
                    pred = v1 if "pred" in v1 else v2
                    event_v_node = FactorGraph.get_node_name("v", pred, "event")
                    event_v_node = fg.variable_nodes[event_v_node]
                    role_v_node = fg.variable_nodes[role_v_node]
                    role_pf_node = fg.factor_nodes[role_pf_node]
                    sem_edge_lf_node = fg.factor_nodes[sem_edge_lf_node]
                    assert fg.has_edge(
                        role_v_node, sem_edge_lf_node
                    ), f"{arg} role has no edge between variable and likelihood factor nodes"
                    assert fg.has_edge(
                        role_v_node, role_pf_node
                    ), f"{arg} role has no edge between variable and prior factor nodes"
                    assert fg.has_edge(
                        event_v_node, role_pf_node
                    ), f"{arg} role has no edge between prior factor node and pred event variable node"
                    assert fg.has_edge(
                        participant_v_node, role_pf_node
                    ), f"{arg} role has no edge between prior factor node and participant variable node"
                    assert fg.has_edge(
                        participant_v_node, arg_lf_node
                    ), f"{(v1, v2)} has no edge between arg variable and edge likelihood factor nodes"

        # Do the same for the document level
        for v1, v2 in test_doc.document_graph.edges:

            # Verify nodes
            relation_v_node = FactorGraph.get_node_name("v", v1, v2, "relation")
            doc_edge_lf_node = FactorGraph.get_node_name("lf", v1, v2)
            relation_pf_node = FactorGraph.get_node_name("pf", v1, v2, "relation")
            assert (
                relation_v_node in fg.variable_nodes
            ), f"{(v1, v2)} has no variable node"
            assert (
                doc_edge_lf_node in fg.factor_nodes
            ), f"{(v1, v2)} has no likelihood factor node"
            assert (
                relation_pf_node in fg.factor_nodes
            ), f"{(v1, v2)} has no prior factor node"

            # Verify edges
            relation_v_node = fg.variable_nodes[relation_v_node]
            doc_edge_lf_node = fg.factor_nodes[doc_edge_lf_node]
            relation_pf_node = fg.factor_nodes[relation_pf_node]
            assert fg.has_edge(
                relation_v_node, doc_edge_lf_node
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

    @unittest.skip("Faster iteration on other tests")
    def test_predicate_node_annotation_likelihood(self):
        uds = self.__class__.uds
        model = self.__class__.model
        pred_node_annotator_confidence = self.__class__.pred_node_annotator_confidence
        pred_ll = PredicateNodeAnnotationLikelihood(
            pred_node_annotator_confidence, uds.metadata.sentence_metadata
        )
        ll = torch.FloatTensor([0.0])
        for node, anno in uds["ewt-train-12"].predicate_nodes.items():
            ll = pred_ll(model.event_mus, anno)
        # for k, v in model.event_mus.items():
        #     print(f"{k}: {v.data}")
        assert False

    @unittest.skip("Faster iteration on other tests")
    def test_semantics_edge_annotation_likelihood(self):
        uds = self.__class__.uds
        model = self.__class__.model
        sem_edge_annotator_confidence = self.__class__.sem_edge_annotator_confidence
        pred_ll = SemanticsEdgeAnnotationLikelihood(
            sem_edge_annotator_confidence, uds.metadata.sentence_metadata
        )
        ll = torch.FloatTensor([0.0])
        for edge, anno in uds["ewt-train-12"].semantics_edges().items():
            ll = pred_ll(model.role_mus, anno)
        assert False

    @unittest.skip("Faster iteration on other tests")
    def test_document_edge_annotation_likelihood(self):
        uds = self.__class__.uds
        model = self.__class__.model
        test_doc = self.__class__.test_doc
        doc_edge_annotator_confidence = self.__class__.doc_edge_annotator_confidence
        doc_ll = DocumentEdgeAnnotationLikelihood(
            doc_edge_annotator_confidence, uds.metadata.document_metadata
        )
        ll = torch.FloatTensor([0.0])
        for edge, anno in test_doc.document_graph.edges().items():
            ll = doc_ll(model.relation_mus, model.relation_covs, anno)
        assert False

    @unittest.skip("Faster iteration on other tests")
    def test_argument_node_annotation_likelihood(self):
        uds = self.__class__.uds
        model = self.__class__.model
        arg_node_annotator_confidence = self.__class__.arg_node_annotator_confidence
        arg_ll = ArgumentNodeAnnotationLikelihood(
            arg_node_annotator_confidence, uds.metadata.sentence_metadata
        )
        ll = torch.FloatTensor([0.0])
        for node, anno in uds["ewt-train-12"].argument_nodes.items():
            ll = arg_ll(model.participant_mus, anno)
        assert False

    @unittest.skip("Faster iteration on other tests")
    def test_node_and_edge_initialization(self):
        """Verify factor graph node and edge attributes"""
        uds = self.__class__.uds
        model = self.__class__.model

        # A test document
        test_doc_id = self.__class__.test_doc_id
        test_doc = self.__class__.test_doc

        # Get the factor graph
        fg = self.__class__.test_fg

        # Verify variable node initialization
        for node_id, node in fg.variable_nodes.items():
            assert (
                node.ntypes is not None
            ), f"Variable node {node_id} was not initialized with a number of types"

            assert torch.equal(
                node.init, torch.zeros(node.ntypes)
            ), f"Incorrect message initialization for variable node {node_id}"

            # Verify neighboring factor nodes
            n_neighbors = 0
            for neighbor in node.neighbors():
                assert node != neighbor, f"Node {node_id} is its own neighbor!"
                assert isinstance(neighbor, LikelihoodFactorNode) or isinstance(
                    neighbor, PriorFactorNode
                ), f"neighbor {neighbor.label} of variable node {node_id} is not a factor node!"
                assert (
                    neighbor.factor is not None
                ), f"Factor node {node_id} has no factor!"
                n_neighbors += 1
            assert n_neighbors > 0, f"Variable node {node_id} has no neighbors!"

        # Verify factor node initialization
        for node_id, node in fg.factor_nodes.items():
            assert node.factor is not None, f"Factor node {node_id} has no factor!"

            # Likelihood factor nodes
            if isinstance(node, LikelihoodFactorNode):
                assert (
                    node.mus is not None
                ), f"Likelihood factor node {node_id} has no mus!"
                assert (
                    node.annotation is not None
                ), f"Likelihood factor node {node_id} has no annotation!"
                assert isinstance(
                    node.factor, Likelihood
                ), f"Likelihood factor node {node_id} has non-Likelihood factor type"

            # Prior factor nodes
            elif isinstance(node, PriorFactorNode):
                assert isinstance(
                    node.factor, torch.Tensor
                ), f"Prior factor node {node_id} has non-Tensor factor type"

            # Verify properties of neighbors
            n_neighbors = 0
            for neighbor in node.neighbors():
                assert node != neighbor, f"Node {node_id} is its own neighbor!"
                assert isinstance(
                    neighbor, VariableNode
                ), f"Factor node {node_id} has non-variable neighbor {neighbor.label}"
                if isinstance(node, PriorFactorNode):
                    # Special case of mismatched dimension for predicate variable nodes
                    # and relation prior factor nodes
                    if "relation-pf" in node_id and "event" in neighbor.label:
                        neighbor_msg_dim = (
                            neighbor.ntypes + self.__class__.n_entity_types
                        )
                    else:
                        neighbor_msg_dim = neighbor.ntypes
                    assert (
                        neighbor_msg_dim in node.factor.shape
                    ), f"No dimension of length {neighbor.ntypes} from neighbor {neighbor.label} in factor for node {node_id}, which has shape {node.factor.shape}."
                n_neighbors += 1
            assert n_neighbors > 0, f"Factor node {node_id} has no neighbors"

            # Verify the number of neighbors
            if isinstance(node, PriorFactorNode):
                assert n_neighbors == len(
                    node.factor.shape
                ), f"Mismatch between number of neighbors {n_neighbors} and factor shape {node.factor.shape} for node {node_id}"
            elif isinstance(node, LikelihoodFactorNode):
                assert (
                    n_neighbors == 1
                ), f"LikelihoodFactorNode {node_id} has {n_neighbors} neighbors but should have only one"

    @unittest.skip("Faster iteration on other tests")
    def test_loopy_sum_product(self):
        """Verify that loopy sum-product (BP) runs without errors"""
        uds = self.__class__.uds
        model = self.__class__.model

        # A test document
        test_doc_id = self.__class__.test_doc_id
        test_doc = self.__class__.test_doc

        # The factor graph
        fg = self.__class__.test_fg

        # Query nodes for belief (marginal) computations
        query_nodes = list(fg.variable_nodes.values())

        # Test loopy sum-product
        fixed, random = model(test_doc)
        assert False

    def test_trainer(self):
        trainer = self.__class__.trainer
        trainer.fit()
        assert False

    @unittest.skip("Faster iteration on other tests")
    def test_loopy_max_product(self):
        """Verify that loopy max-product runs without errors"""
        uds = self.__class__.uds
        model = self.__class__.model

        # A test document
        test_doc_id = self.__class__.test_doc_id
        test_doc = self.__class__.test_doc

        # The factor graph
        fg = self.__class__.test_fg

        # Some query nodes for belief computations
        test_arg_nodes = list(uds["ewt-train-1"].argument_nodes)
        query_node_names = [
            FactorGraph.get_node_name("v", arg, "participant") for arg in test_arg_nodes
        ]
        query_nodes = [fg.variable_nodes[name] for name in query_node_names]

        # Test loopy sum-product
        fg.loopy_max_product(model.bp_iters, query_nodes)
