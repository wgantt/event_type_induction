import unittest

from decomp import UDSCorpus
from event_type_induction.modules.induction import EventTypeInductionModel
from utils import load_event_structure_annotations

class TestEventTypeInductionModel(unittest.TestCase):

	@classmethod
	def setup_class(cls):
		n_event_types = 2
		n_role_types = 3
		n_relation_types = 4
		n_entity_types = 5

		# Load UDS with raw annotations
		uds = UDSCorpus(split='train', annotation_format='raw')
		load_event_structure_annotations(uds)

		cls.uds = uds
		cls.model = EventTypeInductionModel(n_event_types, n_role_types, n_relation_types, n_entity_types, cls.uds)

	def test_factor_graph_construction(self):
		uds = self.__class__.uds
		model = self.__class__.model

		# A test document
		test_doc_id = uds['ewt-train-1'].document_id
		test_doc = uds.documents[test_doc_id]
		test_doc_sentences = test_doc.sentence_ids

		# Construct the factor graph
		fg = model.construct_factor_graph(test_doc)