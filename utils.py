from constants import *
from decomp import UDSCorpus
from typing import Set, Tuple

def load_annotator_ids(uds: UDSCorpus) -> Tuple[Set[str]]:
	"""Fetch all of the annotator IDs from an annotated UDS corpus

	Parameters
	----------
	uds
		The UDSCorpus object from which annotator IDs are to
		be extracted
	"""
	def helper(items, prop_attrs):
		annotators = set()
		prop_domains = prop_attrs.keys()
		for item, annotation in items:
			for domain, props in annotation.items():
				if domain in prop_domains:
					for p in annotation[domain].keys():
						annotators = annotators.union(props[p]['value'].keys())
		return annotators


	# Process sentence-level annotations
	pred_node_annotators = set()
	arg_node_annotators = set()
	sem_edge_annotators = set()
	doc_edge_annotators = set()
	for graph in uds:
		pred_items = uds[graph].predicate_nodes.items()
		arg_items = uds[graph].argument_nodes.items()
		sem_edge_items = uds[graph].semantics_edges().items()

		pred_node_annotators = pred_node_annotators.union(helper(pred_items, PREDICATE_ANNOTATION_ATTRIBUTES))
		arg_node_annotators = arg_node_annotators.union(helper(arg_items, ARGUMENT_ANNOTATION_ATTRIBUTES))
		sem_edge_annotators = sem_edge_annotators.union(helper(sem_edge_items, SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES))

	# Process document-level annotations
	for doc in uds.documents.values():
		doc_edge_items = doc.document_graph.edges.items()
		doc_edge_annotators = doc_edge_annotators.union(helper(doc_edge_items, DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES))

	return pred_node_annotators, arg_node_annotators,\
		   sem_edge_annotators, doc_edge_annotators