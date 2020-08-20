import os

from collections import defaultdict
from decomp import UDSCorpus, RawUDSDataset
from event_type_induction.constants import *
from glob import glob
from pkg_resources import resource_filename
from typing import Dict, Generator, Set, Tuple

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

def get_documents_by_split(uds: UDSCorpus) -> Dict[str, Set[str]]:
	"""Get sets of UDS document IDs based on their split

	This is really a utility UDS should provide on its own.

	Parameters
	----------
	uds
		The UDSCorpus object from which document IDs are
		to be extracted
	"""
	splits = defaultdict(set)
	for doc_id, doc in uds.documents.items():
		sample_sentence = list(doc.sentence_ids)[0]
		split = sample_sentence.split('-')[1]
		splits[split].add(doc_id)
	return splits

def load_event_structure_annotations(uds: UDSCorpus) -> None:
	"""Loads the UDS-EventStructure annotations

	These annotations are not included in v0.1.0 of UDS and
	must therefore be loaded after the corpus has been initialized

	Parameters
	----------
	uds
		The UDSCorpus object into which the annotations will
		be loaded
	"""
	data_dir = resource_filename('event_type_induction', 'data')
	annotation_paths = glob(os.path.join(data_dir, '*.json'))
	for path in annotation_paths:
		annotation = RawUDSDataset.from_json(path)
		uds.add_annotation(annotation)

def parameter_grid(param_dict: Dict[str, Any]) -> Generator[Dict[str, Any]]:
	"""Generator for training hyperparameter grid

	Parameters
	----------
	param_dict
		Dictionary containing the hyperparameters and their possible values
	"""
	ks = list(param_dict.keys())
	vlists = []
	for k, v in param_dict.items():
		if isinstance(v, dict):
			vlists.append(parameter_grid(v))
		elif isinstance(v, list):
			vlists.append(v)
		else:
			errmsg = ("param_dict must be a dictionary contining lists or "
					  "recursively other param_dicts")
			raise ValueError(errmsg)
	for configuration in product(*vlists):
		yield dict(zip(ks, configuration))

def save_model(data_dict, ckpt_dir, file_name):
	if not os.path.isdir(ckpt_dir):
		os.makedirs(ckpt_dir)
	ckpt_path = os.path.join(ckpt_dir, file_name)
	torch.save(data_dict, ckpt_path)
	return ckpt_path

def save_model_with_args(params, model, initargs, ckpt_dir, file_name):
    filtered_args = {}
    for p in inspect.signature(model.__class__.__init__).parameters:
        if p in initargs:
            filtered_args[p] = initargs[p]
    ckpt_dict = dict(params, **{'state_dict': model.state_dict(),
                                'curr_hyper': filtered_args})
    return save_model(ckpt_dict, ckpt_dir, file_name)

def load_model_with_args(cls, ckpt_path):
    ckpt_dict = torch.load(ckpt_path)
    hyper_params = ckpt_dict['curr_hyper']
    model = cls(**hyper_params)
    model.load_state_dict(ckpt_dict['state_dict'])
    return model, hyper_params