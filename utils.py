from pkg_resources import resource_filename
import json
from decomp import UDSCorpus, RawUDSDataset

# Base directory containing the UDS-EventStructure annotations
# TODO: figure out how to select directory more precisely
DATA_DIR = 'data/'

# List of annotation files
UDS_EVENT_STRUCTURE_ANNOTATIONS = [
DATA_BASE_DIR + 'distributivity/train/preprocessed.json'
]

def load_annotations():
	"""Loads UDS-Aspect annotations into UDS graphs

	Annotations are expected in JSON format
	"""
	uds = UDSCorpus()
	for annotation in UDS_EVENT_STRUCTURE_ANNOTATIONS:
		uds.add_annotation(RawUDSDataset.from_json(annotation))

	return uds