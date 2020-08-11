from modules.induction import EventTypeInductionModel
from decomp import UDSCorpus

def main():
	n_event_types = 2
	n_role_types = 3
	n_relation_types = 4
	n_entity_types = 5
	uds_train = UDSCorpus(split='train', annotation_format='raw')
	model = EventTypeInductionModel(n_event_types, n_role_types, n_relation_types, n_entity_types, uds_train)
	test_doc_id = uds_train['ewt-train-13'].document_id
	test_doc = uds_train.documents[test_doc_id]
	ll = model(test_doc)
	print(ll)

if __name__ == '__main__':
	main()