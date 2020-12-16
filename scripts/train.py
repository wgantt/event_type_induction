import argparse

from decomp import UDSCorpus
from event_type_induction.trainers.induction_trainer import EventTypeInductionTrainer
from event_type_induction.utils import load_event_structure_annotations
from scripts.setup_logging import setup_logging

LOG = setup_logging()

def main(args):
    # Just testing for now
    n_event_types = 2
    n_role_types = 3
    n_relation_types = 4
    n_entity_types = 5
    bp_iters = 10

    LOG.info("Loading UDS Corpus...")
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    LOG.info("Loading UDS-EventStructure annotations...")
    load_event_structure_annotations(uds)
    LOG.info("...Complete. Beginning training.")

    trainer = EventTypeInductionTrainer(
        n_event_types, n_role_types, n_relation_types, n_entity_types, bp_iters, uds
    )
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: add args
    args = parser.parse_args()
    main(args)
