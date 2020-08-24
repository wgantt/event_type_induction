import event_type_induction.utils as utils
from event_type_induction.modules.induction import EventTypeInductionModel
from event_type_induction.scripts.setup_logging import setup_logging

import torch
import random
from decomp import UDSCorpus
from torch.nn import NLLLoss
from torch.optim import Adam

LOG = setup_logging()


class EventTypeInductionTrainer:
    def __init__(
        self,
        n_event_types: int,
        n_role_types: int,
        n_relation_types: int,
        n_entity_types: int,
        bp_iters: int,
        device: str = "cpu",
        random_seed=42,
    ):
        self.n_event_types = n_event_types
        self.n_role_types = n_role_types
        self.n_relation_types = n_relation_types
        self.n_entity_types = n_entity_types
        self.bp_iters = bp_iters
        self.device = torch.device(device)
        self.random_seed = random_seed
        self.uds = UDSCorpus(annotation_format="raw")
        self.model = EventTypeInductionModel(
            n_event_types,
            n_role_types,
            n_relation_types,
            n_entity_types,
            bp_iters,
            uds,
            device=device,
        )

    def fit(self, n_epochs: int = 10, lr: float = 1e-3, verbosity: int = 10):

        optimizer = Adam(self.model.parameters(), lr=lr)
        loss = NLLLoss()

        self.model.train()

        LOG.info("Loading UDS corpus for training...")
        uds = UDSCorpus(annotation_format="raw")

        LOG.info("Adding UDS-EventStructure annotations")
        utils.load_event_structure_annotations(uds)

        LOG.info("Finished loading UDS corpus.")

        LOG.info(f"Beginning training for a maximum of {n_epochs} epochs.")
        loss_trace = []

        documents = utils.get_documents_by_split(uds)

        for epoch in range(n_epochs):

            # TODO
            pass
