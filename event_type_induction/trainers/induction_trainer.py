import event_type_induction.utils as utils
from event_type_induction.modules.induction import EventTypeInductionModel

from scripts.setup_logging import setup_logging

import torch
import numpy as np
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
        uds: UDSCorpus,
        model=None,
        batch_size=1,
        lr=1e-3,
        n_epochs=1,
        device: str = "cpu",
        random_seed=42,
    ):
        self.n_event_types = n_event_types
        self.n_role_types = n_role_types
        self.n_relation_types = n_relation_types
        self.n_entity_types = n_entity_types
        self.bp_iters = bp_iters
        self.uds = uds
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = torch.device(device)
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        if model is None:
            self.model = EventTypeInductionModel(
                n_event_types,
                n_role_types,
                n_relation_types,
                n_entity_types,
                bp_iters,
                uds,
                device=self.device,
                random_seed=self.random_seed,
            )
        else:
            self.model = model

    # TODO: incorporate batch size
    def fit(self, verbosity: int = 10):

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        batch_num = 0

        LOG.info("Adding UDS-EventStructure annotations")
        utils.load_event_structure_annotations(self.uds)

        documents_by_split = utils.get_documents_by_split(self.uds)

        LOG.info(f"Beginning training for a maximum of {self.n_epochs} epochs.")
        for epoch in range(10):
            loss_trace = []
            fixed_trace = []
            # Testing on a single document for now
            for doc in sorted(list(documents_by_split["train"]))[:1]:

                # Forward
                self.model.zero_grad()
                fixed_loss, random_loss = self.model(self.uds.documents[doc])
                LOG.info(fixed_loss)
                loss = fixed_loss + random_loss
                print(fixed_loss, random_loss)
                loss_trace.append(fixed_loss + random_loss)
                fixed_trace.append(fixed_loss)

                # Backward + optimizer step
                print("calling backward!")
                loss.backward()
                optimizer.step()

        return self.model.eval()
