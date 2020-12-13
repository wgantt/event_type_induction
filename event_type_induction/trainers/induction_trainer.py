import event_type_induction.utils as utils
from event_type_induction.modules.induction import EventTypeInductionModel

# from scripts.setup_logging import setup_logging

import torch
import random
from decomp import UDSCorpus
from torch.nn import NLLLoss
from torch.optim import Adam

# TODO: figure out imports for logging
# LOG = setup_logging()


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

        if model is None:
            self.model = EventTypeInductionModel(
                n_event_types,
                n_role_types,
                n_relation_types,
                n_entity_types,
                bp_iters,
                uds,
                device=device,
            )
        else:
            self.model = model

    # TODO: incorporate batch size
    def fit(self, verbosity: int = 10):

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        batch_num = 0

        # LOG.info("Adding UDS-EventStructure annotations")
        utils.load_event_structure_annotations(self.uds)

        documents_by_split = utils.get_documents_by_split(self.uds)

        # LOG.info(f"Beginning training for a maximum of {n_epochs} epochs.")
        for epoch in range(self.n_epochs):
            loss_trace = []
            fixed_trace = []
            for doc in documents_by_split["train"]:

                # Forward
                self.model.zero_grad()
                fixed_loss, random_loss = self.model(self.uds.documents[doc])
                print(fixed_loss, random_loss)
                loss = fixed_loss + random_loss
                loss_trace.append(fixed_loss + random_loss)
                fixed_trace.append(fixed_loss)

                # Backward + optimizer step
                print("calling backward!")
                loss.backward()
                optimizer.step()

        return self.model.eval()
