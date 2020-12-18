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
        device: str = "cpu",
        random_seed=42,
    ):
        self.n_event_types = n_event_types
        self.n_role_types = n_role_types
        self.n_relation_types = n_relation_types
        self.n_entity_types = n_entity_types
        self.bp_iters = bp_iters
        self.uds = uds
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

        self.model.to(self.device)

    def fit(
        self,
        batch_size: int = 1,
        n_epochs: int = 10,
        lr: float = 1e-3,
        random_seed: int = 42,
        verbosity: int = 10,
    ):

        optimizer = Adam(self.model.parameters(), lr=lr)
        batch_num = 0

        LOG.info("Adding UDS-EventStructure annotations")
        utils.load_event_structure_annotations(self.uds)

        documents_by_split = utils.get_documents_by_split(self.uds)

        LOG.info(f"Beginning training for a maximum of {n_epochs} epochs.")
        for epoch in range(n_epochs):
            fixed_trace = []
            random_trace = []
            loss = torch.FloatTensor([0.0]).to(self.device)
            for doc in sorted(list(documents_by_split["train"]))[:1]:

                # Forward
                self.model.zero_grad()
                fixed_loss, random_loss = self.model(self.uds.documents[doc])
                loss += fixed_loss + random_loss
                fixed_trace.append(fixed_loss.item())
                random_trace.append(random_loss.item())

                # Backprop + optimizer step
                batch_num += 1
                if (batch_num % batch_size) == 0:
                    loss.backward()
                    optimizer.step()
                    loss = torch.FloatTensor([0.0]).to(self.device)

            epoch_fixed_loss = np.round(np.mean(fixed_trace), 3)
            epoch_random_loss = np.round(np.mean(random_trace), 3)
            LOG.info(
                f"Epoch {epoch} mean loss: {epoch_fixed_loss} (fixed); {epoch_random_loss} (random)"
            )

        return self.model.eval()
