from event_type_induction.modules.induction import EventTypeInductionModel
from event_type_induction.utils import *

from scripts.setup_logging import setup_logging

import torch
import numpy as np
import os
import random
from decomp import UDSCorpus
from torch.optim import Adam
from typing import Dict

LOG = setup_logging()
ckpt_dir = "/data/wgantt/event_type_induction/full_model_w_confidence_weighting"


def save_all_posteriors(
    model: EventTypeInductionModel,
    ckpt_dir: str,
    all_posteriors: Dict[str, torch.Tensor],
    epoch: int,
) -> None:
    event_posteriors_file = (
        "-".join(["ev", str(model.n_event_types), "epoch", str(epoch)]) + ".csv"
    )
    role_posteriors_file = (
        "-".join(["ro", str(model.n_role_types), "epoch", str(epoch)]) + ".csv"
    )
    participant_posteriors_file = (
        "-".join(["pa", str(model.n_participant_types), "epoch", str(epoch)]) + ".csv"
    )
    relation_posteriors_file = (
        "-".join(["re", str(model.n_relation_types), "epoch", str(epoch)]) + ".csv"
    )
    posteriors_files = [
        ckpt_dir + "/" + event_posteriors_file,
        ckpt_dir + "/" + role_posteriors_file,
        ckpt_dir + "/" + participant_posteriors_file,
        ckpt_dir + "/" + relation_posteriors_file,
    ]
    types = [
        (Type.EVENT, model.n_event_types),
        (Type.ROLE, model.n_role_types),
        (Type.PARTICIPANT, model.n_participant_types),
        (Type.RELATION, model.n_relation_types),
    ]
    for (t, n_types), post_file in zip(types, posteriors_files):
        LOG.info(
            f"Saving per-item posteriors for {t.name.lower()} types to {post_file}"
        )
        dump_fg_posteriors(post_file, all_posteriors, t, n_types)


class EventTypeInductionTrainer:
    def __init__(
        self,
        n_event_types: int,
        n_role_types: int,
        n_relation_types: int,
        n_participant_types: int,
        bp_iters: int,
        uds: UDSCorpus,
        use_ordinal: bool = True,
        clip_min_ll: bool = True,
        confidence_weighting: bool = False,
        model=None,
        mmm_ckpts=None,
        device: str = "cpu",
        random_seed=42,
    ):
        self.n_event_types = n_event_types
        self.n_role_types = n_role_types
        self.n_relation_types = n_relation_types
        self.n_participant_types = n_participant_types
        self.bp_iters = bp_iters
        self.uds = uds
        self.use_ordinal = (use_ordinal,)
        self.clip_min_ll = (clip_min_ll,)
        self.confidence_weighting = (confidence_weighting,)
        self.device = torch.device(device)
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        if model is None:
            self.model = EventTypeInductionModel(
                n_event_types,
                n_role_types,
                n_relation_types,
                n_participant_types,
                bp_iters,
                uds,
                use_ordinal,
                clip_min_ll,
                confidence_weighting,
                mmm_ckpts=mmm_ckpts,
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
        random_seed: int = 42,  # currently set elsewhere
        verbosity: int = 10,  # currently unused
        save_posteriors: bool = True,
        start_on_dev: bool = False,
        time: bool = False,
    ):

        optimizer = Adam(self.model.parameters(), lr=lr)

        LOG.info("Adding UDS-EventStructure annotations")
        load_event_structure_annotations(self.uds)

        documents_by_split = get_documents_by_split(self.uds)
        prev_epoch_dev_fixed_loss = float("inf")

        LOG.info(f"Beginning training for a maximum of {n_epochs} epochs.")
        LOG.info(
            f"Training on all {len(documents_by_split['train'])} documents in UDS train split."
        )
        LOG.info(
            f"Evaluating on all {len(documents_by_split['dev'])} documents in UDS dev split."
        )

        for epoch in range(n_epochs):

            # Bookkeeping
            batch_fixed_trace = []
            batch_random_trace = []
            epoch_fixed_trace = []
            epoch_random_trace = []
            batch_num = 0
            loss = torch.FloatTensor([0.0]).to(self.device)
            LOG.info(f"Starting epoch {epoch}")

            # If this is the last epoch, we save all the per-item posteriors
            all_posteriors = {}

            for i, doc in enumerate(sorted(documents_by_split["train"])):
                if start_on_dev and epoch == 0:
                    LOG.info("Evaluating on dev first before training.")
                    break

                # Forward
                self.model.zero_grad()
                fixed_loss, random_loss, posteriors = self.model(
                    self.uds.documents[doc], time, save_posteriors
                )
                loss += fixed_loss + random_loss
                LOG.info(f"Fixed loss for document {i} ({doc}): {fixed_loss.item()}")
                batch_fixed_trace.append(fixed_loss.item())
                batch_random_trace.append(random_loss.item())
                epoch_fixed_trace.append(fixed_loss.item())
                epoch_random_trace.append(random_loss.item())

                # Save per-item posteriors from the current document
                if save_posteriors:
                    all_posteriors[doc] = posteriors

                # Backprop + optimizer step
                # In standard EM, we would obviously want batch size to equal
                # the train set size, but this led to some training runs being
                # killed, so a smaller batch size is required.
                if ((i + 1) % batch_size) == 0:
                    batch_fixed_loss = np.round(np.mean(batch_fixed_trace), 3)
                    batch_random_loss = np.round(np.mean(batch_random_trace), 3)
                    LOG.info(
                        f"Batch {batch_num} mean loss: {batch_fixed_loss} (fixed); {batch_random_loss} (random)"
                    )
                    loss.backward()
                    optimizer.step()
                    loss = torch.FloatTensor([0.0]).to(self.device)
                    batch_fixed_trace = []
                    batch_random_trace = []
                    batch_num += 1

            if (epoch > 0) or (not start_on_dev):
                epoch_fixed_loss = np.round(np.mean(epoch_fixed_trace), 3)
                epoch_random_loss = np.round(np.mean(epoch_random_trace), 3)
                epoch_mean_loss_str = f"Epoch {epoch} mean train loss: {epoch_fixed_loss} (fixed); {epoch_random_loss} (random)"
                LOG.info("-" * len(epoch_mean_loss_str))
                LOG.info(epoch_mean_loss_str)
                LOG.info("-" * len(epoch_mean_loss_str))

            # Evaluate on all of dev at the end of each epoch
            dev_epoch_fixed_trace = []
            LOG.info(f"Beginning dev eval...")
            for i, doc in enumerate(sorted(documents_by_split["dev"])):
                with torch.no_grad():
                    fixed_loss, random_loss, posteriors = self.model(
                        self.uds.documents[doc], time, save_posteriors
                    )
                    dev_epoch_fixed_trace.append(fixed_loss.item())
                    LOG.info(
                        f"Fixed loss for document {i} ({doc}): {fixed_loss.item()}"
                    )
                    all_posteriors[doc] = posteriors
            epoch_dev_fixed_loss = np.mean(dev_epoch_fixed_trace)
            dev_epoch_mean_loss_str = (
                f"Epoch {epoch} mean dev fixed loss: {np.round(epoch_dev_fixed_loss,3)}"
            )
            LOG.info("-" * len(dev_epoch_mean_loss_str))
            LOG.info(dev_epoch_mean_loss_str)
            LOG.info("-" * len(dev_epoch_mean_loss_str))

            no_improvement = prev_epoch_dev_fixed_loss < epoch_dev_fixed_loss
            if no_improvement:
                LOG.info(f"No improvement in dev on epoch {epoch}. Stopping early.")
                break
            prev_epoch_dev_fixed_loss = epoch_dev_fixed_loss

            is_last_epoch = epoch == n_epochs - 1
            if save_posteriors and not (no_improvement or is_last_epoch):
                save_all_posteriors(self.model, ckpt_dir, all_posteriors, epoch)
            model_name = (
                "-".join(
                    [
                        "ev",
                        str(self.model.n_event_types),
                        "ro",
                        str(self.model.n_role_types),
                        "pa",
                        str(self.model.n_participant_types),
                        "re",
                        str(self.model.n_relation_types),
                        "epoch",
                        str(epoch),
                    ]
                )
                + ".pt"
            )
            save_model(self.model.state_dict(), ckpt_dir, model_name)

        # Compute test posteriors before quitting
        test_fixed_loss_trace = []
        LOG.info("Beginning test eval...")
        for i, doc in enumerate(sorted(documents_by_split["test"])):
            with torch.no_grad():
                fixed_loss, random_loss, posteriors = self.model(
                    self.uds.documents[doc], time, save_posteriors
                )
                test_fixed_loss_trace.append(fixed_loss.item())
                LOG.info(f"Fixed loss for document {i} ({doc}): {fixed_loss.item()}")
                all_posteriors[doc] = posteriors

        # Save the final train, dev, and test posteriors
        if save_posteriors:
            save_all_posteriors(self.model, ckpt_dir, all_posteriors, epoch)

        return self.model.eval(), all_posteriors
