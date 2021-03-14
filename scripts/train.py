import argparse
import json

from scripts.setup_logging import setup_logging
from decomp import UDSCorpus
from event_type_induction.trainers.induction_trainer import EventTypeInductionTrainer
from event_type_induction.utils import (
    dump_fg_posteriors,
    load_event_structure_annotations,
    parameter_grid,
    save_model_with_args,
)
from event_type_induction.constants import *

LOG = setup_logging()


def main(args):

    # Load parameters from file
    with open(args.parameters) as f:
        params = json.load(f)

    # Load UDS
    LOG.info("Loading UDS Corpus...")
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    LOG.info("Loading UDS-EventStructure annotations...")
    load_event_structure_annotations(uds)
    LOG.info("...Complete.")

    # Begin training
    for hyperparams in parameter_grid(params["hyper"]):
        for trainparams in parameter_grid(params["training"]):
            for ckpts in parameter_grid(params["checkpoints"]):

                # Model checkpoint file info
                ckpt_dir = ckpts["ckpt_dir"]
                ckpt_file_root = ckpts["ckpt_file_name"]
                save_ckpt = ckpts["save_ckpts"]
                save_posteriors = ckpts["save_posteriors"]
                model_ckpts = {
                    Type.EVENT: ckpts["event_mmm"],
                    Type.PARTICIPANT: ckpts["participant_mmm"],
                    Type.ROLE: ckpts["role_mmm"],
                    Type.RELATION: ckpts["relation_mmm"],
                }

                # Initialize training parameters
                all_trainer_params = {
                    **hyperparams,
                    "uds": uds,
                    "mmm_ckpts": model_ckpts,
                }

                # Do training
                trainer = EventTypeInductionTrainer(**all_trainer_params)
                LOG.info("Beginning training with the following settings:")
                LOG.info(json.dumps(trainparams, indent=4))
                LOG.info("...And hyperparameters:")
                LOG.info(json.dumps(hyperparams, indent=4))
                LOG.info("...And checkpoints:")
                LOG.info(
                    json.dumps({k.name: v for k, v in model_ckpts.items()}, indent=4)
                )
                model, posteriors = trainer.fit(**trainparams)
                LOG.info("Training finished.")

                # Informative model name based on hyperparameter settings
                events = "ev" + str(model.n_event_types)
                roles = "ro" + str(model.n_role_types)
                participants = "en" + str(model.n_participant_types)
                relations = "re" + str(model.n_relation_types)
                model_name = "-".join(
                    [ckpt_file_root, events, roles, participants, relations]
                )

                if save_ckpt:
                    # Save model
                    ckpt_file = model_name + ".pt"
                    LOG.info(f"Saving model to {ckpt_file}")
                    save_model_with_args(
                        params, model, hyperparams, ckpt_dir, model_name
                    )
                    LOG.info("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default="scripts/train.json")
    args = parser.parse_args()
    main(args)
