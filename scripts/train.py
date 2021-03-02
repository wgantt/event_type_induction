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

                if save_posteriors:
                    # Save the posteriors for the variable nodes to a file
                    event_posteriors_file = events + ".csv"
                    role_posteriors_file = roles + ".csv"
                    participant_posteriors_file = participants + ".csv"
                    relations_posteriors_file = relations + ".csv"
                    posteriors_files = [
                        ckpt_dir + "/" + event_posteriors_file,
                        ckpt_dir + "/" + role_posteriors_file,
                        ckpt_dir + "/" + participant_posteriors_file,
                        ckpt_dir + "/" + relations_posteriors_file,
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
                        dump_fg_posteriors(post_file, posteriors, t, n_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default="scripts/train.json")
    args = parser.parse_args()
    main(args)
