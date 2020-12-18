import argparse
import json

from scripts.setup_logging import setup_logging
from decomp import UDSCorpus
from event_type_induction.trainers.induction_trainer import EventTypeInductionTrainer
from event_type_induction.utils import (
    load_event_structure_annotations,
    parameter_grid,
    save_model_with_args,
)

LOG = setup_logging()


def main(args):

    # Load parameters from file
    with open(args.parameters) as f:
        params = json.load(f)

    checkpoints = params["checkpoints"]
    ckpt_dir = checkpoints["ckpt_dir"]
    ckpt_file = checkpoints["ckpt_file_name"] + ".pt"
    save_ckpt = checkpoints["save_ckpts"]

    # Load UDS
    LOG.info("Loading UDS Corpus...")
    uds = UDSCorpus(version="2.0", annotation_format="raw")
    LOG.info("Loading UDS-EventStructure annotations...")
    load_event_structure_annotations(uds)
    LOG.info("...Complete.")

    # Begin training
    for hyperparams in parameter_grid(params["hyper"]):
        for trainparams in parameter_grid(params["training"]):
            all_trainer_params = {
                **hyperparams,
                "uds": uds,
                "random_seed": trainparams["random_seed"],
            }
            trainer = EventTypeInductionTrainer(**all_trainer_params)

            LOG.info("Beginning training with the following settings:")
            LOG.info(json.dumps(trainparams, indent=4))
            LOG.info("...And hyperparameters:")
            LOG.info(json.dumps(hyperparams, indent=4))
            model = trainer.fit(**trainparams)
            LOG.info("Training finished.")

            # TODO: generate checkpoint file name that includes
            # hyperparameter info
            if save_ckpt:
                LOG.info("Saving model...")
                save_model_with_args(params, model, hyperparams, ckpt_dir, ckpt_file)
                LOG.info("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default="scripts/train.json")
    args = parser.parse_args()
    main(args)
