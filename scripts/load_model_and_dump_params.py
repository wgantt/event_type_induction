# Package internal imports
from event_type_induction.utils import dump_property_means
from scripts.cluster_init import MultiviewMixtureModel

# Package external imports
import argparse
import os
import torch


def main(args):
    if args.all:
        model_names = os.listdir(args.model_path)
        ckpts = [os.path.join(args.model_path, f) for f in model_names]
        outfiles = [
            os.path.join(args.out_path, name.split(".")[0] + ".csv")
            for name in model_names
        ]
    else:
        ckpts, outfiles = args.model_path, args.outfile
    for ckpt, out in zip(ckpts, outfiles):
        ckpt_dict = torch.load(ckpt)
        mus = {k.split(".")[-1]: v for k, v in ckpt_dict.items() if "mus" in k}
        dump_property_means(mus, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model whose parameters are to be dumped",
    )
    parser.add_argument(
        "out_path", type=str, help="CSV file to which model parameters are to be dumped"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If specified, process an entire directory of model checkpoints",
    )
    args = parser.parse_args()
    main(args)
