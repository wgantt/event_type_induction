# Package internal imports
from event_type_induction.utils import dump_params

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
        ckpts, outfiles = [args.model_path], [args.out_path]
    for ckpt, out in zip(ckpts, outfiles):
        ckpt_dict = torch.load(ckpt)["state_dict"]
        mu_prefix = args.type + "_mus"
        mus = {k.split(".")[-1]: v for k, v in ckpt_dict.items() if mu_prefix in k}
        dump_params(out, mus)


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
        "type", type=str, choices=["event", "participant", "relation", "role"]
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If specified, process an entire directory of model checkpoints",
    )
    args = parser.parse_args()
    main(args)
