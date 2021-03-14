import argparse
import os
import numpy as np
import pandas as pd
import torch


def main(args):
    np.random.seed(42)
    lower_bounds = []
    upper_bounds = []
    medians = []
    for i in range(args.min_types, args.max_types + 1):
        # Read in unnormalized posteriors, subset to dev
        posteriors_file = (
            "-".join([args.posteriors_file_root, str(i), "posteriors"]) + ".csv"
        )
        posteriors_file = os.path.join(args.posteriors_dir, posteriors_file)
        post = pd.read_csv(posteriors_file)
        post_cols = [c for c in post if "posterior" in c]
        post_dev = np.array(post[post.item_name.str.contains("dev")][post_cols])

        # Compute evidence for all dev items
        evidence = torch.logsumexp(torch.tensor(post_dev), 1)

        # Sample estimates of the log evidence
        estimates = []
        for _ in range(args.n_samples):
            bootstrap_idxs = np.random.choice(
                post_dev.shape[0], post_dev.shape[0], replace=True
            )
            e = (
                torch.logsumexp(torch.tensor(post_dev[bootstrap_idxs]), dim=1)
                .sum()
                .item()
            )
            estimates.append(e)

        # Save confidence intervals for this estimate
        lb, med, ub = np.quantile(estimates, [0.025, 0.5, 0.975])
        lower_bounds.append(int(lb))
        upper_bounds.append(int(ub))
        medians.append(int(med))

    # For each type, determine whether the median estimate for the
    # log-evidence (50% CI) is greater than the 2.5% CI for all greater
    # numbers of types. This is our selection criterion.
    for i,t in enumerate(range(args.min_types, args.max_types + 1)): 
        lb = lower_bounds[i]
        ub = upper_bounds[i]
        med = medians[i]
        med_greater_than_successive_lbs = (np.array(lower_bounds[i+1:]) < med).all()
        print(f"Log evidence estimate CIs for {t} types:")
        print(f"2.5%: {int(lb)} | 50%: {int(med)} | 97.5%: {int(ub)}")
        print(f"Median greater than all successive lower bounds? {med_greater_than_successive_lbs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("posteriors_file_root")
    parser.add_argument(
        "--posteriors_dir", default="/data/wgantt/event_type_induction/checkpoints"
    )
    parser.add_argument("--n_samples", type=int, default=999)
    parser.add_argument("--min_types", type=int, default=2)
    parser.add_argument("--max_types", type=int, default=10)
    args = parser.parse_args()
    main(args)
