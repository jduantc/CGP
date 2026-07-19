#!/usr/bin/env python
"""Bootstrap-resample a dataset into a folder of random subsets, for cross-validation."""

import argparse
import os
import random

import numpy as np


def load_file(path):
    return np.loadtxt(path, skiprows=1)


def generate_subsets(data_file, out_dir, subset_fraction, nsubsets, var_names):
    """Write nsubsets random subsets of data_file (each subset_fraction of its rows) to out_dir."""
    data = load_file(data_file)
    nrows, ncols = data.shape
    subset_size = round(nrows * subset_fraction)
    header = " ".join(var_names)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, nsubsets + 1):
        subset = np.empty((subset_size, ncols))
        eligible_rows = list(range(nrows))
        for j in range(subset_size):
            row_number = random.randrange(len(eligible_rows))
            subset[j] = data[eligible_rows.pop(row_number)]
        np.savetxt(os.path.join(out_dir, f"subset_{i}"), subset, header=header)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Path to the source dataset.")
    parser.add_argument("out_dir", help="Directory to write subset_1, subset_2, ... into.")
    parser.add_argument("--fraction", type=float, default=0.6, help="Subset size as a fraction of the full dataset.")
    parser.add_argument("--nsubsets", type=int, default=40)
    parser.add_argument("--var-names", nargs="+", default=None, help="Column names for the header; defaults to X1, X2, ...")
    args = parser.parse_args()

    ncols = load_file(args.data).shape[1]
    var_names = args.var_names or [f"X{i + 1}" for i in range(ncols)]
    generate_subsets(args.data, args.out_dir, args.fraction, args.nsubsets, var_names)


if __name__ == "__main__":
    main()
