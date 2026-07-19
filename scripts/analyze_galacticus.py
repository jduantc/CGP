#!/usr/bin/env python
"""Exploratory clustering/visualization CLI, originally built for Galacticus
semi-analytic galaxy-formation output (basicMass, diskMassStellar,
spheroidMassStellar, diskMassGas, spheroidMassGas, hotHaloMass,
blackHoleMass), but works on any whitespace-delimited table with a header
row. No dataset is bundled with this repo; point --data at your own table.

This script was exploratory application work, separate from the CGP/AR
pipeline in cgp/ and not part of the main CGP/AR results.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cgp.viz import plot_clusters_corner


def load_file(path):
    with open(path) as f:
        header = f.readline().strip().lstrip("#").split()
    return np.loadtxt(path, skiprows=1), header


def split_by_percentile(data, column, low_cutoff, high_cutoff):
    """Split rows into three subsets by thresholds on `column`'s values."""
    low = data[data[:, column] < low_cutoff]
    mid = data[(data[:, column] >= low_cutoff) & (data[:, column] <= high_cutoff)]
    high = data[data[:, column] > high_cutoff]
    return low, mid, high


def run_kmeans(data, nclusters):
    kmeans = KMeans(n_clusters=nclusters).fit(data)
    clusters = [data[kmeans.labels_ == i] for i in range(nclusters)]
    plot_clusters_corner(clusters).show()


def run_gmm(data, nclusters):
    gmm = GaussianMixture(n_components=nclusters).fit(data)
    labels = gmm.predict(data)
    clusters = [data[labels == i] for i in range(nclusters)]
    plot_clusters_corner(clusters).show()


def show_dendrogram(data):
    import matplotlib.pyplot as plt
    links = linkage(data)
    plt.figure()
    dendrogram(links)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="Path to a whitespace-delimited table with a header row.")
    parser.add_argument("--corner", action="store_true", help="Show a corner (pairwise) plot of the data.")
    parser.add_argument("--kmeans", type=int, metavar="N", help="Cluster into N groups with k-means and show a corner plot.")
    parser.add_argument("--gmm", type=int, metavar="N", help="Cluster into N groups with a Gaussian mixture model and show a corner plot.")
    parser.add_argument("--dendrogram", action="store_true", help="Show a hierarchical clustering dendrogram.")
    parser.add_argument("--split", nargs=3, metavar=("COLUMN", "LOW_CUTOFF", "HIGH_CUTOFF"),
                         help="Split into 3 subsets by percentile thresholds on COLUMN, writing <out>_{low,mid,high}.dat.")
    parser.add_argument("--out", default="split", help="Output filename prefix for --split.")
    args = parser.parse_args()

    data, header = load_file(args.data)

    if args.corner:
        plot_clusters_corner([data]).show()
    if args.kmeans:
        run_kmeans(data, args.kmeans)
    if args.gmm:
        run_gmm(data, args.gmm)
    if args.dendrogram:
        show_dendrogram(data)
    if args.split:
        column_name, low_cutoff, high_cutoff = args.split
        column = header.index(column_name)
        low, mid, high = split_by_percentile(data, column, float(low_cutoff), float(high_cutoff))
        header_line = " ".join(header)
        for suffix, subset in (("low", low), ("mid", mid), ("high", high)):
            np.savetxt(f"{args.out}_{suffix}.dat", subset, header=header_line)


if __name__ == "__main__":
    main()
