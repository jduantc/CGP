"""Causal Graph Probing (CGP).

The pipeline: cluster the dataset, estimate each cluster's "affinity" (swing)
toward each of two candidate DAGs, split clusters into DAG-1 / DAG-2 / neutral
subsets, and search the 2-graph space for the pair that best explains the whole
dataset.

Affinity/swing scoring uses cgp.covstats's sufficient-statistics form of AR
(see docs/notes/covariance_ar.md), which lets `optimize_graphs` score every
cluster's swing toward every candidate DAG ONCE (sharing the same drawn
subsets across all of them) instead of redrawing and rescoring per pair --
both far cheaper and lower-variance than independent draws per pair, since
the shared subsets act as common random numbers for the pairwise difference.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

from cgp.covstats import ar_batch, raw_stats
from cgp.scoring import averaged_residuals, edgelist_to_matrix
# cgp.viz (plotting) is imported lazily inside the functions that plot, so the
# core CGP algorithm can be imported without the plotting dependency `corner`.

# The 11 Markov equivalence classes of 3-node DAGs, represented by one edge
# list per class. Used as the default search space for optimize_graphs in the
# 3-node case.
THREE_NODE_EQUIVALENCE_CLASSES = [
    [],
    [(0, 1)],
    [(1, 2)],
    [(0, 2)],
    [(0, 1), (0, 2)],
    [(0, 2), (2, 1)],
    [(1, 0), (1, 2)],
    [(0, 2), (1, 2)],
    [(1, 0), (2, 0)],
    [(0, 1), (2, 1)],
    [(0, 1), (0, 2), (1, 2)],
]


def generate_linkage_clusters(data, cluster_size, plot=False):
    """Cluster data with hierarchical (average-linkage) clustering."""
    links = linkage(data, method="average")
    indices = fcluster(links, cluster_size, criterion="maxclust")
    if plot:
        nclusters = max(indices)
        cluster_list = [data[indices == (i + 1)] for i in range(nclusters)]
        from cgp.viz import plot_clusters_corner
        plot_clusters_corner(cluster_list)
    return indices


def generate_kmeans_clusters(data, nclusters, plot=False):
    """Cluster data with k-means; returns a list of per-cluster point arrays."""
    kmeans = KMeans(n_clusters=nclusters)
    kmeans.fit(data)
    indices = kmeans.labels_
    cluster_list = [data[indices == i] for i in range(nclusters)]
    if plot:
        from cgp.viz import plot_clusters_corner
        plot_clusters_corner(cluster_list)
    return cluster_list


def _draw_disjoint_subsets(nrows, subset_size, nsubsets, rng):
    """nsubsets independent random samples of subset_size indices, no replacement.

    Vectorized via argsort-of-random-values (one random permutation per row);
    much faster than nsubsets individual np.random.choice calls.
    """
    return np.argsort(rng.random((nsubsets, nrows)), axis=1)[:, :subset_size]


def swing_table(cluster, background, dags, subset_size, nsubsets, rng=None):
    """Swing of `cluster` toward every DAG in `dags`, sharing one batch of
    random subset draws across all of them.

    :param dags: list of adjacency matrices (or edge lists) to score.
    :return: array of shape (len(dags),), the swing toward each dag.
    """
    rng = rng if rng is not None else np.random.default_rng()
    numvars = cluster.shape[1]
    dag_mats = [d if isinstance(d, np.ndarray) else edgelist_to_matrix(d, numvars) for d in dags]

    cn, cs, cQ = raw_stats(cluster)
    idx = _draw_disjoint_subsets(background.shape[0], subset_size, nsubsets, rng)
    pts = background[idx]                                    # (nsubsets, subset_size, d)
    s = pts.sum(axis=1)
    Q = np.matmul(pts.transpose(0, 2, 1), pts)

    swings = np.empty(len(dag_mats))
    for k, dag in enumerate(dag_mats):
        with_cluster = ar_batch(dag, subset_size + cn, s + cs, Q + cQ)
        without = ar_batch(dag, subset_size, s, Q)
        swings[k] = (with_cluster - without).mean()
    return swings


def affinity_table(cluster_list, dags, subset_size, nsubsets, rng=None):
    """Swing of every cluster toward every DAG in `dags`.

    :return: array of shape (len(cluster_list), len(dags)).
    """
    rng = rng if rng is not None else np.random.default_rng()
    numvars = cluster_list[0].shape[1]
    nclusters = len(cluster_list)
    table = np.empty((nclusters, len(dags)))
    for i in range(nclusters):
        rest = np.vstack([cluster_list[j] for j in range(nclusters) if j != i]) if nclusters > 1 \
            else np.empty((0, numvars))
        table[i] = swing_table(cluster_list[i], rest, dags, subset_size, nsubsets, rng)
    return table


def get_affinity(cluster, data, dag1, dag2, subset_size, nsubsets, var_names=None, rng=None):
    """Estimate a single cluster's affinity toward dag1 vs. dag2.

    `data` must already exclude `cluster`'s own points. Returns
    (dag1_swing, dag2_swing); see affinitize_clusters for the combined
    affinity these two feed into.
    """
    swings = swing_table(cluster, data, [dag1, dag2], subset_size, nsubsets, rng=rng)
    return swings[0], swings[1]


def affinitize_clusters(cluster_list, dag1, dag2, subset_size, nsubsets, var_names=None, rng=None):
    """Compute the net affinity (dag1_swing - dag2_swing) of every cluster.

    Negative affinity means the cluster more strongly favors dag1; positive
    means dag2 (swing is optimal at low values, so a cluster that lowers
    dag1's AR more than dag2's AR has affinity < 0).
    """
    table = affinity_table(cluster_list, [dag1, dag2], subset_size, nsubsets, rng=rng)
    return table[:, 0] - table[:, 1]


def probe_graph(dag1, dag2, clusters, plot=True):
    """Compute and optionally display cluster affinities for a single (dag1, dag2) pair."""
    affinities = affinitize_clusters(clusters, dag1, dag2, subset_size=1000, nsubsets=5000)
    if plot:
        from cgp.viz import plot_affinity_colormap
        plot_affinity_colormap(clusters, affinities)
    return affinities


def trichotomize_data(clusters, affinities, threshold):
    """Split clustered points into (neutral, dag1, dag2) subsets by affinity vs. threshold."""
    numvars = clusters[0].shape[1]
    dag1_points = np.empty((0, numvars))
    dag2_points = np.empty((0, numvars))
    neutral_points = np.empty((0, numvars))
    for cluster, affinity in zip(clusters, affinities):
        if abs(affinity) < threshold:
            neutral_points = np.vstack((neutral_points, cluster))
        elif affinity < 0:
            dag1_points = np.vstack((dag1_points, cluster))
        else:
            dag2_points = np.vstack((dag2_points, cluster))
    return neutral_points, dag1_points, dag2_points


def finalize_ar(clusters, affinities, dag1, dag2, threshold, ntrials, var_names=None):
    """Compute the Final AR (FAR) of the partition implied by (dag1, dag2).

    Neutral ("white") points are randomly split in half between the two
    DAGs on each trial, and the resulting FAR is averaged over `ntrials`.
    """
    neutral_points, dag1_points, dag2_points = trichotomize_data(clusters, affinities, threshold)
    nrows = neutral_points.shape[0]
    halfpoint = nrows // 2
    fars = np.zeros(ntrials)
    for i in range(ntrials):
        shuffled = np.random.choice(nrows, size=nrows, replace=False)
        dag1_neutrals = neutral_points[shuffled[:halfpoint]]
        dag2_neutrals = neutral_points[shuffled[halfpoint:]]
        fars[i] = (
            averaged_residuals(dag1, np.vstack((dag1_points, dag1_neutrals)), var_names)
            + averaged_residuals(dag2, np.vstack((dag2_points, dag2_neutrals)), var_names)
        )
    return np.mean(fars)


def optimize_graphs(data, nclusters, subset_size, nsubsets, ntrials,
                     candidate_graphs=None, var_names=None, rng=None, cluster_fn=None):
    """Search the 2-graph space for the (dag1, dag2) pair with the lowest FAR.

    :param candidate_graphs: list of candidate DAGs (edge lists) to pair up;
        defaults to the 11 Markov equivalence classes of 3-node DAGs.
    :param cluster_fn: callable(data, nclusters) -> list of per-cluster point
        arrays; defaults to generate_kmeans_clusters. Swap in an alternative
        (e.g. cgp.feature_clustering.generate_feature_space_clusters) to
        change how points are grouped before affinity is estimated, without
        touching the rest of the search.
    :return: (fars, dag_pairs) where fars[i][j] is the FAR of pairing
        candidate_graphs[i] with candidate_graphs[j] (the diagonal holds
        2 * AR(candidate_graphs[i], data), i.e. the score if the dataset
        were left unpartitioned).
    """
    candidate_graphs = candidate_graphs or THREE_NODE_EQUIVALENCE_CLASSES
    if var_names is None:
        from cgp.viz import generic_var_labels
        var_names = generic_var_labels(data.shape[1])
    rng = rng if rng is not None else np.random.default_rng()
    cluster_fn = cluster_fn or generate_kmeans_clusters
    length = len(candidate_graphs)
    fars = np.full((length, length), np.inf)
    clusters = cluster_fn(data, nclusters)

    # Score every cluster's swing toward every candidate DAG ONCE (sharing
    # subset draws across all of them), rather than redrawing/rescoring per
    # pair -- each pair's affinity below is then just a subtraction.
    table = affinity_table(clusters, candidate_graphs, subset_size, nsubsets, rng)

    for i in range(length):
        for j in range(i):
            dag1, dag2 = candidate_graphs[i], candidate_graphs[j]
            affinities = table[:, i] - table[:, j]
            threshold = min(abs(np.min(affinities)), abs(np.max(affinities))) / 10
            fars[i][j] = finalize_ar(clusters, affinities, dag1, dag2, threshold, ntrials, var_names)

    for i in range(length):
        fars[i][i] = 2 * averaged_residuals(candidate_graphs[i], data, var_names)

    i, j = np.unravel_index(np.argmin(fars), fars.shape)
    return fars, (candidate_graphs[i], candidate_graphs[j])


def evaluate_accuracy(dag1, dag2, data, true_dag1_points, true_dag2_points,
                       nclusters, subset_size, nsubsets, var_names=None):
    """Compare CGP's estimated partition against a known ground-truth partition.

    Requires dag1/dag2 to be listed in the same order as
    true_dag1_points/true_dag2_points. Returns two length-3 arrays
    (correct, neutral, incorrect) counting how many of each DAG's true
    points were assigned to the matching, neutral, or opposite subset.
    """
    clusters = generate_kmeans_clusters(data, nclusters)
    affinities = affinitize_clusters(clusters, dag1, dag2, subset_size, nsubsets, var_names)
    threshold = min(abs(np.min(affinities)), abs(np.max(affinities))) / 10
    neutral_points, estimated_dag1_points, estimated_dag2_points = trichotomize_data(
        clusters, affinities, threshold
    )

    def allocate(true_points, own_estimate, other_estimate):
        counts = np.zeros(3)
        own_list, neutral_list, other_list = own_estimate.tolist(), neutral_points.tolist(), other_estimate.tolist()
        for point in true_points:
            point_list = point.tolist()
            if point_list in own_list:
                counts[0] += 1
            elif point_list in neutral_list:
                counts[1] += 1
            elif point_list in other_list:
                counts[2] += 1
        return counts

    dag1_allocation = allocate(true_dag1_points, estimated_dag1_points, estimated_dag2_points)
    dag2_allocation = allocate(true_dag2_points, estimated_dag2_points, estimated_dag1_points)
    return dag1_allocation, dag2_allocation
