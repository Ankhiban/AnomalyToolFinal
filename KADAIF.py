"""
KADAIF — Kernel-Aware Distance-based Anomaly Isolation Forest

An isolation forest algorithm adapted for microbiome compositional data.
Unlike standard isolation forests that split on raw feature values, KADAIF
uses ecological distance metrics (Bray-Curtis, UniFrac) and dimensionality
reduction (PCoA/PCA) to find meaningful split axes that respect the
compositional structure of microbiome samples.

How it works:
    1. For each tree, a random subset of species (features) is sampled,
       weighted by their overall abundance across locations.
    2. A distance matrix is computed between all locations using the
       sampled species (default: Bray-Curtis dissimilarity).
    3. MDS (PCoA) projects the distance matrix to a low-dimensional space.
    4. A random split point along the first principal coordinate isolates
       a subset of locations into left/right branches.
    5. This recursion continues until each location is alone or max_depth
       is reached. The depth at which a location is isolated is recorded.
    6. After all trees are built, the average isolation depth per location
       is converted to an anomaly score: score = 2^(-avg_depth / c)
       where c is the expected depth for a random sample. Shallow isolation
       (anomalous) → score near 1. Deep isolation (normal) → score near 0.

Supported splitting methods:
    - "pcoa":                    Bray-Curtis distance + MDS (default)
    - "pca":                     PCA on scaled abundance values
    - "unifrac_unweighted_pcoa": Unweighted UniFrac + MDS (requires phylogenetic tree)
    - "unifrac_weighted_pcoa":   Weighted UniFrac + MDS (requires phylogenetic tree)

References:
    Based on the isolation forest concept (Liu et al., 2008) extended for
    microbiome data using ecological distances.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from scipy.spatial.distance import pdist, squareform
from skbio import TreeNode
from skbio.diversity import beta_diversity
from skbio.diversity.beta import unweighted_unifrac, weighted_unifrac

import warnings

# MDS_NUM caps the number of MDS components computed when using multi-component
# methods (pc_method="equal" or "proportion"). Higher values capture more variance
# but increase computation time.
MDS_NUM = 20


class KADAIF:
    """
    Microbiome-aware isolation forest for anomaly detection across ISS locations.

    Builds an ensemble of MicrobiomeIsolationTrees, each of which uses ecological
    distances to recursively partition locations. The anomaly score for each
    location is derived from how quickly it gets isolated (shallow = anomalous).
    """

    def __init__(self, number_of_trees=100, trees=None, min_samples_to_split=2,
                 max_depth=100, weights="proportion", replacement=True,
                 pc_method="first", normalize=True, subsample_size=100,
                 splitting_method="pcoa", paral=True, cpu=None, verbose=True,
                 unifrac_tree=None):
        """
        Args:
            number_of_trees:      Number of isolation trees to build. More trees
                                  give more stable scores but take longer.
            trees:                Pre-built list of trees (used to resume training).
            min_samples_to_split: Stop splitting a node if it has fewer samples.
            max_depth:            Maximum recursion depth per tree.
            weights:              How to weight species during subsampling.
                                  "proportion" = weighted by total abundance (default),
                                  "equal" = all present species equally likely.
            replacement:          Whether to sample species with replacement.
            pc_method:            Which principal coordinate(s) to use for splitting.
                                  "first" = always PC1 (fastest),
                                  "equal" = random PC (uniform),
                                  "proportion" = random PC weighted by variance explained.
            normalize:            If True, normalize subsampled counts to relative
                                  abundance before computing distances.
            subsample_size:       Number of species to subsample per tree node.
            splitting_method:     Distance/projection method for splits. One of:
                                  "pcoa", "pca", "unifrac_unweighted_pcoa",
                                  "unifrac_weighted_pcoa".
            paral:                If True, build trees in parallel using all CPUs.
            cpu:                  Number of CPUs for parallel tree building.
                                  Defaults to all available cores.
            verbose:              Print progress messages during tree building.
            unifrac_tree:         scikit-bio TreeNode required for UniFrac methods.
        """
        self.number_of_trees = number_of_trees
        if trees is None:
            trees = []
        self.trees = trees
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.subsample_size = subsample_size
        self.splitting_method = splitting_method
        self.weights = weights
        self.replacement = replacement
        self.normalize = normalize
        self.paral = paral
        self.cpu = cpu
        self.pc_method = pc_method
        if self.cpu is None:
            self.cpu = cpu_count()

        self.feature_matrix = None
        self.subsample_size_in_each_tree = None
        self.samples_dict_depth = {}  # Accumulates per-tree isolation depths per sample
        self.scores = None
        self.verbose = verbose
        self.unifrac_tree = unifrac_tree

    def fit(self, features_matrix):
        """
        Build all isolation trees on the provided feature matrix.

        Drops species that are zero across all locations before fitting.
        Trees are built in parallel by default using joblib.

        Args:
            features_matrix: DataFrame (locations × species) with organism counts
        """
        self.features_matrix = features_matrix
        # Remove species with zero total abundance — they carry no signal
        self.features_matrix = self.features_matrix.loc[:, self.features_matrix.sum(axis=0) > 0]
        self.subsample_size_in_each_tree = self.features_matrix.shape[0]

        if not self.paral:
            # Sequential build — useful for debugging
            for i in range(len(self.trees), self.number_of_trees):
                if self.verbose:
                    print("Starting tree %i" % i)
                cur_tree = MicrobiomeIsolationTree(
                    min_samples_to_split=self.min_samples_to_split,
                    max_depth=self.max_depth,
                    subsample_size=self.subsample_size,
                    splitting_method=self.splitting_method,
                    pc_method=self.pc_method,
                    weights=self.weights,
                    normalize=self.normalize,
                    replacement=self.replacement,
                    paral=self.paral,
                    cpu=self.cpu,
                    unifrac_tree=self.unifrac_tree
                )
                cur_tree.fit(self.features_matrix)
                self.trees.append(cur_tree)
                if self.verbose:
                    print("Finished tree %i" % i)
        else:
            # Parallel build — each tree is independent so this is embarrassingly parallel
            trees = Parallel(n_jobs=self.cpu)(
                delayed(return_fitted_tree)(
                    data=self.features_matrix,
                    min_samples_to_split=self.min_samples_to_split,
                    max_depth=self.max_depth,
                    subsample_size=self.subsample_size,
                    splitting_method=self.splitting_method,
                    weights=self.weights,
                    normalize=self.normalize,
                    pc_method=self.pc_method,
                    replacement=self.replacement,
                    paral=self.paral,
                    cpu=self.cpu,
                    unifrac_tree=self.unifrac_tree
                ) for _ in range(self.number_of_trees - len(self.trees))
            )
            [self.trees.append(t) for t in trees]

    def score(self):
        """
        Convert per-tree isolation depths into anomaly scores.

        For each sample, collects its isolation depth from every tree, averages them,
        then applies the isolation forest scoring formula:
            score = 2 ^ -(avg_depth / c)

        where c = 2*H(n-1) - 2*(n-1)/n is the expected depth of an isolation tree
        built on n samples (H is the harmonic number). This normalizes scores so
        that 0.5 is expected for a random sample, >0.5 is anomalous, <0.5 is normal.
        """
        harmonic_number = np.sum([1 / i for i in range(1, self.subsample_size_in_each_tree)])
        normalization_const = (
            2 * harmonic_number
            - 2 * (self.subsample_size_in_each_tree - 1) / self.subsample_size_in_each_tree
        )

        for tree in self.trees:
            cur_depths = tree.samples_depth
            for cur_sample, depth in cur_depths.items():
                if cur_sample in self.samples_dict_depth:
                    self.samples_dict_depth[cur_sample].append(depth)
                else:
                    self.samples_dict_depth[cur_sample] = [depth]

        avg_depths = {key: np.mean(value) for key, value in self.samples_dict_depth.items()}
        calc_score = lambda x: 2 ** -(x / normalization_const)
        self.scores = {key: calc_score(value) for key, value in avg_depths.items()}

    def fit_transform(self, features_matrix) -> np.ndarray:
        """
        Fit the model and return anomaly scores in a single call.

        Convenience method that calls fit() then score() and returns scores
        as a numpy array aligned to the input DataFrame's index order.

        Args:
            features_matrix: DataFrame (locations × species) with organism counts

        Returns:
            numpy array of shape (n_locations, 1) with anomaly scores in [0, 1].
            Higher scores indicate more anomalous locations.
        """
        self.fit(features_matrix)
        self.score()
        return np.array(
            pd.DataFrame.from_dict(self.scores, orient='index').loc[self.features_matrix.index]
        )




class MicrobiomeIsolationTree:
    """
    A single isolation tree that recursively partitions ISS locations using
    ecological distances and dimensionality reduction.

    Each node selects a random subset of species, computes a distance matrix
    (Bray-Curtis by default), projects to 1D via MDS, and splits locations at
    a random point along that axis. The depth at which each location is isolated
    is recorded and used by KADAIF to compute anomaly scores.
    """

    def __init__(self, min_samples_to_split=10, max_depth=10, subsample_size=100,
                 splitting_method="pcoa", weights="proportion", pc_method="first",
                 replacement=True, normalize=True, depth=0, left=None, right=None,
                 split_att=None, split_val=None, features_matrix=None,
                 paral=True, cpu=None, parent=None, unifrac_tree=None):
        """
        Args:
            depth:        Current depth of this node in the tree (root = 0).
            left/right:   Child MicrobiomeIsolationTree nodes (set during fit).
            split_att:    The subsampled feature matrix used for the split at this node.
            split_val:    The scalar split point along the first principal coordinate.
            parent:       Reference to the parent node (used for debugging).
            (other args same as KADAIF.__init__)
        """
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.subsample_size = subsample_size
        self.depth = depth
        self.left = left
        self.right = right
        self.split_att = split_att
        self.split_val = split_val
        self.splitting_method = splitting_method
        self.features_matrix = features_matrix
        self.samples_names = None
        self.size = None
        self.samples_depth = {}  # Maps sample name -> depth at which it was isolated
        self.weights = weights
        self.replacement = replacement
        self.normalize = normalize
        self.pc_method = pc_method
        self.paral = paral
        self.cpu = cpu
        if self.cpu is None:
            self.cpu = cpu_count()
        self.parent = parent
        self.unifrac_tree = unifrac_tree

    def fit(self, features_matrix):
        """
        Recursively build this tree node on the given subset of locations.

        Base cases (record current depth for all samples and stop recursing):
          - Fewer samples than min_samples_to_split
          - Reached max_depth

        Otherwise:
          1. Subsample species and compute a split attribute matrix
          2. Retry up to 100 times if all samples are identical (zero distance)
          3. Project to 1D via split_samples() and split at a random point
          4. Recurse on left and right subsets

        Args:
            features_matrix: DataFrame (subset of locations × all species)

        Returns:
            Dict mapping sample name -> isolation depth for all samples in this node
        """
        self.features_matrix = features_matrix
        self.samples_names = list(features_matrix.index)
        self.size = len(self.samples_names)

        # Base case: too few samples or too deep — record depth and stop
        if self.size < self.min_samples_to_split or self.depth >= self.max_depth:
            for sample in self.samples_names:
                self.samples_depth[sample] = self.depth
            return self.samples_depth

        # Sample a subset of species for this node's split
        counter = 0
        self.split_att = subsample(self.features_matrix, self.subsample_size,
                                   weights=self.weights, replacement=self.replacement,
                                   normalize=self.normalize)
        self.split_att.fillna(value=1 / self.split_att.shape[1], inplace=True)

        try:
            # If all samples are identical under this subsample, try a different subsample.
            # This can happen when many species are zero at all locations.
            while (pairwise_distances(self.split_att).max().max() <= 10 ** (-6)
                   or self.split_att.var().max() == 0):
                self.split_att = subsample(self.features_matrix, self.subsample_size,
                                           weights=self.weights, replacement=self.replacement,
                                           normalize=self.normalize)
                self.split_att.fillna(value=1 / self.split_att.shape[1], inplace=True)
                counter += 1
                if counter == 100:
                    raise Microbiome_isolation_forest_error(
                        self, Exception, [self.features_matrix, self.split_att, self.weights]
                    )
        except ValueError:
            print(self.split_att.sum(axis=1))
            raise ValueError
        except Microbiome_isolation_forest_error:
            print(self.features_matrix)
            raise Microbiome_isolation_forest_error
        except Exception as e:
            raise e

        # Project locations to 1D and split at a random point along that axis
        self.split_val, left_samples, right_samples = split_samples(
            self.split_att, method=self.splitting_method,
            pc_method=self.pc_method, unifrac_tree=self.unifrac_tree
        )

        # Create child nodes at depth + 1
        self.left = MicrobiomeIsolationTree(
            min_samples_to_split=self.min_samples_to_split, max_depth=self.max_depth,
            subsample_size=self.subsample_size, splitting_method=self.splitting_method,
            depth=self.depth + 1, weights=self.weights, normalize=self.normalize,
            replacement=self.replacement, paral=self.paral, cpu=self.cpu,
            parent=self, unifrac_tree=self.unifrac_tree
        )
        self.right = MicrobiomeIsolationTree(
            min_samples_to_split=self.min_samples_to_split, max_depth=self.max_depth,
            subsample_size=self.subsample_size, splitting_method=self.splitting_method,
            depth=self.depth + 1, weights=self.weights, normalize=self.normalize,
            replacement=self.replacement, paral=self.paral, cpu=self.cpu,
            parent=self, unifrac_tree=self.unifrac_tree
        )

        if self.paral:
            # Fit both children in parallel
            samples_left_depths, samples_right_depths = Parallel(n_jobs=self.cpu)(
                [delayed(self.left.fit)(input1) for input1 in [self.features_matrix.loc[left_samples]]] +
                [delayed(self.right.fit)(input2) for input2 in [self.features_matrix.loc[right_samples]]]
            )
        else:
            samples_left_depths = self.left.fit(self.features_matrix.loc[left_samples])
            samples_right_depths = self.right.fit(self.features_matrix.loc[right_samples])

        # Merge child depth maps into this node's depth map
        for sample in samples_left_depths:
            self.samples_depth[sample] = self.left.samples_depth[sample]
        for sample in samples_right_depths:
            self.samples_depth[sample] = self.right.samples_depth[sample]

        return self.samples_depth





def split_samples(feature_table, method="pcoa", distance="braycurtis",
                  pc_method="first", unifrac_tree=None):
    """
    Project locations into 1D and choose a random split point.

    Computes an ecological distance matrix between locations, reduces it to
    1D using MDS (PCoA) or PCA, then picks a uniformly random split value
    between the min and max of that 1D projection.

    Args:
        feature_table:  DataFrame (locations × species) — the subsampled matrix
                        for this tree node
        method:         Distance + projection method. One of:
                        "pcoa"                    — Bray-Curtis + MDS (default)
                        "pca"                     — PCA on scaled abundances
                        "unifrac_unweighted_pcoa" — Unweighted UniFrac + MDS
                        "unifrac_weighted_pcoa"   — Weighted UniFrac + MDS
        distance:       Distance metric for pcoa method (default "braycurtis")
        pc_method:      Which principal coordinate to split on:
                        "first"      — always use PC1
                        "equal"      — pick a random PC uniformly
                        "proportion" — pick a random PC weighted by variance explained
        unifrac_tree:   scikit-bio TreeNode (required for UniFrac methods)

    Returns:
        Tuple of (split_value, left_samples, right_samples) where split_value
        is the scalar threshold, and the lists contain location names on each side.
    """
    if method in ["pcoa", "unifrac_unweighted_pcoa", "unifrac_weighted_pcoa"]:
        if method == "pcoa":
            # Standard Bray-Curtis distance matrix between locations
            distance_table = pd.DataFrame(
                pairwise_distances(feature_table, metric=distance),
                index=feature_table.index,
                columns=feature_table.index
            )
        elif method in ["unifrac_unweighted_pcoa", "unifrac_weighted_pcoa"]:
            # UniFrac requires integer counts — scale to smallest nonzero value
            feature_table = feature_table.groupby(feature_table.columns, axis=1).sum()
            feature_table = feature_table / np.min(feature_table[feature_table > 0])
            counts = feature_table.values
            sample_ids = feature_table.index.tolist()
            taxa_ids = feature_table.columns.tolist()

            if method == "unifrac_unweighted_pcoa":
                # Try new skbio API first, fall back to old 'otu_ids' parameter name
                try:
                    unifrac_res = beta_diversity(metric="unweighted_unifrac", counts=counts,
                                                 ids=sample_ids, tree=unifrac_tree, taxa=taxa_ids)
                except ValueError:
                    print("failed due to old skbio version")
                    unifrac_res = beta_diversity(metric="unweighted_unifrac", counts=counts,
                                                 ids=sample_ids, tree=unifrac_tree, otu_ids=taxa_ids)
            elif method == "unifrac_weighted_pcoa":
                try:
                    unifrac_res = beta_diversity(metric="weighted_unifrac", counts=counts,
                                                 ids=sample_ids, tree=unifrac_tree, taxa=taxa_ids)
                except ValueError:
                    print("failed due to old skbio version")
                    unifrac_res = beta_diversity(metric="weighted_unifrac", counts=counts,
                                                 ids=sample_ids, tree=unifrac_tree, otu_ids=taxa_ids)

            distance_table = pd.DataFrame(unifrac_res.data, index=sample_ids, columns=sample_ids)

        if pc_method == "first":
            # Fastest: always project to PC1 only
            mod = MDS(n_components=1, dissimilarity="precomputed")
            mod_first_comp = mod.fit_transform(distance_table)
        else:
            mod = MDS(n_components=np.min([distance_table.shape[0], MDS_NUM]),
                      dissimilarity="precomputed")
            all_comp = mod.fit_transform(distance_table)
            if pc_method == "equal":
                # Pick a random PC with equal probability
                mod_first_comp = all_comp[:, np.random.randint(low=0, high=all_comp.shape[1])]
            elif pc_method == "proportion":
                # Weight PC selection by how much stress it reduces vs. shuffled data.
                # PCs that explain more structure are more likely to be chosen.
                stress_value = []
                feature_table_shuffled = pd.DataFrame(
                    feature_table.apply(lambda row: np.random.permutation(row), axis=1).to_list()
                )
                bray_curtis_matrix_shuffled = squareform(
                    pdist(feature_table_shuffled, metric=distance)
                )
                mod_null = MDS(n_components=1, dissimilarity='precomputed')
                mod_null.fit(bray_curtis_matrix_shuffled)
                stress_value.append(mod_null.stress_)

                for i in range(1, np.min([distance_table.shape[0], MDS_NUM]) + 1):
                    mod_i = MDS(n_components=i, dissimilarity='precomputed')
                    mod_i.fit(distance_table)
                    stress_value.append(mod_i.stress_)

                stress_diffs = [np.max([stress_value[i - 1] - stress_value[i], 0])
                                for i in range(1, len(stress_value))]
                cur_probs = stress_diffs / np.sum(stress_diffs)
                mod_first_comp = all_comp[
                    :, np.random.choice(range(all_comp.shape[1]), p=cur_probs)
                ]

    elif method == "pca":
        # PCA on z-score normalized abundances (no distance matrix needed)
        scaled_feature_table = pd.DataFrame(
            StandardScaler().fit_transform(feature_table),
            index=feature_table.index,
            columns=feature_table.columns
        )
        if pc_method == "first":
            mod = PCA(n_components=1)
            mod_first_comp = mod.fit_transform(scaled_feature_table).T[0]
        elif pc_method == "proportion":
            mod = PCA(n_components=np.min([
                scaled_feature_table.shape[0], MDS_NUM, scaled_feature_table.shape[1]
            ]))
            all_comp = mod.fit_transform(scaled_feature_table)
            cur_probs = mod.explained_variance_ratio_ / mod.explained_variance_ratio_.sum()
            mod_first_comp = all_comp[
                :, np.random.choice(range(all_comp.shape[1]), p=cur_probs)
            ]

    # Random split uniformly between the min and max of the 1D projection
    split_value = np.random.uniform(np.min(mod_first_comp), np.max(mod_first_comp))
    left_samples_list = list(feature_table[mod_first_comp >= split_value].index)
    right_samples_list = list(feature_table[mod_first_comp < split_value].index)
    return split_value, left_samples_list, right_samples_list


def subsample(feature_table, subsample_size, weights="proportion",
              replacement=True, normalize=True):
    """
    Randomly select a subset of species (columns) from the feature table.

    Used at each tree node to introduce randomness, analogous to feature
    subsampling in a random forest.

    Args:
        feature_table:  DataFrame (locations × species)
        subsample_size: Number of species to select. Pass "random" to pick
                        a random size between 1 and the total number of species.
        weights:        Sampling weights for species:
                        "proportion" — weight by total abundance (common species
                                       more likely, better for microbiome data)
                        "equal"      — all present species equally likely
                        "None"       — return full table without subsampling
        replacement:    Whether to sample with replacement (allows duplicates).
                        Sampling without replacement caps subsample_size at the
                        number of present species.
        normalize:      If True, convert counts to relative abundance within
                        each location after subsampling.

    Returns:
        DataFrame with the same rows but only the selected species columns.
        If normalize=True, each row sums to 1.
    """
    if subsample_size == "random":
        subsample_size = np.random.randint(1, feature_table.shape[1])

    if weights == "equal":
        # Equal probability for all species that are present at least once
        weights = list((feature_table.sum(axis=0) > 0) / (feature_table.sum(axis=0) > 0).sum())
    elif weights == "proportion":
        # Probability proportional to total abundance across all locations
        weights = list(feature_table.sum(axis=0) / feature_table.sum(axis=0).sum())
    elif weights == "None":
        return feature_table

    if not replacement:
        # Can't sample more species without replacement than species that exist
        subsample_size = np.min([
            subsample_size,
            np.sum([w != 0 for w in weights])
        ])

    try:
        columns_to_select = np.random.choice(
            feature_table.columns, size=subsample_size, replace=replacement, p=weights
        )
    except ValueError:
        raise ValueError

    cur_feature_table = feature_table[columns_to_select]

    if normalize:
        # Convert to relative abundance — temporarily use integer column indices
        # to handle duplicate column names when sampling with replacement
        real_columns = list(cur_feature_table.columns)
        cur_feature_table.columns = list(range(subsample_size))
        cur_feature_table = cur_feature_table.div(cur_feature_table.sum(axis=1), axis=0)
        cur_feature_table.columns = real_columns

    return cur_feature_table


def return_fitted_tree(data, min_samples_to_split, max_depth, subsample_size,
                       splitting_method, weights, normalize, replacement,
                       pc_method, paral, cpu, unifrac_tree=None):
    """
    Build and return a single fitted MicrobiomeIsolationTree.

    This is a module-level function (not a method) so it can be pickled and
    sent to worker processes by joblib for parallel tree building.

    Args:
        data: Full locations × species DataFrame
        (other args passed directly to MicrobiomeIsolationTree.__init__)

    Returns:
        A fitted MicrobiomeIsolationTree with samples_depth populated
    """
    t = MicrobiomeIsolationTree(
        min_samples_to_split=min_samples_to_split, max_depth=max_depth,
        subsample_size=subsample_size, splitting_method=splitting_method,
        weights=weights, normalize=normalize, replacement=replacement,
        pc_method=pc_method, paral=paral, cpu=cpu, unifrac_tree=unifrac_tree
    )
    t.fit(data)
    return t


class Microbiome_isolation_forest_error(Exception):
    """
    Raised when a tree node fails to find a valid split after 100 attempts.

    This typically occurs when all locations in a node are identical under
    every possible species subsample (e.g. all counts are zero).
    """

    def __init__(self, t, e, features=None):
        """
        Args:
            t:        The MicrobiomeIsolationTree node that failed
            e:        The underlying exception
            features: Debug info — [features_matrix, split_att, weights]
        """
        self.t = t
        self.error = e
        self.features = features