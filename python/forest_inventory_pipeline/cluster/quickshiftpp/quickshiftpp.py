# MIT License
#
# Copyright (c) 2024 Meher Malladi, Luca Lobefaro, Tiziano Guadagnino, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors

from ...utils import logger

# quickshift_pybind CMake's install handles installing the so file to this directory
from . import _quickshift_pybind


class QuickshiftPP:
    """
    Quickshift++ implementation.
    Usage: with any 2 axis array of points. Can be n-dimensional points.

    labels = self.cluster_to_cores(data, self.cluster_cores(data))
    or
    labels = self.cluster(data)
    """

    def __init__(self, k, beta, epsilon=0.0, knn_algo="kd_tree", leaf_size=30):
        """

        :param k: The number of neighbors (i.e. the k in k-NN density)
        :param beta: Ranges from 0 to 1. We choose points that have
        kernel density of at least (1 - beta) * F where F is the mode of
        the empirical density of the cluster
        :param epsilon: For pruning. Sets how much deeper in the cluster
        tree to look in order to connect clusters together. Must be at
        least 0.
        :param leaf_size: kdtree leaf node size.
        """
        self.k = k
        self.beta = beta
        self.epsilon = epsilon
        assert knn_algo in ["kd_tree", "ball_tree", "auto"]
        self.knn_algo = knn_algo
        self.leaf_size = leaf_size

    def _fit_and_query_knn(self, data, k, knn_algo, leaf_size):
        logger.trace("fitting kdtree.")
        nbrs = SkNearestNeighbors(
            n_neighbors=k,
            algorithm=knn_algo,
            metric="euclidean",
            leaf_size=leaf_size,
            n_jobs=-1,
        ).fit(data)
        logger.trace("Kdtree fit complete.")
        knn_radius, neighbors = nbrs.kneighbors(data)
        logger.trace("kdtree query complete.")
        return knn_radius, neighbors

    def cluster_cores(self, data, k=None, beta=None, epsilon=None):
        """
        Determines the cores of possible clusters in the data matrix.
        Points not in cores are given label -1.

        :param data: Data matrix. Each row should represent a datapoint in
        euclidean space
        :param k: Optionally overwrite the class instance setting.
        :param beta: Optionally overwrite the class instance setting.
        :param epsilon: Optionally overwrite the class instance setting.
        :return: np array of core labels.
        """
        data = np.array(data)
        n_points, dim = data.shape
        # knn_density = None
        dist, neighbors = self._fit_and_query_knn(data, k or self.k, self.knn_algo, self.leaf_size)
        knn_radius = dist[:, -1].astype(np.float64, order="C", copy=False)
        neighbors = neighbors.astype(np.int32, order="C", copy=False)
        cores = _quickshift_pybind.compute_cores(
            dim,
            knn_radius,
            neighbors,
            float(beta or self.beta),
            float(epsilon or self.epsilon),
        )
        logger.info("computed cores.")
        return cores

    def cluster_to_cores(self, data, cores, k=None):
        """
        Clusters all remaining points to the cores identified by
        cluster_cores().

        :param data: array of points to cluster
        :param cores: array of cores, one dimensional labels
        :param k: Optionally overwrite the class instance setting.
        :return: array of labels per point
        """
        data = np.array(data).astype(np.float64, order="C", copy=False)
        # knn_density = None
        dist, neighbors = self._fit_and_query_knn(data, k or self.k, self.knn_algo, self.leaf_size)
        knn_radius = dist[:, -1].astype(np.float64, order="C", copy=False)
        neighbors = neighbors.astype(np.int32, order="C", copy=False)
        cores = cores.astype(np.int32, order="C", copy=False)

        result = _quickshift_pybind.cluster_remaining(data, knn_radius, neighbors, cores)
        logger.info("All points assigned to cluster cores.")
        return result

    def cluster(self, data: np.ndarray):
        """
        clusters data as per quickshift++ and the settings in the
        initialization. Does both cluster_cores and cluster_to_cores,
        but loses the flexibility of using different points in both
        steps. Which is used for GDQuickshift++.

        :param data: points to cluster
        :type data: np.ndarray
        :return: array of labels of length data.shape[0]
        """
        n_points, dim = data.shape
        # knn_density = None
        dist, neighbors = self._fit_and_query_knn(data, self.k, self.knn_algo, self.leaf_size)
        knn_radius = dist[:, -1].astype(np.float64, order="C", copy=False)
        neighbors = neighbors.astype(np.int32, order="C", copy=False)

        cores = _quickshift_pybind.compute_cores(
            dim,
            knn_radius,
            neighbors,
            self.beta,
            self.epsilon,
        )
        logger.info("computed cores.")
        data = np.array(data).astype(np.float64, order="C", copy=False)
        result = _quickshift_pybind.cluster_remaining(data, knn_radius, neighbors, cores)
        logger.info("All points assigned to cluster cores.")
        return result
