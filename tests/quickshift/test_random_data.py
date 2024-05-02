import numpy as np
import pytest

from forest_inventory_pipeline.cluster.quickshiftpp import QuickshiftPP


class TestRandomData:
    def test_random(self):
        cluster_len = 100
        cluster1 = np.random.multivariate_normal(
            [0, 0], [[100, 50], [50, 100]], cluster_len
        )
        cluster2 = np.random.multivariate_normal(
            [1000, 1000], [[100, 50], [50, 100]], cluster_len
        )
        data = np.vstack((cluster1, cluster2))

        qpp = QuickshiftPP(20, 0.5)
        cores = qpp.cluster_cores(data)
        res_clusters = qpp.cluster_to_cores(data, cores)
        assert res_clusters[res_clusters == 0].shape[0] == cluster_len
        assert res_clusters[res_clusters == 1].shape[0] == cluster_len
        assert np.all(res_clusters[:cluster_len] == 0)
        assert np.all(res_clusters[cluster_len:] == 1)

    def test_equivalence_of_methods(self):
        # this test seems like it can fail rarely
        cluster_len = 100
        cluster1 = np.random.multivariate_normal(
            [0, 0], [[100, 50], [50, 100]], cluster_len
        )
        cluster2 = np.random.multivariate_normal(
            [1000, 1000], [[100, 50], [50, 100]], cluster_len
        )
        data = np.vstack((cluster1, cluster2))

        qpp = QuickshiftPP(20, 0.5)

        assert np.all(
            qpp.cluster_to_cores(data, qpp.cluster_cores(data)) == qpp.cluster(data)
        )

    def test_different_params(self):
        cluster_len = 100
        cluster1 = np.random.multivariate_normal(
            [0, 0], [[100, 50], [50, 100]], cluster_len
        )
        cluster2 = np.random.multivariate_normal(
            [1000, 1000], [[100, 50], [50, 100]], cluster_len
        )
        data = np.vstack((cluster1, cluster2))

        qpp = QuickshiftPP(20, 0.5)
        cores = qpp.cluster_cores(data)
        diff_cores = qpp.cluster_cores(data, 10, 0.1)
        assert not np.all(cores == diff_cores)
        res_clusters = qpp.cluster_to_cores(data, cores)
        assert not np.all(res_clusters == qpp.cluster_to_cores(data, diff_cores, 5))

    # beta = 1 clusters nothing. make it a test?
    @pytest.mark.parametrize("beta", np.arange(0, 1, 0.1))
    def test_beta_range(self, beta):
        cluster_len = 100
        cluster1 = np.random.multivariate_normal(
            [0, 0], [[100, 50], [50, 100]], cluster_len
        )
        cluster2 = np.random.multivariate_normal(
            [1000, 1000], [[100, 50], [50, 100]], cluster_len
        )
        data = np.vstack((cluster1, cluster2))

        qpp = QuickshiftPP(20, beta)
        cores = qpp.cluster_cores(data)
        # theres at least one core identified
        assert np.any(cores >= 0)
