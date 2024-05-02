/* This code has been modified from the original quickshift distribution */
/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>

#include "functions.hpp"
#include "graph.hpp"
#include "progress.hpp"

namespace quickshift {

struct KnnPoint {
    double dist;
    int32_t idx;
};

/**
 * @brief Cluster cores from points.
 *
 * Given the kNN density and neighbors, build the k-NN graph, cluster tree and return the estimated
 * modes. Note that here, we don't require the dimension of the dataset
 *
 * @param dim
 * @param radii
 * @param neighbors
 * @param beta
 * @param epsilon
 * @return VectorXi array of estimated mode membership, where each index cosrresponds the respective
 * index in the density array. Points without membership are assigned -1
 */
VectorXi compute_cores(int dim,
                       const Eigen::Ref<const VectorXd> &radii,
                       const Eigen::Ref<const RowMatrixXi> &neighbors,
                       double beta,
                       double epsilon) {
    int32_t n_points = radii.size();

    if (n_points != neighbors.rows()) {
        throw std::runtime_error("neighbors.rows() != n_points");
    }

    std::vector<KnnPoint> knn_radii(n_points);
    // for (int i = 0; i < n_points; i++){
    for (progress::Progress bar(n_points, 100); auto i : bar.name("populating neighbors")) {
        knn_radii[i].dist = radii[i];
        knn_radii[i].idx = i;
        // knn_radii gets sorted, so the idx wont be maintained
    }
    std::sort(knn_radii.begin(), knn_radii.end(),
              [](const auto &left, const auto &right) { return left.dist < right.dist; });

    std::cout << "Sorting the neighbor idx matrix" << std::endl;
    RowMatrixXi sorted_neigbors(neighbors);
    for (auto row : sorted_neigbors.rowwise()) {
        std::sort(row.begin(), row.end());
    }

    std::vector<int> m_hat(n_points);
    std::vector<int> cluster_membership(n_points);
    int n_chosen_points{};
    int n_chosen_clusters{};

    Graph graph;

    int last_considered{0};
    int last_pruned{0};

    for (progress::Progress bar(n_points, 100); auto i : bar.name("clustering cores")) {
        while (last_pruned < n_points &&
               pow(1. + epsilon, 1. / dim) * knn_radii[i].dist > knn_radii[last_pruned].dist) {
            graph.add_node(knn_radii[last_pruned].idx);

            // eigen sorted neighbors
            for (auto neighbor_idx : neighbors.row(knn_radii[last_pruned].idx)) {
                if (graph.M.contains(neighbor_idx) &&
                    std::binary_search(sorted_neigbors.row(neighbor_idx).begin(),
                                       sorted_neigbors.row(neighbor_idx).end(),
                                       knn_radii[last_pruned].idx)) {
                    graph.add_edge(knn_radii[last_pruned].idx, neighbor_idx);
                }
            }
            last_pruned++;
        }

        while (knn_radii[i].dist * pow(1. - beta, 1. / dim) > knn_radii[last_considered].dist) {
            if (!graph.component_seen(knn_radii[last_considered].idx)) {
                auto res = graph.get_connected_component(knn_radii[last_considered].idx);
                for (const auto &comp_idx : res) {
                    if (radii[comp_idx] <= knn_radii[i].dist) {
                        cluster_membership[n_chosen_points] = n_chosen_clusters;
                        m_hat[n_chosen_points++] = comp_idx;
                    }
                }
                n_chosen_clusters++;
            }
            last_considered++;
        }
    }

    VectorXi result = VectorXi::Constant(n_points, -1);

    for (int i = 0; i < n_chosen_points; ++i) {
        result[m_hat[i]] = cluster_membership[i];
    }
    return result;
}
}  // namespace quickshift
