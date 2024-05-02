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
#include <Eigen/Core>
#include <iostream>
#include <map>

#include "functions.hpp"
#include "graph_basic.hpp"
#include "progress.hpp"

namespace quickshift {

/**
 * @brief cluster all (unclustered) points to cores
 *
 * To cluster The remaining points, for each sample not in a cluster-core, we must find its
 * nearest sample of higher k-NN density
 *
 * @param dataset
 * @param radii
 * @param neighbors
 * @param cores
 * @return VectorXi
 */
VectorXi cluster_remaining(const Eigen::Ref<const RowMatrixXd> &dataset,
                           const Eigen::Ref<const VectorXd> &radii,
                           const Eigen::Ref<const RowMatrixXi> &neighbors,
                           const Eigen::Ref<const VectorXi> &cores) {
    auto n_points = static_cast<int32_t>(dataset.rows());

    if (neighbors.rows() != n_points) {
        throw std::length_error("neighbors.rows()!= n_points");
    }

    std::cout << "clustering all points" << std::endl;

    // Final clusters.
    GraphBasic graphb = GraphBasic(n_points);

    int n_chosen_clusters = cores.maxCoeff() + 1;

    std::vector<std::vector<int>> modal_sets(n_chosen_clusters);

    for (int i = 0; i < n_points; ++i) {
        if (cores[i] >= 0) {
            modal_sets[cores[i]].emplace_back(i);
        }
    }
    for (int c = 0; c < n_chosen_clusters; ++c) {
        for (size_t i = 0; i < modal_sets[c].size() - 1; ++i) {
            graphb.add_edge(modal_sets[c][i], modal_sets[c][i + 1]);
        }
    }
    std::cout << "graph with only core edges built" << std::endl;

    int next = -1;
    for (progress::Progress bar(n_points, 100); const int i : bar.name("building edges")) {
        if (cores[i] >= 0) {
            continue;
        }
        next = -1;
        // scan the neighbors of the current point to find the first neighbor point in a denser
        // region (lesser r_k) than the current point. add an edge from current point to the
        // neighbor, so pointing in direction of increasing density.
        for (int j = 0; j < neighbors.cols(); ++j) {
            if (radii[neighbors(i, j)] < radii[i]) {
                next = neighbors(i, j);
                break;
            }
        }
        if (next < 0) {
            // no other point in the neighborhood has a higher density than the current point.

            // check if there's any point with a higher density than the current point.
            if ((radii.array() < radii[i]).any()) {
                // find distance of all points to current point.
                VectorXd dist = (dataset.rowwise() - dataset.row(i)).rowwise().squaredNorm();
                // this is an extra scan through the entire list. adds
                // to complexity
                for (int j = 0; j < n_points; ++j) {
                    if (radii[j] >= radii[i]) {
                        dist[j] = std::numeric_limits<double>::infinity();
                    }
                }
                Eigen::Index index = -1;
                dist.minCoeff(&index);
                if (index > -1) {
                    next = static_cast<int>(index);
                }
            }
            // for (int j = 0; j < n_points; ++j) {
            //     if (radii[j] >= radii[i]) {
            //         continue;
            //     }
            //     // whatever dt means
            //     // euclidean distance squared?
            //     double dist = (dataset.row(i) - dataset.row(j)).array().square().sum();
            //     if (best_dist > dist) {
            //         best_dist = dist;
            //         next = j;
            //     }
            //     // so add an edge between current point and next point in the list of
            //     // points which is closest and is also denser than current point
            // }
        }
        if (next < 0) {
            // still no point found. causes a silent failure next if not careful and passed as is
            continue;
        }
        graphb.add_edge(i, next);
    }

    VectorXi result = VectorXi::Constant(n_points, -1);

    int n_clusters = 0;
    std::map<int, int> label_mapping;
    for (progress::Progress bar(n_points, 100); const int i : bar.name("assigning clusters")) {
        int label = (graphb.get_root(graphb.M[i]))->index;
        if (label_mapping.contains(label)) {
            result[i] = label_mapping[label];
        } else {
            label_mapping[label] = n_clusters;
            result[i] = n_clusters;
            n_clusters++;
        }
    }
    return result;
}
}  // namespace quickshift
