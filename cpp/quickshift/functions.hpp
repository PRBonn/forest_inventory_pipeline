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
#include <cstdint>
#include <Eigen/Core>

namespace quickshift {

using Eigen::VectorXd;
using Eigen::VectorXi;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

VectorXi compute_cores(int dim,
                       const Eigen::Ref<const VectorXd> &radii,
                       const Eigen::Ref<const RowMatrixXi> &neighbors,
                       double beta,
                       double epsilon);

VectorXi cluster_remaining(const Eigen::Ref<const RowMatrixXd> &dataset,
                           const Eigen::Ref<const VectorXd> &radii,
                           const Eigen::Ref<const RowMatrixXi> &neighbors,
                           const Eigen::Ref<const VectorXi> &cores);

}  // namespace quickshift
