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
#include "graph_basic.hpp"

namespace quickshift {
GraphBasic::GraphBasic(const int32_t n) {
    M.clear();
    for (int32_t i = 0; i < n; ++i) {
        M.push_back(std::make_shared<NodeBasic>(i));
    }
}

NodeBasic::Ptr GraphBasic::get_root(NodeBasic::Ptr node) {
    if (!node) {
        return nullptr;
    }
    if (!node->parent) {
        return node;
    }
    node->parent = get_root(node->parent);
    return node->parent;
}

void GraphBasic::add_edge(const int32_t node1, const int32_t node2) {
    if (node1 < 0 || node2 < 0) {
        return;
    }
    NodeBasic::Ptr root1 = get_root(M[node1]);
    NodeBasic::Ptr root2 = get_root(M[node2]);
    if (!root1 || !root2) {
        return;
    }
    if (root1 != root2) {
        (root1->rank > root2->rank) ? root2->parent = root1 : root1->parent = root2;
        if (root1->rank == root2->rank) {
            root2->rank++;
        }
    }
}

}  // namespace quickshift
