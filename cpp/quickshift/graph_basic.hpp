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
#pragma once
#include <vector>

#include "node_basic.hpp"

namespace quickshift {
struct GraphBasic {
    /*
    Basic disjoint set data structure.    */
    std::vector<NodeBasic::Ptr> M;

    explicit GraphBasic(int32_t n);
    NodeBasic::Ptr get_root(NodeBasic::Ptr node);
    void add_edge(int32_t node1, int32_t node2);
};
}  // namespace quickshift
