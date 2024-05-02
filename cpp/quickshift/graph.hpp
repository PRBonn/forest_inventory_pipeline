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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "node.hpp"

namespace quickshift {

struct Graph {
    /*
    Graph struct.
    Allows us to build the graph one node at a time
    */
    std::vector<Node::Ptr> nodes;
    std::unordered_map<int32_t, Node::Ptr> M;
    std::unordered_set<Node::Ptr> intersecting_sets;
    Graph() = default;

    Node::Ptr get_root(Node::Ptr node);
    void add_node(int32_t idx);
    void add_edge(int32_t node1, int32_t node2);
    void merge_components(const Node::Ptr &parent, const Node::Ptr &child);
    std::vector<int32_t> get_connected_component(int32_t n);
    bool component_seen(int32_t n);
};
}  // namespace quickshift
