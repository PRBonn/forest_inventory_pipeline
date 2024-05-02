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
#include "graph.hpp"

namespace quickshift {
Node::Ptr Graph::get_root(Node::Ptr node) {
    if (node->parent) {
        node->parent->children.erase(node);
        node->parent = get_root(node->parent);
        node->parent->children.insert(node);
        return node->parent;
    }
    return node;
}

void Graph::add_node(int32_t idx) {
    nodes.emplace_back(std::make_shared<Node>(idx));
    M[idx] = nodes.back();
}

void Graph::add_edge(int32_t node1, int32_t node2) {
    Node::Ptr root1 = get_root(M[node1]);
    Node::Ptr root2 = get_root(M[node2]);
    if (root1 != root2) {
        (root1->rank > root2->rank) ? merge_components(root1, root2)
                                    : merge_components(root2, root1);
    }
}

void Graph::merge_components(const Node::Ptr &parent, const Node::Ptr &child) {
    child->parent = parent;
    parent->children.emplace(child);
    if (intersecting_sets.contains(child)) {
        intersecting_sets.erase(child);
        intersecting_sets.emplace(parent);
    }
    if (parent->rank == child->rank) {
        parent->rank++;
    }
}

std::vector<int32_t> Graph::get_connected_component(int32_t n) {
    auto root = get_root(M[n]);
    std::vector<int32_t> conn_comp;
    std::vector<Node::Ptr> node_stack;
    node_stack.emplace_back(root);
    while (!node_stack.empty()) {
        auto top = node_stack.back();
        node_stack.pop_back();
        conn_comp.emplace_back(top->index);
        for (const auto &node_ptr : top->children) {
            node_stack.emplace_back(node_ptr);
        }
    }
    return conn_comp;
}

bool Graph::component_seen(int32_t n) {
    auto root = get_root(M[n]);
    // insertion succeeds (second is bool) only if the node didnt already exist
    return !intersecting_sets.emplace(root).second;
}

}  // namespace quickshift
