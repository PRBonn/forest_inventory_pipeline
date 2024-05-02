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
#include <memory>
#include <unordered_set>

namespace quickshift {

struct Node {
    using Ptr = std::shared_ptr<Node>;
    /*
    Node struct for our k-NN or neighborhood Graph
    */
    int index;
    int rank{};
    Ptr parent;
    std::unordered_set<Ptr> children;
    explicit Node(int idx) : index(idx), parent(nullptr) { children.clear(); };
};
}  // namespace quickshift
