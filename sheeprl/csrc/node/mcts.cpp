#include "mcts.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

namespace mcts {

    MinMaxStats::MinMaxStats() {
        this->minimum = 99999.9;
        this->maximum = -99999.9;
    }

    void MinMaxStats::update(float value) {
        if (value > maximum) {
            this->maximum = value;
        }
        if (value < minimum) {
            this->minimum = value;
        }
    }

    float MinMaxStats::normalize(float value) {
        return (value - this->minimum) / (this->maximum - this->minimum + 1e-8);
    }

    std::vector<float> ucb_score(Node::Node parent, float pbc_base, float pbc_init, float gamma, MinMaxStats stats) {
        std::vector<float> ucb_scores;
        std::cout << "Starting ucb score computation" << std::endl;
        std::cout << "Parent has " << parent.children.size() << " children" << std::endl;
        for (size_t i = 0; i < parent.children.size(); i++) {
            Node::Node child = *parent.children[i];
            float pb_c = std::log((parent.visit_count + pbc_base + 1) / pbc_base) + pbc_init;
            pb_c *= std::sqrt(parent.visit_count / (child.visit_count + 1));

            float prior_score = pb_c * child.prior;
            float value_score = stats.normalize(child.value());
            float score = prior_score + value_score;
            ucb_scores.push_back(score);
        }
        std::cout << "Finished ucb score computation" << std::endl;
        return ucb_scores;
    }

    void backpropagate(std::vector<Node::Node> search_path, float value, float gamma, MinMaxStats stats) {
        for (size_t i = 0; i < search_path.size(); i++) {
            Node::Node visited_node = search_path[i];
            visited_node.value_sum += value;
            visited_node.visit_count += 1;
            stats.update(visited_node.value());
            value = gamma * value + visited_node.reward;
        }
    }

    std::vector<Node::Node> rollout(Node::Node root, float pbc_base, float pbc_init, float gamma, MinMaxStats stats) {
        std::vector<Node::Node> search_path;
        search_path.push_back(root);
        Node::Node node = root;

        while (node.expanded()) {
            std::cout << "Visiting a node with " << node.children.size() << " children" << std::endl;
            std::cout << "Computing ucb scores" << std::endl;
            std::vector<float> ucb_scores = ucb_score(node, pbc_base, pbc_init, gamma, stats);
            std::cout << "Ucb scores have " << ucb_scores.size() << " elements" << std::endl;
            std::cout << "Selecting action" << std::endl;
            auto maxElementIterator = std::max_element(ucb_scores.begin(), ucb_scores.end());
            int action = std::distance(ucb_scores.begin(), maxElementIterator);
            node.imagined_action = action;
            std::cout << "Selected action " << action << std::endl;
            std::cout << "Updating node" << std::endl;
            Node::Node child = *node.children[action];
            std::cout << "Adding child to search path" << std::endl;
            search_path.push_back(child);
            node = child;
        }
        return search_path;
    }
}; // namespace mcts
