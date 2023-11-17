#include "mcts.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

namespace mcts {

    MinMaxStats::MinMaxStats() {
        this->minimum = std::numeric_limits<double>::infinity();
        this->maximum = -std::numeric_limits<double>::infinity();
    }

    void MinMaxStats::update(double value) {
        if (value > maximum) {
            this->maximum = value;
        }
        if (value < minimum) {
            this->minimum = value;
        }
    }

    double MinMaxStats::normalize(double value) const {
        return (value - this->minimum) / (this->maximum - this->minimum + 1e-8);
    }

    std::vector<double> ucb_score(Node::Node* parent, double pbc_base, double pbc_init, double gamma, MinMaxStats &stats) {
        std::vector<double> ucb_scores;

        for (size_t i = 0; i < parent->children.size(); i++) {
            Node::Node* child = parent->children[i];

            double pb_c = std::log((parent->visit_count + pbc_base + 1) / pbc_base) + pbc_init;
            pb_c *= std::sqrt(parent->visit_count / (child->visit_count + 1));

            double prior_score = pb_c * child->prior;
            double value_score = child->reward + gamma * stats.normalize(child->value());
            double score = prior_score + value_score;
            ucb_scores.push_back(score);
        }
        return ucb_scores;
    }

    void backpropagate(std::vector<Node::Node*> &search_path, std::vector<double> priors, double value, double gamma, MinMaxStats &stats) {
        Node::Node* visited_node;
        Node::Node* leaf = search_path.back();
        leaf->expand(std::move(priors));
        int path_length = search_path.size();
        for (int i = path_length - 1; i >= 0; --i) {
            visited_node = search_path[i];
            visited_node->value_sum += value;
            visited_node->visit_count += 1;
            stats.update(visited_node->value());

            // Use a separate variable for the updated value
            double updated_value = gamma * value + visited_node->reward;
            value = updated_value;
        }
    }

    std::vector<Node::Node*> rollout(Node::Node* root, double pbc_base, double pbc_init, double gamma, MinMaxStats &stats) {
        std::vector<Node::Node*> search_path;
        search_path.push_back(root);
        Node::Node* node = root;

        while (node->expanded()) {
            std::vector<double> ucb_scores = ucb_score(node, pbc_base, pbc_init, gamma, stats);
            auto maxElementIterator = std::max_element(ucb_scores.begin(), ucb_scores.end());
            int action = std::distance(ucb_scores.begin(), maxElementIterator);
            node->imagined_action = action;
            Node::Node* child = node->children[action];
            search_path.push_back(child);
            node = child;
        }
        return search_path;
    }
}; // namespace mcts