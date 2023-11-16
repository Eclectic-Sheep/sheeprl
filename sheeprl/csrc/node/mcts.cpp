#include "mcts.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>

#

namespace mcts {

    MinMaxStats::MinMaxStats() {
        this->minimum = 100.0;
        this->maximum = -100.0;
    }

    void MinMaxStats::update(double value) {
        if (value > maximum) {
            this->maximum = value;
        }
        if (value < minimum) {
            this->minimum = value;
        }
    }

    double MinMaxStats::normalize(double value) {
        return (value - this->minimum) / (this->maximum - this->minimum + 1e-8);
    }

    std::vector<double> ucb_score(Node::Node* parent, double pbc_base, double pbc_init, double gamma, MinMaxStats &stats) {
        std::vector<double> ucb_scores;
//        //std::cout << "Starting ucb score computation" << std::endl;
//        //std::cout << "Parent has " << parent->children.size() << " children" << std::endl;
        for (size_t i = 0; i < parent->children.size(); i++) {
            Node::Node* child = parent->children[i];
            double pb_c = std::log((parent->visit_count + pbc_base + 1) / pbc_base) + pbc_init;
            pb_c *= std::sqrt(parent->visit_count / (child->visit_count + 1));

            double prior_score = pb_c * child->prior;
            double value_score = stats.normalize(child->value());
            double score = prior_score + value_score;
            ucb_scores.push_back(score);
        }
//        //std::cout << "Finished ucb score computation" << std::endl;
        return ucb_scores;
    }

    void backpropagate(std::vector<Node::Node*> &search_path, std::vector<double> priors, double value, double gamma, MinMaxStats &stats) {
        //std::cout << "Expanding the node" << std::endl;
        Node::Node* visited_node;
        Node::Node* leaf = search_path.back();
        leaf->expand(priors);
        //std::cout << "Starting backpropagation" << std::endl;
        int path_length = search_path.size();
        for (int i = path_length - 1; i >= 0; --i) {
            //std::cout << "Visiting node in position " << i << std::endl;
            visited_node = search_path[i];
            visited_node->value_sum += value;
            visited_node->visit_count += 1;
            stats.update(visited_node->value());

            // Use a separate variable for the updated value
            double updated_value = gamma * value + visited_node->reward;
            value = updated_value;

            //std::cout << "Updated node " << visited_node << std::endl;
        }
    }

    std::vector<Node::Node*> rollout(Node::Node* root, double pbc_base, double pbc_init, double gamma, MinMaxStats &stats) {
        std::vector<Node::Node*> search_path;
        search_path.push_back(root);
        Node::Node* node = root;

        while (node->expanded()) {
            //std::cout << "Visiting a node with " << node->children.size() << " children" << std::endl;
            //std::cout << "Computing ucb scores" << std::endl;
            std::vector<double> ucb_scores = ucb_score(node, pbc_base, pbc_init, gamma, stats);
            //std::cout << "Ucb scores are ";
            for (auto const n : ucb_scores) {
                //std::cout << n << " ";
            }
            //std::cout << "Selecting action" << std::endl;
            auto maxElementIterator = std::max_element(ucb_scores.begin(), ucb_scores.end());
            int action = std::distance(ucb_scores.begin(), maxElementIterator);
            node->imagined_action = action;
            //std::cout << "Selected action " << node->imagined_action << std::endl;
            //std::cout << "Updating node" << std::endl;

            Node::Node* child = node->children[action];
            //std::cout << "Adding child to search path: " << child << std::endl;
            search_path.push_back(child);
            //std::cout << "Search path has " << search_path.size() << " nodes" << std::endl;
            node = child;

            //std::cout << "Search path: ";
            for (auto const n : search_path) {
                //std::cout << n << " ";
            }
            //std::cout << "Is node expanded? " << node->expanded() << std::endl;
        }
        return search_path;
    }
}; // namespace mcts

namespace tester{
    torch::Tensor forward_model(torch::nn::Module* model, torch::Tensor* input) {
        torch::Tensor output = torch::ones({1, 1, 8, 8});
        std::cout << "Model is: " << *model << std::endl;
        return output;
    }
}; // namespace tester
