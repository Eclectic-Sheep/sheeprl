
#include "node.h"

#include <iostream>

#include "torch/extension.h"
#include "torch/torch.h"

namespace Node {
    Node::Node(double prior){
        this->prior = prior;
        this->reward = 0.0;
        this->value_sum = 0.0;
        this->visit_count = 0;
        this->children = std::vector<Node*>();
        this->imagined_action = -1;
    }

    Node::~Node(){
        for (auto & i : this->children){
            delete i;
        }
    }

    int Node::expanded() const{
        int num_children = this->children.size();
        if (num_children > 0){
            return 1;
        }
        else{
            return 0;
        }
    }

    double Node::value() const{
        if (this->visit_count == 0){
            return 0.0;
        }
        else{
            return this->value_sum / this->visit_count;
        }
    }

    void Node::expand(std::vector<double> priors){
        int num_children = priors.size();
        for (int i = 0; i < num_children; i++){
            Node* child = new Node(priors[i]);
            this->children.push_back(child);
        }
    }

    void Node::add_exploration_noise(std::vector<double> noise, double exploration_fraction){
        int num_children = this->children.size();
        if (num_children > 0){
            for (int i = 0; i < num_children; i++){
                this->children[i]->prior = (1 - exploration_fraction) * this->children[i]->prior + exploration_fraction * noise[i];
            }
        }
    }

    Node::Node() = default;
};  // namespace Node