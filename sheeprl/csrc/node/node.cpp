
#include "node.h"

#include <iostream>

#include "torch/extension.h"
#include "torch/torch.h"

namespace Node {
    Node::Node(float prior){
        this->prior = prior;
        this->reward = 0.0;
        this->value_sum = 0.0;
        this->visit_count = 0;
        this->children = std::vector<Node*>();
        this->imagined_action = -1;
    }

    int Node::expanded(){
	    int num_children = this->children.size();
	    if (num_children > 0){
            return 1;
        }
        else{
            return 0;
        }
    }

    int Node::value(){
        if (this->visit_count == 0){
            return 0;
        }
        else{
            return this->value_sum / this->visit_count;
        }
    }

    void Node::expand(std::vector<float> priors){
        int num_children = priors.size();
        for (int i = 0; i < num_children; i++){
            this->children.push_back(new Node(priors[i]));
        }
    }

    void Node::add_exploration_noise(std::vector<float> noise, float exploration_fraction){
        int num_children = this->children.size();
        if (num_children > 0){
            for (int i = 0; i < num_children; i++){
                this->children[i]->prior = (1 - exploration_fraction) * this->children[i]->prior + exploration_fraction * noise[i];
            }
        }
    }

    torch::Tensor Node::Node::get_hidden_state() { return this->hidden_state; }
    void Node::Node::set_hidden_state(torch::Tensor hidden_state) { this->hidden_state = hidden_state; }
};	// namespace Node