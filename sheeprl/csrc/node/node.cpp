
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

    void Node::expand(std::vector<double> priors){
        int num_children = priors.size();
        for (int i = 0; i < num_children; i++){
            Node* child = new Node(priors[i]);
            this->children.push_back(child);
        }
        //std::cout << std::endl;
        //std::cout << "Added the following children to node " << this << std::endl;
        for (int i = 0; i < num_children; i++){
            //std::cout << this->children[i] << std::endl;
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

    torch::Tensor Node::Node::get_hidden_state() { return this->hidden_state; }
    void Node::Node::set_hidden_state(torch::Tensor hidden_state) { this->hidden_state = hidden_state; }
};	// namespace Node