#include "torch/torch.h"
namespace Node {

    class Node {
        public:
            int visit_count;
            double prior;
            double value_sum;
            double reward;
            int imagined_action;
            std::vector<Node*> children;

            Node(double prior); //constructor
//            ~Node(); //destructor

            void set_hidden_state(torch::Tensor hidden_state);
            torch::Tensor get_hidden_state();
            torch::Tensor hidden_state;

            int expanded();
            int value();
            void expand(std::vector<double> action_priors);
            void add_exploration_noise(std::vector<double> noise, double exploration_fraction);
    };

}  // namespace Node