#include "torch/torch.h"
namespace Node {

    class Node {
        public:
            int visit_count;
            float prior;
            float value_sum;
            float reward;
            int imagined_action;
            std::vector<Node*> children;

            Node(float prior); //constructor
//            ~Node(); //destructor

            void set_hidden_state(torch::Tensor hidden_state);
            torch::Tensor get_hidden_state();
            torch::Tensor hidden_state;

            int expanded();
            int value();
            void expand(std::vector<float> action_priors);
            void add_exploration_noise(std::vector<float> noise, float exploration_fraction);
    };

}  // namespace Node