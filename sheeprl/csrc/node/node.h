#include "torch/torch.h"
namespace Node {

    class Node {
    public:
        Node();
        ~Node();

        int visit_count;
        double prior;
        double value_sum;
        double reward;
        int imagined_action;
        std::vector<Node*> children;

        explicit Node(double prior); //constructor

        void set_hidden_state(torch::Tensor hidden_state);
        torch::Tensor get_hidden_state();
        torch::Tensor hidden_state;

        [[nodiscard]] int expanded() const;
        [[nodiscard]] double value() const;
        void expand(std::vector<double> action_priors);
        void add_exploration_noise(std::vector<double> noise, double exploration_fraction);
    };

}  // namespace Node