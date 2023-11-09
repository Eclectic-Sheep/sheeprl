#include "node.h"

namespace mcts {

    class MinMaxStats {
        public:
            double minimum;
            double maximum;
            MinMaxStats();
//            ~MinMaxStats();

            void update(double value);
            double normalize(double value);
    };

    std::vector<double> ucb_score(Node::Node* parent, double pbc_base, double pbc_init, double gamma, MinMaxStats &stats);
    void backpropagate(std::vector<Node::Node*> &search_path, std::vector<double> priors, double value, double gamma, MinMaxStats &stats);
    std::vector<Node::Node*> rollout(Node::Node* root, double pbc_base, double pbc_init, double gamma, MinMaxStats &stats);
} // namespace mcts