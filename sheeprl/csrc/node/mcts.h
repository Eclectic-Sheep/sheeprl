#include "node.h"

namespace mcts {

    class MinMaxStats {
        public:
            float minimum;
            float maximum;
            MinMaxStats();
//            ~MinMaxStats();

            void update(float value);
            float normalize(float value);
    };

    std::vector<float> ucb_score(Node::Node parent, float pbc_base, float pbc_init, float gamma, MinMaxStats stats);
    void backpropagate(std::vector<Node::Node> search_path, float value, float gamma, MinMaxStats stats);
    std::vector<Node::Node> rollout(Node::Node root, float pbc_base, float pbc_init, float gamma, MinMaxStats stats);
} // namespace mcts