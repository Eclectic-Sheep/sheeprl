<style>
r { color: Red }
o { color: Orange }
g { color: Green }
</style>

# MuZero

## Self-play
The agents need to know the <r>action_space_size</r>. You can set a custom number of <r>num_actors</r>, meaning the number of parallel environments.
An episode terminate if a final state is reached or if the <r>max_moves</r> number of actions is reached
While playing, the agents select actions based on their **mcts** exploration.

### Playing an episode
The action selection is based on the mcts exploration. When the agent start playing, it starts from a root <g>Node</g> with no prior knowledge. The current state of the game is used to create an <o>image</o> of it, that can be fed to the <g>Network</g> to perform <o>initial_inference</o>, meaning the **representation** + **prediction**.
Using the <g>root, (player *for multiplayer games*), legal_actions, and the output of the Network</g>, we <o>expand_node</o> the root. This creates a new <g>Node</g> that is a child of the root, where each `child.prior` is set using the output of the Network's output policy. Meanwhile, the root is updated with the prediction of the Network. Especially: 
  * `node.hidden_state` = **representation**
  * `node.reward` = :warning: problem: the **representation** and **prediction** do not give a reward, **dynamics** does
  * `node.policy` = **prediction**.policy

At this point, we can apply <o>add_exploration_noise</o> to the root, which adds dirichlet noise to the prior of the root's children. Then MCTS starts from the root.

#### MCTS
MCTS explores the possible path (sequences of actions) starting from the root node, in order to decide what action to perform next.
Using <r>known_bounds</r> on the possible value of nodes, it does a <r>num_simulations</r>, where each simulation ends when a leaf node is encountered (leaf node = no children).
In MCTS, the way we <o>select_child</o> as the next node, is by using the <o>ucb_score</o>.
The <o>ucb_score</o> is computed using the <g>parent, child, min_max_stats</g> and the <r>pb_c_base</r> and <r>pb_c_init</r> and <r>discount</r> hyperparameters.

:construction: how to compute UCB score coming soon :construction:

The child with the higher score is selected as the next node in the <g>search_path</g>. When a leaf node is encountered, the <g>Network</g> output is computed using its <g>parent.hidden_state</g> and the <g>history.last_action</g>.
We then <o>expand_node</o> the leaf node, computing its `node.hidden_state`, `node.reward` and `node.policy`. 
The **dyamics** of the <g>Network</g> returns also the value and reward of the leaf node. The value of the <g>Network</g> is used to initialize the <o>backpropagate</o> on the search path, that updates the `node.value_sum` and `node.visit_count` of each node in the search path.

#### Action selection
When the <r>num_simulations</r> are done, the action selection is performed using the <o>select_action</o> function. This function selects the action based on the <g>root</g>'s `child.visit_count` and the <r>temperature</r> hyperparameter. The next action is computed using the <o>softmax_sample</o> on the <g>visit_counts</g> and the updated <r>temperature</r>.

### Storing in the buffer
When a full episode is played or the <r>max_moves</r> is reached, the episode is stored in the <g>ReplayBuffer</g>. Whenever the buffer is full, the oldest episode is removed to not exceed the <r>window_size</r>.
Of an <g>Episode</g>, we save:
  * history: list of the performed actions
  * rewards: list of the rewards collected
  * child_visits: list of the visit counts of the children of the root node
  * root_values: list of the values of the root node

## Training
When all the <r>num_actors</r> have finished playing, we can <o>train_network</o> using the data collected in the buffer.

For a fixed amount of <r>training_steps</r>, we <o>sample_batch</o> <r>batch_size</r> chunks of episodes of length <r>num_unroll_steps</r>. Each chunk can start from a random *position* inside an episode. The batch is made of
  * The <o>state.image</o> of the first state of the chunk (not of the episode!)
  * The `episode.history` of the actions starting from *position* up to *position* + <r>num_unroll_steps</r>
  * The <o>episode.make_targets</o> for each action in the chunk. Targets are:
    - the value, that is the `root_values` discounted by <r>discount<sup>td_steps</sup></r>
    - the reward
    - the `child_visits`

Using this batch, the <o>update_weights</o> is called, where the loss is computed and then the weights of the <g>Network</g> are updated using the <r>learning_rate</r> and the <r>weight_decay</r> hyperparameters.

### Computing the loss
For each chunk in the batch, the <o>network.initial_inference</o> is called, using the sampled image. This gives predicted value, reward, policy_logits and hidden state for the first observation in the chunk. 

The same is done for each time step in the chunk but using the <o>network.recurrent_inference</o>, on the <g>hidden_state</g> freshly computed and on the <g>action</g> sampled. The loss is computed as the sum of the <o>scalar_loss</o> between the predicted and the target value, the <o>scalar_loss</o> between the predicted and the target reward and the <o>cross_entropy</o> between the predicted and target policy. The sum is done also over all the steps of the chunk.
NOTE: each step of the chunk's gradient is weighted differently using the <o>scale_gradient</o>. Also the <g>hidden_state</g> of the first time step is weighted with weight 0.5.

Finally, to the loss a l2 regularization is applied on the weights of the <g>network</g> using the <r>weight_decay</r> hyperparameter.

### Completing the training
When all the chunks in the batch has been processed, the optimization step is performed and one training step is completed. 
When all the training steps are completed, the updated <g>network</g> is sent to the players to start a new episode.