<style>
r { color: Red }
o { color: Orange }
g { color: Green }
</style>

# Replay buffer
First thing we introduce is the `TrajectoryReplayBuffer`.
This is like a Replay Buffer, i.e. it stores transitions with a FIFO logic when the maximum capacity is reached. But a transition is a whole trajectory and not just a single step.

## `TrajectoryReplayBuffer`
The `TrajectoryReplayBuffer` is basically a list of TensorDicts, where each TensorDict represents a trajectory.

It stores the `Trajectory` class, which is a TensorDict with an additional sample method that allows for slicing the trajectory.

Right now each Trajectory is sampled with the same probability. Later one could add the weights of the trajectories to the `add` method and sample with the weights in the `random.choices` call in the `sample` method.

We add the `TrajectoryReplayBuffer` to the `data/buffer.py` file.

# MCTS
For the MCTS algorithm, we needed to add some extra classes and auxiliary functions like `Node`, `visit_softmax_temperature` and `ucb_score`.

The `Node` class represents a node in the MCTS tree. It stores the prior probability of the action, the visit count, the value sum and the children of the node, which are the actions that can be taken from the current state.
It also has some extra methods like `has_child` and `value`, to quickly check if the node is a leaf node and to get the *target* value of the node, based on its visit_count.

