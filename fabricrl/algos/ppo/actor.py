from tensordict.nn import TensorDictModule, TensorDictSequential


class Actor(TensorDictSequential):
    """Actor model with a policy module."""

    def __init__(self, *tensor_dict_modules: TensorDictModule, policy):
        """Extends the models of the Actor with a specified policy.

        The TensorDictModules in input should add at least a "features" key to the
        input TensorDict.
        """
        super().__init__(
            *tensor_dict_modules,
            TensorDictModule(
                policy, in_keys=["actor_features", "greedy"], out_keys=["actions"]
            )
        )
        self.policy = policy
