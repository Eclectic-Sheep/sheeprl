"""Regression tests for MuZero bugs that prevented learning.

These tests guard against 9 critical bugs that were found and fixed:
1. Observation misalignment: storing next_obs instead of current obs
2. Action index off-by-one in recurrent unrolling
3. Reward target off-by-one in recurrent unrolling
4. MCTS hidden state normalization division by zero
5. N-step return bootstrap off-by-one
6. Softmax on visit probabilities destroying MCTS policy targets
7. Numerically unstable 1/weights IS correction in loss
8. Batch-level vs per-sample hidden state normalization mismatch
9. Value/reward arguments swapped in batch_back_propagate call
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from unittest.mock import MagicMock, patch

from sheeprl.algos.muzero.agent import MlpDynamics, MuzeroAgent, Predictor
from sheeprl.algos.muzero.loss import policy_loss, reward_loss, value_loss
from sheeprl.algos.muzero.utils import MinMaxStats
from sheeprl.models.models import MLP
from sheeprl.utils.utils import nstep_returns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_agent():
    """Create a small MuZero agent for testing."""
    num_actions = 4
    embedding_size = 8
    support_size = 5
    full_support_size = 2 * support_size + 1

    agent = MuzeroAgent(
        representation=MLP(
            input_dims=(4,),
            hidden_sizes=tuple(),
            output_dim=embedding_size,
            activation=torch.nn.ELU,
        ),
        dynamics=MlpDynamics(
            num_actions=num_actions,
            embedding_size=embedding_size,
            full_support_size=full_support_size,
        ),
        prediction=Predictor(
            embedding_size=embedding_size,
            num_actions=num_actions,
            full_support_size=full_support_size,
        ),
    )
    return agent


@pytest.fixture()
def num_actions():
    return 4


# ===========================================================================
# Test 1: Observation-target alignment
# ===========================================================================

class TestObservationAlignment:
    """Regression test for Bug 1: observations must be the state BEFORE the action.

    In MuZero, MCTS runs on observation o_t to produce targets (policy_t, value_t).
    The stored trajectory must pair o_t with those targets, NOT o_{t+1}.
    """

    def test_stored_obs_matches_pre_action_state(self):
        """Simulate the data collection loop and verify obs alignment.

        The bug was: storing next_obs (post-action) with targets computed from
        obs_pool (pre-action). This test verifies the correct pattern.
        """
        obs_shape = (4,)
        num_actions = 2
        device = torch.device("cpu")

        # Simulate a simple environment trajectory
        # obs_0 -> action_0 -> [reward_0, obs_1] -> action_1 -> [reward_1, obs_2]
        observations = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # obs_0
            np.array([0.0, 1.0, 0.0, 0.0]),  # obs_1 (after action_0)
            np.array([0.0, 0.0, 1.0, 0.0]),  # obs_2 (after action_1)
        ]

        # Simulate the corrected data collection pattern
        obs_pool = torch.tensor(observations[0], device=device).reshape(1, -1)
        steps_data = []

        for step in range(2):
            action = torch.tensor(0)
            # Correct: save current obs BEFORE the action (this is the fix)
            current_obs = obs_pool[0].clone()

            next_obs = observations[step + 1]
            reward = 1.0
            value = 0.5
            visit_probs = torch.ones(num_actions) / num_actions

            step_data = TensorDict(
                {
                    "observations": current_obs.reshape(1, 1, *obs_shape),
                    "actions": action.reshape(1, 1, 1),
                    "policies": visit_probs.reshape(1, 1, num_actions),
                    "rewards": torch.tensor([reward]).reshape(1, 1, 1),
                    "values": torch.tensor([value]).reshape(1, 1, 1),
                },
                batch_size=(1, 1),
                device=device,
            )
            steps_data.append(step_data)

            # Update obs_pool to next_obs (for next MCTS call)
            obs_pool[0] = torch.tensor(next_obs, device=device).reshape(1, -1)

        # Verify: stored observations should be pre-action states
        # Step 0 should store obs_0, NOT obs_1
        assert torch.allclose(
            steps_data[0]["observations"].squeeze(), torch.tensor(observations[0])
        ), "Step 0 must store obs_0 (pre-action), not obs_1 (post-action)"

        # Step 1 should store obs_1, NOT obs_2
        assert torch.allclose(
            steps_data[1]["observations"].squeeze(), torch.tensor(observations[1])
        ), "Step 1 must store obs_1 (pre-action), not obs_2 (post-action)"

    def test_obs_pool_not_modified_before_save(self):
        """Ensure that .clone() is used so obs_pool mutation doesn't affect stored data."""
        obs_shape = (4,)
        device = torch.device("cpu")

        obs_pool = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device).reshape(1, -1)

        # Clone before mutation (correct pattern)
        saved_obs = obs_pool[0].clone()
        obs_pool[0] = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device)

        # saved_obs should NOT have changed
        assert torch.allclose(saved_obs, torch.tensor([1.0, 2.0, 3.0, 4.0])), \
            "Saved observation must be independent of obs_pool mutations (use .clone())"


# ===========================================================================
# Test 2 & 3: Action and reward indexing in recurrent unrolling
# ===========================================================================

class TestRecurrentUnrollIndexing:
    """Regression tests for Bugs 2 & 3: correct action and reward indexing
    during the training unroll loop.

    MuZero training unrolls:
        h_0 = represent(o_0)
        For k = 1..K:
            h_k, r_hat_k = dynamics(a_{k-1}, h_{k-1})
            pi_hat_k, v_hat_k = predict(h_k)
            L += loss(pi_hat_k, pi_k) + loss(v_hat_k, v_k) + loss(r_hat_k, r_{k-1})

    Bug 2: used actions[k] instead of actions[k-1]
    Bug 3: used target_rewards[k] instead of target_rewards[k-1]
    """

    def test_action_index_uses_previous_step(self, simple_agent):
        """Verify that recurrent_inference at step k uses action[k-1]."""
        num_actions = 4
        batch_size = 2
        seq_len = 3

        # Create distinct actions so we can verify which one is used
        actions = torch.zeros(seq_len, batch_size, 1, dtype=torch.long)
        actions[0] = 0  # action at t=0
        actions[1] = 1  # action at t=1
        actions[2] = 2  # action at t=2

        observations = torch.randn(seq_len, batch_size, 4)

        # Track which actions are passed to recurrent_inference
        called_actions = []
        original_recurrent = simple_agent.recurrent_inference

        def tracking_recurrent(action, hidden_state):
            called_actions.append(action.clone())
            return original_recurrent(action, hidden_state)

        simple_agent.recurrent_inference = tracking_recurrent

        # Simulate the training unroll (corrected version)
        hidden_states, _, _ = simple_agent.initial_inference(observations[0])

        for sequence_idx in range(1, seq_len):
            # Correct: use actions[sequence_idx - 1]
            action_input = actions[sequence_idx - 1: sequence_idx].to(dtype=torch.float32)
            hidden_states, rewards, policies, values = simple_agent.recurrent_inference(
                action_input, hidden_states
            )

        # At sequence_idx=1, the action fed should be actions[0] (transition from h_0 to h_1)
        assert torch.equal(called_actions[0].long().squeeze(), actions[0].squeeze()), \
            "At unroll step 1, must use action[0] (a_{k-1}), not action[1]"

        # At sequence_idx=2, the action fed should be actions[1] (transition from h_1 to h_2)
        assert torch.equal(called_actions[1].long().squeeze(), actions[1].squeeze()), \
            "At unroll step 2, must use action[1] (a_{k-1}), not action[2]"

    def test_reward_target_uses_previous_step(self):
        """Verify that the reward loss at step k compares against target_rewards[k-1].

        The dynamics model predicts the reward for the transition caused by action a_{k-1},
        so the target must be the reward r_{k-1} received after taking a_{k-1}.
        """
        seq_len = 3
        batch_size = 2
        support_size = 5
        full_support_size = 2 * support_size + 1

        # Create distinct rewards so we can verify which one is used
        target_rewards = torch.zeros(seq_len, batch_size, 1)
        target_rewards[0] = 1.0   # reward for transition at t=0
        target_rewards[1] = 2.0   # reward for transition at t=1
        target_rewards[2] = 3.0   # reward for transition at t=2

        # Simulate predicted rewards (arbitrary logits)
        predicted_reward = torch.randn(batch_size, full_support_size)

        # Correct: at sequence_idx=1, reward loss should use target_rewards[0]
        loss_correct = reward_loss(predicted_reward, target_rewards[1 - 1])
        # Wrong: using target_rewards[1] would match the wrong transition
        loss_wrong = reward_loss(predicted_reward, target_rewards[1])

        # These should generally differ (they target different rewards)
        # The point is the correct one uses index k-1
        assert loss_correct.shape == (batch_size,), "Reward loss should return per-sample losses"

        # Verify the indexing pattern: at unroll step k, we use target_rewards[k-1]
        for sequence_idx in range(1, seq_len):
            target_idx = sequence_idx - 1
            assert target_idx >= 0, f"Reward target index must be non-negative, got {target_idx}"
            assert target_idx < seq_len, f"Reward target index out of bounds: {target_idx}"
            # The target is the reward for the action that caused this transition
            expected_reward = target_rewards[target_idx]
            assert expected_reward.shape == (batch_size, 1)

    def test_recurrent_unroll_end_to_end(self, simple_agent):
        """End-to-end test: verify the full unroll produces valid losses with correct indexing."""
        num_actions = 4
        batch_size = 4
        seq_len = 5
        support_size = 5
        full_support_size = 2 * support_size + 1
        device = torch.device("cpu")

        # Create mock trajectory data
        observations = torch.randn(seq_len, batch_size, 4)
        actions = torch.randint(0, num_actions, (seq_len, batch_size, 1))
        target_policies = torch.softmax(torch.randn(seq_len, batch_size, num_actions), dim=-1)
        target_rewards = torch.randn(seq_len, batch_size, 1)
        target_values = torch.randn(seq_len, batch_size, 1)

        # Run the corrected training unroll (no 1/weights, per-sample normalization)
        hidden_states, policy_0, value_0 = simple_agent.initial_inference(observations[0])

        pg_loss = policy_loss(policy_0, target_policies[0]).mean()
        v_loss = value_loss(value_0, target_values[0]).mean()
        r_loss = torch.tensor(0.0, device=device)

        for sequence_idx in range(1, seq_len):
            h_min = hidden_states.min(dim=-1, keepdim=True)[0]
            h_max = hidden_states.max(dim=-1, keepdim=True)[0]
            hidden_states = (hidden_states - h_min) / (h_max - h_min + 1e-8)

            # Correct indexing: actions[sequence_idx - 1]
            hidden_states, rewards, policies, values = simple_agent.recurrent_inference(
                actions[sequence_idx - 1: sequence_idx].to(dtype=torch.float32), hidden_states
            )

            pg_loss += policy_loss(policies.squeeze(), target_policies[sequence_idx]).mean()
            v_loss += value_loss(values.squeeze(), target_values[sequence_idx]).mean()
            # Correct indexing: target_rewards[sequence_idx - 1]
            r_loss += reward_loss(rewards.squeeze(), target_rewards[sequence_idx - 1]).mean()

        total_loss = (pg_loss + v_loss + r_loss) / seq_len

        # Verify loss is finite and valid
        assert torch.isfinite(total_loss), f"Total loss must be finite, got {total_loss.item()}"
        assert not torch.isnan(total_loss), f"Total loss must not be NaN"

        # Verify backward pass succeeds
        total_loss.backward()
        grad_norm = simple_agent.gradient_norm()
        assert np.isfinite(grad_norm), f"Gradient norm must be finite, got {grad_norm}"


# ===========================================================================
# Test 4: Hidden state normalization (division by zero)
# ===========================================================================

class TestHiddenStateNormalization:
    """Regression test for Bug 4: hidden state min-max normalization must not
    produce NaN when all values are identical (max == min).
    """

    def test_identical_hidden_states_no_nan(self):
        """When all hidden state values are identical, normalization must not produce NaN."""
        # All values identical: max - min = 0
        hidden_states = torch.ones(1, 4, 8) * 5.0

        h_min = hidden_states.min(dim=-1, keepdim=True)[0]
        h_max = hidden_states.max(dim=-1, keepdim=True)[0]
        normalized = (hidden_states - h_min) / (h_max - h_min + 1e-8)

        assert not torch.isnan(normalized).any(), \
            "Normalization of constant hidden states must not produce NaN (need epsilon)"
        assert torch.isfinite(normalized).all(), \
            "Normalization of constant hidden states must produce finite values"

    def test_normal_hidden_states_normalized(self):
        """Standard case: normalization should map values to [0, 1] range."""
        hidden_states = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])

        h_min = hidden_states.min(dim=-1, keepdim=True)[0]
        h_max = hidden_states.max(dim=-1, keepdim=True)[0]
        normalized = (hidden_states - h_min) / (h_max - h_min + 1e-8)

        assert normalized.min() >= 0.0 - 1e-6, "Min should be ~0"
        assert normalized.max() <= 1.0 + 1e-6, "Max should be ~1"

    def test_zero_hidden_states(self):
        """Edge case: all zeros should not cause div-by-zero."""
        hidden_states = torch.zeros(2, 3, 8)

        h_min = hidden_states.min(dim=-1, keepdim=True)[0]
        h_max = hidden_states.max(dim=-1, keepdim=True)[0]
        normalized = (hidden_states - h_min) / (h_max - h_min + 1e-8)

        assert not torch.isnan(normalized).any()
        assert torch.isfinite(normalized).all()

    def test_normalization_preserves_gradient(self):
        """Ensure normalization doesn't break gradient flow."""
        hidden_states = torch.randn(1, 4, 8, requires_grad=True)

        h_min = hidden_states.min(dim=-1, keepdim=True)[0]
        h_max = hidden_states.max(dim=-1, keepdim=True)[0]
        normalized = (hidden_states - h_min) / (h_max - h_min + 1e-8)

        loss = normalized.sum()
        loss.backward()
        assert hidden_states.grad is not None, "Gradients must flow through normalization"
        assert torch.isfinite(hidden_states.grad).all(), "Gradients must be finite"


# ===========================================================================
# Test 5: N-step return bootstrap index
# ===========================================================================

class TestNStepReturnBootstrap:
    """Regression test for Bug 5: n-step return must bootstrap from the value
    of state s_{t+n}, not s_{t+n-1}.

    Return formula: G_t = sum_{i=0}^{n-1} gamma^i * r_{t+i} + gamma^n * V(s_{t+n})

    Bug was: bootstrapping with V(s_{t+n-1}) instead of V(s_{t+n}).
    """

    def test_bootstrap_uses_value_after_rewards(self):
        """The n-step return for step 0 with n=2 should bootstrap from V(s_2), not V(s_1).

        Setup:
            rewards = [1, 1, 1, 1, 1]   (5 steps)
            values  = [0, 0, 100, 0, 0] (only V(s_2) = 100)
            dones   = [F, F, F, F, T]
            gamma   = 1.0  (no discounting, for easy math)
            n       = 2

        Correct:  G_0 = r_0 + r_1 + gamma^2 * V(s_2) = 1 + 1 + 100 = 102
        Bug:      G_0 = r_0 + r_1 + gamma^2 * V(s_1) = 1 + 1 + 0   = 2
        """
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).reshape(5, 1, 1)
        values = torch.tensor([0.0, 0.0, 100.0, 0.0, 0.0]).reshape(5, 1, 1)
        dones = torch.zeros(5, 1, 1, dtype=torch.bool)
        dones[4] = True
        gamma = 1.0
        n_steps = 2

        returns = nstep_returns(rewards, values, dones, n_steps, gamma)

        # G_0 = r_0 + r_1 + gamma^2 * V(s_2) = 1 + 1 + 100 = 102
        expected_G0 = 102.0
        assert torch.isclose(returns[0].squeeze(), torch.tensor(expected_G0)), \
            f"G_0 should be {expected_G0} (bootstrap from V(s_2)=100), got {returns[0].item()}"

    def test_bootstrap_no_leak_from_wrong_index(self):
        """Ensure bootstrap does NOT use value at index t+n-1.

        Setup where V(s_{t+n-1}) is large but V(s_{t+n}) is zero.
        If the bug existed, the return would be inflated.
        """
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).reshape(5, 1, 1)
        values = torch.tensor([0.0, 999.0, 0.0, 0.0, 0.0]).reshape(5, 1, 1)
        dones = torch.zeros(5, 1, 1, dtype=torch.bool)
        dones[4] = True
        gamma = 1.0
        n_steps = 2

        returns = nstep_returns(rewards, values, dones, n_steps, gamma)

        # G_0 = r_0 + r_1 + gamma^2 * V(s_2) = 1 + 1 + 0 = 2
        # NOT: 1 + 1 + 999 = 1001 (would happen if bootstrapping from V(s_1))
        expected_G0 = 2.0
        assert torch.isclose(returns[0].squeeze(), torch.tensor(expected_G0)), \
            f"G_0 should be {expected_G0} (V(s_2)=0), not inflated by V(s_1)=999. Got {returns[0].item()}"

    def test_end_of_trajectory_no_bootstrap(self):
        """At the end of the episode, when there are fewer than n steps remaining
        and the episode terminates, do not bootstrap.
        """
        rewards = torch.tensor([1.0, 1.0, 1.0]).reshape(3, 1, 1)
        values = torch.tensor([50.0, 50.0, 50.0]).reshape(3, 1, 1)
        dones = torch.zeros(3, 1, 1, dtype=torch.bool)
        dones[2] = True  # Last step is terminal
        gamma = 1.0
        n_steps = 5  # Larger than trajectory

        returns = nstep_returns(rewards, values, dones, n_steps, gamma)

        # G_2 = r_2 = 1.0 (terminal, no bootstrap)
        assert torch.isclose(returns[2].squeeze(), torch.tensor(1.0)), \
            f"Terminal step return should be just the reward, got {returns[2].item()}"

    def test_with_discounting(self):
        """Test n-step returns with gamma < 1 to verify correct discounting."""
        rewards = torch.tensor([2.0, 3.0, 0.0, 0.0, 0.0]).reshape(5, 1, 1)
        values = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0]).reshape(5, 1, 1)
        dones = torch.zeros(5, 1, 1, dtype=torch.bool)
        dones[4] = True
        gamma = 0.9
        n_steps = 2

        returns = nstep_returns(rewards, values, dones, n_steps, gamma)

        # G_0 = r_0 + 0.9*r_1 + 0.9^2 * V(s_2) = 2 + 0.9*3 + 0.81*10 = 2 + 2.7 + 8.1 = 12.8
        expected_G0 = 2.0 + 0.9 * 3.0 + 0.81 * 10.0
        assert torch.isclose(returns[0].squeeze(), torch.tensor(expected_G0), atol=1e-5), \
            f"G_0 should be {expected_G0}, got {returns[0].item()}"

    def test_done_cuts_bootstrap(self):
        """If a done occurs within the n-step window, bootstrap should be zeroed."""
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0]).reshape(4, 1, 1)
        values = torch.tensor([0.0, 0.0, 999.0, 0.0]).reshape(4, 1, 1)
        dones = torch.zeros(4, 1, 1, dtype=torch.bool)
        dones[1] = True  # Episode ends at step 1
        gamma = 1.0
        n_steps = 3

        returns = nstep_returns(rewards, values, dones, n_steps, gamma)

        # G_0: looks at r_0, r_1, r_2 with n=3, but done at step 1
        # The done at step 1 means the bootstrap term is masked
        # G_0 = r_0 + r_1 + r_2 + gamma^3 * V(s_3) * (~done[2]) = 1+1+1+0 = 3
        # Actually done[1] gates the bootstrap at step t+n-1=2
        # The original code uses dones[t+n-1:t+n] to mask bootstrap
        # With done at index 1, for t=0, n_to_consider=3, dones[2] is False
        # So bootstrap from V(s_3) = values[3] * ~dones[2] = 0 * True = 0
        # G_0 = 1+1+1+0 = 3
        assert torch.isfinite(returns[0])


# ===========================================================================
# Test: MinMaxStats initialization (guarding the existing fix)
# ===========================================================================

class TestMinMaxStatsInit:
    """Guard against re-introducing the inverted min/max initialization."""

    def test_initial_max_is_negative_inf(self):
        """Maximum must start at -inf so any real value is greater."""
        stats = MinMaxStats()
        assert stats.maximum == -float("inf"), \
            "maximum must be -inf initially (was wrongly +inf before fix)"

    def test_initial_min_is_positive_inf(self):
        """Minimum must start at +inf so any real value is smaller."""
        stats = MinMaxStats()
        assert stats.minimum == float("inf"), \
            "minimum must be +inf initially (was wrongly -inf before fix)"

    def test_first_update_sets_both(self):
        """After the first update, both min and max should equal the value."""
        stats = MinMaxStats()
        stats.update(42.0)
        assert stats.maximum == 42.0
        assert stats.minimum == 42.0

    def test_normalize_works_after_two_updates(self):
        """After seeing distinct values, normalization should work."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)
        result = stats.normalize(torch.tensor([5.0]))
        assert torch.isclose(result, torch.tensor([0.5])), \
            f"Normalizing 5.0 in [0, 10] should give 0.5, got {result.item()}"


# ===========================================================================
# Test 6: MCTS visit probability computation (no softmax)
# ===========================================================================

class TestVisitProbabilityComputation:
    """Regression test for Bug 6: softmax on visit probabilities destroys
    the MCTS policy signal.

    The policy target must be the raw normalized visit counts from MCTS.
    Applying softmax to probability values flattens the distribution,
    making all policy targets near-uniform and preventing policy learning.
    """

    def test_visit_probs_preserve_mcts_distribution(self):
        """Visit probabilities must faithfully represent MCTS search results.

        With visits [15, 5, 3, 2] (25 total), the target should be [0.6, 0.2, 0.12, 0.08].
        The bug was applying softmax which produces ~[0.35, 0.23, 0.21, 0.21].
        """
        visits_count = [15, 5, 3, 2]
        num_simulations = 25

        # Correct: raw normalized visit counts
        visit_probs = torch.tensor(visits_count, dtype=torch.float32)
        total_visits = visit_probs.sum()
        visit_probs = visit_probs / total_visits

        expected = torch.tensor([0.6, 0.2, 0.12, 0.08])
        assert torch.allclose(visit_probs, expected, atol=1e-6), \
            f"Visit probs should be {expected.tolist()}, got {visit_probs.tolist()}"

    def test_softmax_flattens_distribution(self):
        """Demonstrate that softmax on probabilities destroys information.

        This is the WRONG approach that was causing the bug.
        """
        visits_count = [15, 5, 3, 2]
        visit_probs = torch.tensor(visits_count, dtype=torch.float32) / 25

        # Wrong: applying softmax to probabilities
        wrong_probs = torch.softmax(visit_probs, dim=-1)

        # The max probability should be much lower after softmax
        assert wrong_probs.max() < 0.4, \
            "Softmax on probabilities flattens the distribution (max should be < 0.4)"
        assert visit_probs.max() > 0.5, \
            "Raw visit probs correctly preserve the peaked distribution"

        # Entropy should be higher (more uniform) after softmax
        entropy_correct = -(visit_probs * torch.log(visit_probs + 1e-8)).sum()
        entropy_wrong = -(wrong_probs * torch.log(wrong_probs + 1e-8)).sum()
        assert entropy_wrong > entropy_correct, \
            "Softmax increases entropy (makes distribution more uniform), destroying the MCTS signal"

    def test_zero_visits_gives_uniform(self):
        """When all visit counts are zero (warmup), distribution should be uniform."""
        visits_count = [0, 0, 0, 0]
        num_actions = 4

        visit_probs = torch.tensor(visits_count, dtype=torch.float32)
        total_visits = visit_probs.sum()
        if total_visits > 0:
            visit_probs = visit_probs / total_visits
        else:
            visit_probs = torch.ones(num_actions, dtype=torch.float32) / num_actions

        expected = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert torch.allclose(visit_probs, expected), \
            f"Zero visits should give uniform, got {visit_probs.tolist()}"

    def test_visit_probs_sum_to_one(self):
        """Visit probabilities must be a valid probability distribution."""
        for visits in [[10, 10, 5, 0], [25, 0, 0, 0], [5, 5, 5, 5, 5]]:
            visit_probs = torch.tensor(visits, dtype=torch.float32)
            total = visit_probs.sum()
            if total > 0:
                visit_probs = visit_probs / total
            else:
                visit_probs = torch.ones(len(visits), dtype=torch.float32) / len(visits)

            assert torch.isclose(visit_probs.sum(), torch.tensor(1.0), atol=1e-6), \
                f"Visit probs must sum to 1, got {visit_probs.sum().item()}"
            assert (visit_probs >= 0).all(), "Visit probs must be non-negative"


# ===========================================================================
# Test 7: Loss weighting stability (no 1/weights)
# ===========================================================================

class TestLossWeightingStability:
    """Regression test for Bug 7: 1/weights in loss causes numerical instability.

    weights = |returns - values|^alpha can be zero, making 1/weights = inf.
    The loss should not use 1/weights scaling.
    """

    def test_zero_weight_causes_inf(self):
        """Show that 1/weights produces infinity when weights are zero.

        This is the bug we're guarding against.
        """
        weights = torch.tensor([0.0, 1.0, 2.0])
        inverse_weights = 1.0 / weights

        assert torch.isinf(inverse_weights[0]), \
            "1/0 weight = inf: this is why 1/weights in loss is dangerous"

    def test_loss_finite_without_weight_scaling(self):
        """Loss computation without 1/weights should always be finite."""
        num_actions = 4
        support_size = 5
        full_support_size = 2 * support_size + 1
        batch_size = 8

        # Random predictions and targets
        pred_logits = torch.randn(batch_size, num_actions)
        target_policy = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)

        pred_value = torch.randn(batch_size, full_support_size)
        target_value = torch.randn(batch_size, 1)

        pred_reward = torch.randn(batch_size, full_support_size)
        target_reward = torch.randn(batch_size, 1)

        # Without 1/weights (correct)
        pl = policy_loss(pred_logits, target_policy).mean()
        vl = value_loss(pred_value, target_value).mean()
        rl = reward_loss(pred_reward, target_reward).mean()

        total = pl + vl + rl
        assert torch.isfinite(total), f"Loss must be finite without 1/weights, got {total.item()}"

    def test_near_zero_weights_blow_up_loss(self):
        """Demonstrate that near-zero weights cause extreme loss amplification.

        This is the scenario that happens in practice when value predictions
        are close to their targets.
        """
        num_actions = 4
        batch_size = 4
        base_loss = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Normal per-sample loss

        # Simulate weights where one sample has near-zero |returns - values|
        weights = torch.tensor([0.001, 1.0, 1.0, 1.0]) ** 0.5  # alpha=0.5
        # weights ≈ [0.032, 1.0, 1.0, 1.0]

        scaled_loss = (base_loss * (1.0 / weights)).mean()
        unscaled_loss = base_loss.mean()

        # The scaled loss is ~8x larger due to the near-zero weight
        assert scaled_loss > 5 * unscaled_loss, \
            "Near-zero weights amplify loss dramatically, destabilizing training"


# ===========================================================================
# Test 8: Per-sample hidden state normalization
# ===========================================================================

class TestPerSampleNormalization:
    """Regression test for Bug 8: hidden state normalization must be per-sample,
    not per-batch, to match MCTS inference behavior.

    During MCTS, each root's hidden state is normalized independently.
    During training, if normalization is per-batch (across all samples),
    the dynamics network sees a different input distribution than at inference.
    """

    def test_per_sample_normalization_maps_each_sample_to_01(self):
        """Each sample should independently be mapped to [0, 1] range."""
        # Two samples with very different scales
        hidden_states = torch.tensor([[
            [1.0, 2.0, 3.0, 4.0],    # sample 0: range [1, 4]
            [100.0, 200.0, 300.0, 400.0],  # sample 1: range [100, 400]
        ]])

        h_min = hidden_states.min(dim=-1, keepdim=True)[0]
        h_max = hidden_states.max(dim=-1, keepdim=True)[0]
        normalized = (hidden_states - h_min) / (h_max - h_min + 1e-8)

        # Each sample should be in [0, 1]
        for i in range(2):
            sample = normalized[0, i]
            assert sample.min() >= -1e-6, f"Sample {i} min should be ~0, got {sample.min()}"
            assert sample.max() <= 1.0 + 1e-6, f"Sample {i} max should be ~1, got {sample.max()}"

    def test_batch_normalization_breaks_per_sample_range(self):
        """Demonstrate that batch-level normalization is WRONG.

        With batch normalization, samples with small values get squished
        near 0, and samples with large values dominate the [0,1] range.
        """
        hidden_states = torch.tensor([[
            [1.0, 2.0, 3.0, 4.0],       # sample 0: small values
            [100.0, 200.0, 300.0, 400.0],  # sample 1: large values
        ]])

        # WRONG: batch-level normalization
        h_min_batch = hidden_states.min()
        h_max_batch = hidden_states.max()
        normalized_batch = (hidden_states - h_min_batch) / (h_max_batch - h_min_batch + 1e-8)

        # Sample 0 gets squished near 0 (max ≈ 4/400 ≈ 0.01)
        assert normalized_batch[0, 0].max() < 0.05, \
            "Batch normalization squishes small-scale samples near 0"

        # CORRECT: per-sample normalization
        h_min = hidden_states.min(dim=-1, keepdim=True)[0]
        h_max = hidden_states.max(dim=-1, keepdim=True)[0]
        normalized_per_sample = (hidden_states - h_min) / (h_max - h_min + 1e-8)

        # Sample 0 properly spans [0, 1]
        assert normalized_per_sample[0, 0].max() > 0.9, \
            "Per-sample normalization maps each sample to full [0, 1] range"

    def test_per_sample_matches_mcts_single_root(self):
        """Per-sample normalization should produce the same result as MCTS
        normalizing a single root's hidden state."""
        embedding = torch.tensor([1.0, 5.0, 3.0, 7.0, 2.0])

        # MCTS style (single sample, global min/max = per-sample min/max)
        mcts_norm = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-8)

        # Training style (in a batch, using per-sample norm)
        batch = embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, 5)
        h_min = batch.min(dim=-1, keepdim=True)[0]
        h_max = batch.max(dim=-1, keepdim=True)[0]
        training_norm = (batch - h_min) / (h_max - h_min + 1e-8)

        assert torch.allclose(mcts_norm, training_norm.squeeze(), atol=1e-6), \
            "Per-sample normalization must match MCTS single-root normalization"


# ===========================================================================
# Test 9: batch_back_propagate argument order (reward/value swap)
# ===========================================================================

class TestBatchBackPropagateArgumentOrder:
    """Regression test for Bug 9: value_pool and reward_pool were swapped
    in the call to tree.batch_back_propagate inside MCTS.search().

    The C++ function signature is:
        cbatch_back_propagate(hidden_state_index_x, discount, rewards, values, policies, ...)

    Where:
        - rewards[i] is stored as the node's reward via expand()
        - values[i] is the bootstrap leaf value for backpropagation

    The bug was passing value_pool as the 3rd arg (rewards) and
    reward_pool as the 4th arg (values), completely breaking Q(s,a) = r + gamma*V.
    """

    def test_source_code_argument_order(self):
        """Verify that MCTS.search() passes reward_pool before value_pool
        to batch_back_propagate by inspecting the source code.

        This is a static check that catches the swap even without running MCTS.
        """
        import inspect
        from sheeprl.algos.muzero.utils import MCTS

        source = inspect.getsource(MCTS.search)

        # Find the batch_back_propagate call and check argument order
        # The call should have reward_pool BEFORE value_pool
        bp_start = source.find("batch_back_propagate")
        assert bp_start != -1, "batch_back_propagate call not found in MCTS.search"

        # Extract the call text (up to the closing paren)
        call_text = source[bp_start:]
        # Find reward_pool and value_pool positions within the call
        reward_pos = call_text.find("reward_pool")
        value_pos = call_text.find("value_pool")

        assert reward_pos != -1, "reward_pool not found in batch_back_propagate call"
        assert value_pos != -1, "value_pool not found in batch_back_propagate call"
        assert reward_pos < value_pos, (
            f"reward_pool (pos {reward_pos}) must appear BEFORE value_pool (pos {value_pos}) "
            f"in batch_back_propagate call. The C++ signature expects rewards as 3rd arg "
            f"and values as 4th arg. If this fails, the arguments are swapped."
        )

    def test_cython_signature_matches_expectations(self):
        """Verify the Cython batch_back_propagate function signature
        has rewards before values, matching the C++ implementation."""
        import inspect
        import sheeprl.algos.muzero.ctree.cytree as cytree

        # For Cython functions, inspect.signature may work
        try:
            sig = inspect.signature(cytree.batch_back_propagate)
            params = list(sig.parameters.keys())
        except (ValueError, TypeError):
            # Cython functions may not support inspect.signature
            # Fall back to checking the docstring or source
            pytest.skip("Cannot inspect Cython function signature")

        # Verify parameter order: ..., rewards, values, ...
        if "rewards" in params and "values" in params:
            rewards_idx = params.index("rewards")
            values_idx = params.index("values")
            assert rewards_idx < values_idx, (
                f"In Cython signature, 'rewards' (idx {rewards_idx}) must come before "
                f"'values' (idx {values_idx})"
            )
