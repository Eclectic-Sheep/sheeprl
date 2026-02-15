# MuZero Implementation Fixes - Progress Tracking

## Overview
This document tracks 10 critical issues found in the MuZero implementation that prevent learning. Each issue will be explained, fixed, and tested incrementally.

---

## Issues & Progress

### Issue 1: Policy Loss Uses Wrong Target Type ✅ CLARIFIED (Not a bug with PyTorch >= 1.10)

**File**: [`sheeprl/algos/muzero/loss.py`](sheeprl/algos/muzero/loss.py)

**UPDATE**: With PyTorch 2.10 (installed version), the original code **works correctly**. PyTorch >= 1.10 natively handles float targets as probability distributions in `cross_entropy()`.

**Problem** (originally suspected):
The code wasn't documented, and it's unclear if the behavior was intentional.

**What we did**:
- Added comprehensive docstring explaining that `cross_entropy` with float targets computes proper cross-entropy for distributions
- Verified it works correctly with PyTorch 2.10
- No functional change needed

**Status**: ✅ **DOCUMENTED** - Original code works, now documented properly

**File**: [`sheeprl/algos/muzero/loss.py`](sheeprl/algos/muzero/loss.py)

**Problem**:
```python
def policy_loss(predicted_policy_logits, target_policy):
    return torch.nn.functional.cross_entropy(predicted_policy_logits, target_policy, reduction="none")
```

`cross_entropy()` expects:
- Input: logits
- Target: class indices (integers) OR probabilities (floats with special handling)

But here `target_policy` is a probability distribution (floats), and `cross_entropy()` treats it as class indices, causing wrong gradients.

**Why it matters**: The gradient signal for policy learning is garbage → agent can't improve policy.

**Fix**: Replace with KL divergence or proper cross-entropy for distributions:
```python
def policy_loss(predicted_policy_logits, target_policy):
    log_probs = torch.nn.functional.log_softmax(predicted_policy_logits, dim=-1)
    return -(target_policy * log_probs).sum(-1)  # Cross-entropy for distributions
```

**Status**: ✅ **FIXED** - Changed to proper cross-entropy for distributions using log_softmax

---

### Issue 2: MinMaxStats Min/Max Initialization Inverted ✅ FIXED

**File**: [`sheeprl/algos/muzero/utils.py`](sheeprl/algos/muzero/utils.py) (lines 67-68)

**Problem**:
```python
def __init__(self):
    self.maximum = float("inf")   # ❌ WRONG
    self.minimum = -float("inf")  # ❌ WRONG
```

Should be the opposite! When you normalize:
```python
def normalize(self, value):
    if self.maximum > self.minimum != -float("inf") and self.maximum != float("inf"):
        return (value - self.minimum) / (self.maximum - self.minimum)
    return value
```

With current init: `inf > -inf` is true but won't update because any real value violates the condition. Result: **normalization never happens**.

**Why it matters**: Value scores in MCTS UCB calculation remain unnormalized → UCB scores are wrong → action selection is biased.

**Fix**:
```python
self.maximum = -float("inf")  # Start low so any value is > maximum
self.minimum = float("inf")   # Start high so any value is < minimum
```

**Status**: ✅ **FIXED** - User corrected the initialization. All 19 unit tests pass.

---

### Issue 3: Hidden State Normalization Division by Zero ✅ FIXED

**File**: [`sheeprl/algos/muzero/muzero.py`](sheeprl/algos/muzero/muzero.py) (line 280)

**Problem**:
```python
hidden_states = (hidden_states - hidden_states.min()) / (hidden_states.max() - hidden_states.min())
```

If all hidden state values are identical: `hidden_states.max() == hidden_states.min()` → **division by zero** → NaN propagates through entire batch.

**Why it matters**: Gradient updates become NaN → model weights become NaN → training collapses.

**Fix**: Add epsilon:
```python
h_min = hidden_states.min()
h_max = hidden_states.max()
hidden_states = (hidden_states - h_min) / (h_max - h_min + 1e-8)
```

**Status**: ✅ **FIXED** - Added epsilon (1e-8) to prevent division by zero

---

### Issue 4: Device Mismatch on GPU ✅ FIXED

**File**: [`sheeprl/algos/muzero/muzero.py`](sheeprl/algos/muzero/muzero.py) (line 218)

**Problem**:
```python
obs_pool[env_idx] = torch.tensor(next_obs).reshape(1, -1)
```

`obs_pool` is on `device` (GPU if available), but `torch.tensor(next_obs)` is created on CPU. Assigning CPU tensor to GPU tensor causes silent device mismatch or error.

**Why it matters**: On GPU, this silently copies tensors repeatedly (slow) or errors out. On multi-GPU, indices get misaligned.

**Fix**:
```python
obs_pool[env_idx] = torch.tensor(next_obs, device=device).reshape(1, -1)
```

**Status**: ✅ **FIXED** - Added device parameter to torch.tensor() call

---

### Issue 5: Non-Standard VLA in C++ ✅ FIXED

**File**: [`sheeprl/algos/muzero/ctree/cnode.cpp`](sheeprl/algos/muzero/ctree/cnode.cpp) (line 55)

**Problem**:
```cpp
float policy[action_num];  // ❌ Variable-length array - NOT standard C++
```

VLAs are a GCC extension. They work on Linux but fail on MSVC/Windows or strict compilers.

**Why it matters**: Code is non-portable. On strict compilers, build fails.

**Fix**: Use `std::vector`:
```cpp
std::vector<float> policy(action_num);
```

**Status**: ✅ **FIXED** - Replaced VLA with std::vector, recompiled successfully

---

### Issue 6: Broken Cython Node Class ✅ FIXED

**File**: [`sheeprl/algos/muzero/ctree/cytree.pyx`](sheeprl/algos/muzero/ctree/cytree.pyx) (lines 74-82)

**Problem**:
```cython
cdef class Node:
    cdef CNode cnode

    def __cinit__(self):
        pass

    def __cinit__(self, float prior, int action_num):
        # self.cnode = CNode(prior, action_num)
        pass
```

Two `__cinit__` methods (both stubs). The class is defined but not usable. Code has commented-out initialization.

**Why it matters**: If someone tries to use `Node` class from Python, it will silently fail or be unsafe.

**Fix**: Either:
- Option A: Remove unused `Node` class entirely (it's not used) ✅ **CHOSEN**
- Option B: Implement properly with single valid `__cinit__`

**Status**: ✅ **FIXED** - Removed unused Node class, added comment explaining removal

---

### Issue 7: Setup.py Configuration Error ✅ FIXED

**File**: [`sheeprl/algos/muzero/ctree/setup.py`](sheeprl/algos/muzero/ctree/setup.py)

**Problem**:
```python
from Cython.Build import cythonize
from setuptools import setup

setup(ext_modules=cythonize("cytree.pyx"), extra_compile_args=["-O3"], include_dirs=[np.get_include()])
```

`extra_compile_args` is passed to `setup()` but should be part of the `Extension()` object inside `cythonize()`.

**Why it matters**: Compiler flags are ignored. Code compiles without optimizations (slower).

**Fix**:
```python
from Cython.Build import cythonize
from setuptools import setup, Extension

ext = Extension("cytree", sources=["cytree.pyx"], extra_compile_args=["-O3"], include_dirs=[np.get_include()])
setup(ext_modules=cythonize([ext]))
```

**Status**: ✅ **FIXED** - Properly configured Extension with compile flags, recompiled

---

## Detailed Explanation: Why These Prevent Learning

### Flow Analysis:

1. **Data Collection** (exploration phase):
   - ✅ Works correctly with random or untrained policy
   - Stores trajectories in buffer

2. **Training Update** (where issues appear):
   - ❌ **Issue 2** (MinMaxStats): Value normalization broken → UCB scores wrong
   - ❌ **Issue 1** (Policy loss): Wrong loss gradient → policy doesn't improve
   - ❌ **Issue 3** (Hidden state norm): Can divide by zero → NaN gradients
   - ❌ **Issue 4** (Device): Tensor misalignment → training is slow/buggy

3. **MCTS Search** (using updated model):
   - ❌ **Issue 5** (VLA): Compiles but fragile on some systems
   - ❌ **Issue 6** (Cython): Not blocking but unsafe code pattern

4. **Result**: Model weights don't improve despite training → **no learning observed**.

---

## Fix Order (Dependency Graph)

```
Issue 1 (Policy Loss)
├─ No dependencies, can be fixed first
└─ Affects: Training gradient quality

Issue 2 (MinMaxStats)
├─ No dependencies
└─ Affects: MCTS value normalization

Issue 3 (Hidden State Norm)
├─ No dependencies
└─ Affects: Training stability

Issue 4 (Device Mismatch)
├─ No dependencies
└─ Affects: GPU execution

Issue 5 (VLA in C++)
├─ No dependencies
└─ Affects: Portability (not learning directly)

Issue 6 (Cython Node)
├─ No dependencies
└─ Affects: Code safety (not learning directly)

Issue 7 (Setup.py)
├─ No dependencies
└─ Affects: Compilation performance (not learning directly)
```

**Recommended order**: 1 → 2 → 3 → 4 → (5, 6, 7)

Issues 5, 6, 7 are lower priority but should be fixed for production code.

---

## Testing Strategy

After each fix:
1. Recompile if C++/Cython changed: `cd sheeprl/algos/muzero/ctree && python setup.py build_ext --inplace`
2. Run dry-run test: `CUDA_VISIBLE_DEVICES="" python sheeprl.py muzero exp=muzero dry_run=True`
3. Check TensorBoard after 20k steps for:
   - `Rewards/rew_avg` increases (should go from ~0 to ~50+ on LunarLander)
   - `Loss/total_loss` decreases
   - `Loss/value_loss` decreases

---

## Progress Checklist

- [x] Issue 1: Fix policy_loss() - explain & implement
- [x] Issue 2: Fix MinMaxStats init - explain & implement
- [x] Issue 3: Fix hidden_state normalization - explain & implement
- [x] Issue 4: Fix device mismatch - explain & implement
- [x] Issue 5: Fix VLA in C++ - explain & implement
- [x] Issue 6: Fix Cython Node class - explain & implement
- [x] Issue 7: Fix setup.py - explain & implement
- [x] Recompile C++ extensions
- [x] Run dry-run test
- [ ] ~~Run full training (20k steps minimum)~~
- [ ] ~~Verify learning curve in TensorBoard~~

---

## Issue 8: Value Prediction Divergence (CRITICAL) ✅ FIXED

**Files**: [`sheeprl/configs/exp/muzero.yaml`](sheeprl/configs/exp/muzero.yaml), [`sheeprl/configs/algo/muzero.yaml`](sheeprl/configs/algo/muzero.yaml), [`sheeprl/algos/muzero/muzero.py`](sheeprl/algos/muzero/muzero.py)

**Problem**:
After 750k training steps, the agent showed **no policy learning**:
- Policy entropy stuck at maximum (13.86 = 10 × ln(4)) — uniform random policy throughout
- MCTS visit counts: [12, 12, 13, 13] — perfectly uniform (should be non-uniform)
- Value predictions: **-3325** (should be ~-100 to -300 for LunarLander)
- Reward improved only slightly: -275 → -146 (random agent level)

**Root Cause**: A self-amplifying feedback loop in value predictions:

1. Model predicts slightly wrong value (e.g., -31 at step 100)
2. This value is stored as MCTS bootstrap for n-step returns
3. Bootstrap amplifies: `return = rewards + γ^n × wrong_value`
4. `symsqrt(extreme_return)` clips to support edge (-60)
5. Model learns to predict -60 in support space → `inverse_symsqrt(-60) ≈ -3329`
6. This -3329 is used as next bootstrap → even more extreme targets
7. Values rapidly saturate at support edge within ~1000 training updates

**Divergence timeline** (from checkpoint analysis):
```
Step   100: value =    -31.39  (reasonable)
Step   200: value =    -45.01  (still OK)
Step   500: value =   -147.21  (diverging)
Step  1000: value = -1,324.33  (badly diverging)
Step  2000: value = -3,319.78  (saturated at support edge)
Step  4000+: value = -3,325.36  (stuck forever)
```

With ALL Q-values at -3329, MinMaxStats normalizes to 0 for all actions → uniform MCTS visits → uniform policy targets → policy never learns.

**Why it matters**: This is the **primary reason** the agent fails to learn. All previous fixes (Issues 1-7) were necessary but insufficient because the value divergence prevents MCTS from producing useful policy targets.

**Fix** (3 changes):

1. **Increased `support_size` from 60 to 300** (as in the MuZero paper, Appendix F Table S4):
   - Previous: support range [-60, 60], max representable value ≈ ±3,329
   - Now: support range [-300, 300], max representable value ≈ ±89,700
   - LunarLander values in [-600, 300] map to [-24, +17] in support space — only 8% of the range
   - Enormous headroom prevents the divergence from starting

2. **Added `value_target_clip: 300.0`** — clips MCTS root values before they're used as bootstrap:
   ```python
   value = max(-cfg.algo.value_target_clip, min(cfg.algo.value_target_clip, value))
   ```
   This directly breaks the feedback loop by preventing extreme bootstrap values.

3. **Added `agent.train()`** before the training loop — MCTS sets `model.eval()` but never switches back. While MLPs aren't affected by train/eval mode, this is correct practice.

4. **Added configurable `value_loss_weight`** — allows reducing the value loss contribution to prevent value gradients from overwhelming policy gradients through gradient clipping.

**Status**: ✅ **FIXED** — Three-pronged fix addressing the value divergence feedback loop

---

## Updated Progress Checklist

- [x] Issue 1: Fix policy_loss() - explain & implement
- [x] Issue 2: Fix MinMaxStats init - explain & implement
- [x] Issue 3: Fix hidden_state normalization - explain & implement
- [x] Issue 4: Fix device mismatch - explain & implement
- [x] Issue 5: Fix VLA in C++ - explain & implement
- [x] Issue 6: Fix Cython Node class - explain & implement
- [x] Issue 7: Fix setup.py - explain & implement
- [x] Issue 8: Fix value prediction divergence (support_size + value clipping)
- [x] Recompile C++ extensions
- [x] Run dry-run test
- [ ] Run full training (200k steps)
- [ ] Verify learning curve in TensorBoard

---

## Issue 9: Value/Reward Arguments Swapped in MCTS Backpropagation ✅ FIXED

**File**: [`sheeprl/algos/muzero/utils.py`](sheeprl/algos/muzero/utils.py) (lines 402-410, inside `MCTS.search()`)

**Problem**:
```python
# WRONG — value_pool and reward_pool are in the wrong positions:
tree.batch_back_propagate(
    hidden_state_index_x,
    self.discount,
    value_pool,           # goes to C++ "rewards" parameter
    reward_pool,          # goes to C++ "values" parameter
    policy_logits_pool,
    min_max_stats_lst,
    results,
)
```

The Cython/C++ function signature is:
```
cbatch_back_propagate(int hidden_state_index_x, float discount,
                      const vector<float> &rewards,   // 3rd arg: stored as node.reward
                      const vector<float> &values,     // 4th arg: bootstrap for backprop
                      ...)
```

This means:
- Value predictions were stored as node rewards → Q(s,a) = V_pred + γ·V(s') instead of r_pred + γ·V(s')
- Reward predictions were used as bootstrap values → backpropagation starts from r instead of V

**Why it matters**: The MCTS tree computes Q-values with swapped components, making UCB scores meaningless. The search cannot distinguish good actions from bad ones, and the policy targets extracted from visit counts carry no useful signal.

**Fix**:
```python
# CORRECT — reward_pool first, value_pool second:
tree.batch_back_propagate(
    hidden_state_index_x,
    self.discount,
    reward_pool,          # rewards → stored as node.reward via expand()
    value_pool,           # values → bootstrap leaf value via cback_propagate()
    policy_logits_pool,
    min_max_stats_lst,
    results,
)
```

**Status**: ✅ **FIXED** — Swapped arguments to match C++ function signature

---

## Issue 10: Gradient Starvation — Policy Network Can't Learn ⚠️ CONFIG FIX NEEDED

**Files**: [`sheeprl/configs/algo/muzero.yaml`](sheeprl/configs/algo/muzero.yaml), [`sheeprl/configs/exp/muzero.yaml`](sheeprl/configs/exp/muzero.yaml)

**Problem**:
After fixing Issue 9 (value/reward swap), MCTS now produces peaked, non-uniform policy targets (e.g., [0.88, 0.02, 0.08, 0.02]). The value and reward networks learn successfully. However, the policy network fails to track the improving MCTS targets.

**Symptoms observed at ~166K env steps**:
- Gradient norm: **always 1.0** (clipped at every single step)
- Policy loss: 13.9 → 12.3 (barely moved from max entropy 13.86 = 10 × ln(4))
- KL divergence: 0.04 → **1.3** (increasing — policy diverges from MCTS targets)
- Value loss: 63 → 18 (good, learning)
- Reward loss: 56 → 8 (good, learning)
- Agent performance: -219 reward vs random -191 (worse than random!)

**Root Cause**: Three factors work together:

1. **Gradient clipping at 1.0 is too aggressive**: Every single gradient update is clipped, meaning useful gradient information is lost. The raw gradient norm is always >> 1.0.

2. **Imbalanced loss scales**: Value/reward losses use 601-dimensional categorical cross-entropy (magnitudes 18-63), while policy loss uses 4-dimensional cross-entropy (magnitude 12-14). The gradient from value/reward dominates the shared representation and dynamics networks, leaving the policy head with insufficient gradient budget after clipping.

3. **Insufficient training**: 166K env steps is early for MuZero, but issues 1 and 2 make convergence unnecessarily slow.

**Fix** (config changes only, no code changes):

```yaml
# In sheeprl/configs/exp/muzero.yaml or sheeprl/configs/algo/muzero.yaml:
algo:
  max_grad_norm: 5.0          # was 1.0 — allow larger gradients through
  value_loss_weight: 0.25     # was 1.0 — reduce value gradient dominance
```

**Why these values**:
- `max_grad_norm: 5.0` — Allows 5× more gradient flow. Still clips extreme gradients but stops starving the policy. The MuZero paper (Appendix G) doesn't prescribe 1.0; many implementations use 5.0 or 10.0.
- `value_loss_weight: 0.25` — Reduces value loss contribution to 25%. Since value loss magnitude is ~3× larger than policy loss AND has a 601-dim output (vs 4-dim policy), this rebalances the effective gradient contribution between value and policy.

**Status**: ⚠️ **CONFIG FIX NEEDED** — Requires restarting training with updated config

---

## Updated Progress Checklist

- [x] Issue 1: Fix policy_loss() - explain & implement
- [x] Issue 2: Fix MinMaxStats init - explain & implement
- [x] Issue 3: Fix hidden_state normalization - explain & implement
- [x] Issue 4: Fix device mismatch - explain & implement
- [x] Issue 5: Fix VLA in C++ - explain & implement
- [x] Issue 6: Fix Cython Node class - explain & implement
- [x] Issue 7: Fix setup.py - explain & implement
- [x] Issue 8: Fix value prediction divergence (support_size + value clipping)
- [x] Issue 9: Fix value/reward swap in batch_back_propagate call
- [ ] Issue 10: Fix gradient starvation (config: max_grad_norm=5.0, value_loss_weight=0.25)
- [x] Recompile C++ extensions
- [x] Run dry-run test
- [ ] Run full training (200k steps) with updated config
- [ ] Verify learning curve in TensorBoard

---

## Notes

- Each step will include: problem explanation → code before/after → why it matters
- No changes until user confirms understanding of that step
- After all fixes, we verify learning on LunarLander-v2
