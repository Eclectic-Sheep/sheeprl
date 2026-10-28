"""Record what an experiment computes, or check that a change keeps it: the reference of a run.

The experiment runs in-process on the CPU in `32-true`, on one thread, with fixed seeds and the given Hydra overrides
(single device, synchronous environments, no checkpoints, no videos, logs in a temporary directory). The reference
is a JSON with:

- `acts`: the `env_actions` and the stored `columns` of every `Act` of every environment step, the random ones of the
  warm-up (`Policy.random`) and the ones of the test episode included, in order;
- `metrics`: the metrics returned by every `train_step`, in order;
- `state`: the shape, dtype, SHA-256 and first values of every tensor of the final `TrainState.state_dict()`, and the
  value of every other entry.

With `--compare <json>` the run is compared with the reference, with the overrides of the command line if any (e.g. the
same experiment in a new configuration schema), else with the ones of the reference. The comparison is exact by
default: the tensors of the state are then matched by content, so a renamed parameter still matches. With `--rtol`
(e.g. on another BLAS) the values are compared within a relative tolerance and the tensors of the state by name, on
their first values and sums.

Example:
    python benchmarks/reference_run.py --out results/refs/ppo.json -- exp=ppo env.num_envs=2 algo.total_steps=128
    python benchmarks/reference_run.py --compare results/refs/ppo.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest import mock

import numpy as np
import torch
from hydra import compose, initialize_config_module

from sheeprl import ROOT_DIR
from sheeprl.cli import run
from sheeprl.core.algorithm import Algorithm

# The settings every reference runs with: what changes the computation is in the overrides of the experiment
DEFAULT_OVERRIDES = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "fabric.accelerator=cpu",
    "fabric.devices=1",
    "fabric.precision=32-true",
    "env.sync_env=True",
    "env.capture_video=False",
    "seed=42",
    # Some algorithms compute more metrics when they are logged (e.g. the entropies of DreamerV3)
    "metric.log_level=1",
    "metric.disable_timer=True",
    "checkpoint.every=1000000000",
    "checkpoint.save_last=False",
    "model_manager.disabled=True",
    "run_name=reference",
]
# The values of a tensor kept in the reference, besides its hash and its sums
HEAD = 8


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    argv = sys.argv[1:]
    overrides = []
    if "--" in argv:
        sep = argv.index("--")
        argv, overrides = argv[:sep], argv[sep + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="Path of the JSON to record the reference in")
    mode.add_argument("--compare", help="Path of the reference JSON to compare the run with")
    parser.add_argument("--rtol", type=float, default=None, help="Relative tolerance (default: exact comparison)")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance, with --rtol")
    return parser.parse_args(argv), overrides


def git_commit() -> Dict[str, Any]:
    """The commit of the code and whether the files that run differ from it."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT_DIR).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", os.path.abspath(__file__))
    return {"commit": git("rev-parse", "HEAD"), "dirty": dirty.splitlines()}


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu()
        # NumPy has no bfloat16, whose values are exact in float32
        return (x.float() if x.dtype == torch.bfloat16 else x).numpy()
    return np.asarray(x)


def dtype_name(x: Any) -> str:
    """The name of the dtype of a tensor or an array, the same for both (e.g. `float32`)."""
    return str(x.dtype).removeprefix("torch.") if torch.is_tensor(x) else str(np.asarray(x).dtype)


def encode_array(x: Any) -> Dict[str, Any]:
    """An array with its values: the floats of `tolist` are written exactly by `json`."""
    array = to_numpy(x)
    return {"dtype": dtype_name(x), "shape": list(array.shape), "values": array.tolist()}


def digest_tensor(x: Any) -> Dict[str, Any]:
    """A tensor summarized: shape, dtype, SHA-256 of its bytes, first values, sum and sum of the absolute values."""
    if torch.is_tensor(x):
        t = x.detach().cpu().contiguous()
        raw = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
    else:
        raw = np.ascontiguousarray(x).tobytes()
    values = to_numpy(x).astype(np.float64)
    flat = values.reshape(-1)
    return {
        "shape": list(values.shape),
        "dtype": dtype_name(x),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "head": flat[:HEAD].tolist(),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
    }


def flatten_state(value: Any, path: str = "") -> Iterator[Tuple[str, Dict[str, Any]]]:
    """The leaves of a state dictionary with their paths: tensors and arrays digested, the other values as they are."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten_state(v, f"{path}/{k}" if path else str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from flatten_state(v, f"{path}/{i}")
    elif torch.is_tensor(value) or isinstance(value, np.ndarray):
        yield path, {"tensor": digest_tensor(value)}
    elif value is None or isinstance(value, (bool, int, float, str)):
        yield path, {"value": value}
    elif isinstance(value, np.generic):
        yield path, {"value": value.item()}
    else:
        yield path, {"value": repr(value)}


class Recorder:
    """Records the actions of the policies, the metrics of the training steps and the final training state of the
    algorithm it is attached to (`attach`, called when the algorithm is created)."""

    def __init__(self) -> None:
        self.acts: List[Dict[str, Any]] = []
        self.metrics: List[Dict[str, Any]] = []
        self.state: Any = None
        self.phase = "train"
        self._wrapped: set = set()

    def attach(self, algo: Algorithm) -> None:
        # Instance attributes shadow the methods of the class: every call of the training loop and of the test goes
        # through them, also the ones an algorithm makes to itself (e.g. `self.policy(state)`)
        policy_fn, train_step_fn, test_fn = algo.policy, algo.train_step, algo.test

        def policy(state, *args, **kwargs):
            self.state = state
            p = policy_fn(state, *args, **kwargs)
            self.wrap_policy(p)
            return p

        def train_step(state, batch, step):
            metrics = train_step_fn(state, batch, step)
            self.metrics.append({k: encode_array(v) for k, v in metrics.items()})
            return metrics

        def test(*args, **kwargs):
            self.phase = "test"
            try:
                return test_fn(*args, **kwargs)
            finally:
                self.phase = "train"

        algo.policy, algo.train_step, algo.test = policy, train_step, test

    def wrap_policy(self, policy: Any) -> None:
        # Some algorithms return the same policy at every call
        if id(policy) in self._wrapped:
            return
        self._wrapped.add(id(policy))
        act_fn, random_fn = policy.act, policy.random

        def act(obs, greedy: bool = False):
            out = act_fn(obs, greedy=greedy)
            self.record(out, "greedy" if greedy else "act")
            return out

        def random(env):
            out = random_fn(env)
            self.record(out, "random")
            return out

        # `object.__setattr__`: a policy may be an `nn.Module`, whose `__setattr__` is not needed for functions
        object.__setattr__(policy, "act", act)
        object.__setattr__(policy, "random", random)

    def record(self, act: Any, kind: str) -> None:
        # Copied now: the writer may change the arrays of the step in place
        self.acts.append(
            {
                "phase": self.phase,
                "kind": kind,
                "env_actions": encode_array(np.array(to_numpy(act.env_actions), copy=True)),
                "columns": {k: encode_array(np.array(to_numpy(v), copy=True)) for k, v in act.columns.items()},
            }
        )

    def final_state(self) -> Dict[str, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("No training state was recorded: the algorithm never created its policy")
        return dict(flatten_state(self.state.state_dict()))


def run_experiment(overrides: List[str]) -> Recorder:
    """Run the experiment with `overrides` and record it."""
    # The environment variables of Lightning override the devices and the precision of the configuration
    for name in [k for k in os.environ if k.startswith("LT_")]:
        os.environ.pop(name)
    torch.set_num_threads(1)
    recorder = Recorder()
    algorithm_init = Algorithm.__init__

    def init(self, *args, **kwargs):
        algorithm_init(self, *args, **kwargs)
        recorder.attach(self)

    with tempfile.TemporaryDirectory() as log_root:
        argv = [os.path.join(ROOT_DIR, "__main__.py"), *DEFAULT_OVERRIDES, *overrides, f"log_root={log_root}"]
        with mock.patch.object(Algorithm, "__init__", init), mock.patch.object(sys, "argv", argv):
            run()
    return recorder


def flat(array: Dict[str, Any]) -> Dict[str, Any]:
    """An encoded array flattened."""
    values = np.asarray(array["values"]).reshape(-1)
    return {"dtype": array["dtype"], "shape": list(values.shape), "values": values.tolist()}


def arrays_equal(a: Dict[str, Any], b: Dict[str, Any], rtol: Optional[float], atol: float) -> bool:
    if a["shape"] != b["shape"]:
        return False
    x, y = np.asarray(a["values"], dtype=np.float64), np.asarray(b["values"], dtype=np.float64)
    if rtol is None:
        return a["dtype"] == b["dtype"] and np.array_equal(x, y, equal_nan=True)
    return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True)


def compare(reference: Dict[str, Any], recorder: Recorder, rtol: Optional[float], atol: float) -> List[str]:
    """The differences between the run and the reference, at most a few per section."""
    errors: List[str] = []
    limit = 10

    # Actions
    acts = recorder.acts
    if len(acts) != len(reference["acts"]):
        errors.append(f"acts: {len(acts)} steps, the reference has {len(reference['acts'])}")
    act_errors = []
    for i, (got, ref) in enumerate(zip(acts, reference["acts"])):
        where = f"acts[{i}] ({ref['phase']}, {ref['kind']})"
        if (got["phase"], got["kind"]) != (ref["phase"], ref["kind"]):
            act_errors.append(f"{where}: played as ({got['phase']}, {got['kind']})")
        # The environments reshape the actions to their action space: the layout of the array is not compared, the
        # actions are (their values in order and their dtype)
        if not arrays_equal(flat(got["env_actions"]), flat(ref["env_actions"]), rtol, atol):
            act_errors.append(f"{where}: env_actions differ")
        if sorted(got["columns"]) != sorted(ref["columns"]):
            act_errors.append(f"{where}: columns {sorted(got['columns'])}, the reference has {sorted(ref['columns'])}")
        else:
            act_errors += [
                f"{where}: column {k!r} differs"
                for k in ref["columns"]
                if not arrays_equal(got["columns"][k], ref["columns"][k], rtol, atol)
            ]
    errors += act_errors[:limit]

    # Metrics of the training steps
    metrics = recorder.metrics
    if len(metrics) != len(reference["metrics"]):
        errors.append(f"metrics: {len(metrics)} training steps, the reference has {len(reference['metrics'])}")
    metric_errors = []
    for i, (got, ref) in enumerate(zip(metrics, reference["metrics"])):
        if sorted(got) != sorted(ref):
            metric_errors.append(f"metrics[{i}]: names {sorted(got)}, the reference has {sorted(ref)}")
            continue
        metric_errors += [
            f"metrics[{i}][{k!r}]: {got[k]['values']} != {ref[k]['values']}"
            for k in ref
            if not arrays_equal(got[k], ref[k], rtol, atol)
        ]
    errors += metric_errors[:limit]

    # Final training state
    state, ref_state = recorder.final_state(), reference["state"]
    if rtol is None:
        # Matched by content: the multisets of the tensors (dtype, shape, hash) and of the other values
        def key(entry: Dict[str, Any]) -> str:
            if "tensor" in entry:
                t = entry["tensor"]
                return f"tensor {t['dtype']} {t['shape']} {t['sha256']}"
            return f"value {json.dumps(entry['value'], sort_keys=True)}"

        got_keys = collections.Counter(key(e) for e in state.values())
        ref_keys = collections.Counter(key(e) for e in ref_state.values())
        missing, extra = ref_keys - got_keys, got_keys - ref_keys
        missing_names = [name for name, e in ref_state.items() if key(e) in missing]
        extra_names = [name for name, e in state.items() if key(e) in extra]
        # The same name on both sides: the entry changed; otherwise it is only in one of them
        changed = [name for name in missing_names if name in set(extra_names)]
        missing_names = [name for name in missing_names if name not in changed]
        extra_names = [name for name in extra_names if name not in changed]
        if changed:
            errors.append(f"state: {len(changed)} entries differ: {changed[:limit]}")
        if missing_names:
            errors.append(f"state: {len(missing_names)} entries only in the reference: {missing_names[:limit]}")
        if extra_names:
            errors.append(f"state: {len(extra_names)} entries only in the run: {extra_names[:limit]}")
    else:
        if sorted(state) != sorted(ref_state):
            errors.append(f"state: other names: {sorted(set(state) ^ set(ref_state))[:limit]}")
        state_errors = []
        for name in sorted(set(state) & set(ref_state)):
            got, ref = state[name], ref_state[name]
            if ("tensor" in got) != ("tensor" in ref):
                state_errors.append(f"state[{name!r}]: a tensor in one run only")
            elif "tensor" in ref:
                g, r = got["tensor"], ref["tensor"]
                same = g["shape"] == r["shape"] and all(
                    np.allclose(g[k], r[k], rtol=rtol, atol=atol, equal_nan=True) for k in ("head", "sum", "abs_sum")
                )
                if not same:
                    state_errors.append(f"state[{name!r}]: differs")
            elif got["value"] != ref["value"]:
                state_errors.append(f"state[{name!r}]: {got['value']!r} != {ref['value']!r}")
        errors += state_errors[:limit]
    return errors


def main() -> None:
    args, overrides = parse_args()
    reference = None
    if args.compare:
        with open(args.compare) as f:
            reference = json.load(f)
        overrides = overrides or reference["overrides"]
    elif not overrides:
        raise SystemExit("Give the overrides of the experiment after `--`")

    # The configuration composes: a wrong override fails here, before the run
    with initialize_config_module(version_base="1.3", config_module="sheeprl.configs"):
        compose(config_name="config", overrides=DEFAULT_OVERRIDES + overrides + ["log_root=."])

    recorder = run_experiment(overrides)
    if reference is None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        result = {
            "overrides": overrides,
            "default_overrides": DEFAULT_OVERRIDES,
            **git_commit(),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "acts": recorder.acts,
            "metrics": recorder.metrics,
            "state": recorder.final_state(),
        }
        with open(args.out, "w") as f:
            json.dump(result, f)
        print(
            f"Recorded {len(recorder.acts)} steps, {len(recorder.metrics)} training steps and "
            f"{len(result['state'])} state entries in {args.out}"
        )
        return

    errors = compare(reference, recorder, args.rtol, args.atol)
    mode = "exactly" if args.rtol is None else f"within rtol={args.rtol}, atol={args.atol}"
    if errors:
        print(f"DIFFERENT from {args.compare} (commit {reference['commit'][:8]}), compared {mode}:")
        print("\n".join(f"  {e}" for e in errors))
        sys.exit(1)
    print(
        f"IDENTICAL to {args.compare} (commit {reference['commit'][:8]}), compared {mode}: {len(recorder.acts)} steps, "
        f"{len(recorder.metrics)} training steps, {len(reference['state'])} state entries"
    )


if __name__ == "__main__":
    main()
