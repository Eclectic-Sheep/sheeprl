import pathlib
import re

import sheeprl
from sheeprl.utils.callback import CheckpointCallback


def test_every_called_hook_is_a_method_of_the_checkpoint_callback():
    # `fabric.call(hook, ...)` does nothing when no callback has the method `hook`: a misspelled or renamed hook would
    # silently stop saving the checkpoints
    hooks = set()
    for path in pathlib.Path(sheeprl.__file__).parent.rglob("*.py"):
        hooks |= set(re.findall(r'fabric\.call\(\s*"(\w+)"', path.read_text(encoding="utf-8")))
    assert len(hooks) > 0
    missing = {hook for hook in hooks if not callable(getattr(CheckpointCallback, hook, None))}
    assert len(missing) == 0, missing
