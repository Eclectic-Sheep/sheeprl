import sys

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(["-s", "--cov=sheeprl", "-vv", "tests/test_algos/test_algos.py::test_dreamer_v3"]))
