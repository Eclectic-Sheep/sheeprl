import os
import subprocess
import sys

import pytest

if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    p = subprocess.Popen(["mlflow", "ui", "--port", "5000"])
    exit_code = pytest.main(
        ["-s", "--cov=sheeprl", "-vv", "tests/test_algos/test_algos.py::test_p2e_dv3", "--timeout", "180"]
    )
    p.terminate()
    sys.exit(exit_code)
