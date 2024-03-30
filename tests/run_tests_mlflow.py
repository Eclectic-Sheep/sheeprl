import os
import subprocess
import sys

import pytest

if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    p = subprocess.Popen(["mlflow", "ui", "--port", "5000"])
    exit_code = pytest.main(["-s", "--cov=sheeprl", "-vv"])
    p.terminate()
    sys.exit(exit_code)
