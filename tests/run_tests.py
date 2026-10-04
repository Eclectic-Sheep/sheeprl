import sys

import pytest

if __name__ == "__main__":
    # The tests run in parallel, one worker per core (at most 8): a worker takes the next test when it finishes its own
    # (`worksteal`), since the trainings take much longer than the other tests. Coverage is not measured: it doubled
    # the time of the tests
    sys.exit(pytest.main(["-n", "logical", "--maxprocesses", "8", "--dist", "worksteal", "-vv"]))
