import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from serving.core.controller import Controller  # noqa: E402


def _backend(script):
    return subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_read_wait_returns_at_waiting():
    p = _backend("print('sys[0] iteration 0 finished, 5 cycles, exposed communication 0 cycles.'); print('Waiting')")
    out = Controller(1).read_wait(p)
    assert "Waiting" in out[-1]
    assert Controller(1).parse_output(out[-2]) == {"sys": 0, "id": 0, "cycle": 5}
    p.wait()


def test_read_wait_raises_when_backend_exits():
    # A backend that dies before its prompt: without the guard read_wait
    # spins on "" forever.
    p = _backend("import sys; print('partial report'); sys.exit(139)")
    with pytest.raises(RuntimeError, match=r"exited \(return code 139\)"):
        Controller(1).read_wait(p)


def test_check_end_raises_when_backend_exits():
    p = _backend("print('Checking Non-Exited Systems ...')")
    with pytest.raises(RuntimeError, match="exited"):
        Controller(1).check_end(p)
