import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "serving" / "core" / "htsim_astra_shim.py"


def _run_shim(stdin: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SHIM), "--fixed-cycles=1234", *extra_args],
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_pass_then_exit_protocol():
    proc = _run_shim("pass\nexit\n")

    assert proc.returncode == 0
    assert proc.stderr == ""
    assert proc.stdout.splitlines() == [
        "Waiting",
        "Waiting",
        "All Request Has Been Exited",
        "HTSim ASTRA shim exiting",
    ]


def test_initial_workload_completion_protocol():
    proc = _run_shim(
        "exit\n",
        "--workload-configuration=/tmp/inputs/workload/gpu/model/instance2_batch7/llm",
    )

    assert proc.returncode == 0
    assert proc.stderr == ""
    assert proc.stdout.splitlines() == [
        "sys[2] iteration 7 finished, 1234 cycles, exposed communication 1234 cycles.",
        "Waiting",
        "All Request Has Been Exited",
        "HTSim ASTRA shim exiting",
    ]


def test_stdin_workload_completion_protocol():
    proc = _run_shim("/tmp/inputs/workload/gpu/model/instance2_batch7/llm\nexit\n")

    assert proc.returncode == 0
    assert proc.stderr == ""
    assert proc.stdout.splitlines() == [
        "Waiting",
        "sys[2] iteration 7 finished, 1234 cycles, exposed communication 1234 cycles.",
        "Waiting",
        "All Request Has Been Exited",
        "HTSim ASTRA shim exiting",
    ]
