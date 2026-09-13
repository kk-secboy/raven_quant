from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.run_rdagent_scenario import _run_streaming_redacted

pytestmark = [pytest.mark.no_database,
              pytest.mark.skipif(sys.platform != "linux", reason="Production process groups")]


def _alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split(") ", 1)[1].split()[0] != "Z"


@pytest.mark.parametrize("exit_code", [0, 7, -9])
def test_exit_is_reported_when_grandchild_keeps_stdout_open(tmp_path, exit_code, capsys):
    pid_file = tmp_path / "orphan.pid"
    child = (
        "import subprocess,sys,os,signal; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
        f"Path({str(pid_file)!r}).write_text(str(p.pid)); "
        "print('last research log',flush=True); "
        + ("os.kill(os.getpid(),signal.SIGKILL)" if exit_code == -9
           else f"sys.exit({exit_code})")
    )
    before = time.monotonic()
    # An unrelated process must survive cleanup of the research session.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        if exit_code:
            with pytest.raises(subprocess.CalledProcessError) as error:
                _run_streaming_redacted([sys.executable, "-c", child],
                                        timeout=None, env=dict(os.environ))
            assert error.value.returncode == exit_code
        else:
            _run_streaming_redacted([sys.executable, "-c", child],
                                    timeout=None, env=dict(os.environ))
        assert time.monotonic() - before < 10
        assert not _alive(int(pid_file.read_text()))
        assert unrelated.poll() is None
        assert "last research log" in capsys.readouterr().out
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_worker_cancellation_reaches_owned_research_session(tmp_path):
    pid_file = tmp_path / "research.pid"
    child = ("import os,time; from pathlib import Path; "
             f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)")
    wrapper = (
        "import os,sys; from scripts.run_rdagent_scenario import _run_streaming_redacted; "
        f"_run_streaming_redacted([sys.executable,'-c',{child!r}],"
        "timeout=None,env=dict(os.environ))"
    )
    process = subprocess.Popen([sys.executable, "-c", wrapper])
    try:
        until = time.monotonic() + 15
        while not pid_file.exists() and time.monotonic() < until:
            time.sleep(0.05)
        assert pid_file.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 128 + signal.SIGTERM
        assert not _alive(int(pid_file.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
