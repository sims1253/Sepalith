"""CPU regression for profile child cleanup on supervisor SIGTERM.

The child process is a temporary sleep only.  No model, TRAIN, provider, or
CUDA path is touched.  The test monkeypatches the profile implementation so
that the real execute_profile signal wrapper and run_stage cleanup are tested
without entering input verification or training.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


PACKET = Path(__file__).resolve().parent
PROFILE_DIR = PACKET / "source/docs/campaign/work/r2-draft-cloud-profile-v2"
REPORT = PACKET / "termination-regression.json"


def _profile():
    sys.path.insert(0, str(PROFILE_DIR))
    import profile_orchestrator  # type: ignore
    return profile_orchestrator


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _child(run_dir: Path) -> int:
    profile = _profile()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "stages").mkdir()
    pid_file = run_dir / "child.pid"
    code = (
        "import os, pathlib, time; "
        "pathlib.Path(__import__('sys').argv[1]).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )

    def fake_impl(**_kwargs):
        stage = profile.StageSpec(
            name="termination-probe",
            argv=(sys.executable, "-c", code, str(pid_file)),
            cwd=run_dir,
            wall_limit_seconds=60,
        )
        return profile.run_stage(stage, run=run_dir, environment=dict(os.environ))

    profile._execute_profile_impl = fake_impl
    try:
        profile.execute_profile(
            run_dir=run_dir,
            target_manifest_path=run_dir / "unused-manifest.json",
            train_data_path=run_dir / "unused-train.jsonl",
            source_root=run_dir,
            python_executable=sys.executable,
        )
    except BaseException as exc:
        print(type(exc).__name__, str(exc), flush=True)
        return 1
    return 0


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "child":
        raise SystemExit(_child(Path(sys.argv[2])))

    with tempfile.TemporaryDirectory(prefix="r2-profile-termination-") as temporary:
        run_dir = Path(temporary) / "run"
        env = dict(os.environ)
        env.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(PROFILE_DIR),
        })
        child = subprocess.Popen(
            [sys.executable, __file__, "child", str(run_dir)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        pid_file = run_dir / "child.pid"
        deadline = time.monotonic() + 8
        while not pid_file.exists() and time.monotonic() < deadline:
            if child.poll() is not None:
                break
            time.sleep(0.05)
        if not pid_file.exists():
            child.kill()
            child.wait(timeout=5)
            raise AssertionError("termination probe child did not start")
        stage_pid = int(pid_file.read_text())
        os.kill(child.pid, signal.SIGTERM)
        exit_code = child.wait(timeout=10)
        output = child.stdout.read() if child.stdout is not None else ""
        time.sleep(0.2)
        stage_alive = _alive(stage_pid)
        if stage_alive:
            try:
                os.killpg(stage_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        terminal_path = run_dir / "stages/termination-probe.terminal.json"
        terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
        assert exit_code == 1, (exit_code, output)
        assert not stage_alive, f"owned stage process survived supervisor SIGTERM: {stage_pid}"
        assert isinstance(terminal, dict) and terminal.get("status") == "failed", terminal
        assert terminal.get("error_type") == "ProfileError", terminal
        result = {
            "status": "PASS",
            "cpu_only": True,
            "cuda_visible_devices": "",
            "orchestrator_exit_code": exit_code,
            "stage_pid": stage_pid,
            "stage_alive_after_cleanup": stage_alive,
            "failure_receipt": str(terminal_path),
            "failure_receipt_status": terminal.get("status"),
            "failure_receipt_error_type": terminal.get("error_type"),
            "model_loaded": False,
            "train_rows_read": False,
            "provider_or_gpu_used": False,
        }
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
