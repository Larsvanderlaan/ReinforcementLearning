"""Two-worker subprocess supervisor with exact atomic-fold timeouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from occupancy_ratio_benchmark.calibration_artifacts import (
    quarantine_stale_state,
    write_unit_state,
)


CommandBuilder = Callable[[Mapping[str, Any], int], Sequence[str]]
CompletionCheck = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class UnitOutcome:
    unit_id: str
    status: str
    failure_type: str | None
    error: str | None
    attempts: int
    retry_count: int
    runtime_sec: float
    returncode: int | None
    resumed: bool = False


@dataclass
class _Running:
    unit: Mapping[str, Any]
    attempt: int
    process: subprocess.Popen[bytes]
    started_monotonic: float
    started_unix: float
    last_heartbeat: float
    stdout_handle: Any
    stderr_handle: Any


def run_supervised_units(
    *,
    units: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    run_root: str | Path,
    command_builder: CommandBuilder,
    completion_check: CompletionCheck,
    maximum_concurrency: int,
    timeout_sec: float,
    heartbeat_interval_sec: float,
    stale_after_sec: float,
    maximum_transient_retries: int,
    worker_environment: Mapping[str, str] | None = None,
    poll_interval_sec: float = 0.1,
) -> list[UnitOutcome]:
    """Run atomic units with bounded concurrency and killable processes."""

    concurrency = int(maximum_concurrency)
    if concurrency <= 0:
        raise ValueError("maximum_concurrency must be positive")
    timeout = float(timeout_sec)
    heartbeat = float(heartbeat_interval_sec)
    stale = float(stale_after_sec)
    retries = int(maximum_transient_retries)
    poll = float(poll_interval_sec)
    if (
        timeout <= 0.0
        or heartbeat <= 0.0
        or stale + 1e-12 < 3.0 * heartbeat
        or retries < 0
        or poll <= 0.0
    ):
        raise ValueError("invalid supervisor timing or retry configuration")
    root = Path(run_root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Mapping[str, Any], int]] = []
    outcomes: list[UnitOutcome] = []
    seen = set()
    for unit in units:
        unit_id = str(unit.get("unit_id", ""))
        if not unit_id or unit_id in seen:
            raise ValueError("supervised units require unique nonempty unit ids")
        seen.add(unit_id)
        try:
            ready = bool(completion_check(unit))
        except Exception as error:
            raise ValueError(f"cannot validate existing artifact for {unit_id}: {error}") from error
        if ready:
            outcomes.append(
                UnitOutcome(unit_id, "ok", None, None, 0, 0, 0.0, 0, resumed=True)
            )
            continue
        quarantine_stale_state(
            run_root=root,
            unit_id=unit_id,
            stale_after_sec=stale,
        )
        pending.append((unit, 0))

    running: dict[str, _Running] = {}
    try:
        while pending or running:
            while pending and len(running) < concurrency:
                unit, attempt = pending.pop(0)
                worker = _launch(
                    unit=unit,
                    attempt=attempt,
                    manifest=manifest,
                    run_root=root,
                    command=command_builder(unit, attempt),
                    worker_environment=worker_environment,
                )
                running[str(unit["unit_id"])] = worker

            now = time.monotonic()
            for unit_id, worker in list(running.items()):
                elapsed = now - worker.started_monotonic
                returncode = worker.process.poll()
                if returncode is None and elapsed >= timeout:
                    _terminate(worker.process)
                    _close_logs(worker)
                    del running[unit_id]
                    _retry_or_finish(
                        unit=worker.unit,
                        attempt=worker.attempt,
                        runtime=elapsed,
                        returncode=None,
                        failure_type="timeout",
                        error=f"atomic fold exceeded {timeout:g} seconds",
                        retryable=True,
                        retries=retries,
                        pending=pending,
                        outcomes=outcomes,
                        manifest=manifest,
                        run_root=root,
                        started_at=worker.started_unix,
                    )
                    continue
                if returncode is None:
                    if now - worker.last_heartbeat >= heartbeat:
                        write_unit_state(
                            run_root=root,
                            manifest=manifest,
                            unit=worker.unit,
                            status="running",
                            attempt=worker.attempt,
                            started_at=worker.started_unix,
                        )
                        worker.last_heartbeat = now
                    continue

                _close_logs(worker)
                del running[unit_id]
                if returncode == 0:
                    try:
                        ready = bool(completion_check(worker.unit))
                    except Exception as error:
                        ready = False
                        completion_error = f"artifact validation failed: {type(error).__name__}: {error}"
                    else:
                        completion_error = "worker exited zero without a valid artifact"
                    if ready:
                        outcome = UnitOutcome(
                            unit_id=unit_id,
                            status="ok",
                            failure_type=None,
                            error=None,
                            attempts=worker.attempt + 1,
                            retry_count=worker.attempt,
                            runtime_sec=elapsed,
                            returncode=0,
                        )
                        outcomes.append(outcome)
                        write_unit_state(
                            run_root=root,
                            manifest=manifest,
                            unit=worker.unit,
                            status="ok",
                            attempt=worker.attempt,
                            started_at=worker.started_unix,
                        )
                    else:
                        _retry_or_finish(
                            unit=worker.unit,
                            attempt=worker.attempt,
                            runtime=elapsed,
                            returncode=0,
                            failure_type="transient_io",
                            error=completion_error,
                            retryable=True,
                            retries=retries,
                            pending=pending,
                            outcomes=outcomes,
                            manifest=manifest,
                            run_root=root,
                            started_at=worker.started_unix,
                        )
                    continue

                failure_type, retryable = classify_returncode(returncode)
                _retry_or_finish(
                    unit=worker.unit,
                    attempt=worker.attempt,
                    runtime=elapsed,
                    returncode=returncode,
                    failure_type=failure_type,
                    error=f"worker exited with code {returncode}",
                    retryable=retryable,
                    retries=retries,
                    pending=pending,
                    outcomes=outcomes,
                    manifest=manifest,
                    run_root=root,
                    started_at=worker.started_unix,
                )
            _write_summary(root, outcomes, pending_count=len(pending), running_count=len(running))
            if pending or running:
                time.sleep(min(poll, 1.0))
    except BaseException:
        for worker in running.values():
            _terminate(worker.process)
            _close_logs(worker)
        raise
    _write_summary(root, outcomes, pending_count=0, running_count=0)
    return outcomes


def classify_returncode(returncode: int) -> tuple[str, bool]:
    """Map worker exit codes to structured retry policy."""

    code = int(returncode)
    if code in {2, 3, 4}:
        return ({2: "validation", 3: "missing_backend", 4: "numerical"}[code], False)
    if code == 75:
        return "transient_io", True
    if code < 0 or code >= 128:
        return "worker_lost", True
    return "worker_lost", True


def summarize_outcomes(outcomes: Sequence[UnitOutcome]) -> dict[str, Any]:
    runtimes = [row.runtime_sec for row in outcomes if row.status == "ok" and not row.resumed]
    status_counts = {
        status: sum(row.status == status for row in outcomes)
        for status in sorted({row.status for row in outcomes})
    }
    return {
        "unit_count": len(outcomes),
        "status_counts": status_counts,
        "resumed_count": sum(row.resumed for row in outcomes),
        "retry_count": sum(row.retry_count for row in outcomes),
        "runtime_p50_sec": float(np.quantile(runtimes, 0.50)) if runtimes else None,
        "runtime_p95_sec": float(np.quantile(runtimes, 0.95)) if runtimes else None,
        "outcomes": [asdict(row) for row in outcomes],
    }


def worker_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the crash-resistant deterministic child environment."""

    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "TF_DETERMINISTIC_OPS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    if overrides:
        environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def _launch(
    *,
    unit: Mapping[str, Any],
    attempt: int,
    manifest: Mapping[str, Any],
    run_root: Path,
    command: Sequence[str],
    worker_environment: Mapping[str, str] | None,
) -> _Running:
    if not command:
        raise ValueError("worker command cannot be empty")
    unit_id = str(unit["unit_id"])
    log_dir = run_root / "logs" / unit_id
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_handle = (log_dir / f"attempt-{attempt}.stdout.log").open("wb")
    stderr_handle = (log_dir / f"attempt-{attempt}.stderr.log").open("wb")
    started_unix = time.time()
    write_unit_state(
        run_root=run_root,
        manifest=manifest,
        unit=unit,
        status="running",
        attempt=attempt,
        started_at=started_unix,
    )
    try:
        process = subprocess.Popen(
            [str(value) for value in command],
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=worker_environment_fn(worker_environment),
            start_new_session=True,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    now = time.monotonic()
    return _Running(
        unit=unit,
        attempt=attempt,
        process=process,
        started_monotonic=now,
        started_unix=started_unix,
        last_heartbeat=now,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )


def worker_environment_fn(overrides: Mapping[str, str] | None) -> dict[str, str]:
    return worker_environment(overrides)


def _retry_or_finish(
    *,
    unit: Mapping[str, Any],
    attempt: int,
    runtime: float,
    returncode: int | None,
    failure_type: str,
    error: str,
    retryable: bool,
    retries: int,
    pending: list[tuple[Mapping[str, Any], int]],
    outcomes: list[UnitOutcome],
    manifest: Mapping[str, Any],
    run_root: Path,
    started_at: float,
) -> None:
    if retryable and attempt < retries:
        write_unit_state(
            run_root=run_root,
            manifest=manifest,
            unit=unit,
            status="pending",
            attempt=attempt + 1,
            failure_type=failure_type,
            error=error,
            started_at=started_at,
        )
        pending.append((unit, attempt + 1))
        return
    outcomes.append(
        UnitOutcome(
            unit_id=str(unit["unit_id"]),
            status="failed",
            failure_type=failure_type,
            error=error,
            attempts=attempt + 1,
            retry_count=attempt,
            runtime_sec=float(runtime),
            returncode=returncode,
        )
    )
    write_unit_state(
        run_root=run_root,
        manifest=manifest,
        unit=unit,
        status="failed",
        attempt=attempt,
        failure_type=failure_type,
        error=error,
        started_at=started_at,
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait(timeout=5.0)


def _close_logs(worker: _Running) -> None:
    worker.stdout_handle.close()
    worker.stderr_handle.close()


def _write_summary(
    run_root: Path,
    outcomes: Sequence[UnitOutcome],
    *,
    pending_count: int,
    running_count: int,
) -> None:
    payload = summarize_outcomes(outcomes)
    payload.update({"pending_count": int(pending_count), "running_count": int(running_count)})
    path = run_root / "supervisor_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "UnitOutcome",
    "classify_returncode",
    "run_supervised_units",
    "summarize_outcomes",
    "worker_environment",
]
