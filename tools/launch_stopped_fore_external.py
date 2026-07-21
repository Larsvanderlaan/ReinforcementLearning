"""Supervise the sharded external-test stopped-FORE confirmatory run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Sequence


@dataclass
class ShardJob:
    """Mutable supervisor state for one resumable benchmark shard."""

    label: str
    config: Path
    output_dir: Path
    shard_index: int
    num_shards: int
    attempts: int = 0
    return_codes: list[int] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    log_handle: object | None = None
    started_unix: float | None = None
    finished_unix: float | None = None

    @property
    def key(self) -> str:
        return f"{self.label}_shard_{self.shard_index:02d}"


def supervise(
    *,
    configs: Sequence[tuple[str, Path]],
    output_root: Path,
    num_shards: int,
    max_workers: int,
    max_restarts: int,
) -> int:
    """Run shards with bounded parallelism, restart failures, then merge."""
    repository = Path(__file__).resolve().parents[1]
    package_root = repository / "packages" / "occupancy-ratio"
    output_root.mkdir(parents=True, exist_ok=True)
    # Longest-processing-time first keeps late long shards off the critical path.
    ordered_configs = sorted(
        configs,
        key=lambda item: _configured_training_size(item[1]),
        reverse=True,
    )
    jobs = [
        ShardJob(
            label=label,
            config=config.resolve(),
            output_dir=(output_root / label).resolve(),
            shard_index=shard_index,
            num_shards=num_shards,
        )
        for label, config in ordered_configs
        for shard_index in range(num_shards)
    ]
    pending = list(jobs)
    running: dict[str, ShardJob] = {}
    completed: dict[str, ShardJob] = {}
    failed: dict[str, ShardJob] = {}
    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status_path = output_root / "supervisor_status.json"
    started = time.time()
    while pending or running:
        while pending and len(running) < max_workers and not stop_requested:
            job = pending.pop(0)
            job.output_dir.mkdir(parents=True, exist_ok=True)
            logs = output_root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            job.log_handle = (logs / f"{job.key}.log").open("a")
            command = _run_command(job)
            job.attempts += 1
            job.started_unix = time.time()
            job.process = subprocess.Popen(
                command,
                cwd=repository,
                env=_worker_environment(package_root),
                stdout=job.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[job.key] = job

        for key, job in list(running.items()):
            assert job.process is not None
            return_code = job.process.poll()
            if return_code is None:
                continue
            job.return_codes.append(int(return_code))
            job.finished_unix = time.time()
            if job.log_handle is not None:
                job.log_handle.close()
                job.log_handle = None
            running.pop(key)
            if return_code == 0:
                completed[key] = job
            elif job.attempts <= max_restarts:
                pending.append(job)
            else:
                failed[key] = job

        _write_status(
            status_path,
            jobs=jobs,
            pending=pending,
            running=running,
            completed=completed,
            failed=failed,
            started=started,
            stop_requested=stop_requested,
        )
        if stop_requested:
            for job in running.values():
                assert job.process is not None
                job.process.terminate()
            for job in running.values():
                assert job.process is not None
                try:
                    job.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    job.process.kill()
            return 130
        if pending or running:
            time.sleep(2.0)

    if failed:
        _write_status(
            status_path,
            jobs=jobs,
            pending=pending,
            running=running,
            completed=completed,
            failed=failed,
            started=started,
            stop_requested=False,
            final_state="failed",
        )
        return 1

    merge_results: dict[str, int] = {}
    for label, config in configs:
        output_dir = output_root / label
        log_path = output_root / "logs" / f"merge_{label}.log"
        with log_path.open("a") as handle:
            result = subprocess.run(
                _merge_command(config=config, output_dir=output_dir),
                cwd=repository,
                env=_worker_environment(package_root),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        merge_results[label] = int(result.returncode)
    final_state = "complete" if all(code == 0 for code in merge_results.values()) else "merge_failed"
    _write_status(
        status_path,
        jobs=jobs,
        pending=pending,
        running=running,
        completed=completed,
        failed=failed,
        started=started,
        stop_requested=False,
        final_state=final_state,
        merge_results=merge_results,
    )
    return 0 if final_state == "complete" else 1


def _run_command(job: ShardJob) -> list[str]:
    return [
        sys.executable,
        "-m",
        "occupancy_ratio_benchmark.stopped_fore_external",
        "run",
        "--config",
        str(job.config),
        "--output-dir",
        str(job.output_dir),
        "--shard-index",
        str(job.shard_index),
        "--num-shards",
        str(job.num_shards),
        "--fail-fast",
    ]


def _merge_command(*, config: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "occupancy_ratio_benchmark.stopped_fore_external",
        "merge",
        "--config",
        str(config.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]


def _worker_environment(package_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(package_root) + (
        os.pathsep + prior_pythonpath if prior_pythonpath else ""
    )
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONFAULTHANDLER": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        }
    )
    return environment


def _configured_training_size(config: Path) -> int:
    payload = json.loads(config.read_text())
    return int(payload.get("n_train", 0))


def _write_status(
    path: Path,
    *,
    jobs: Sequence[ShardJob],
    pending: Sequence[ShardJob],
    running: dict[str, ShardJob],
    completed: dict[str, ShardJob],
    failed: dict[str, ShardJob],
    started: float,
    stop_requested: bool,
    final_state: str | None = None,
    merge_results: dict[str, int] | None = None,
) -> None:
    payload = {
        "schema": "stopped-fore-supervisor-v1",
        "state": final_state or ("stopping" if stop_requested else "running"),
        "started_unix": started,
        "updated_unix": time.time(),
        "elapsed_sec": time.time() - started,
        "total_jobs": len(jobs),
        "pending_jobs": [job.key for job in pending],
        "running_jobs": {
            key: {
                "pid": None if job.process is None else job.process.pid,
                "attempt": job.attempts,
                "started_unix": job.started_unix,
            }
            for key, job in running.items()
        },
        "completed_jobs": sorted(completed),
        "failed_jobs": {
            key: {
                "attempts": job.attempts,
                "return_codes": job.return_codes,
            }
            for key, job in failed.items()
        },
        "jobs": {
            job.key: {
                "config": str(job.config),
                "output_dir": str(job.output_dir),
                "attempts": job.attempts,
                "return_codes": job.return_codes,
                "started_unix": job.started_unix,
                "finished_unix": job.finished_unix,
            }
            for job in jobs
        },
        "merge_results": merge_results or {},
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    config_root = (
        repository
        / "packages"
        / "occupancy-ratio"
        / "occupancy_ratio_benchmark"
        / "configs"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument(
        "--config-n2000",
        default=str(config_root / "stopped_fore_external_confirmatory_n2000.json"),
    )
    parser.add_argument(
        "--config-n10000",
        default=str(config_root / "stopped_fore_external_confirmatory_n10000.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the overnight supervisor."""
    args = _build_parser().parse_args(argv)
    if args.num_shards <= 0 or args.max_workers <= 0 or args.max_restarts < 0:
        raise ValueError("shard, worker, and restart counts are invalid.")
    configs = [
        ("n2000", Path(args.config_n2000)),
        ("n10000", Path(args.config_n10000)),
    ]
    if any(not config.exists() for _, config in configs):
        raise FileNotFoundError("one or more stopped-FORE configs do not exist.")
    return supervise(
        configs=configs,
        output_root=Path(args.output_root).resolve(),
        num_shards=args.num_shards,
        max_workers=args.max_workers,
        max_restarts=args.max_restarts,
    )


if __name__ == "__main__":
    raise SystemExit(main())
