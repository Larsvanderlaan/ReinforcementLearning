"""Reproducible artifact helpers for the clipped-coverage benchmark."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "clipped-coverage-v2"


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for configuration and identity hashing."""
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def configuration_hash(config: Any) -> str:
    """Hash a complete benchmark configuration."""
    value = asdict(config) if is_dataclass(config) else config
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def benchmark_cell_id(
    *,
    config: Any,
    repetition: int,
    requested_mass: float,
    backend: str,
    seed: int,
    fold: int,
) -> str:
    """Return a deterministic identifier for one fitted benchmark fold."""
    if int(fold) not in {0, 1}:
        raise ValueError("fold must be 0 or 1")
    payload = {
        "schema": SCHEMA_VERSION,
        "config_hash": configuration_hash(config),
        "repetition": int(repetition),
        "requested_mass": float(requested_mass),
        "backend": str(backend),
        "seed": int(seed),
        "fold": int(fold),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:24]


def write_cell(
    path: Path, *, cell_id: str, config_hash: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Atomically write one deterministic cell artifact."""
    payload = {
        "schema": SCHEMA_VERSION,
        "cell_id": cell_id,
        "config_hash": config_hash,
        "rows": _json_safe(list(rows)),
    }
    atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def read_cell(path: Path, *, cell_id: str, config_hash: str) -> list[dict[str, Any]]:
    """Load a compatible cell artifact or reject it explicitly."""
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"incompatible cell schema in {path}")
    if payload.get("cell_id") != cell_id:
        raise ValueError(f"cell id mismatch in {path}")
    if payload.get("config_hash") != config_hash:
        raise ValueError(f"configuration hash mismatch in {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid rows in {path}")
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    """Write text using an atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def repository_metadata(start: Path) -> dict[str, Any]:
    """Return revision and a content-sensitive dirty-tree hash."""
    start = Path(start)
    if start.is_file():
        start = start.parent
    root_text = _git(start, "rev-parse", "--show-toplevel", allow_failure=True)
    if not root_text:
        return {
            "git_root": None,
            "git_revision": None,
            "git_dirty": None,
            "git_diff_hash": None,
        }
    root = Path(root_text)
    revision = _git(root, "rev-parse", "HEAD")
    status = _git_bytes(root, "status", "--porcelain=v1", "-z")
    digest = hashlib.sha256(status)
    digest.update(_git_bytes(root, "diff", "--binary", "HEAD"))
    untracked = _git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z")
    for relative in sorted(item for item in untracked.split(b"\0") if item):
        digest.update(relative)
        path = root / os.fsdecode(relative)
        if path.is_file():
            digest.update(path.read_bytes())
    return {
        "git_root": str(root),
        "git_revision": revision,
        "git_dirty": bool(status),
        "git_diff_hash": digest.hexdigest() if status else None,
    }


def environment_metadata(*, include_torch: bool) -> dict[str, Any]:
    """Collect dependency and deterministic-backend metadata."""
    packages = {}
    for name in ("numpy", "occupancy-ratio", "torch", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    torch_metadata: dict[str, Any] = {"requested": bool(include_torch)}
    if include_torch:
        try:
            import torch

            torch_metadata.update(
                {
                    "version": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_version": torch.version.cuda,
                    "deterministic_algorithms": bool(
                        torch.are_deterministic_algorithms_enabled()
                    ),
                }
            )
        except Exception as exc:
            torch_metadata["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch": torch_metadata,
    }


def _git(start: Path, *args: str, allow_failure: bool = False) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=start,
            check=not allow_failure,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _git_bytes(start: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=start,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return b""


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "SCHEMA_VERSION",
    "atomic_write_text",
    "benchmark_cell_id",
    "canonical_json",
    "configuration_hash",
    "environment_metadata",
    "read_cell",
    "repository_metadata",
    "write_cell",
]
