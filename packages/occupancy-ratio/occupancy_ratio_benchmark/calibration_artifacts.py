"""Content-addressed atomic artifacts for calibration compute units."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


Array = np.ndarray
ARTIFACT_SCHEMA_VERSION = 1


class ArtifactConflictError(ValueError):
    """Raised when resume encounters a modified or incompatible artifact."""


def write_unit_artifact(
    *,
    run_root: str | Path,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    arrays: Mapping[str, Array],
    metadata: Mapping[str, Any],
    attempt: int,
    started_at: float,
) -> dict[str, Any]:
    """Write one successful unit artifact or accept an identical resume."""

    json_path = expected_artifact_path(run_root, unit)
    npz_path = json_path.with_suffix(".npz")
    array_payload = {str(name): np.asarray(value) for name, value in arrays.items()}
    _validate_array_payload(array_payload)
    if json_path.exists():
        existing = read_unit_artifact(run_root=run_root, manifest=manifest, unit=unit)
        existing_arrays = read_unit_arrays(run_root=run_root, manifest=manifest, unit=unit)
        if _arrays_equal(existing_arrays, array_payload):
            return existing
        raise ArtifactConflictError(f"refusing to overwrite incompatible artifact {json_path}")
    _atomic_write_npz(npz_path, array_payload)
    array_sha = file_sha256(npz_path)
    identity = _mapping(unit, "identity")
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": str(manifest["run_id"]),
        "unit_id": str(unit["unit_id"]),
        "kind": str(identity.get("kind", "unknown")),
        "status": "ok",
        "config_sha256": str(identity["config_sha256"]),
        "provenance_fingerprint": str(_mapping(manifest, "provenance")["fingerprint"]),
        "manifest_payload_sha256": str(manifest["manifest_payload_sha256"]),
        "array_file": npz_path.name,
        "array_sha256": array_sha,
        "array_names": sorted(array_payload),
        "metadata": _json_safe(metadata),
        "attempt": int(attempt),
        "started_at_unix": float(started_at),
        "finished_at_unix": float(time.time()),
        "runtime_sec": max(0.0, float(time.time() - started_at)),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    _atomic_write_json(json_path, payload)
    return payload


def read_unit_artifact(
    *,
    run_root: str | Path,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and validate all resume identity and payload digests."""

    json_path = expected_artifact_path(run_root, unit)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactConflictError(f"cannot read artifact {json_path}: {error}") from error
    identity = _mapping(unit, "identity")
    expected = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": str(manifest["run_id"]),
        "unit_id": str(unit["unit_id"]),
        "status": "ok",
        "config_sha256": str(identity["config_sha256"]),
        "provenance_fingerprint": str(_mapping(manifest, "provenance")["fingerprint"]),
        "manifest_payload_sha256": str(manifest["manifest_payload_sha256"]),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ArtifactConflictError(f"artifact {json_path} has mismatched {key}")
    stored_digest = payload.get("payload_sha256")
    without_digest = dict(payload)
    without_digest.pop("payload_sha256", None)
    if stored_digest != payload_sha256(without_digest):
        raise ArtifactConflictError(f"artifact metadata digest mismatch: {json_path}")
    npz_path = json_path.with_suffix(".npz")
    if payload.get("array_file") != npz_path.name:
        raise ArtifactConflictError(f"artifact array filename mismatch: {json_path}")
    if payload.get("array_sha256") != file_sha256(npz_path):
        raise ArtifactConflictError(f"artifact array digest mismatch: {npz_path}")
    return payload


def read_unit_arrays(
    *,
    run_root: str | Path,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> dict[str, Array]:
    payload = read_unit_artifact(run_root=run_root, manifest=manifest, unit=unit)
    path = expected_artifact_path(run_root, unit).with_suffix(".npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if sorted(arrays) != payload["array_names"]:
        raise ArtifactConflictError(f"artifact array member mismatch: {path}")
    return arrays


def artifact_is_ready(
    *,
    run_root: str | Path,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> bool:
    path = expected_artifact_path(run_root, unit)
    if not path.exists():
        return False
    read_unit_artifact(run_root=run_root, manifest=manifest, unit=unit)
    return True


def write_unit_state(
    *,
    run_root: str | Path,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    status: str,
    attempt: int,
    failure_type: str | None = None,
    error: str | None = None,
    started_at: float | None = None,
) -> Path:
    """Atomically update pending/running/failed state or a heartbeat."""

    if status not in {"pending", "running", "failed", "ok"}:
        raise ValueError("invalid unit status")
    path = Path(run_root) / "unit_state" / f"{unit['unit_id']}.json"
    payload = {
        "schema_version": 1,
        "run_id": str(manifest["run_id"]),
        "unit_id": str(unit["unit_id"]),
        "status": status,
        "attempt": int(attempt),
        "failure_type": failure_type,
        "error": error,
        "started_at_unix": started_at,
        "heartbeat_at_unix": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "provenance_fingerprint": str(_mapping(manifest, "provenance")["fingerprint"]),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    _atomic_write_json(path, payload)
    return path


def quarantine_stale_state(
    *,
    run_root: str | Path,
    unit_id: str,
    stale_after_sec: float,
    now: float | None = None,
) -> Path | None:
    """Move, never delete, a stale running-state record."""

    state = Path(run_root) / "unit_state" / f"{unit_id}.json"
    if not state.exists():
        return None
    try:
        payload = json.loads(state.read_text(encoding="utf-8"))
        heartbeat = float(payload["heartbeat_at_unix"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        heartbeat = 0.0
    current = time.time() if now is None else float(now)
    if current - heartbeat <= float(stale_after_sec):
        return None
    quarantine = Path(run_root) / "quarantine" / "stale_state"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / f"{unit_id}.{int(current)}.{os.getpid()}.json"
    os.replace(state, destination)
    return destination


def expected_artifact_path(run_root: str | Path, unit: Mapping[str, Any]) -> Path:
    relative = unit.get("expected_artifact")
    if not isinstance(relative, str) or not relative.endswith(".json"):
        raise ValueError("unit expected_artifact must be a relative JSON path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unit expected_artifact must stay inside the run root")
    return Path(run_root) / path


def payload_sha256(payload: Mapping[str, Any]) -> str:
    safe = _json_safe(payload)
    return hashlib.sha256(
        json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_array_payload(arrays: Mapping[str, Array]) -> None:
    for name, value in arrays.items():
        if not name or "/" in name:
            raise ValueError("array artifact names must be nonempty flat strings")
        if np.asarray(value).dtype == object:
            raise ValueError(f"object array {name!r} is forbidden")


def _arrays_equal(left: Mapping[str, Array], right: Mapping[str, Array]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        np.asarray(left[name]).dtype == np.asarray(right[name]).dtype
        and np.array_equal(np.asarray(left[name]), np.asarray(right[name]), equal_nan=True)
        for name in left
    )


def _atomic_write_npz(path: Path, arrays: Mapping[str, Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return None
        if np.isposinf(number):
            return "+inf"
        if np.isneginf(number):
            return "-inf"
        return number
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


__all__: Sequence[str] = (
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactConflictError",
    "artifact_is_ready",
    "expected_artifact_path",
    "file_sha256",
    "payload_sha256",
    "quarantine_stale_state",
    "read_unit_arrays",
    "read_unit_artifact",
    "write_unit_artifact",
    "write_unit_state",
)
