"""JSONL rollout log.

Append-only and flushed per record: a campaign that dies halfway (shared GPU,
OOM, someone killing a stray process) must leave every completed rollout
readable. Raw generations are kept in the log itself so scoring can be redone
without the model.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, RolloutRecord


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


class RolloutLogger:
    def __init__(self, path: str | Path, run_id: str, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._fh = open(self.path, "a", encoding="utf-8")
        self.n = 0
        self._write({
            "type": "run_meta",
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "started_at": time.time(),
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": platform.node(),
            "pid": os.getpid(),
            "git": _git_rev(),
            **(meta or {}),
        })

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def log(self, record: RolloutRecord) -> None:
        self._write(asdict(record))
        self.n += 1

    def note(self, **kwargs: Any) -> None:
        self._write({"type": "note", "run_id": self.run_id, "ts": time.time(), **kwargs})

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RolloutLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_rollouts(path: str | Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "rollout":
                out.append(obj)
    return out
