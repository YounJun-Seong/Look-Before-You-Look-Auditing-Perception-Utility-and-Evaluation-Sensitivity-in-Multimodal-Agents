"""Generation cache.

Gate 1 is ~22k generations and will be re-analyzed many times; re-scoring must
never require the GPU again. The cache key covers everything that could change
a generation (model, prompt version, decode config, served question, action,
view budgets), so a prompt edit produces a miss instead of silently reusing an
answer produced under different conditions.

SQLite with WAL so several runner shards can share one cache file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schema import config_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    key         TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    text        TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    latency_s   REAL DEFAULT 0.0,
    created_at  REAL DEFAULT (strftime('%s','now'))
);
"""


@dataclass
class CachedGeneration:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class GenerationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(payload: dict) -> str:
        return config_hash(payload)

    def get(self, key: str) -> Optional[CachedGeneration]:
        with self._lock:
            row = self._conn.execute(
                "SELECT text, prompt_tokens, completion_tokens, latency_s "
                "FROM generations WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return CachedGeneration(row[0], row[1], row[2], row[3])

    def put(self, key: str, model_id: str, payload: dict, gen) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO generations "
                "(key, model_id, payload, text, prompt_tokens, completion_tokens, latency_s) "
                "VALUES (?,?,?,?,?,?,?)",
                (key, model_id, json.dumps(payload, sort_keys=True, ensure_ascii=False),
                 gen.text, gen.prompt_tokens, gen.completion_tokens, gen.latency_s),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
