"""Request metrics store (SQLite) + in-process pub/sub for live dashboard updates.

stdlib only. One row per request, safe for concurrent use (one lock, WAL mode).
Every number stored here comes from a measurement or from Ollama's final event.
Unknown values are stored as NULL and shown as N/A - never invented.
"""
from __future__ import annotations

import asyncio
import csv
import io
import math
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

DB_PATH = os.environ.get("HORIZON_METRICS_DB", "metrics.db")

COLUMNS = [
    "request_id", "created_at", "prompt_preview", "model", "mode", "backend", "policy",
    "profile", "status", "latency_s", "ttft_s", "prompt_tokens", "output_tokens",
    "tokens_per_second", "prompt_eval_s", "load_s", "ollama_total_s", "retries", "error",
    "estimated_input_tokens", "max_output_tokens", "routing_score", "routing_explanation",
    "fallback_occurred", "queue_wait_s",
]
FILTERS = ("model", "mode", "backend", "policy", "status")
_IMMUTABLE = {"request_id", "created_at"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  request_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, prompt_preview TEXT, model TEXT,
  mode TEXT NOT NULL, backend TEXT, policy TEXT, profile TEXT, status TEXT NOT NULL,
  latency_s REAL, ttft_s REAL, prompt_tokens INTEGER, output_tokens INTEGER,
  tokens_per_second REAL, prompt_eval_s REAL, load_s REAL, ollama_total_s REAL,
  retries INTEGER DEFAULT 0, error TEXT,
  estimated_input_tokens INTEGER, max_output_tokens INTEGER,
  routing_score REAL, routing_explanation TEXT,
  fallback_occurred INTEGER DEFAULT 0, queue_wait_s REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);

CREATE TABLE IF NOT EXISTS benchmark_runs (
  run_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  baseline_policy TEXT NOT NULL,
  adaptive_policy TEXT NOT NULL,
  mode TEXT NOT NULL,
  workload_preset TEXT NOT NULL,
  request_count INTEGER NOT NULL,
  concurrency INTEGER NOT NULL,
  warmup_count INTEGER NOT NULL,
  summary_json TEXT NOT NULL,
  raw_results_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bench_created ON benchmark_runs(created_at);
"""


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    """Nearest-rank percentile (p in 0..100). None for an empty sample."""
    s = sorted(values)
    if not s:
        return None
    return s[max(1, math.ceil(p / 100.0 * len(s))) - 1]


def _vals(rows: list, key: str) -> list:
    return [r[key] for r in rows if r.get(key) is not None]


def _avg(v: list) -> Optional[float]:
    return sum(v) / len(v) if v else None


def aggregate(rows: list) -> dict:
    """Aggregate finished rows. Averages use SUCCESS rows only; failures are counted separately.

    Each average states its own sample size (n_*), because TTFT / throughput may be
    unavailable for some successful requests. Percentiles: nearest-rank.
    """
    rows = [dict(r) for r in rows]
    ok = [r for r in rows if r.get("status") == "SUCCESS"]
    bad = [r for r in rows if r.get("status") == "FAILED"]
    lat, ttft = _vals(ok, "latency_s"), _vals(ok, "ttft_s")
    out, tps = _vals(ok, "output_tokens"), _vals(ok, "tokens_per_second")
    finished = len(ok) + len(bad)
    return {
        "total": finished, "success": len(ok), "failed": len(bad),
        "success_rate": (len(ok) / finished) if finished else None,
        "avg_latency_s": _avg(lat), "median_latency_s": percentile(lat, 50),
        "p95_latency_s": percentile(lat, 95), "n_latency": len(lat),
        "avg_ttft_s": _avg(ttft), "n_ttft": len(ttft),
        "avg_output_tokens": _avg(out), "n_output_tokens": len(out),
        "avg_tokens_per_second": _avg(tps), "n_tokens_per_second": len(tps),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MetricsStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or DB_PATH
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            # Migrate existing tables if columns are missing
            cur = self._db.execute("PRAGMA table_info(requests)")
            existing_cols = {r[1] for r in cur.fetchall()}
            for col_name, col_type in [
                ("estimated_input_tokens", "INTEGER"),
                ("max_output_tokens", "INTEGER"),
                ("routing_score", "REAL"),
                ("routing_explanation", "TEXT"),
                ("fallback_occurred", "INTEGER DEFAULT 0"),
                ("queue_wait_s", "REAL"),
            ]:
                if col_name not in existing_cols:
                    try:
                        self._db.execute(f"ALTER TABLE requests ADD COLUMN {col_name} {col_type}")
                    except Exception:
                        pass
            # Requests still IN_PROGRESS at startup belonged to a previous process: they can never finish.
            self._db.execute(
                "UPDATE requests SET status='FAILED', error=? WHERE status='IN_PROGRESS'",
                ("interrupted: gateway restarted before the request finished",),
            )
            self._db.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._db.close()
            except Exception:
                pass

    # ---- writes -------------------------------------------------------------------
    def start(self, request_id: str, *, prompt: str, model: Optional[str], mode: str,
              backend: Optional[str] = None, policy: Optional[str] = None,
              profile: Optional[str] = None, **extra: Any) -> dict:
        flat = " ".join((prompt or "").split())
        preview = flat if len(flat) <= 120 else flat[:119] + "\u2026"
        row = {"request_id": request_id or uuid.uuid4().hex, "created_at": _now(),
               "prompt_preview": preview, "model": model, "mode": mode, "backend": backend,
               "policy": policy, "profile": profile, "status": "IN_PROGRESS", "retries": 0}
        for k, v in extra.items():
            if k in COLUMNS and k not in _IMMUTABLE:
                row[k] = v
        with self._lock:
            try:
                self._insert(row)
            except sqlite3.IntegrityError:  # caller reused an id: keep it recognisable but unique
                row["request_id"] = f"{row['request_id']}-{uuid.uuid4().hex[:6]}"
                self._insert(row)
            self._db.commit()
        return row

    def _insert(self, row: dict) -> None:
        cols = list(row)
        self._db.execute(
            f"INSERT INTO requests ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols],
        )

    def finish(self, request_id: str, *, status: str, **fields: Any) -> Optional[dict]:
        """Complete a request. Idempotent: only the first call for an id changes the row,
        so a request can never be counted twice. Returns the final row, or None if ignored."""
        sets, vals = ["status=?"], [status]
        for k, v in fields.items():
            if k in COLUMNS and k not in _IMMUTABLE:
                sets.append(f"{k}=?")
                vals.append(v)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE requests SET {', '.join(sets)} WHERE request_id=? AND status='IN_PROGRESS'",
                vals + [request_id],
            )
            self._db.commit()
            if cur.rowcount == 0:
                return None
            return dict(self._db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone())

    # ---- reads --------------------------------------------------------------------
    def _all(self) -> list:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM requests ORDER BY created_at, rowid")]

    def recent(self, limit: int = 50, **filters: Any) -> list:
        where, vals = [], []
        for k in FILTERS:
            if filters.get(k):
                where.append(f"{k}=?")
                vals.append(filters[k])
        sql = "SELECT * FROM requests" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, vals + [max(1, min(int(limit), 1000))])]

    def summary(self) -> dict:
        rows = self._all()
        finished = [r for r in rows if r["status"] != "IN_PROGRESS"]
        overall = aggregate(finished)
        return {
            "total": overall["total"], "success": overall["success"], "failed": overall["failed"],
            "success_rate": overall["success_rate"],
            "in_progress": sum(1 for r in rows if r["status"] == "IN_PROGRESS"),
            "latest": finished[-1] if finished else None,
            # REAL and SIMULATED are never blended into one average.
            "by_mode": {m: aggregate([r for r in finished if r["mode"] == m]) for m in ("REAL", "SIMULATED")},
            "percentile_method": "nearest-rank",
        }

    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(COLUMNS)
        for r in self._all():
            w.writerow(["" if r[c] is None else r[c] for c in COLUMNS])
        return buf.getvalue()

    # ---- benchmark persistence ---------------------------------------------------
    def save_benchmark_run(self, run_data: dict) -> dict:
        import json
        row = {
            "run_id": run_data["run_id"],
            "created_at": run_data.get("created_at") or _now(),
            "baseline_policy": run_data["baseline_policy"],
            "adaptive_policy": run_data["adaptive_policy"],
            "mode": run_data.get("mode", "SIMULATED"),
            "workload_preset": run_data.get("workload_preset", "balanced"),
            "request_count": int(run_data.get("request_count", 0)),
            "concurrency": int(run_data.get("concurrency", 1)),
            "warmup_count": int(run_data.get("warmup_count", 0)),
            "summary_json": json.dumps(run_data.get("summary", {}), default=str),
            "raw_results_json": json.dumps(run_data.get("raw_results", []), default=str),
        }
        with self._lock:
            cols = list(row)
            self._db.execute(
                f"INSERT OR REPLACE INTO benchmark_runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [row[c] for c in cols],
            )
            self._db.commit()
        return row

    def get_benchmark_runs(self, limit: int = 50) -> list:
        import json
        with self._lock:
            rows = self._db.execute(
                "SELECT run_id, created_at, baseline_policy, adaptive_policy, mode, workload_preset, "
                "request_count, concurrency, warmup_count, summary_json FROM benchmark_runs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),)
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["summary"] = json.loads(d.pop("summary_json", "{}"))
                except Exception:
                    d["summary"] = {}
                result.append(d)
            return result

    def get_benchmark_run(self, run_id: str) -> Optional[dict]:
        import json
        with self._lock:
            r = self._db.execute("SELECT * FROM benchmark_runs WHERE run_id=?", (run_id,)).fetchone()
            if not r:
                return None
            d = dict(r)
            try:
                d["summary"] = json.loads(d.pop("summary_json", "{}"))
            except Exception:
                d["summary"] = {}
            try:
                d["raw_results"] = json.loads(d.pop("raw_results_json", "[]"))
            except Exception:
                d["raw_results"] = []
            return d

    def export_benchmark_csv(self, run_id: str) -> Optional[str]:
        run = self.get_benchmark_run(run_id)
        if not run or not run.get("raw_results"):
            return None
        buf = io.StringIO()
        raw = run["raw_results"]
        if raw:
            keys = list(raw[0].keys())
            w = csv.DictWriter(buf, fieldnames=keys)
            w.writeheader()
            w.writerows(raw)
        return buf.getvalue()



class MetricsHub:
    """Fan-out of 'something changed' events to SSE subscribers.

    Call publish() from the event-loop thread (instrument_stream does)."""

    def __init__(self) -> None:
        self._subs: set = set()

    def subscribe(self, maxsize: int = 100) -> "asyncio.Queue":
        q: asyncio.Queue = asyncio.Queue(maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue") -> None:
        self._subs.discard(q)

    def publish(self, event: dict) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # slow viewer; the dashboard also re-polls
                pass
