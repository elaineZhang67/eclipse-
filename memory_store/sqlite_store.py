import json
import os
import sqlite3
import uuid
from datetime import datetime


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class SurveillanceMemoryStore:
    def __init__(self, db_path):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    results_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def save_run(self, results, camera_id, video_path, run_id=None):
        run_id = str(run_id or uuid.uuid4().hex[:12])
        created_at = _utc_now()
        config_json = json.dumps(results.get("config", {}), ensure_ascii=True)
        results_json = json.dumps(results, ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, camera_id, video_path, created_at, config_json, results_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, camera_id, video_path, created_at, config_json, results_json),
            )
        return run_id

    def list_runs(self, camera_id=None, limit=50):
        query = """
            SELECT run_id, camera_id, video_path, created_at, config_json
            FROM runs
        """
        params = []
        if camera_id:
            query += " WHERE camera_id = ?"
            params.append(camera_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        items = []
        for row in rows:
            items.append(
                {
                    "run_id": row["run_id"],
                    "camera_id": row["camera_id"],
                    "video_path": row["video_path"],
                    "created_at": row["created_at"],
                    "config": json.loads(row["config_json"]),
                }
            )
        return items

    def load_run(self, run_id):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, camera_id, video_path, created_at, results_json
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "run_id": row["run_id"],
            "camera_id": row["camera_id"],
            "video_path": row["video_path"],
            "created_at": row["created_at"],
            "results": json.loads(row["results_json"]),
        }

    def load_runs(self, run_ids=None, camera_id=None, limit=50):
        if run_ids:
            bundles = []
            for run_id in run_ids:
                bundle = self.load_run(run_id)
                if bundle is not None:
                    bundles.append(bundle)
            return bundles

        items = self.list_runs(camera_id=camera_id, limit=limit)
        return [self.load_run(item["run_id"]) for item in items if item.get("run_id")]

    def append_message(self, session_id, role, content, metadata=None):
        message_id = uuid.uuid4().hex
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id, session_id, role, content, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    created_at,
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )
        return message_id

    def load_messages(self, session_id, limit=12):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at, metadata_json
                FROM (
                    SELECT role, content, created_at, metadata_json
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC
                """,
                (session_id, int(limit)),
            ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]
