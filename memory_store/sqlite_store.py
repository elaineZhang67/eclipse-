import json
import os
import sqlite3
import uuid
from datetime import datetime


_UNSET = object()


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
                    video_id TEXT,
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    label TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    job_id TEXT PRIMARY KEY,
                    video_id TEXT,
                    camera_id TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    run_id TEXT,
                    error TEXT,
                    config_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    video_id TEXT,
                    job_id TEXT,
                    run_id TEXT,
                    camera_id TEXT NOT NULL,
                    video_path TEXT,
                    label TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "runs", "video_id", "TEXT")

    def _table_columns(self, conn, table_name):
        rows = conn.execute("PRAGMA table_info({table_name})".format(table_name=table_name)).fetchall()
        return {row["name"] for row in rows}

    def _ensure_column(self, conn, table_name, column_name, column_type):
        if column_name in self._table_columns(conn, table_name):
            return
        conn.execute(
            "ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}".format(
                table_name=table_name,
                column_name=column_name,
                column_type=column_type,
            )
        )

    def register_video(self, video_path, camera_id, video_id=None, label=None, metadata=None):
        video_id = str(video_id or uuid.uuid4().hex[:12])
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO videos (
                    video_id, camera_id, video_path, label, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    camera_id,
                    video_path,
                    label,
                    created_at,
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )
        return video_id

    def list_videos(self, camera_id=None, limit=100):
        query = """
            SELECT video_id, camera_id, video_path, label, created_at, metadata_json
            FROM videos
        """
        params = []
        if camera_id:
            query += " WHERE camera_id = ?"
            params.append(camera_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "video_id": row["video_id"],
                "camera_id": row["camera_id"],
                "video_path": row["video_path"],
                "label": row["label"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def load_video(self, video_id):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT video_id, camera_id, video_path, label, created_at, metadata_json
                FROM videos
                WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "video_id": row["video_id"],
            "camera_id": row["camera_id"],
            "video_path": row["video_path"],
            "label": row["label"],
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"]),
        }

    def create_processing_job(self, video_path, camera_id, video_id=None, config=None, job_id=None):
        job_id = str(job_id or uuid.uuid4().hex[:12])
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processing_jobs (
                    job_id, video_id, camera_id, video_path, status, created_at, updated_at,
                    run_id, error, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    video_id,
                    camera_id,
                    video_path,
                    "queued",
                    created_at,
                    created_at,
                    None,
                    None,
                    json.dumps(config or {}, ensure_ascii=True),
                ),
            )
        return job_id

    def update_processing_job(self, job_id, status=None, run_id=None, error=None, config=None):
        existing = self.load_processing_job(job_id)
        if existing is None:
            return None

        next_status = status if status is not None else existing["status"]
        next_run_id = run_id if run_id is not None else existing.get("run_id")
        next_error = error if error is not None else existing.get("error")
        next_config = config if config is not None else existing.get("config", {})
        updated_at = _utc_now()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE processing_jobs
                SET status = ?, updated_at = ?, run_id = ?, error = ?, config_json = ?
                WHERE job_id = ?
                """,
                (
                    next_status,
                    updated_at,
                    next_run_id,
                    next_error,
                    json.dumps(next_config, ensure_ascii=True),
                    job_id,
                ),
            )
        return self.load_processing_job(job_id)

    def load_processing_job(self, job_id):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, video_id, camera_id, video_path, status, created_at, updated_at,
                       run_id, error, config_json
                FROM processing_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "video_id": row["video_id"],
            "camera_id": row["camera_id"],
            "video_path": row["video_path"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "run_id": row["run_id"],
            "error": row["error"],
            "config": json.loads(row["config_json"]),
        }

    def list_processing_jobs(self, camera_id=None, video_id=None, limit=50):
        query = """
            SELECT job_id, video_id, camera_id, video_path, status, created_at, updated_at,
                   run_id, error, config_json
            FROM processing_jobs
        """
        filters = []
        params = []
        if camera_id:
            filters.append("camera_id = ?")
            params.append(camera_id)
        if video_id:
            filters.append("video_id = ?")
            params.append(video_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "job_id": row["job_id"],
                "video_id": row["video_id"],
                "camera_id": row["camera_id"],
                "video_path": row["video_path"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "run_id": row["run_id"],
                "error": row["error"],
                "config": json.loads(row["config_json"]),
            }
            for row in rows
        ]

    def save_run(self, results, camera_id, video_path, run_id=None, video_id=None):
        run_id = str(run_id or uuid.uuid4().hex[:12])
        created_at = _utc_now()
        config_json = json.dumps(results.get("config", {}), ensure_ascii=True)
        results_json = json.dumps(results, ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, video_id, camera_id, video_path, created_at, config_json, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, video_id, camera_id, video_path, created_at, config_json, results_json),
            )
        return run_id

    def list_runs(self, camera_id=None, video_id=None, limit=50):
        query = """
            SELECT run_id, video_id, camera_id, video_path, created_at, config_json
            FROM runs
        """
        filters = []
        params = []
        if camera_id:
            filters.append("camera_id = ?")
            params.append(camera_id)
        if video_id:
            filters.append("video_id = ?")
            params.append(video_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        items = []
        for row in rows:
            items.append(
                {
                    "run_id": row["run_id"],
                    "video_id": row["video_id"],
                    "camera_id": row["camera_id"],
                    "video_path": row["video_path"],
                    "created_at": row["created_at"],
                    "config": json.loads(row["config_json"]),
                }
            )
        return items

    def list_cameras(self):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT camera_id, COUNT(*) AS run_count, MAX(created_at) AS latest_created_at
                FROM runs
                GROUP BY camera_id
                ORDER BY latest_created_at DESC
                """
            ).fetchall()

        return [
            {
                "camera_id": row["camera_id"],
                "run_count": int(row["run_count"]),
                "latest_created_at": row["latest_created_at"],
            }
            for row in rows
        ]

    def load_run(self, run_id):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, video_id, camera_id, video_path, created_at, results_json
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "run_id": row["run_id"],
            "video_id": row["video_id"],
            "camera_id": row["camera_id"],
            "video_path": row["video_path"],
            "created_at": row["created_at"],
            "results": json.loads(row["results_json"]),
        }

    def load_runs(self, run_ids=None, camera_id=None, video_id=None, limit=50):
        if run_ids:
            bundles = []
            for run_id in run_ids:
                bundle = self.load_run(run_id)
                if bundle is not None:
                    bundles.append(bundle)
            return bundles

        items = self.list_runs(camera_id=camera_id, video_id=video_id, limit=limit)
        return [self.load_run(item["run_id"]) for item in items if item.get("run_id")]

    def create_chat_session(
        self,
        camera_id,
        session_id=None,
        video_id=None,
        job_id=None,
        run_id=None,
        video_path=None,
        label=None,
        status="uploaded",
        error=None,
        metadata=None,
    ):
        session_id = str(session_id or uuid.uuid4().hex[:12])
        created_at = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (
                    session_id, video_id, job_id, run_id, camera_id, video_path, label,
                    status, created_at, updated_at, error, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    video_id,
                    job_id,
                    run_id,
                    camera_id,
                    video_path,
                    label,
                    status,
                    created_at,
                    created_at,
                    error,
                    json.dumps(metadata or {}, ensure_ascii=True),
                ),
            )
        return session_id

    def update_chat_session(
        self,
        session_id,
        video_id=_UNSET,
        job_id=_UNSET,
        run_id=_UNSET,
        camera_id=_UNSET,
        video_path=_UNSET,
        label=_UNSET,
        status=_UNSET,
        error=_UNSET,
        metadata=_UNSET,
    ):
        existing = self.load_chat_session(session_id)
        if existing is None:
            return None

        next_metadata = existing.get("metadata", {})
        if metadata is not _UNSET:
            next_metadata = dict(next_metadata)
            next_metadata.update(metadata or {})

        values = {
            "video_id": existing.get("video_id") if video_id is _UNSET else video_id,
            "job_id": existing.get("job_id") if job_id is _UNSET else job_id,
            "run_id": existing.get("run_id") if run_id is _UNSET else run_id,
            "camera_id": existing.get("camera_id") if camera_id is _UNSET else camera_id,
            "video_path": existing.get("video_path") if video_path is _UNSET else video_path,
            "label": existing.get("label") if label is _UNSET else label,
            "status": existing.get("status") if status is _UNSET else status,
            "error": existing.get("error") if error is _UNSET else error,
            "metadata": next_metadata,
        }
        updated_at = _utc_now()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET video_id = ?, job_id = ?, run_id = ?, camera_id = ?, video_path = ?,
                    label = ?, status = ?, updated_at = ?, error = ?, metadata_json = ?
                WHERE session_id = ?
                """,
                (
                    values["video_id"],
                    values["job_id"],
                    values["run_id"],
                    values["camera_id"],
                    values["video_path"],
                    values["label"],
                    values["status"],
                    updated_at,
                    values["error"],
                    json.dumps(values["metadata"], ensure_ascii=True),
                    session_id,
                ),
            )
        return self.load_chat_session(session_id)

    def load_chat_session(self, session_id):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, video_id, job_id, run_id, camera_id, video_path,
                       label, status, created_at, updated_at, error, metadata_json
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row["session_id"],
            "video_id": row["video_id"],
            "job_id": row["job_id"],
            "run_id": row["run_id"],
            "camera_id": row["camera_id"],
            "video_path": row["video_path"],
            "label": row["label"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "error": row["error"],
            "metadata": json.loads(row["metadata_json"]),
        }

    def list_chat_sessions(self, limit=50):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, video_id, job_id, run_id, camera_id, video_path,
                       label, status, created_at, updated_at, error, metadata_json
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "video_id": row["video_id"],
                "job_id": row["job_id"],
                "run_id": row["run_id"],
                "camera_id": row["camera_id"],
                "video_path": row["video_path"],
                "label": row["label"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "error": row["error"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

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
            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (created_at, session_id),
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
