import argparse

from backend.services import PipelineJobService
from memory_store.sqlite_store import SurveillanceMemoryStore


def run_session_job(db_path, session_id, job_id):
    store = SurveillanceMemoryStore(db_path)
    store.update_chat_session(session_id, status="running", error=None)
    try:
        run_id = PipelineJobService(store).run_job(job_id)
    except Exception as exc:
        store.update_chat_session(session_id, status="failed", error=str(exc))
        raise
    store.update_chat_session(session_id, status="completed", run_id=run_id, error=None)
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Run a queued pipeline job for one chat session.")
    parser.add_argument("--db", required=True, help="SQLite memory database path.")
    parser.add_argument("--session-id", required=True, help="Chat session id to update.")
    parser.add_argument("--job-id", required=True, help="Processing job id to run.")
    args = parser.parse_args()
    run_session_job(args.db, args.session_id, args.job_id)


if __name__ == "__main__":
    main()
