import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.schemas import ChatAskRequest, ChatSessionResponse, PipelineOptions, ProcessVideoRequest
from backend.services import ChatSessionService, PipelineJobService
from backend.settings import DEFAULT_MEMORY_DB, VIDEO_STORAGE_DIR
from memory_store.sqlite_store import SurveillanceMemoryStore


router = APIRouter(prefix="/chat", tags=["chat"])


def get_store():
    return SurveillanceMemoryStore(DEFAULT_MEMORY_DB)


def get_chat_service(store=Depends(get_store)):
    return ChatSessionService(store)


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _parse_pipeline_options(options_json, device, summary_backend, llm_model):
    payload = {}
    if options_json:
        try:
            payload = json.loads(options_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="options_json must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="options_json must be a JSON object")

    if device:
        payload["device"] = device
    if summary_backend:
        payload["summary_backend"] = summary_backend
    if llm_model:
        payload["llm_model"] = llm_model
    try:
        return PipelineOptions(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _start_worker(session_id, job_id):
    repo_root = Path(__file__).resolve().parents[1]
    log_dir = Path(
        os.environ.get(
            "SURVEILLANCE_WORKER_LOG_DIR",
            str(Path(DEFAULT_MEMORY_DB).resolve().parent / "worker_logs"),
        )
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "{session_id}.log".format(session_id=session_id)
    env = dict(os.environ)
    env["SURVEILLANCE_MEMORY_DB"] = DEFAULT_MEMORY_DB
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "backend.worker",
                "--db",
                DEFAULT_MEMORY_DB,
                "--session-id",
                session_id,
                "--job-id",
                job_id,
            ],
            cwd=str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid, str(log_path)


@router.post("/upload", response_model=ChatSessionResponse)
def upload_for_chat(
    file: UploadFile = File(...),
    camera_id: str = Form("camera_1"),
    label: str | None = Form(None),
    options_json: str | None = Form(None),
    device: str | None = Form(None),
    summary_backend: str | None = Form(None),
    llm_model: str | None = Form(None),
    store=Depends(get_store),
    chat_service=Depends(get_chat_service),
):
    options = _parse_pipeline_options(options_json, device, summary_backend, llm_model)
    session_id = uuid.uuid4().hex[:12]
    video_id = uuid.uuid4().hex[:12]
    expected_run_id = uuid.uuid4().hex[:12]

    original_name = os.path.basename(file.filename or "video.mp4")
    suffix = Path(original_name).suffix or ".mp4"
    storage_dir = Path(VIDEO_STORAGE_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)
    output_path = storage_dir / "{video_id}{suffix}".format(video_id=video_id, suffix=suffix)

    with output_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    display_label = label or original_name
    store.register_video(
        video_path=str(output_path),
        camera_id=camera_id,
        video_id=video_id,
        label=display_label,
        metadata={
            "source": "chat_upload",
            "original_filename": original_name,
            "session_id": session_id,
        },
    )
    store.create_chat_session(
        session_id=session_id,
        camera_id=camera_id,
        video_id=video_id,
        video_path=str(output_path),
        label=display_label,
        status="uploaded",
        metadata={
            "original_filename": original_name,
            "pipeline_options": _model_dump(options),
            "expected_run_id": expected_run_id,
        },
    )

    try:
        job_id = PipelineJobService(store).create_job(
            ProcessVideoRequest(
                video_id=video_id,
                camera_id=camera_id,
                run_id=expected_run_id,
                options=options,
            )
        )
        store.update_chat_session(session_id, job_id=job_id, status="queued")
        worker_pid, worker_log = _start_worker(session_id, job_id)
        store.update_chat_session(
            session_id,
            metadata={
                "worker_pid": worker_pid,
                "worker_log": worker_log,
            },
        )
    except Exception as exc:
        store.update_chat_session(session_id, status="failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return chat_service.load_session(session_id)


@router.get("/sessions", response_model=list[ChatSessionResponse])
def list_sessions(limit: int = 50, chat_service=Depends(get_chat_service)):
    return chat_service.list_sessions(limit=limit)


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(session_id: str, chat_service=Depends(get_chat_service)):
    session = chat_service.load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return session


@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    limit: int = 100,
    chat_service=Depends(get_chat_service),
):
    if chat_service.load_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return {"messages": chat_service.load_messages(session_id, limit=limit)}


@router.post("/sessions/{session_id}/ask")
def ask_session(
    session_id: str,
    request: ChatAskRequest,
    chat_service=Depends(get_chat_service),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    if request.end_sec is not None and request.start_sec is not None and request.end_sec <= request.start_sec:
        raise HTTPException(status_code=400, detail="end_sec must be greater than start_sec")
    try:
        return chat_service.ask(session_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        session = chat_service.load_session(session_id)
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "session": session,
            },
        ) from exc
