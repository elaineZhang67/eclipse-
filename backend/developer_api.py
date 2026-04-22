import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi import File, Form, UploadFile

from backend.schemas import ProcessVideoRequest, RegisterVideoRequest
from backend.services import PipelineJobService
from backend.settings import DEFAULT_MEMORY_DB, VIDEO_STORAGE_DIR
from memory_store.sqlite_store import SurveillanceMemoryStore


router = APIRouter(prefix="/developer", tags=["developer"])


def get_store():
    return SurveillanceMemoryStore(DEFAULT_MEMORY_DB)


def get_pipeline_service(store=Depends(get_store)):
    return PipelineJobService(store)


@router.post("/videos")
def register_video(request: RegisterVideoRequest, store=Depends(get_store)):
    video_id = store.register_video(
        video_path=request.video_path,
        camera_id=request.camera_id,
        video_id=request.video_id,
        label=request.label,
        metadata=request.metadata,
    )
    return store.load_video(video_id)


@router.post("/videos/upload")
def upload_video(
    file: UploadFile = File(...),
    camera_id: str = Form("camera_1"),
    video_id: str | None = Form(None),
    label: str | None = Form(None),
    store=Depends(get_store),
):
    video_id = video_id or uuid.uuid4().hex[:12]
    storage_dir = Path(VIDEO_STORAGE_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)

    original_name = os.path.basename(file.filename or "video.mp4")
    suffix = Path(original_name).suffix or ".mp4"
    output_path = storage_dir / "{video_id}{suffix}".format(video_id=video_id, suffix=suffix)

    with output_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    stored_video_id = store.register_video(
        video_path=str(output_path),
        camera_id=camera_id,
        video_id=video_id,
        label=label or original_name,
        metadata={
            "original_filename": original_name,
            "storage": "local",
        },
    )
    return store.load_video(stored_video_id)


@router.get("/videos")
def list_videos(camera_id: str | None = None, limit: int = 100, store=Depends(get_store)):
    return {"videos": store.list_videos(camera_id=camera_id, limit=limit)}


@router.post("/process")
def process_video(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    service=Depends(get_pipeline_service),
    store=Depends(get_store),
):
    try:
        job_id = service.create_job(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(service.run_job, job_id)
    return store.load_processing_job(job_id)


@router.post("/process-sync")
def process_video_sync(
    request: ProcessVideoRequest,
    service=Depends(get_pipeline_service),
    store=Depends(get_store),
):
    try:
        job_id = service.create_job(request)
        service.run_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return store.load_processing_job(job_id)


@router.get("/jobs")
def list_jobs(
    camera_id: str | None = None,
    video_id: str | None = None,
    limit: int = 50,
    store=Depends(get_store),
):
    return {"jobs": store.list_processing_jobs(camera_id=camera_id, video_id=video_id, limit=limit)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, store=Depends(get_store)):
    job = store.load_processing_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job


@router.get("/runs")
def list_runs(
    camera_id: str | None = None,
    video_id: str | None = None,
    limit: int = 50,
    store=Depends(get_store),
):
    return {"runs": store.list_runs(camera_id=camera_id, video_id=video_id, limit=limit)}


@router.get("/runs/{run_id}")
def get_run(run_id: str, store=Depends(get_store)):
    run = store.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return run
