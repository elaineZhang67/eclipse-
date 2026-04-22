from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import AskRequest, AskResponse
from backend.services import TimeframeQAService
from backend.settings import DEFAULT_MEMORY_DB
from memory_store.sqlite_store import SurveillanceMemoryStore


router = APIRouter(prefix="/user", tags=["user"])


def get_store():
    return SurveillanceMemoryStore(DEFAULT_MEMORY_DB)


def get_qa_service(store=Depends(get_store)):
    return TimeframeQAService(store)


@router.get("/videos")
def list_videos(camera_id: str | None = None, limit: int = 100, store=Depends(get_store)):
    return {"videos": store.list_videos(camera_id=camera_id, limit=limit)}


@router.get("/runs")
def list_runs(
    camera_id: str | None = None,
    video_id: str | None = None,
    limit: int = 50,
    store=Depends(get_store),
):
    return {"runs": store.list_runs(camera_id=camera_id, video_id=video_id, limit=limit)}


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest, service=Depends(get_qa_service)):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    if request.end_sec is not None and request.start_sec is not None and request.end_sec <= request.start_sec:
        raise HTTPException(status_code=400, detail="end_sec must be greater than start_sec")
    return service.ask(request)
