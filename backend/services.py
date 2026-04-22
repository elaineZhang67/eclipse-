import uuid
from functools import lru_cache
from types import SimpleNamespace

from backend.schemas import AskRequest, PipelineOptions, ProcessVideoRequest
from memory_store.sqlite_store import SurveillanceMemoryStore
from pipeline import run_pipeline
from retrieval.event_rag import EventRAG
from summarization.qwen_summary import build_summarizer


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _overlaps(doc_start, doc_end, query_start, query_end):
    if query_start is None and query_end is None:
        return True
    if doc_start is None and doc_end is None:
        return True

    left_start = float("-inf") if doc_start is None else float(doc_start)
    left_end = float("inf") if doc_end is None else float(doc_end)
    right_start = float("-inf") if query_start is None else float(query_start)
    right_end = float("inf") if query_end is None else float(query_end)
    return max(left_start, right_start) < min(left_end, right_end)


def _filter_documents_by_time(documents, start_sec=None, end_sec=None):
    return [
        doc
        for doc in documents
        if _overlaps(doc.get("start"), doc.get("end"), start_sec, end_sec)
    ]


def _build_pipeline_args(video_path, camera_id, run_id, options):
    payload = _model_dump(options)
    payload.update(
        {
            "video": video_path,
            "camera_id": camera_id,
            "run_id": run_id,
            "save_json": None,
            "memory_db": None,
            "question": None,
            "question_top_k": 4,
        }
    )
    return SimpleNamespace(**payload)


@lru_cache(maxsize=4)
def _cached_summarizer(backend, model_id, device):
    return build_summarizer(
        backend=backend,
        model_id=model_id or None,
        device=device,
    )


class PipelineJobService:
    def __init__(self, store):
        self.store = store

    def create_job(self, request):
        video = None
        video_id = request.video_id
        video_path = request.video_path
        camera_id = request.camera_id

        if video_id:
            video = self.store.load_video(video_id)
            if video is None:
                raise ValueError("Unknown video_id: {video_id}".format(video_id=video_id))
            video_path = video["video_path"]
            camera_id = camera_id or video["camera_id"]

        if not video_path:
            raise ValueError("Either video_id or video_path is required.")

        camera_id = camera_id or "camera_1"
        if video_id is None:
            video_id = self.store.register_video(
                video_path=video_path,
                camera_id=camera_id,
                metadata={"source": "process_request"},
            )

        config = _model_dump(request.options)
        if request.run_id:
            config["requested_run_id"] = request.run_id
        return self.store.create_processing_job(
            video_path=video_path,
            camera_id=camera_id,
            video_id=video_id,
            config=config,
            job_id=request.job_id,
        )

    def run_job(self, job_id):
        job = self.store.load_processing_job(job_id)
        if job is None:
            raise ValueError("Unknown job_id: {job_id}".format(job_id=job_id))

        self.store.update_processing_job(job_id, status="running", error=None)
        try:
            options = PipelineOptions(**job.get("config", {}))
            run_id = job.get("config", {}).get("requested_run_id") or uuid.uuid4().hex[:12]
            args = _build_pipeline_args(
                video_path=job["video_path"],
                camera_id=job["camera_id"],
                run_id=run_id,
                options=options,
            )
            results = run_pipeline(args)
            persisted_run_id = self.store.save_run(
                results,
                camera_id=job["camera_id"],
                video_path=job["video_path"],
                run_id=run_id,
                video_id=job.get("video_id"),
            )
            self.store.update_processing_job(
                job_id,
                status="completed",
                run_id=persisted_run_id,
                error=None,
            )
        except Exception as exc:
            self.store.update_processing_job(
                job_id,
                status="failed",
                error=str(exc),
            )
            raise


class TimeframeQAService:
    def __init__(self, store):
        self.store = store
        self.rag = EventRAG()

    def _load_run_bundles(self, request):
        if request.run_id:
            bundle = self.store.load_run(request.run_id)
            return [] if bundle is None else [bundle]
        return self.store.load_runs(
            camera_id=request.camera_id,
            video_id=request.video_id,
            limit=20,
        )

    def _build_documents(self, bundles):
        documents = []
        for bundle in bundles:
            if bundle is None:
                continue
            documents.extend(
                self.rag.build_documents(
                    bundle["results"].get("event_log", []),
                    bundle["results"].get("tracks", {}),
                    bundle["results"].get("interval_summaries", []),
                    source_meta={
                        "run_id": bundle.get("run_id"),
                        "camera_id": bundle.get("camera_id"),
                        "video_path": bundle.get("video_path"),
                    },
                )
            )
        return documents

    def ask(self, request):
        session_id = request.session_id or uuid.uuid4().hex[:12]
        bundles = self._load_run_bundles(request)
        documents = self._build_documents(bundles)
        documents = _filter_documents_by_time(
            documents,
            start_sec=request.start_sec,
            end_sec=request.end_sec,
        )
        retrieved = self.rag.retrieve(request.question, documents, top_k=request.top_k)
        history = self.store.load_messages(session_id, limit=request.history_turns)

        timeframe = None
        resolved_question = request.question
        if request.start_sec is not None or request.end_sec is not None:
            timeframe = {"start_sec": request.start_sec, "end_sec": request.end_sec}
            resolved_question = (
                "{question}\n"
                "Only answer using evidence that overlaps the requested timeframe: "
                "{start_sec} to {end_sec} seconds."
            ).format(
                question=request.question,
                start_sec=request.start_sec,
                end_sec=request.end_sec,
            )

        if not retrieved:
            answer = (
                "I could not find relevant processed evidence for that video/timeframe. "
                "Make sure the developer backend has processed the video first, then ask again with a valid timeframe."
            )
        else:
            summarizer = _cached_summarizer(
                request.answer_backend,
                request.answer_model or "",
                request.device,
            )
            answer = summarizer.answer_question(
                resolved_question,
                retrieved,
                conversation_history=history,
            )

        metadata = {
            "video_id": request.video_id,
            "run_id": request.run_id,
            "camera_id": request.camera_id,
            "timeframe": timeframe,
            "retrieved_context": retrieved,
        }
        self.store.append_message(session_id, "user", request.question, metadata=metadata)
        self.store.append_message(session_id, "assistant", answer, metadata=metadata)

        return {
            "session_id": session_id,
            "answer": answer,
            "resolved_question": resolved_question,
            "retrieved_context": retrieved,
            "run_ids": [bundle["run_id"] for bundle in bundles if bundle is not None],
            "video_id": request.video_id,
            "timeframe": timeframe,
        }
