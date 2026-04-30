import uuid
from functools import lru_cache
import re
from types import SimpleNamespace

from backend.schemas import AskRequest, ChatAskRequest, PipelineOptions, ProcessVideoRequest
from memory_store.sqlite_store import SurveillanceMemoryStore
from retrieval.event_rag import EventRAG


_TRACK_PATTERN = re.compile(r"\btrack\s+(\d+)\b", re.IGNORECASE)
_PERSON_PATTERN = re.compile(r"\bperson\s+\d+\b", re.IGNORECASE)


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


def _contains_pronoun(question):
    lowered = " " + str(question).lower().strip() + " "
    pronouns = [" he ", " she ", " they ", " them ", " him ", " her ", " that person ", " this person "]
    return any(token in lowered for token in pronouns)


def _question_has_explicit_identity(question):
    return bool(_TRACK_PATTERN.search(str(question)) or _PERSON_PATTERN.search(str(question)))


def _extract_track_refs_from_docs(documents):
    refs = []
    for doc in documents or []:
        track_id = doc.get("track_id")
        if track_id is None:
            continue
        display_name = doc.get("display_name")
        if display_name:
            refs.append("{name} (track {track_id})".format(name=display_name, track_id=track_id))
        else:
            refs.append("track {track_id}".format(track_id=track_id))
    deduped = []
    seen = set()
    for ref in refs:
        if ref in seen:
            continue
        deduped.append(ref)
        seen.add(ref)
    return deduped


def _resolve_followup_question(question, history):
    if not _contains_pronoun(question) or _question_has_explicit_identity(question):
        return question

    for item in reversed(history or []):
        if item.get("role") != "assistant":
            continue
        metadata = item.get("metadata") or {}
        refs = _extract_track_refs_from_docs(metadata.get("retrieved_context", []))
        if refs:
            return (
                "{question}\n"
                "Likely referent from prior turn: {track_refs}."
            ).format(
                question=question,
                track_refs=", ".join(refs),
            )
    return question


@lru_cache(maxsize=4)
def _cached_summarizer(backend, model_id, device):
    from summarization.qwen_summary import build_summarizer

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
            from pipeline import run_pipeline

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
            return persisted_run_id
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
                    bundle["results"].get("window_summaries", []),
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
        history = self.store.load_messages(session_id, limit=request.history_turns)
        resolved_question = _resolve_followup_question(request.question, history)

        timeframe = None
        answer_question = resolved_question
        if request.start_sec is not None or request.end_sec is not None:
            timeframe = {"start_sec": request.start_sec, "end_sec": request.end_sec}
            answer_question = (
                "{question}\n"
                "The user is asking about the requested timeframe: {start_sec} to {end_sec} seconds. "
                "Use adjacent stitched context only to understand continuity, but answer the timeframe directly."
            ).format(
                question=resolved_question,
                start_sec=request.start_sec,
                end_sec=request.end_sec,
            )
        retrieved = self.rag.retrieve_with_stitching(
            resolved_question,
            documents,
            top_k=request.top_k,
            focus_start_sec=request.start_sec,
            focus_end_sec=request.end_sec,
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
                answer_question,
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
            "resolved_question": answer_question,
            "retrieved_context": retrieved,
            "run_ids": [bundle["run_id"] for bundle in bundles if bundle is not None],
            "video_id": request.video_id,
            "timeframe": timeframe,
        }


class ChatSessionService:
    def __init__(self, store):
        self.store = store

    def _attach_processing_job(self, session):
        if session is None:
            return None
        payload = dict(session)
        job_id = payload.get("job_id")
        payload["processing_job"] = self.store.load_processing_job(job_id) if job_id else None
        return payload

    def load_session(self, session_id):
        return self._attach_processing_job(self.store.load_chat_session(session_id))

    def list_sessions(self, limit=50):
        return [
            self._attach_processing_job(session)
            for session in self.store.list_chat_sessions(limit=limit)
        ]

    def load_messages(self, session_id, limit=50):
        return self.store.load_messages(session_id, limit=limit)

    def ask(self, session_id, request):
        if isinstance(request, ChatAskRequest):
            payload = _model_dump(request)
        else:
            payload = dict(request)

        session = self.store.load_chat_session(session_id)
        if session is None:
            raise ValueError("Unknown session_id: {session_id}".format(session_id=session_id))
        if session.get("status") != "completed" or not session.get("run_id"):
            raise RuntimeError(
                "Session is still processing. Current status: {status}".format(
                    status=session.get("status", "unknown"),
                )
            )

        ask_request = AskRequest(
            question=payload["question"],
            video_id=session.get("video_id"),
            run_id=session.get("run_id"),
            camera_id=session.get("camera_id"),
            start_sec=payload.get("start_sec"),
            end_sec=payload.get("end_sec"),
            session_id=session_id,
            top_k=payload.get("top_k", 4),
            history_turns=payload.get("history_turns", 8),
            answer_backend=payload.get("answer_backend", "text"),
            answer_model=payload.get("answer_model"),
            device=payload.get("device", "auto"),
        )
        return TimeframeQAService(self.store).ask(ask_request)
