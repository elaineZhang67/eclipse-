import re
import uuid

from retrieval.event_rag import EventRAG


_TRACK_PATTERN = re.compile(r"\btrack\s+(\d+)\b", re.IGNORECASE)


def _extract_track_ids(documents):
    track_ids = []
    for doc in documents:
        if doc.get("track_id") is not None:
            track_ids.append(int(doc["track_id"]))
            continue

        text = str(doc.get("text", ""))
        for match in _TRACK_PATTERN.findall(text):
            track_ids.append(int(match))
    return sorted(set(track_ids))


def _contains_pronoun(question):
    lowered = str(question).lower()
    pronouns = [" he ", " she ", " they ", " them ", " him ", " her ", " that person ", " this person "]
    padded = " " + lowered + " "
    return any(token in padded for token in pronouns)


class SurveillanceChatSession:
    def __init__(
        self,
        store,
        summarizer,
        session_id=None,
        camera_id=None,
        run_ids=None,
        top_k=4,
        history_turns=8,
    ):
        self.store = store
        self.summarizer = summarizer
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.camera_id = camera_id
        self.run_ids = list(run_ids or [])
        self.top_k = max(1, int(top_k))
        self.history_turns = max(0, int(history_turns))
        self.rag = EventRAG()
        self.last_track_ids = []

    def set_camera_filter(self, camera_id):
        self.camera_id = camera_id

    def set_run_filter(self, run_ids):
        self.run_ids = list(run_ids or [])

    def list_runs(self, limit=20):
        return self.store.list_runs(camera_id=self.camera_id, limit=limit)

    def _load_bundles(self):
        if self.run_ids:
            return self.store.load_runs(run_ids=self.run_ids)
        return self.store.load_runs(camera_id=self.camera_id, limit=50)

    def _build_documents(self):
        documents = []
        for bundle in self._load_bundles():
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

    def _resolve_question(self, question):
        if self.last_track_ids and _contains_pronoun(question) and not _TRACK_PATTERN.search(question):
            return (
                "{question}\n"
                "Likely referent from prior turn: track IDs {track_ids}."
            ).format(
                question=question,
                track_ids=", ".join(str(item) for item in self.last_track_ids),
            )
        return question

    def history(self):
        return self.store.load_messages(self.session_id, limit=self.history_turns)

    def ask(self, question):
        resolved_question = self._resolve_question(question)
        documents = self._build_documents()
        if not documents:
            answer = (
                "No persisted runs are available for the current camera or run filter. "
                "Run the pipeline with --memory_db first, then select the matching camera or run in the frontend."
            )
            metadata = {
                "resolved_question": resolved_question,
                "retrieved_context": [],
                "camera_id": self.camera_id,
                "run_ids": self.run_ids,
                "last_track_ids": [],
            }
            self.store.append_message(self.session_id, "user", question, metadata={"resolved_question": resolved_question})
            self.store.append_message(self.session_id, "assistant", answer, metadata=metadata)
            self.last_track_ids = []
            return {
                "session_id": self.session_id,
                "question": question,
                "resolved_question": resolved_question,
                "retrieved_context": [],
                "answer": answer,
                "last_track_ids": [],
            }

        retrieved = self.rag.retrieve(resolved_question, documents, top_k=self.top_k)
        history = self.history()
        if not retrieved:
            answer = (
                "I could not find relevant context for that question in the selected runs. "
                "Try selecting a different camera or run, or ask with track IDs, time ranges, objects, or interactions."
            )
        else:
            answer = self.summarizer.answer_question(
                resolved_question,
                retrieved,
                conversation_history=history,
            )
        self.last_track_ids = _extract_track_ids(retrieved)

        metadata = {
            "resolved_question": resolved_question,
            "retrieved_context": retrieved,
            "camera_id": self.camera_id,
            "run_ids": self.run_ids,
            "last_track_ids": self.last_track_ids,
        }
        self.store.append_message(self.session_id, "user", question, metadata={"resolved_question": resolved_question})
        self.store.append_message(self.session_id, "assistant", answer, metadata=metadata)

        return {
            "session_id": self.session_id,
            "question": question,
            "resolved_question": resolved_question,
            "retrieved_context": retrieved,
            "answer": answer,
            "last_track_ids": list(self.last_track_ids),
        }
