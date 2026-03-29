import re
from collections import Counter


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _prefix_text(source_meta):
    if not source_meta:
        return ""

    parts = []
    if source_meta.get("camera_id"):
        parts.append("camera {camera_id}".format(camera_id=source_meta["camera_id"]))
    if source_meta.get("run_id"):
        parts.append("run {run_id}".format(run_id=source_meta["run_id"]))
    if source_meta.get("video_path"):
        parts.append("video {video_path}".format(video_path=source_meta["video_path"]))
    if not parts:
        return ""
    return ". ".join(parts) + ". "


def _format_track_payload(track_id, payload):
    metadata = payload.get("metadata", {})
    segments = payload.get("segments", [])
    parts = [
        f"track {track_id}",
        f"first seen {metadata.get('first_seen')}",
        f"last seen {metadata.get('last_seen')}",
    ]
    if metadata.get("objects"):
        parts.append("objects " + ", ".join(metadata["objects"]))
    for segment in segments:
        parts.append(
            "action {action} from {start} to {end}".format(
                action=segment.get("action"),
                start=segment.get("start"),
                end=segment.get("end"),
            )
        )
    for interaction in metadata.get("interactions", []):
        parts.append(
            "interaction with track {other_track_id} {kinds} count {count}".format(
                other_track_id=interaction.get("other_track_id"),
                kinds=" ".join(interaction.get("kinds", [])),
                count=interaction.get("count"),
            )
        )
    return ". ".join(parts)


class EventRAG:
    def build_documents(self, event_log, track_payload, interval_summaries, source_meta=None):
        documents = []
        prefix = _prefix_text(source_meta)
        for interval in interval_summaries:
            documents.append(
                {
                    "type": "interval",
                    "camera_id": None if not source_meta else source_meta.get("camera_id"),
                    "run_id": None if not source_meta else source_meta.get("run_id"),
                    "start": interval["start"],
                    "end": interval["end"],
                    "text": (
                        prefix +
                        "interval {start} to {end}. active tracks {tracks}. objects {objects}. "
                        "interaction count {interaction_count}. summary {summary}".format(
                            start=interval["start"],
                            end=interval["end"],
                            tracks=", ".join(str(item) for item in interval.get("active_tracks", [])) or "none",
                            objects=", ".join(interval.get("objects", [])) or "none",
                            interaction_count=interval.get("interaction_count", 0),
                            summary=interval.get("summary", ""),
                        )
                    ),
                }
            )

        for track_id, payload in sorted(track_payload.items()):
            documents.append(
                {
                    "type": "track",
                    "camera_id": None if not source_meta else source_meta.get("camera_id"),
                    "run_id": None if not source_meta else source_meta.get("run_id"),
                    "track_id": int(track_id),
                    "start": payload.get("metadata", {}).get("first_seen"),
                    "end": payload.get("metadata", {}).get("last_seen"),
                    "text": prefix + _format_track_payload(track_id, payload),
                }
            )

        for window in event_log:
            documents.append(
                {
                    "type": "window",
                    "camera_id": None if not source_meta else source_meta.get("camera_id"),
                    "run_id": None if not source_meta else source_meta.get("run_id"),
                    "start": window["start"],
                    "end": window["end"],
                    "text": prefix + "window {start} to {end}. active tracks {tracks}. events {events}".format(
                        start=window["start"],
                        end=window["end"],
                        tracks=", ".join(str(item) for item in window.get("active_tracks", [])) or "none",
                        events=window.get("events", []),
                    ),
                }
            )

        return documents

    def retrieve(self, question, documents, top_k=4):
        query_counts = Counter(_tokenize(question))
        scored = []
        for document in documents:
            doc_counts = Counter(_tokenize(document["text"]))
            overlap = sum(min(query_counts[token], doc_counts[token]) for token in query_counts)
            if overlap <= 0:
                continue
            score = float(overlap)
            if document.get("type") == "interval":
                score += 0.25
            scored.append((score, document))

        scored.sort(key=lambda item: (-item[0], item[1].get("start") or 0.0))
        return [doc for _, doc in scored[: max(1, int(top_k))]]
