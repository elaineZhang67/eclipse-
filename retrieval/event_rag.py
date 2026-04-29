import os
import re
from collections import Counter
from functools import lru_cache

import numpy as np


_QUERY_EXPANSIONS = {
    "pick": ["picked", "pickup", "grab", "grabbed", "take", "took", "associated", "attached", "near"],
    "picked": ["pick", "pickup", "grabbed", "took", "associated", "attached"],
    "grab": ["pick", "picked", "take", "took", "associated", "attached"],
    "hold": ["held", "holding", "carry", "carried", "associated", "attached"],
    "carrying": ["carry", "held", "holding", "associated", "attached"],
    "near": ["nearby", "close", "interaction", "associated"],
    "with": ["associated", "attached", "near"],
    "bag": ["backpack", "handbag", "suitcase"],
    "phone": ["cell", "cellphone", "mobile"],
    "leave": ["left", "last", "exit", "exited"],
    "enter": ["entered", "entry", "arrive", "arrived"],
}


@lru_cache(maxsize=1)
def _embedding_model():
    enabled = str(os.environ.get("ECLIPSE_ENABLE_EMBEDDING_RAG", "")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _expanded_query_text(text):
    tokens = _tokenize(text)
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_QUERY_EXPANSIONS.get(token, []))
    return " ".join(expanded)


def _extract_track_ids(text):
    return {int(item) for item in re.findall(r"\btrack\s+(\d+)\b", str(text).lower())}


def _extract_time_mentions(text):
    return [float(item) for item in re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)\b", str(text).lower())]


def _document_structural_boost(question, document):
    question_tokens = set(_tokenize(question))
    doc_tokens = set(_tokenize(document.get("text", "")))
    boost = 0.0

    for track_id in _extract_track_ids(question):
        if document.get("track_id") == track_id or "track {track_id}".format(track_id=track_id) in document.get("text", "").lower():
            boost += 2.0

    for t_sec in _extract_time_mentions(question):
        start = document.get("start")
        end = document.get("end")
        if start is not None and end is not None and float(start) <= t_sec <= float(end):
            boost += 1.5

    object_like_tokens = {
        token
        for token in question_tokens
        if token in doc_tokens and token not in {"who", "what", "when", "where", "which", "did", "the", "a", "an"}
    }
    boost += min(1.5, 0.25 * len(object_like_tokens))

    if document.get("type") == "track" and any(token in question_tokens for token in {"who", "person", "track"}):
        boost += 0.5
    if document.get("type") == "window" and any(token in question_tokens for token in {"when", "time", "timestamp"}):
        boost += 0.5
    if document.get("type") == "interval":
        boost += 0.25

    return boost


def _semantic_scores(question, documents):
    model = _embedding_model()
    if model is None or not documents:
        return {}

    texts = [document.get("text", "") for document in documents]
    try:
        doc_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        query_embedding = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    except Exception:
        return {}

    doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    try:
        import faiss

        index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        scores, indices = index.search(query_embedding.reshape(1, -1), len(documents))
        return {int(index): float(score) for index, score in zip(indices[0], scores[0]) if int(index) >= 0}
    except Exception:
        return {
            idx: float(np.dot(query_embedding, doc_embedding))
            for idx, doc_embedding in enumerate(doc_embeddings)
        }


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


def _track_order_key(item):
    track_id, payload = item
    metadata = payload.get("metadata", {})
    first_seen = metadata.get("first_seen")
    return (first_seen is None, float(first_seen or 0.0), int(track_id))


def _build_alias_map(track_payload):
    alias_map = {}
    for idx, (track_id, payload) in enumerate(sorted(track_payload.items(), key=_track_order_key), start=1):
        metadata = payload.get("metadata", {})
        alias_map[int(track_id)] = metadata.get("display_name") or "Person {idx}".format(idx=idx)
    return alias_map


def _display_name(track_id, payload, alias_map):
    metadata = payload.get("metadata", {})
    return metadata.get("display_name") or alias_map.get(int(track_id), "Person")


def _track_ref(track_id, payload, alias_map):
    metadata = payload.get("metadata", {})
    return metadata.get("track_ref") or "{name} (track {track_id})".format(
        name=_display_name(track_id, payload, alias_map),
        track_id=int(track_id),
    )


def _track_refs_for_ids(track_ids, track_payload, alias_map):
    refs = []
    for track_id in track_ids or []:
        tid = int(track_id)
        refs.append(_track_ref(tid, track_payload.get(tid, {}), alias_map))
    return refs


def _format_profile(profile):
    if not isinstance(profile, dict):
        return ""

    appearance = profile.get("appearance") or {}
    parts = []
    if appearance.get("top_color"):
        parts.append("top color {value}".format(value=appearance["top_color"]))
    if appearance.get("bottom_color"):
        parts.append("bottom color {value}".format(value=appearance["bottom_color"]))
    if appearance.get("outerwear"):
        parts.append("outerwear {value}".format(value=appearance["outerwear"]))
    if appearance.get("helmet") is not None:
        parts.append("helmet {value}".format(value=appearance["helmet"]))
    carried = [
        item.get("label")
        for item in profile.get("carried_objects", [])
        if isinstance(item, dict) and item.get("label")
    ]
    if carried:
        parts.append("carried objects " + ", ".join(carried))
    if profile.get("behavior_overview"):
        parts.append("behavior overview {value}".format(value=profile["behavior_overview"]))
    return ". ".join(parts)


def _format_window_event(event, track_payload, alias_map):
    if event.get("type") == "enter":
        track_id = int(event.get("track_id"))
        payload = track_payload.get(track_id, {})
        return "{track_ref} entered the scene".format(track_ref=_track_ref(track_id, payload, alias_map))

    if event.get("type") == "last_seen":
        track_id = int(event.get("track_id"))
        payload = track_payload.get(track_id, {})
        return "{track_ref} was last seen at {time}".format(
            track_ref=_track_ref(track_id, payload, alias_map),
            time=event.get("time"),
        )

    if event.get("type") in {"attributed_objects", "near_objects"}:
        track_id = int(event.get("track_id"))
        payload = track_payload.get(track_id, {})
        objects = []
        for item in event.get("objects", []):
            if isinstance(item, dict):
                label = item.get("label")
                object_track_id = item.get("object_track_id")
                confidence = item.get("confidence")
                suffix = ""
                if confidence is not None:
                    suffix = " confidence {confidence}".format(confidence=confidence)
                if label and object_track_id is not None:
                    objects.append(
                        "{label} object track {object_track_id}{suffix}".format(
                            label=label,
                            object_track_id=object_track_id,
                            suffix=suffix,
                        )
                    )
                elif label:
                    objects.append("{label}{suffix}".format(label=label, suffix=suffix))
            elif item:
                objects.append(str(item))
        if objects:
            verb = "was associated with" if event.get("type") == "attributed_objects" else "was near"
            return "{track_ref} {verb} {objects}".format(
                track_ref=_track_ref(track_id, payload, alias_map),
                verb=verb,
                objects=", ".join(objects),
            )

    if event.get("type") == "interaction":
        refs = event.get("track_refs") or []
        if not refs:
            refs = [
                "{name} (track {track_id})".format(
                    name=alias_map.get(int(track_id), "Person"),
                    track_id=int(track_id),
                )
                for track_id in event.get("track_ids", [])
            ]
        if refs:
            return "{kind} interaction between {refs}".format(
                kind=event.get("kind", "unknown"),
                refs=" and ".join(refs),
            )

    return str(event)


def _format_track_payload(track_id, payload, alias_map):
    metadata = payload.get("metadata", {})
    segments = payload.get("segments", [])
    parts = [
        "{track_ref}. entity type {entity_type}".format(
            track_ref=_track_ref(track_id, payload, alias_map),
            entity_type=metadata.get("entity_type", "person"),
        ),
        f"first seen {metadata.get('first_seen')}",
        f"last seen {metadata.get('last_seen')}",
    ]
    if metadata.get("objects"):
        parts.append("objects " + ", ".join(metadata["objects"]))
    if metadata.get("memory_track_id") is not None:
        parts.append("memory track id {memory_track_id}".format(memory_track_id=metadata.get("memory_track_id")))
    if metadata.get("tracker_track_ids"):
        parts.append(
            "linked raw tracker ids {tracker_track_ids}".format(
                tracker_track_ids=", ".join(str(item) for item in metadata.get("tracker_track_ids", [])),
            )
        )
    if metadata.get("identity_history"):
        parts.append("identity history {value}".format(value=metadata["identity_history"]))
    if metadata.get("appearance_mean_similarity") is not None:
        parts.append(
            "appearance mean similarity {value}".format(
                value=metadata["appearance_mean_similarity"],
            )
        )
    for obj in metadata.get("object_tracks", []):
        parts.append(
            "object track {object_track_id} label {label} first seen {first_seen} last seen {last_seen} hits {hits}".format(
                object_track_id=obj.get("object_track_id"),
                label=obj.get("label"),
                first_seen=obj.get("first_seen"),
                last_seen=obj.get("last_seen"),
                hits=obj.get("hits"),
            )
        )
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
            "interaction with {other_track_ref} {kinds} count {count}".format(
                other_track_ref=interaction.get("other_track_ref") or "track {other_track_id}".format(
                    other_track_id=interaction.get("other_track_id"),
                ),
                kinds=" ".join(interaction.get("kinds", [])),
                count=interaction.get("count"),
            )
        )
    profile_text = _format_profile(payload.get("profile"))
    if profile_text:
        parts.append("profile " + profile_text)
    if payload.get("summary"):
        parts.append("summary {value}".format(value=payload["summary"]))
    return ". ".join(parts)


class EventRAG:
    def build_documents(self, event_log, track_payload, interval_summaries, source_meta=None):
        documents = []
        prefix = _prefix_text(source_meta)
        alias_map = _build_alias_map(track_payload)
        for interval in interval_summaries:
            active_track_refs = interval.get("active_track_refs") or _track_refs_for_ids(
                interval.get("active_tracks", []),
                track_payload,
                alias_map,
            )
            documents.append(
                {
                    "type": "interval",
                    "camera_id": None if not source_meta else source_meta.get("camera_id"),
                    "run_id": None if not source_meta else source_meta.get("run_id"),
                    "start": interval["start"],
                    "end": interval["end"],
                    "text": (
                        prefix +
                        "interval {start} to {end}. active people {tracks}. objects {objects}. "
                        "interaction count {interaction_count}. summary {summary}".format(
                            start=interval["start"],
                            end=interval["end"],
                            tracks=", ".join(active_track_refs) or "none",
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
                    "display_name": _display_name(track_id, payload, alias_map),
                    "text": prefix + _format_track_payload(track_id, payload, alias_map),
                }
            )

        for window in event_log:
            active_track_refs = window.get("active_track_refs") or _track_refs_for_ids(
                window.get("active_tracks", []),
                track_payload,
                alias_map,
            )
            event_text = "; ".join(
                _format_window_event(event, track_payload, alias_map)
                for event in window.get("events", [])
            ) or "none"
            documents.append(
                {
                    "type": "window",
                    "camera_id": None if not source_meta else source_meta.get("camera_id"),
                    "run_id": None if not source_meta else source_meta.get("run_id"),
                    "start": window["start"],
                    "end": window["end"],
                    "text": prefix + "window {start} to {end}. active people {tracks}. events {events}".format(
                        start=window["start"],
                        end=window["end"],
                        tracks=", ".join(active_track_refs) or "none",
                        events=event_text,
                    ),
                }
            )

        return documents

    def retrieve(self, question, documents, top_k=4):
        expanded_question = "{question} {expanded}".format(
            question=question,
            expanded=_expanded_query_text(question),
        )
        query_counts = Counter(_tokenize(expanded_question))
        semantic_scores = _semantic_scores(expanded_question, documents)
        scored = []
        for idx, document in enumerate(documents):
            doc_counts = Counter(_tokenize(document["text"]))
            overlap = sum(min(query_counts[token], doc_counts[token]) for token in query_counts)
            semantic = semantic_scores.get(idx, 0.0)
            boost = _document_structural_boost(question, document)
            if overlap <= 0 and semantic <= 0.0 and boost <= 0.0:
                continue
            score = float(overlap)
            score += 3.0 * max(0.0, float(semantic))
            score += boost
            scored.append((score, document))

        scored.sort(key=lambda item: (-item[0], item[1].get("start") or 0.0))
        return [doc for _, doc in scored[: max(1, int(top_k))]]
