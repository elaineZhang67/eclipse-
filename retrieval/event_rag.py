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

_WINDOW_DOC_TYPES = {"window_summary", "window"}


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


def _extract_person_numbers(text):
    return {int(item) for item in re.findall(r"\bperson\s+(\d+)\b", str(text).lower())}


def _track_ids_for_person_refs(question, documents):
    person_numbers = _extract_person_numbers(question)
    if not person_numbers:
        return set()

    track_ids = set()
    for person_number in person_numbers:
        pattern = re.compile(
            r"\bperson\s+{person_number}\s+\(track\s+(\d+)\)".format(person_number=int(person_number)),
            re.IGNORECASE,
        )
        for document in documents or []:
            for match in pattern.findall(str(document.get("text", ""))):
                track_ids.add(int(match))
    return track_ids


def _extract_time_mentions(text):
    return [
        float(item)
        for item in re.findall(
            r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds|秒)",
            str(text).lower(),
        )
    ]


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _overlaps(start_a, end_a, start_b, end_b):
    if start_b is None and end_b is None:
        return True
    left_start = float("-inf") if start_a is None else _safe_float(start_a, float("-inf"))
    left_end = float("inf") if end_a is None else _safe_float(end_a, float("inf"))
    right_start = float("-inf") if start_b is None else _safe_float(start_b, float("-inf"))
    right_end = float("inf") if end_b is None else _safe_float(end_b, float("inf"))
    return max(left_start, right_start) < min(left_end, right_end)


def _document_track_ids(document):
    track_ids = set()
    if document.get("track_id") is not None:
        track_ids.add(int(document["track_id"]))
    for key in ("track_ids", "active_tracks"):
        for item in document.get(key, []) or []:
            try:
                track_ids.add(int(item))
            except (TypeError, ValueError):
                continue
    track_ids.update(_extract_track_ids(document.get("text", "")))
    return track_ids


def _ordered_track_ids_from_documents(documents, seed_text=""):
    ordered = []
    seen = set()
    for track_id in sorted(_extract_track_ids(seed_text)):
        ordered.append(track_id)
        seen.add(track_id)
    for document in documents or []:
        for track_id in sorted(_document_track_ids(document)):
            if track_id in seen:
                continue
            ordered.append(track_id)
            seen.add(track_id)
    return ordered


def _context_time_window(question, focus_start_sec=None, focus_end_sec=None, padding_sec=10.0):
    focus_start = _safe_float(focus_start_sec)
    focus_end = _safe_float(focus_end_sec)
    if focus_start is None and focus_end is None:
        time_mentions = _extract_time_mentions(question)
        if not time_mentions:
            return None, None, None, None
        focus_start = min(time_mentions)
        focus_end = max(time_mentions)
    elif focus_start is None:
        focus_start = focus_end
    elif focus_end is None:
        focus_end = focus_start

    if focus_end < focus_start:
        focus_start, focus_end = focus_end, focus_start

    padding = max(0.0, float(padding_sec))
    context_start = max(0.0, focus_start - padding)
    context_end = focus_end + padding
    return context_start, context_end, focus_start, focus_end


def _doc_sort_key(document):
    type_rank = {"stitched_track_timeline": 0, "window_summary": 1, "track": 2, "window": 3, "interval": 4}
    return (
        type_rank.get(document.get("type"), 9),
        float(document.get("start") or 0.0),
        int(document.get("track_id") or 0),
    )


def _doc_dedupe_key(document):
    return (
        document.get("type"),
        document.get("run_id"),
        document.get("camera_id"),
        document.get("track_id"),
        document.get("start"),
        document.get("end"),
        str(document.get("text", ""))[:120],
    )


def _dedupe_documents(documents):
    deduped = []
    seen = set()
    for document in documents or []:
        key = _doc_dedupe_key(document)
        if key in seen:
            continue
        deduped.append(document)
        seen.add(key)
    return deduped


def _dedupe_window_documents(documents):
    deduped = []
    seen = set()
    for document in sorted(documents or [], key=_doc_sort_key):
        if document.get("type") not in _WINDOW_DOC_TYPES:
            deduped.append(document)
            continue
        key = (
            document.get("run_id"),
            document.get("camera_id"),
            document.get("start"),
            document.get("end"),
        )
        if key in seen:
            continue
        deduped.append(document)
        seen.add(key)
    return deduped


def _compact_text(text, max_chars=1400):
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _display_name_from_text(track_id, text):
    pattern = re.compile(r"\b(Person\s+\d+)\s+\(track\s+{track_id}\)".format(track_id=int(track_id)), re.IGNORECASE)
    match = pattern.search(str(text or ""))
    if match:
        return match.group(1)
    return "Person"


def _select_time_context_documents(documents, context_start, context_end, track_ids=None, max_docs=6):
    selected = []
    wanted_track_ids = set(track_ids or [])
    for document in documents or []:
        if document.get("type") not in _WINDOW_DOC_TYPES:
            continue
        if not _overlaps(document.get("start"), document.get("end"), context_start, context_end):
            continue
        if wanted_track_ids and not (_document_track_ids(document) & wanted_track_ids):
            continue
        selected.append(document)

    selected = _dedupe_window_documents(selected)
    return selected[: max(0, int(max_docs))]


def _find_track_document(documents, track_id):
    for document in documents or []:
        if document.get("type") == "track" and document.get("track_id") == int(track_id):
            return document
    return None


def _build_track_timeline_document(
    track_id,
    documents,
    context_start=None,
    context_end=None,
    focus_start=None,
    focus_end=None,
    max_windows=12,
):
    window_docs = [
        document
        for document in documents or []
        if document.get("type") in _WINDOW_DOC_TYPES
        and int(track_id) in _document_track_ids(document)
        and _overlaps(document.get("start"), document.get("end"), context_start, context_end)
    ]
    window_docs = _dedupe_window_documents(window_docs)
    if not window_docs:
        return None

    if focus_start is not None or focus_end is not None:
        focus_center = _safe_float(focus_start if focus_start is not None else focus_end, 0.0)
        if focus_end is not None and focus_start is not None:
            focus_center = (float(focus_start) + float(focus_end)) / 2.0
        window_docs.sort(
            key=lambda document: (
                abs(((float(document.get("start") or 0.0) + float(document.get("end") or 0.0)) / 2.0) - focus_center),
                _doc_sort_key(document),
            )
        )
        window_docs = sorted(window_docs[: max(1, int(max_windows))], key=_doc_sort_key)
    else:
        window_docs = sorted(window_docs, key=_doc_sort_key)[: max(1, int(max_windows))]

    track_doc = _find_track_document(documents, track_id)
    display_name = (
        track_doc.get("display_name")
        if track_doc is not None and track_doc.get("display_name")
        else _display_name_from_text(track_id, " ".join(document.get("text", "") for document in window_docs))
    )
    track_ref = "{name} (track {track_id})".format(name=display_name, track_id=int(track_id))

    timeline_lines = []
    for document in window_docs:
        timeline_lines.append(
            "[{start}-{end}s] {text}".format(
                start=document.get("start"),
                end=document.get("end"),
                text=_compact_text(document.get("text", ""), max_chars=650),
            )
        )

    parts = [
        "stitched track timeline for {track_ref}. This context links the same person across adjacent 5-second windows.".format(
            track_ref=track_ref,
        )
    ]
    if context_start is not None or context_end is not None:
        parts.append(
            "timeline context window {start} to {end} seconds around the requested time.".format(
                start=context_start,
                end=context_end,
            )
        )
    if track_doc is not None:
        parts.append("track-level evidence: {text}".format(text=_compact_text(track_doc.get("text", ""), max_chars=1400)))
    parts.append("stitched window timeline: " + " | ".join(timeline_lines))

    return {
        "type": "stitched_track_timeline",
        "track_id": int(track_id),
        "track_ids": [int(track_id)],
        "display_name": display_name,
        "start": min(document.get("start") for document in window_docs),
        "end": max(document.get("end") for document in window_docs),
        "camera_id": window_docs[0].get("camera_id"),
        "run_id": window_docs[0].get("run_id"),
        "text": " ".join(parts),
    }


def _document_structural_boost(question, document):
    question_tokens = set(_tokenize(question))
    doc_tokens = set(_tokenize(document.get("text", "")))
    boost = 0.0

    for track_id in _extract_track_ids(question):
        if track_id in _document_track_ids(document):
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
    if document.get("type") in {"window", "window_summary"} and any(
        token in question_tokens for token in {"when", "time", "timestamp", "doing", "happening", "action"}
    ):
        boost += 0.5
    if document.get("type") == "window_summary":
        boost += 0.35
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
        refs.append(_track_ref(tid, track_payload.get(tid) or track_payload.get(str(tid), {}), alias_map))
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
    def build_documents(self, event_log, track_payload, interval_summaries, window_summaries=None, source_meta=None):
        documents = []
        prefix = _prefix_text(source_meta)
        alias_map = _build_alias_map(track_payload)
        for window_summary in window_summaries or []:
            active_track_refs = window_summary.get("active_track_refs") or _track_refs_for_ids(
                window_summary.get("active_tracks", []),
                track_payload,
                alias_map,
            )
            summary = window_summary.get("summary") or window_summary.get("structured_summary") or ""
            documents.append(
                {
                    "type": "window_summary",
                    "camera_id": None if not source_meta else source_meta.get("camera_id"),
                    "run_id": None if not source_meta else source_meta.get("run_id"),
                    "start": window_summary["start"],
                    "end": window_summary["end"],
                    "active_tracks": list(window_summary.get("active_tracks", [])),
                    "track_ids": list(window_summary.get("active_tracks", [])),
                    "active_track_refs": active_track_refs,
                    "text": (
                        prefix +
                        "5-second window summary {start} to {end}. active people {tracks}. "
                        "event count {event_count}. summary {summary}".format(
                            start=window_summary["start"],
                            end=window_summary["end"],
                            tracks=", ".join(active_track_refs) or "none",
                            event_count=window_summary.get("event_count", 0),
                            summary=summary,
                        )
                    ),
                }
            )

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
                    "active_tracks": list(interval.get("active_tracks", [])),
                    "track_ids": list(interval.get("active_tracks", [])),
                    "active_track_refs": active_track_refs,
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
                    "track_ids": [int(track_id)],
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
                    "active_tracks": list(window.get("active_tracks", [])),
                    "track_ids": list(window.get("active_tracks", [])),
                    "active_track_refs": active_track_refs,
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

    def retrieve_with_stitching(
        self,
        question,
        documents,
        top_k=4,
        focus_start_sec=None,
        focus_end_sec=None,
        time_padding_sec=10.0,
        max_track_windows=12,
        max_timeline_docs=3,
    ):
        base_limit = max(int(top_k), 8)
        base_docs = self.retrieve(question, documents, top_k=base_limit)
        context_start, context_end, focus_start, focus_end = _context_time_window(
            question,
            focus_start_sec=focus_start_sec,
            focus_end_sec=focus_end_sec,
            padding_sec=time_padding_sec,
        )

        time_docs = []
        if context_start is not None or context_end is not None:
            time_docs = _select_time_context_documents(
                documents,
                context_start,
                context_end,
                max_docs=max(6, int(top_k)),
            )

        ordered_track_ids = _ordered_track_ids_from_documents(
            list(base_docs) + list(time_docs),
            seed_text=question,
        )
        explicit_track_ids = sorted(_extract_track_ids(question) | _track_ids_for_person_refs(question, documents))
        timeline_track_ids = explicit_track_ids or ordered_track_ids
        if (context_start is not None or context_end is not None) and ordered_track_ids:
            focused_time_docs = _select_time_context_documents(
                documents,
                context_start,
                context_end,
                track_ids=timeline_track_ids,
                max_docs=max(6, int(top_k)),
            )
            if explicit_track_ids:
                time_docs = focused_time_docs
            else:
                time_docs = _dedupe_documents(list(focused_time_docs) + list(time_docs))

        timeline_docs = []
        for track_id in timeline_track_ids[: max(0, int(max_timeline_docs))]:
            timeline_doc = _build_track_timeline_document(
                track_id,
                documents,
                context_start=context_start,
                context_end=context_end,
                focus_start=focus_start,
                focus_end=focus_end,
                max_windows=max_track_windows,
            )
            if timeline_doc is not None:
                timeline_docs.append(timeline_doc)

        combined = _dedupe_documents(_dedupe_window_documents(list(timeline_docs) + list(time_docs) + list(base_docs)))
        if not combined:
            return []

        context_budget = max(int(top_k), 1) + len(timeline_docs) + min(len(time_docs), 6)
        return combined[: max(context_budget, int(top_k))]
