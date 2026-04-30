import os
from collections import OrderedDict

import cv2


DEFAULT_QA_MAX_FRAMES = 8
DEFAULT_QA_FRAME_LONG_EDGE = 768


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resize_long_edge(image_bgr, max_long_edge):
    max_long_edge = int(max_long_edge or 0)
    if max_long_edge <= 0:
        return image_bgr

    height, width = image_bgr.shape[:2]
    current_long_edge = max(height, width)
    if current_long_edge <= max_long_edge:
        return image_bgr

    scale = float(max_long_edge) / float(current_long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _windows_from_document(document):
    video_path = document.get("video_path")
    for window in document.get("windows", []) or []:
        start = _safe_float(window.get("start"))
        end = _safe_float(window.get("end"))
        if start is None or end is None:
            continue
        yield {
            "video_path": window.get("video_path") or video_path,
            "start": start,
            "end": end,
            "source_type": document.get("type"),
            "track_id": document.get("track_id"),
            "run_id": document.get("run_id"),
        }

    if document.get("type") in {"window_summary", "window", "interval"}:
        start = _safe_float(document.get("start"))
        end = _safe_float(document.get("end"))
        if start is not None and end is not None:
            yield {
                "video_path": video_path,
                "start": start,
                "end": end,
                "source_type": document.get("type"),
                "track_id": document.get("track_id"),
                "run_id": document.get("run_id"),
            }


def _dedupe_windows(windows):
    deduped = OrderedDict()
    for window in windows:
        video_path = window.get("video_path")
        if not video_path:
            continue
        start = _safe_float(window.get("start"))
        end = _safe_float(window.get("end"))
        if start is None or end is None:
            continue
        key = (video_path, round(start, 3), round(end, 3))
        deduped.setdefault(key, {**window, "start": start, "end": end})
    return list(deduped.values())


def _select_windows(windows, max_frames):
    if max_frames <= 0 or not windows:
        return []
    windows = sorted(windows, key=lambda item: (item.get("video_path") or "", item["start"], item["end"]))
    if len(windows) <= max_frames:
        return windows
    if max_frames == 1:
        return [windows[len(windows) // 2]]

    step = float(len(windows) - 1) / float(max_frames - 1)
    selected = []
    seen = set()
    for idx in range(max_frames):
        source_idx = int(round(idx * step))
        source_idx = max(0, min(len(windows) - 1, source_idx))
        if source_idx in seen:
            continue
        selected.append(windows[source_idx])
        seen.add(source_idx)
    return selected


def _read_frame_at(video_path, time_sec, long_edge):
    if not video_path or not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if fps > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(float(time_sec) * fps))))
                ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            return None
        resized = _resize_long_edge(frame_bgr, long_edge)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def collect_qa_visual_context(retrieved_context, max_frames=DEFAULT_QA_MAX_FRAMES, frame_long_edge=DEFAULT_QA_FRAME_LONG_EDGE):
    windows = _dedupe_windows(
        window
        for document in retrieved_context or []
        for window in _windows_from_document(document)
    )
    selected_windows = _select_windows(windows, int(max_frames or 0))

    images = []
    frames = []
    for window in selected_windows:
        time_sec = 0.5 * (float(window["start"]) + float(window["end"]))
        image_rgb = _read_frame_at(window["video_path"], time_sec, frame_long_edge)
        if image_rgb is None:
            continue
        images.append(image_rgb)
        frames.append(
            {
                "video_path": window["video_path"],
                "time_sec": round(time_sec, 3),
                "window_start": round(float(window["start"]), 3),
                "window_end": round(float(window["end"]), 3),
                "source_type": window.get("source_type"),
                "track_id": window.get("track_id"),
                "run_id": window.get("run_id"),
            }
        )

    return {
        "images": images,
        "frames": frames,
        "frame_count": len(images),
        "candidate_windows": len(windows),
    }
