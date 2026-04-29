import json
import os
from pathlib import Path

import cv2


def _round_box(box):
    return [round(float(value), 1) for value in box]


def _center(box):
    x1, y1, x2, y2 = box
    return int(0.5 * (float(x1) + float(x2))), int(0.5 * (float(y1) + float(y2)))


def _draw_label(frame, text, x, y, color):
    y = max(16, int(y))
    cv2.rectangle(frame, (int(x), y - 15), (int(x) + 7 * len(text) + 8, y + 4), color, -1)
    cv2.putText(
        frame,
        text,
        (int(x) + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


class DebugEvidenceWriter:
    def __init__(self, output_dir=None, stride_sec=2.0, max_frames=80, enabled=True):
        self.enabled = bool(enabled)
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.stride_sec = max(0.1, float(stride_sec))
        self.max_frames = max(1, int(max_frames))
        self._last_saved_t = None
        self.frames = []

        if self.enabled and self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _should_save(self, t_sec):
        if not self.enabled or self.output_dir is None:
            return False
        if len(self.frames) >= self.max_frames:
            return False
        if self._last_saved_t is None:
            return True
        return (float(t_sec) - float(self._last_saved_t)) >= self.stride_sec

    def update(self, frame_bgr, t_sec, person_tracks, object_tracks, assignments, window_idx):
        if not self._should_save(t_sec):
            return

        annotated = frame_bgr.copy()
        person_by_id = {}
        for person in person_tracks:
            track_id = int(person.get("track_id"))
            person_by_id[track_id] = person
            x1, y1, x2, y2 = [int(round(float(v))) for v in person["xyxy"]]
            color = (43, 170, 73)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            _draw_label(
                annotated,
                "person {track_id}".format(track_id=track_id),
                x1,
                y1,
                color,
            )

        for obj in object_tracks:
            x1, y1, x2, y2 = [int(round(float(v))) for v in obj["xyxy"]]
            color = (52, 132, 219)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = "{label} {track_id}".format(
                label=obj.get("label", "object"),
                track_id=obj.get("track_id", "?"),
            )
            _draw_label(annotated, label, x1, y1, color)

        attribution_records = []
        for track_id, objects in (assignments or {}).items():
            person = person_by_id.get(int(track_id))
            if person is None:
                continue
            person_center = _center(person["xyxy"])
            for obj in objects:
                obj_center = _center(obj["xyxy"])
                kind = obj.get("attribution_kind", "near")
                confidence = obj.get("attribution_confidence")
                color = (0, 190, 255) if kind == "attached" else (0, 165, 255)
                cv2.line(annotated, person_center, obj_center, color, 2)
                label = "{kind} {confidence}".format(
                    kind=kind,
                    confidence="" if confidence is None else confidence,
                ).strip()
                _draw_label(
                    annotated,
                    label,
                    int(0.5 * (person_center[0] + obj_center[0])),
                    int(0.5 * (person_center[1] + obj_center[1])),
                    color,
                )
                attribution_records.append(
                    {
                        "track_id": int(track_id),
                        "object_track_id": None
                        if obj.get("track_id") is None
                        else int(obj.get("track_id")),
                        "label": obj.get("label"),
                        "kind": kind,
                        "confidence": confidence,
                        "stable_hits": obj.get("attribution_hits"),
                        "mask_overlap": obj.get("mask_overlap"),
                    }
                )

        filename = "frame_{idx:04d}_{t_ms:08d}ms.jpg".format(
            idx=len(self.frames) + 1,
            t_ms=int(round(float(t_sec) * 1000.0)),
        )
        path = self.output_dir / filename
        cv2.imwrite(str(path), annotated)
        record = {
            "path": str(path),
            "time_sec": round(float(t_sec), 3),
            "window_index": int(window_idx),
            "people": [
                {
                    "track_id": int(item.get("track_id")),
                    "label": item.get("label", "person"),
                    "box": _round_box(item["xyxy"]),
                }
                for item in person_tracks
            ],
            "objects": [
                {
                    "track_id": None if item.get("track_id") is None else int(item.get("track_id")),
                    "label": item.get("label", "object"),
                    "box": _round_box(item["xyxy"]),
                    "conf": round(float(item.get("conf", 0.0)), 3),
                }
                for item in object_tracks
            ],
            "attributions": attribution_records,
        }
        self.frames.append(record)
        self._last_saved_t = float(t_sec)

    def close(self):
        if not self.enabled or self.output_dir is None:
            return None
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w") as handle:
            json.dump({"frames": self.frames}, handle, indent=2)
        return {
            "dir": str(self.output_dir),
            "manifest": str(manifest_path),
            "frames": len(self.frames),
        }


def default_debug_evidence_dir(run_id):
    explicit_root = os.environ.get("SURVEILLANCE_DEBUG_EVIDENCE_DIR")
    if explicit_root:
        return str(Path(explicit_root) / str(run_id or "run"))

    memory_db = os.environ.get("SURVEILLANCE_MEMORY_DB")
    if memory_db:
        return str(Path(memory_db).resolve().parent / "debug_evidence" / str(run_id or "run"))

    return str(Path(".debug_evidence") / str(run_id or "run"))
