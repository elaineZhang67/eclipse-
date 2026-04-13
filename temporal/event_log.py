import math
from collections import defaultdict
from itertools import combinations


def _center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _width(box):
    return max(1.0, float(box[2]) - float(box[0]))


def _height(box):
    return max(1.0, float(box[3]) - float(box[1]))


def _area(box):
    return _width(box) * _height(box)


def _center_in_box(point, box):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _union_box(box_a, box_b):
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def _round_box(box):
    return [round(float(v), 1) for v in box]


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    denom = _area(box_a) + _area(box_b) - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _normalized_center_distance(box_a, box_b):
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    distance = math.hypot(ax - bx, ay - by)
    scale = max(1.0, 0.5 * (_width(box_a) + _width(box_b)))
    return distance / scale


class SceneEventBuilder:
    def __init__(
        self,
        window_sec=5.0,
        combine_iou=0.05,
        combine_dist=1.2,
        nearby_dist=2.5,
    ):
        self.window_sec = window_sec
        self.combine_iou = combine_iou
        self.combine_dist = combine_dist
        self.nearby_dist = nearby_dist
        self.track_state = defaultdict(
            lambda: {
                "first_seen": None,
                "last_seen": None,
                "object_track_hits": defaultdict(int),
                "object_track_labels": {},
                "object_label_hits": defaultdict(int),
                "interaction_counts": defaultdict(int),
                "interaction_kinds": defaultdict(set),
            }
        )
        self.object_state = defaultdict(
            lambda: {
                "label": None,
                "first_seen": None,
                "last_seen": None,
                "hits": 0,
                "owner_counts": defaultdict(int),
                "last_owner_track_id": None,
            }
        )
        self.windows = {}

    def _get_window(self, t_sec):
        window_idx = int(t_sec // self.window_sec)
        if window_idx not in self.windows:
            start = window_idx * self.window_sec
            self.windows[window_idx] = {
                "index": window_idx,
                "start": start,
                "end": start + self.window_sec,
                "active_tracks": set(),
                "events": [],
                "signatures": set(),
            }
        return self.windows[window_idx]

    def _add_event(self, window, event, signature):
        if signature in window["signatures"]:
            return False
        window["signatures"].add(signature)
        window["events"].append(event)
        return True

    def _assignment_score(self, person_box, obj_box, obj_center):
        score = 0.0
        if _center_in_box(obj_center, person_box):
            score += 3.0
        score += 2.0 * _iou(obj_box, person_box)
        score += max(0.0, 1.5 - _normalized_center_distance(obj_box, person_box))
        return score

    def _record_object_presence(self, t_sec, object_tracks):
        for obj in object_tracks:
            object_id = obj.get("track_id")
            if object_id is None:
                continue

            state = self.object_state[object_id]
            state["label"] = obj.get("label", state["label"])
            if state["first_seen"] is None:
                state["first_seen"] = t_sec
            state["last_seen"] = t_sec
            state["hits"] += 1

    def _assign_objects(self, person_tracks, object_tracks):
        assigned = defaultdict(list)
        for obj in object_tracks:
            obj_box = obj["xyxy"]
            obj_center = _center(obj_box)
            object_id = obj.get("track_id")
            best_tid = None
            best_score = float("-inf")
            scores = {}

            for person in person_tracks:
                person_box = person["xyxy"]
                score = self._assignment_score(person_box, obj_box, obj_center)
                scores[person["track_id"]] = score

                if score > best_score:
                    best_score = score
                    best_tid = person["track_id"]

            if object_id is not None:
                sticky_owner = self.object_state[object_id]["last_owner_track_id"]
                sticky_score = scores.get(sticky_owner)
                if sticky_score is not None and sticky_score >= max(0.5, best_score - 0.25):
                    best_tid = sticky_owner
                    best_score = sticky_score

            if best_tid is not None and best_score >= 0.5:
                assigned[best_tid].append(obj)
                if object_id is not None:
                    state = self.object_state[object_id]
                    state["owner_counts"][best_tid] += 1
                    state["last_owner_track_id"] = best_tid
        return assigned

    def _record_track_presence(self, t_sec, person_tracks):
        window = self._get_window(t_sec)
        for track in person_tracks:
            tid = track["track_id"]
            window["active_tracks"].add(tid)

            state = self.track_state[tid]
            if state["first_seen"] is None:
                state["first_seen"] = t_sec
                inserted = self._add_event(
                    window,
                    {
                        "type": "enter",
                        "track_id": tid,
                        "label": track.get("label", "person"),
                        "box": _round_box(track["xyxy"]),
                    },
                    ("enter", tid),
                )
            state["last_seen"] = t_sec

    def _record_object_events(self, window, assignments):
        for tid, objects in assignments.items():
            for obj in objects:
                label = obj.get("label", "unknown")
                self.track_state[tid]["object_label_hits"][label] += 1
                object_id = obj.get("track_id")
                if object_id is not None:
                    self.track_state[tid]["object_track_labels"][int(object_id)] = label
                    self.track_state[tid]["object_track_hits"][int(object_id)] += 1

            event_objects = []
            for obj in sorted(
                objects,
                key=lambda item: (
                    item.get("label", "unknown"),
                    -1 if item.get("track_id") is None else int(item["track_id"]),
                ),
            ):
                object_id = obj.get("track_id")
                event_objects.append(
                    {
                        "object_track_id": None if object_id is None else int(object_id),
                        "label": obj.get("label", "unknown"),
                    }
                )

            self._add_event(
                window,
                {
                    "type": "attributed_objects",
                    "track_id": tid,
                    "objects": event_objects,
                },
                (
                    "objects",
                    tid,
                    tuple(
                        (obj["label"], obj["object_track_id"])
                        for obj in event_objects
                    ),
                ),
            )

    def _record_interactions(self, window, person_tracks):
        for left, right in combinations(person_tracks, 2):
            tid_a = left["track_id"]
            tid_b = right["track_id"]
            pair_key = tuple(sorted((tid_a, tid_b)))
            left_box = left["xyxy"]
            right_box = right["xyxy"]
            iou = _iou(left_box, right_box)
            distance = _normalized_center_distance(left_box, right_box)

            if iou >= self.combine_iou or distance <= self.combine_dist:
                event = {
                    "type": "interaction",
                    "kind": "close",
                    "track_ids": list(pair_key),
                    "combined_box": _round_box(_union_box(left_box, right_box)),
                    "distance": round(distance, 3),
                    "iou": round(iou, 3),
                }
            elif distance <= self.nearby_dist:
                event = {
                    "type": "interaction",
                    "kind": "nearby",
                    "track_ids": list(pair_key),
                    "boxes": [_round_box(left_box), _round_box(right_box)],
                    "distance": round(distance, 3),
                    "iou": round(iou, 3),
                }
            else:
                continue

            inserted = self._add_event(
                window,
                event,
                ("interaction", event["kind"], pair_key),
            )
            if inserted:
                self.track_state[tid_a]["interaction_counts"][tid_b] += 1
                self.track_state[tid_b]["interaction_counts"][tid_a] += 1
                self.track_state[tid_a]["interaction_kinds"][tid_b].add(event["kind"])
                self.track_state[tid_b]["interaction_kinds"][tid_a].add(event["kind"])

    def update(self, t_sec, person_tracks, object_tracks):
        window = self._get_window(t_sec)
        self._record_track_presence(t_sec, person_tracks)
        self._record_object_presence(t_sec, object_tracks)
        assignments = self._assign_objects(person_tracks, object_tracks)
        self._record_object_events(window, assignments)
        self._record_interactions(window, person_tracks)

    def finalize(self):
        event_log = []
        for state_tid, state in self.track_state.items():
            if state["last_seen"] is None:
                continue
            exit_window = self._get_window(state["last_seen"])
            self._add_event(
                exit_window,
                {
                    "type": "last_seen",
                    "track_id": state_tid,
                    "time": round(state["last_seen"], 2),
                },
                ("last_seen", state_tid),
            )

        for idx in sorted(self.windows):
            window = self.windows[idx]
            event_log.append(
                {
                    "start": round(window["start"], 2),
                    "end": round(window["end"], 2),
                    "active_tracks": sorted(window["active_tracks"]),
                    "events": window["events"],
                }
            )

        per_track_metadata = {}
        for tid in sorted(self.track_state):
            state = self.track_state[tid]
            if state["first_seen"] is None:
                continue

            object_tracks = []
            label_counts = defaultdict(int)
            for object_id, label in sorted(
                state["object_track_labels"].items(),
                key=lambda item: (item[1], item[0]),
            ):
                object_state = self.object_state.get(object_id, {})
                label_counts[label] += 1
                object_tracks.append(
                    {
                        "object_track_id": int(object_id),
                        "label": label,
                        "first_seen": None
                        if object_state.get("first_seen") is None
                        else round(object_state["first_seen"], 2),
                        "last_seen": None
                        if object_state.get("last_seen") is None
                        else round(object_state["last_seen"], 2),
                        "hits": int(state["object_track_hits"].get(object_id, 0)),
                    }
                )

            if label_counts:
                object_counts = {
                    label: count
                    for label, count in sorted(
                        label_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                }
            else:
                object_counts = {
                    label: count
                    for label, count in sorted(
                        state["object_label_hits"].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                }
            interactions = []
            for other_tid, count in sorted(
                state["interaction_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            ):
                interactions.append(
                    {
                        "other_track_id": int(other_tid),
                        "count": int(count),
                        "kinds": sorted(state["interaction_kinds"][other_tid]),
                    }
                )

            per_track_metadata[tid] = {
                "first_seen": round(state["first_seen"], 2),
                "last_seen": round(state["last_seen"], 2),
                "object_counts": object_counts,
                "objects": list(object_counts.keys()),
                "object_tracks": object_tracks,
                "interactions": interactions,
            }

        return event_log, per_track_metadata
