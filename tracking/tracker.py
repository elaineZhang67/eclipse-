from ultralytics import YOLO


def _normalize_label(value):
    return str(value).strip().lower().replace("_", " ")


def _extract_names(model):
    raw_names = getattr(model, "names", {})
    if isinstance(raw_names, dict):
        return {int(idx): str(name) for idx, name in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {int(idx): str(name) for idx, name in enumerate(raw_names)}
    return {}


class MultiObjectTracker:
    def __init__(self, weights="yolov8n.pt", tracker_cfg="botsort.yaml", conf=0.25, classes=None):
        self.yolo_for_track = YOLO(weights)
        self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.names = _extract_names(self.yolo_for_track)
        self.requested_classes = list(classes or [])
        self.class_ids, self.missing_classes = self._resolve_classes(self.requested_classes)
        self._persist = True

    def _resolve_classes(self, classes):
        if not classes:
            return None, []

        name_to_id = {_normalize_label(name): idx for idx, name in self.names.items()}
        class_ids = []
        missing = []
        for item in classes:
            if isinstance(item, int):
                class_ids.append(int(item))
                continue

            item_txt = str(item).strip()
            if item_txt.isdigit():
                class_ids.append(int(item_txt))
                continue

            match = name_to_id.get(_normalize_label(item_txt))
            if match is None:
                missing.append(item_txt)
                continue
            class_ids.append(match)
        return sorted(set(class_ids)), missing

    def update(self, frame_bgr):
        if self.class_ids == []:
            return []

        results = self.yolo_for_track.track(
            source=frame_bgr,
            tracker=self.tracker_cfg,
            persist=self._persist,
            classes=self.class_ids,
            conf=self.conf,
            verbose=False
        )
        r = results[0]
        boxes = r.boxes
        if boxes is None or boxes.id is None:
            return []

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else [None] * len(xyxy)

        tracks = []
        for tid, b, c, class_id in zip(ids, xyxy, conf, cls):
            tracks.append({
                "track_id": int(tid),
                "xyxy": b.tolist(),
                "conf": float(c),
                "class_id": None if class_id is None else int(class_id),
                "label": self.names.get(int(class_id), str(class_id)) if class_id is not None else "unknown",
            })
        return tracks
