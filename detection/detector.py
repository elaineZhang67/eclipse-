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


class YoloDetector:
    def __init__(self, weights="yolov8n.pt", conf=0.25, classes=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.names = _extract_names(self.model)
        self.requested_classes = list(classes or [])
        self.class_ids, self.missing_classes = self._resolve_classes(self.requested_classes)

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

    def detect(self, frame_bgr):
        if self.class_ids == []:
            return []

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            classes=self.class_ids,
            verbose=False
        )
        r = results[0]
        dets = []
        if r.boxes is None:
            return dets

        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else [None] * len(xyxy)
        for b, c, class_id in zip(xyxy, conf, cls):
            dets.append({
                "xyxy": b.tolist(),
                "conf": float(c),
                "class_id": None if class_id is None else int(class_id),
                "label": self.names.get(int(class_id), str(class_id)) if class_id is not None else "unknown",
            })
        return dets
