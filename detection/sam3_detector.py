import os

import cv2
import torch

from runtime.device import default_torch_dtype, resolve_device


def _iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(1.0, float(box_a[2]) - float(box_a[0])) * max(1.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(1.0, float(box_b[2]) - float(box_b[0])) * max(1.0, float(box_b[3]) - float(box_b[1]))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _move_to_device(batch, device, float_dtype=None):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point() and float_dtype is not None:
                moved[key] = value.to(device=device, dtype=float_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _hub_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _is_gated_repo_error(exc):
    text = str(exc).lower()
    return "403 forbidden" in text or "gated" in text or "public gated repositories" in text


def _sam3_access_error(model_id, exc):
    return RuntimeError(
        "Could not load SAM3 model '{model_id}' from Hugging Face. "
        "The model files are gated, so the runtime needs an HF_TOKEN/HUGGING_FACE_HUB_TOKEN "
        "that has access to public gated repositories and has accepted the facebook/sam3 terms. "
        "Original error: {error}".format(model_id=model_id, error=exc)
    )


def _load_sam3_processor(model_id, token):
    from transformers import CLIPTokenizer, Sam3ImageProcessor, Sam3Processor

    try:
        return Sam3Processor.from_pretrained(model_id, token=token)
    except OSError as exc:
        first_error = exc

    try:
        image_processor = Sam3ImageProcessor()
        tokenizer = CLIPTokenizer.from_pretrained(model_id, token=token)
        return Sam3Processor(image_processor=image_processor, tokenizer=tokenizer)
    except Exception as exc:
        if _is_gated_repo_error(exc):
            raise _sam3_access_error(model_id, exc) from exc
        raise RuntimeError(
            "Could not load SAM3 processor for '{model_id}'. The Hugging Face repo currently "
            "ships processor_config.json, while some transformers builds still look for "
            "preprocessor_config.json; the fallback processor construction also failed. "
            "Original errors: {first_error}; {second_error}".format(
                model_id=model_id,
                first_error=first_error,
                second_error=exc,
            )
        ) from exc


def _load_sam3_model(model_cls, model_id, device, token):
    try:
        model = model_cls.from_pretrained(
            model_id,
            token=token,
            torch_dtype=default_torch_dtype(device),
        )
    except Exception as exc:
        if _is_gated_repo_error(exc):
            raise _sam3_access_error(model_id, exc) from exc
        raise

    return model.to(device)


class Sam3Detector:
    def __init__(
        self,
        model_id="facebook/sam3",
        conf=0.25,
        mask_threshold=0.5,
        classes=None,
        device="auto",
        track_iou=0.3,
        track_ttl=12,
    ):
        self.model_id = model_id
        self.conf = float(conf)
        self.mask_threshold = float(mask_threshold)
        self.track_iou = float(track_iou)
        self.track_ttl = max(1, int(track_ttl))
        self.requested_classes = [str(item).strip() for item in (classes or []) if str(item).strip()]
        self.missing_classes = []
        self.device = resolve_device(device)
        self._frame_index = 0
        self._next_track_id = 1
        self._active_tracks = {}
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("SAM3 detection requires Pillow.") from exc

        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "SAM3 detection requires a newer transformers installation that provides "
                "Sam3Model and Sam3Processor."
            ) from exc

        self.image_cls = Image
        token = _hub_token()
        self.processor = _load_sam3_processor(model_id, token)
        self.model = _load_sam3_model(Sam3Model, model_id, self.device, token)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype

    def detect(self, frame_bgr):
        if not self.requested_classes:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.image_cls.fromarray(frame_rgb)
        detections = []

        for label in self.requested_classes:
            inputs = self.processor(images=image, text=label, return_tensors="pt")
            target_sizes = inputs["original_sizes"].cpu().tolist()
            inputs = _move_to_device(inputs, self.device, float_dtype=self.model_dtype)

            with torch.inference_mode():
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.conf,
                mask_threshold=self.mask_threshold,
                target_sizes=target_sizes,
            )[0]

            boxes = results.get("boxes")
            scores = results.get("scores")
            if boxes is None or scores is None:
                continue

            boxes = boxes.detach().cpu().tolist() if torch.is_tensor(boxes) else boxes
            scores = scores.detach().cpu().tolist() if torch.is_tensor(scores) else scores
            for box, score in zip(boxes, scores):
                detections.append(
                    {
                        "xyxy": [float(v) for v in box],
                        "conf": float(score),
                        "class_id": None,
                        "label": label,
                    }
                )

        return detections

    def _prune_tracks(self):
        stale_ids = [
            track_id
            for track_id, state in self._active_tracks.items()
            if (self._frame_index - state["last_frame_index"]) > self.track_ttl
        ]
        for track_id in stale_ids:
            self._active_tracks.pop(track_id, None)

    def _assign_track_ids(self, detections):
        self._frame_index += 1
        self._prune_tracks()

        detections = sorted(
            detections,
            key=lambda item: (item.get("label", ""), -float(item.get("conf", 0.0))),
        )
        used_track_ids = set()
        tracked = []

        for det in detections:
            label = det.get("label", "unknown")
            box = det.get("xyxy", [0.0, 0.0, 0.0, 0.0])
            best_track_id = None
            best_iou = 0.0
            for track_id, state in self._active_tracks.items():
                if track_id in used_track_ids or state.get("label") != label:
                    continue
                score = _iou(box, state.get("xyxy", box))
                if score >= self.track_iou and score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self._next_track_id
                self._next_track_id += 1

            self._active_tracks[best_track_id] = {
                "label": label,
                "xyxy": list(box),
                "last_frame_index": self._frame_index,
            }
            used_track_ids.add(best_track_id)
            tracked.append(
                {
                    **det,
                    "track_id": int(best_track_id),
                }
            )

        return tracked

    def update(self, frame_bgr):
        return self._assign_track_ids(self.detect(frame_bgr))
