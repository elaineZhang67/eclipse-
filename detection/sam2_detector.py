import cv2
import torch

from detection.detector import YoloDetector
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


def _select_masks(mask_tensor):
    masks = mask_tensor
    if torch.is_tensor(masks):
        masks = masks.detach().cpu()

    if masks.ndim == 5:
        masks = masks[0]
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    return masks


def _mask_to_box(mask, fallback_box, threshold):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu()
    binary = mask > float(threshold)
    coords = torch.nonzero(binary, as_tuple=False)
    if coords.numel() == 0:
        return [float(v) for v in fallback_box]

    y1 = int(coords[:, 0].min().item())
    x1 = int(coords[:, 1].min().item())
    y2 = int(coords[:, 0].max().item()) + 1
    x2 = int(coords[:, 1].max().item()) + 1
    return [float(x1), float(y1), float(x2), float(y2)]


def _score_for_index(scores, idx, default_score):
    if scores is None:
        return float(default_score)
    values = scores.detach().cpu() if torch.is_tensor(scores) else scores
    try:
        score = values[0, idx, 0]
    except Exception:
        try:
            score = values[idx, 0]
        except Exception:
            try:
                score = values[idx]
            except Exception:
                return float(default_score)
    if torch.is_tensor(score):
        score = score.item()
    return float(score)


class Sam2Detector:
    """
    SAM2 is promptable segmentation, not open-vocabulary detection.

    This detector therefore uses YOLO for class-aware object proposals, then uses
    SAM2 box prompts to refine each proposed object into a mask-derived box.
    """

    def __init__(
        self,
        model_id="facebook/sam2.1-hiera-large",
        yolo_weights="yolov8n.pt",
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
        self.device = resolve_device(device)
        self._frame_index = 0
        self._next_track_id = 1
        self._active_tracks = {}
        self.proposal_detector = YoloDetector(
            weights=yolo_weights,
            conf=conf,
            classes=classes,
        )
        self.missing_classes = list(getattr(self.proposal_detector, "missing_classes", []))

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("SAM2 detection requires Pillow.") from exc

        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 detection requires a transformers installation that provides "
                "Sam2Model and Sam2Processor."
            ) from exc

        self.image_cls = Image
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(
            model_id,
            torch_dtype=default_torch_dtype(self.device),
        ).to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype

    def detect(self, frame_bgr):
        proposals = self.proposal_detector.detect(frame_bgr)
        if not proposals:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.image_cls.fromarray(frame_rgb)
        input_boxes = [[proposal["xyxy"] for proposal in proposals]]
        inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt")
        original_sizes = inputs["original_sizes"]
        inputs = _move_to_device(inputs, self.device, float_dtype=self.model_dtype)

        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            original_sizes,
        )[0]
        masks = _select_masks(masks)

        detections = []
        for idx, proposal in enumerate(proposals):
            mask = masks[min(idx, masks.shape[0] - 1)]
            refined_box = _mask_to_box(mask, proposal["xyxy"], self.mask_threshold)
            sam_score = _score_for_index(
                getattr(outputs, "iou_scores", None),
                idx,
                default_score=1.0,
            )
            detections.append(
                {
                    "xyxy": refined_box,
                    "proposal_xyxy": [float(v) for v in proposal["xyxy"]],
                    "conf": float(proposal["conf"]) * float(sam_score),
                    "class_id": proposal.get("class_id"),
                    "label": proposal.get("label", "unknown"),
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
