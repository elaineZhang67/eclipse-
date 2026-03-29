import random

import cv2

from preprocessing.clip_builder import safe_crop


def _resize_long_edge(image_bgr, max_long_edge):
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


class _ReservoirImageBank:
    def __init__(self, max_items, rng):
        self.max_items = max(0, int(max_items))
        self._rng = rng
        self._seen = 0
        self.items = []

    def add(self, sample):
        if self.max_items <= 0:
            return

        self._seen += 1
        if len(self.items) < self.max_items:
            self.items.append(sample)
            return

        idx = self._rng.randrange(self._seen)
        if idx < self.max_items:
            self.items[idx] = sample


class VisualEvidenceCollector:
    def __init__(
        self,
        max_track_images=4,
        max_scene_images=4,
        track_gap_sec=1.5,
        scene_gap_sec=3.0,
        track_long_edge=256,
        scene_long_edge=768,
    ):
        self.max_track_images = max(0, int(max_track_images))
        self.max_scene_images = max(0, int(max_scene_images))
        self.track_gap_sec = max(0.0, float(track_gap_sec))
        self.scene_gap_sec = max(0.0, float(scene_gap_sec))
        self.track_long_edge = max(0, int(track_long_edge))
        self.scene_long_edge = max(0, int(scene_long_edge))

        self._rng = random.Random(0)
        self._scene_bank = _ReservoirImageBank(self.max_scene_images, self._rng)
        self._track_banks = {}
        self._last_scene_t = None
        self._last_track_t = {}

    def _track_bank(self, track_id):
        bank = self._track_banks.get(track_id)
        if bank is None:
            bank = _ReservoirImageBank(self.max_track_images, self._rng)
            self._track_banks[track_id] = bank
        return bank

    def update_scene(self, frame_bgr, t_sec):
        if self.max_scene_images <= 0:
            return
        if self._last_scene_t is not None and (t_sec - self._last_scene_t) < self.scene_gap_sec:
            return

        resized = _resize_long_edge(frame_bgr, self.scene_long_edge)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._scene_bank.add({"t_sec": float(t_sec), "image_rgb": frame_rgb})
        self._last_scene_t = float(t_sec)

    def update_tracks(self, frame_bgr, person_tracks, t_sec):
        if self.max_track_images <= 0:
            return

        for track in person_tracks:
            track_id = track["track_id"]
            last_t = self._last_track_t.get(track_id)
            if last_t is not None and (t_sec - last_t) < self.track_gap_sec:
                continue

            crop = safe_crop(frame_bgr, track["xyxy"])
            if crop is None:
                continue

            resized = _resize_long_edge(crop, self.track_long_edge)
            crop_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self._track_bank(track_id).add({"t_sec": float(t_sec), "image_rgb": crop_rgb})
            self._last_track_t[track_id] = float(t_sec)

    def get_track_images(self, track_id, start_sec=None, end_sec=None):
        bank = self._track_banks.get(track_id)
        if bank is None:
            return []
        samples = sorted(bank.items, key=lambda item: item["t_sec"])
        if start_sec is not None:
            samples = [item for item in samples if item["t_sec"] >= float(start_sec)]
        if end_sec is not None:
            samples = [item for item in samples if item["t_sec"] <= float(end_sec)]
        return [item["image_rgb"] for item in samples]

    def get_scene_images(self, start_sec=None, end_sec=None):
        samples = sorted(self._scene_bank.items, key=lambda item: item["t_sec"])
        if start_sec is not None:
            samples = [item for item in samples if item["t_sec"] >= float(start_sec)]
        if end_sec is not None:
            samples = [item for item in samples if item["t_sec"] <= float(end_sec)]
        return [item["image_rgb"] for item in samples]
