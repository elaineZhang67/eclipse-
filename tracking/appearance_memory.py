from collections import deque

import cv2
import numpy as np

from preprocessing.clip_builder import safe_crop


def _l2_normalize(vector):
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return array
    return array / norm


def _cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return None
    return float(np.dot(vec_a, vec_b))


class AppearanceEmbedder:
    """
    Lightweight appearance descriptor for person crops.

    The goal here is not perfect re-identification; it is a compact, dependency-light
    embedding that is good enough to stabilize short-term identity continuity on top of
    the existing tracker.
    """

    def __init__(self, crop_width=48, crop_height=96, hist_bins=16, grid_size=2):
        self.crop_width = max(16, int(crop_width))
        self.crop_height = max(32, int(crop_height))
        self.hist_bins = max(4, int(hist_bins))
        self.grid_size = max(1, int(grid_size))

    def extract(self, frame_bgr, xyxy):
        crop = safe_crop(frame_bgr, xyxy)
        if crop is None or crop.size == 0:
            return None
        return self.extract_from_crop(crop)

    def extract_from_crop(self, crop_bgr):
        resized = cv2.resize(
            crop_bgr,
            (self.crop_width, self.crop_height),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] /= 179.0
        hsv[:, :, 1] /= 255.0
        hsv[:, :, 2] /= 255.0
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        features = []

        for image in (rgb, hsv):
            features.extend(image.mean(axis=(0, 1)).tolist())
            features.extend(image.std(axis=(0, 1)).tolist())

            for row_chunk in np.array_split(image, self.grid_size, axis=0):
                for patch in np.array_split(row_chunk, self.grid_size, axis=1):
                    features.extend(patch.mean(axis=(0, 1)).tolist())

        hist, _ = np.histogram(gray, bins=self.hist_bins, range=(0.0, 1.0), density=True)
        features.extend(hist.astype(np.float32).tolist())

        height, width = crop_bgr.shape[:2]
        aspect_ratio = float(width) / float(max(height, 1))
        features.append(aspect_ratio)

        return _l2_normalize(np.asarray(features, dtype=np.float32))


class TrackMemoryBank:
    def __init__(
        self,
        similarity_threshold=0.82,
        mapping_ttl_sec=8.0,
        reassoc_gap_sec=20.0,
        prototype_momentum=0.85,
        history_size=8,
        rewrite_track_ids=True,
        embedder=None,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.mapping_ttl_sec = max(0.0, float(mapping_ttl_sec))
        self.reassoc_gap_sec = max(self.mapping_ttl_sec, float(reassoc_gap_sec))
        self.prototype_momentum = min(0.99, max(0.0, float(prototype_momentum)))
        self.history_size = max(1, int(history_size))
        self.rewrite_track_ids = bool(rewrite_track_ids)
        self.embedder = embedder or AppearanceEmbedder()

        self._next_memory_id = 1
        self._states = {}
        self._raw_to_memory = {}
        self._raw_last_seen = {}

    def _expire_raw_mappings(self, t_sec):
        stale_raw_ids = [
            raw_id
            for raw_id, last_seen in self._raw_last_seen.items()
            if (float(t_sec) - float(last_seen)) > self.mapping_ttl_sec
        ]
        for raw_id in stale_raw_ids:
            self._raw_to_memory.pop(raw_id, None)
            self._raw_last_seen.pop(raw_id, None)

    def _new_state(self, embedding, raw_track_id, t_sec):
        memory_track_id = self._next_memory_id
        self._next_memory_id += 1
        self._states[memory_track_id] = {
            "memory_track_id": int(memory_track_id),
            "embedding": np.asarray(embedding, dtype=np.float32),
            "first_seen": float(t_sec),
            "last_seen": float(t_sec),
            "hits": 0,
            "tracker_track_ids": set(),
            "recent_similarities": deque(maxlen=self.history_size),
            "recent_times": deque(maxlen=self.history_size),
        }
        self._bind_raw_track(raw_track_id, memory_track_id, t_sec)
        return memory_track_id

    def _bind_raw_track(self, raw_track_id, memory_track_id, t_sec):
        self._raw_to_memory[int(raw_track_id)] = int(memory_track_id)
        self._raw_last_seen[int(raw_track_id)] = float(t_sec)

    def _match_embedding(self, embedding, t_sec, claimed_memory_ids):
        best_memory_id = None
        best_score = None

        for memory_track_id, state in self._states.items():
            if memory_track_id in claimed_memory_ids:
                continue
            if (float(t_sec) - float(state["last_seen"])) > self.reassoc_gap_sec:
                continue

            score = _cosine_similarity(embedding, state.get("embedding"))
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_memory_id = memory_track_id

        if best_memory_id is None or best_score is None or best_score < self.similarity_threshold:
            return None, None
        return int(best_memory_id), float(best_score)

    def _update_state(self, memory_track_id, raw_track_id, embedding, similarity, t_sec):
        state = self._states[int(memory_track_id)]
        if state.get("embedding") is None:
            state["embedding"] = np.asarray(embedding, dtype=np.float32)
        else:
            updated = (
                (self.prototype_momentum * state["embedding"]) +
                ((1.0 - self.prototype_momentum) * np.asarray(embedding, dtype=np.float32))
            )
            state["embedding"] = _l2_normalize(updated)

        state["hits"] += 1
        state["last_seen"] = float(t_sec)
        state["tracker_track_ids"].add(int(raw_track_id))
        if similarity is not None:
            state["recent_similarities"].append(float(similarity))
        state["recent_times"].append(float(t_sec))
        self._bind_raw_track(raw_track_id, memory_track_id, t_sec)

    def _state_summary(self, memory_track_id):
        state = self._states[int(memory_track_id)]
        similarities = list(state["recent_similarities"])
        mean_similarity = None
        if similarities:
            mean_similarity = round(float(sum(similarities) / len(similarities)), 4)

        return {
            "memory_track_id": int(memory_track_id),
            "tracker_track_ids": sorted(int(item) for item in state["tracker_track_ids"]),
            "first_seen": round(float(state["first_seen"]), 2),
            "last_seen": round(float(state["last_seen"]), 2),
            "hits": int(state["hits"]),
            "mean_similarity": mean_similarity,
            "last_similarity": None if not similarities else round(float(similarities[-1]), 4),
            "recent_update_times": [round(float(item), 2) for item in state["recent_times"]],
        }

    def update(self, frame_bgr, person_tracks, t_sec):
        self._expire_raw_mappings(t_sec)

        annotated = []
        claimed_memory_ids = set()
        for track in person_tracks:
            raw_track_id = int(track["track_id"])
            embedding = self.embedder.extract(frame_bgr, track["xyxy"])
            if embedding is None:
                bound_memory_track_id = self._raw_to_memory.get(raw_track_id)
                annotated_track = dict(track)
                annotated_track["raw_track_id"] = raw_track_id
                annotated_track["memory_track_id"] = bound_memory_track_id
                if self.rewrite_track_ids and bound_memory_track_id is not None:
                    annotated_track["track_id"] = int(bound_memory_track_id)
                annotated.append(annotated_track)
                continue

            memory_track_id = None
            similarity = None
            bound_memory_track_id = self._raw_to_memory.get(raw_track_id)
            if bound_memory_track_id is not None and bound_memory_track_id not in claimed_memory_ids:
                memory_track_id = int(bound_memory_track_id)
                similarity = _cosine_similarity(embedding, self._states[memory_track_id].get("embedding"))
            else:
                memory_track_id, similarity = self._match_embedding(
                    embedding,
                    t_sec=t_sec,
                    claimed_memory_ids=claimed_memory_ids,
                )
                if memory_track_id is None:
                    memory_track_id = self._new_state(embedding, raw_track_id, t_sec)
                    similarity = 1.0

            self._update_state(
                memory_track_id,
                raw_track_id=raw_track_id,
                embedding=embedding,
                similarity=similarity,
                t_sec=t_sec,
            )
            claimed_memory_ids.add(int(memory_track_id))

            annotated_track = dict(track)
            annotated_track["raw_track_id"] = raw_track_id
            annotated_track["memory_track_id"] = int(memory_track_id)
            annotated_track["appearance_similarity"] = None if similarity is None else round(float(similarity), 4)
            annotated_track["track_memory"] = self._state_summary(memory_track_id)
            if self.rewrite_track_ids:
                annotated_track["track_id"] = int(memory_track_id)
            annotated.append(annotated_track)

        return annotated

    def export_metadata(self):
        return {
            int(memory_track_id): {
                "memory_track_id": snapshot["memory_track_id"],
                "tracker_track_ids": snapshot["tracker_track_ids"],
                "appearance_updates": snapshot["hits"],
                "appearance_mean_similarity": snapshot["mean_similarity"],
                "appearance_last_similarity": snapshot["last_similarity"],
                "appearance_recent_times": snapshot["recent_update_times"],
                "identity_history": (
                    "memory track {memory_track_id} linked tracker ids {tracker_ids} "
                    "from {first_seen}s to {last_seen}s across {hits} appearance updates"
                ).format(
                    memory_track_id=snapshot["memory_track_id"],
                    tracker_ids=", ".join(str(item) for item in snapshot["tracker_track_ids"]) or "none",
                    first_seen=snapshot["first_seen"],
                    last_seen=snapshot["last_seen"],
                    hits=snapshot["hits"],
                ),
            }
            for memory_track_id, snapshot in (
                (memory_track_id, self._state_summary(memory_track_id))
                for memory_track_id in sorted(self._states)
            )
        }

    def export_bank(self):
        return [
            self._state_summary(memory_track_id)
            for memory_track_id in sorted(self._states)
        ]
