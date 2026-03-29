import cv2
from collections import defaultdict, deque

def safe_crop(frame_bgr, xyxy):
    x1, y1, x2, y2 = xyxy
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w-1, int(x1)))
    y1 = max(0, min(h-1, int(y1)))
    x2 = max(0, min(w-1, int(x2)))
    y2 = max(0, min(h-1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]

class ClipBuilder:
    def __init__(self, clip_len=16, stride=8, crop_size=224):
        self.clip_len = clip_len
        self.stride = stride
        self.crop_size = crop_size

        self.buffers = defaultdict(lambda: deque(maxlen=self.clip_len))
        self.time_buffers = defaultdict(lambda: deque(maxlen=self.clip_len))
        self.counts = defaultdict(int)

    def push(self, frame_bgr, tracks, t_sec):
        """
        For each track_id, crop person patch, resize to 224, convert to RGB, push.
        When buffer full and stride reached, emit a clip for that person.
        """
        ready = []
        for tr in tracks:
            tid = tr["track_id"]
            crop = safe_crop(frame_bgr, tr["xyxy"])
            if crop is None:
                continue
            crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            self.buffers[tid].append(crop_rgb)
            self.time_buffers[tid].append(t_sec)
            self.counts[tid] += 1

            if len(self.buffers[tid]) == self.clip_len and (self.counts[tid] % self.stride == 0):
                clip_rgb = list(self.buffers[tid])
                t_start = float(self.time_buffers[tid][0])
                t_end = float(self.time_buffers[tid][-1])
                ready.append({
                    "track_id": tid,
                    "clip_rgb": clip_rgb,
                    "t_start": t_start,
                    "t_end": t_end
                })
        return ready
