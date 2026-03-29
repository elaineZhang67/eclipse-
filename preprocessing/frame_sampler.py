import cv2
import math

class FrameSampler:
    def __init__(self, target_fps=12.0):
        self.target_fps = target_fps
        self.cap = None
        self.native_fps = 0.0
        self.total_frames = 0
        self.duration_sec = 0.0
        self.estimated_sampled_frames = 0
        self._frame_period = 1
        self._frame_idx = -1
        self._last_t_sec = 0.0

    def open(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.native_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.native_fps <= 0:
            self.native_fps = 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_period = max(1, int(round(self.native_fps / self.target_fps)))
        self.duration_sec = (
            float(self.total_frames) / max(self.native_fps, 1e-6)
            if self.total_frames > 0
            else 0.0
        )
        self.estimated_sampled_frames = (
            int(math.ceil(float(self.total_frames) / float(self._frame_period)))
            if self.total_frames > 0
            else 0
        )
        self._frame_idx = -1
        self._last_t_sec = 0.0
        return self.cap

    def _timestamp_sec(self):
        computed_t = max(0.0, float(self._frame_idx) / max(self.native_fps, 1e-6))
        pos_msec_t = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

        # Some encoded videos report unstable POS_MSEC values; fall back to frame index.
        if pos_msec_t <= 0.0 or abs(pos_msec_t - computed_t) > max(0.5, 2.0 / self.native_fps):
            t_sec = computed_t
        else:
            t_sec = pos_msec_t

        t_sec = max(t_sec, self._last_t_sec)
        self._last_t_sec = t_sec
        return t_sec

    def read(self):
        """
        Read frames with downsampling.
        Returns: (ok, frame_bgr, t_sec)
        """
        if self.cap is None:
            return False, None, 0.0

        while True:
            ok, frame = self.cap.read()
            if not ok:
                return False, None, 0.0
            self._frame_idx += 1
            # take every _frame_period frame
            if (self._frame_idx % self._frame_period) == 0:
                return True, frame, self._timestamp_sec()
