class EventSegmenter:
    """
    把平滑后的窗口序列合并成更长的事件段：
    输入：[(ts, te, label, conf), ...]
    输出：[{start,end,action,avg_conf}, ...]
    """
    def __init__(self, min_seg_len_sec=1.0):
        self.min_seg_len_sec = min_seg_len_sec

    def segment(self, smooth_preds):
        if not smooth_preds:
            return []

        segments = []
        cur_label = smooth_preds[0][2]
        cur_start = smooth_preds[0][0]
        cur_end = smooth_preds[0][1]
        conf_sum = smooth_preds[0][3]
        n = 1

        def flush():
            nonlocal cur_label, cur_start, cur_end, conf_sum, n
            if (cur_end - cur_start) >= self.min_seg_len_sec:
                segments.append({
                    "start": round(cur_start, 2),
                    "end": round(cur_end, 2),
                    "action": cur_label,
                    "avg_conf": round(conf_sum / max(1, n), 3),
                })

        for (ts, te, label, conf) in smooth_preds[1:]:
            if label == cur_label:
                cur_end = te
                conf_sum += conf
                n += 1
            else:
                flush()
                cur_label = label
                cur_start = ts
                cur_end = te
                conf_sum = conf
                n = 1

        flush()
        return segments
