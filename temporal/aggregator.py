from collections import deque, Counter

class TemporalAggregator:
    """
    输入：一堆滑窗预测 (t_start, t_end, label, conf)
    输出：平滑后的同样格式列表
    """
    def __init__(self, vote_k=5):
        self.vote_k = vote_k

    def smooth(self, preds):
        # 简单 majority vote smoothing：对最近 K 个窗口投票
        hist = deque(maxlen=self.vote_k)
        smoothed = []
        for (ts, te, label, conf) in preds:
            hist.append(label)
            voted = Counter(hist).most_common(1)[0][0]
            # conf 这里只保留原 conf，也可以改为平均
            smoothed.append((ts, te, voted, conf))
        return smoothed
