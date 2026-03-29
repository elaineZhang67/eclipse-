import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

from runtime.device import model_load_kwargs, resolve_device


class VideoMAEActionModel:
    """
    直接使用 VideoMAE Kinetics-400 预训练模型做 per-clip 分类
    """
    def __init__(self, model_id="MCG-NJU/videomae-base-finetuned-kinetics", device="auto"):
        self.device = resolve_device(device)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_id,
            **model_load_kwargs(self.device),
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.expected_num_frames = int(getattr(self.model.config, "num_frames", 16))

    def _match_num_frames(self, clip_rgb_frames):
        if not clip_rgb_frames:
            raise ValueError("clip_rgb_frames cannot be empty")

        if len(clip_rgb_frames) == self.expected_num_frames:
            return clip_rgb_frames

        if len(clip_rgb_frames) > self.expected_num_frames:
            idx = torch.linspace(
                0,
                len(clip_rgb_frames) - 1,
                steps=self.expected_num_frames,
            ).round().to(torch.long).tolist()
            return [clip_rgb_frames[i] for i in idx]

        padded = list(clip_rgb_frames)
        while len(padded) < self.expected_num_frames:
            padded.append(padded[-1])
        return padded

    @torch.inference_mode()
    def predict_label(self, clip_rgb_frames):
        # clip_rgb_frames: list of numpy RGB images [224,224,3]
        clip_rgb_frames = self._match_num_frames(clip_rgb_frames)
        inputs = self.processor(clip_rgb_frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        label = self.model.config.id2label[idx]
        return label, conf
