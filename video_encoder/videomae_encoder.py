import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors_file
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

from runtime.device import model_load_kwargs, resolve_device


def _load_videomae_state_dict(model_id):
    try:
        checkpoint_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        return load_safetensors_file(checkpoint_path)
    except Exception:
        checkpoint_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        return torch.load(checkpoint_path, map_location="cpu")


def _convert_legacy_attention_bias_keys(state_dict):
    converted = {}
    converted_q_bias = 0
    converted_v_bias = 0
    inserted_key_bias = 0

    for key, value in state_dict.items():
        if key.endswith(".attention.attention.q_bias"):
            new_key = key[:-len("q_bias")] + "query.bias"
            converted[new_key] = value
            converted_q_bias += 1
            continue

        if key.endswith(".attention.attention.v_bias"):
            new_key = key[:-len("v_bias")] + "value.bias"
            converted[new_key] = value
            converted_v_bias += 1
            continue

        converted[key] = value

    for key, value in list(converted.items()):
        if not key.endswith(".attention.attention.key.weight"):
            continue
        bias_key = key[:-len("weight")] + "bias"
        if bias_key in converted:
            continue
        converted[bias_key] = torch.zeros(
            value.shape[0],
            dtype=value.dtype,
        )
        inserted_key_bias += 1

    return converted, {
        "converted_q_bias": converted_q_bias,
        "converted_v_bias": converted_v_bias,
        "inserted_key_bias": inserted_key_bias,
    }


def _load_videomae_model(model_id, device):
    raw_state_dict = _load_videomae_state_dict(model_id)
    state_dict, compat_report = _convert_legacy_attention_bias_keys(raw_state_dict)
    model, loading_info = VideoMAEForVideoClassification.from_pretrained(
        model_id,
        state_dict=state_dict,
        output_loading_info=True,
        **model_load_kwargs(device),
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    model._compat_loading_info = loading_info
    model._compat_bias_report = compat_report
    return model


class VideoMAEActionModel:
    """
    直接使用 VideoMAE Kinetics-400 预训练模型做 per-clip 分类
    """
    def __init__(self, model_id="MCG-NJU/videomae-base-finetuned-kinetics", device="auto"):
        self.device = resolve_device(device)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.model = _load_videomae_model(model_id, self.device)
        self.expected_num_frames = int(getattr(self.model.config, "num_frames", 16))
        self.model_dtype = next(self.model.parameters()).dtype

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
        inputs = {
            key: (
                value.to(device=self.device, dtype=self.model_dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value.to(device=self.device)
            )
            for key, value in inputs.items()
        }
        out = self.model(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        label = self.model.config.id2label[idx]
        return label, conf
