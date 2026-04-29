import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoProcessor

from runtime.device import model_load_kwargs, resolve_device
from summarization.qwen_summary import QwenVLSummarizer, _move_to_device


DEFAULT_GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_GEMMA4_MAX_PROMPT_CHARS = 12000
DEFAULT_GEMMA4_IMAGE_LONG_EDGE = 384


def _safe_rgb_image(image_rgb, max_long_edge):
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        return None

    height, width = image.shape[:2]
    if height < 16 or width < 16:
        return None

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    max_long_edge = max(0, int(max_long_edge))
    current_long_edge = max(height, width)
    if max_long_edge > 0 and current_long_edge > max_long_edge:
        scale = float(max_long_edge) / float(current_long_edge)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(image)


def _limit_prompt(prompt, max_chars):
    max_chars = max(0, int(max_chars))
    if not max_chars or len(prompt) <= max_chars:
        return prompt
    return (
        prompt[:max_chars].rstrip()
        + "\n\n[Structured context truncated to keep Gemma4 generation within GPU memory.]\n\nSummary:"
    )


def _load_gemma4_model(model_id, device):
    try:
        from transformers import Gemma4ForConditionalGeneration

        model_cls = Gemma4ForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 4 requires a newer transformers installation with "
                "Gemma4ForConditionalGeneration or AutoModelForImageTextToText support."
            ) from exc
        model_cls = AutoModelForImageTextToText

    try:
        model = model_cls.from_pretrained(
            model_id,
            **model_load_kwargs(device),
        )
    except ValueError as exc:
        if "model type `gemma4`" in str(exc):
            raise RuntimeError(
                "Your installed transformers version does not recognize the Gemma 4 architecture yet. "
                "Upgrade transformers to a release that includes Gemma4 support, or install the latest "
                "transformers from source."
            ) from exc
        raise

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model


class Gemma4VLSummarizer(QwenVLSummarizer):
    uses_visual_inputs = True

    def __init__(
        self,
        model_id=DEFAULT_GEMMA4_MODEL_ID,
        max_track_images=1,
        max_scene_images=1,
        image_long_edge=DEFAULT_GEMMA4_IMAGE_LONG_EDGE,
        max_prompt_chars=DEFAULT_GEMMA4_MAX_PROMPT_CHARS,
        device="auto",
    ):
        self.model_id = model_id
        self.max_track_images = max(0, int(max_track_images))
        self.max_scene_images = max(0, int(max_scene_images))
        self.image_long_edge = max(0, int(image_long_edge))
        self.max_prompt_chars = max(0, int(max_prompt_chars))
        self.device = resolve_device(device)
        self.input_device = self.device
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        self.model = _load_gemma4_model(model_id, self.device)
        self.model_dtype = next(self.model.parameters()).dtype

    def _generate(self, prompt, image_arrays=None, max_new_tokens=220):
        prompt = _limit_prompt(prompt, self.max_prompt_chars)
        image_arrays = [
            image
            for image in (_safe_rgb_image(item, self.image_long_edge) for item in list(image_arrays or []))
            if image is not None
        ]

        with tempfile.TemporaryDirectory() as tmpdir_name:
            image_refs = []
            for idx, image_rgb in enumerate(image_arrays):
                path = Path(tmpdir_name) / f"frame_{idx:02d}.png"
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(path), image_bgr)
                image_refs.append(str(path))

            content = [{"type": "image", "image": item} for item in image_refs]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = _move_to_device(inputs, self.input_device, float_dtype=self.model_dtype)

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = generated[:, prompt_len:]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


def build_gemma4_summarizer(
    model_id=None,
    max_track_images=1,
    max_scene_images=1,
    image_long_edge=DEFAULT_GEMMA4_IMAGE_LONG_EDGE,
    max_prompt_chars=DEFAULT_GEMMA4_MAX_PROMPT_CHARS,
    device="auto",
):
    return Gemma4VLSummarizer(
        model_id=model_id or DEFAULT_GEMMA4_MODEL_ID,
        max_track_images=max_track_images,
        max_scene_images=max_scene_images,
        image_long_edge=image_long_edge,
        max_prompt_chars=max_prompt_chars,
        device=device,
    )
