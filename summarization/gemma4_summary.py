import tempfile
from pathlib import Path

import cv2
import torch
from transformers import AutoProcessor

from runtime.device import model_load_kwargs, resolve_device
from summarization.qwen_summary import QwenVLSummarizer, _move_to_device


DEFAULT_GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"


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
        max_track_images=4,
        max_scene_images=4,
        device="auto",
    ):
        self.model_id = model_id
        self.max_track_images = max(0, int(max_track_images))
        self.max_scene_images = max(0, int(max_scene_images))
        self.device = resolve_device(device)
        self.input_device = self.device
        self.processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        self.model = _load_gemma4_model(model_id, self.device)
        self.model_dtype = next(self.model.parameters()).dtype

    def _generate(self, prompt, image_arrays=None, max_new_tokens=220):
        image_arrays = list(image_arrays or [])

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
    max_track_images=4,
    max_scene_images=4,
    device="auto",
):
    return Gemma4VLSummarizer(
        model_id=model_id or DEFAULT_GEMMA4_MODEL_ID,
        max_track_images=max_track_images,
        max_scene_images=max_scene_images,
        device=device,
    )
