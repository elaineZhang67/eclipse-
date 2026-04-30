import json
import re
import tempfile
from pathlib import Path

import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from runtime.device import model_load_kwargs, resolve_device


DEFAULT_TEXT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_QWEN_VL_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_VL_MODEL_ID = "google/gemma-4-E4B-it"


def _move_to_device(batch, device, float_dtype=None):
    if hasattr(batch, "items"):
        return {
            key: (
                value.to(device=device, dtype=float_dtype)
                if hasattr(value, "to") and torch.is_tensor(value) and value.is_floating_point() and float_dtype is not None
                else value.to(device=device)
                if hasattr(value, "to")
                else value
            )
            for key, value in batch.items()
        }
    if hasattr(batch, "to"):
        kwargs = {"device": device}
        if float_dtype is not None and torch.is_tensor(batch) and batch.is_floating_point():
            kwargs["dtype"] = float_dtype
        return batch.to(**kwargs)
    return batch


def _select_model_id(model_id, backend):
    if model_id:
        return model_id
    if backend == "vl":
        return DEFAULT_VL_MODEL_ID
    return DEFAULT_TEXT_MODEL_ID


def _is_gemma4_model_id(model_id):
    return "gemma-4" in str(model_id or "").lower() or "gemma4" in str(model_id or "").lower()


def _extract_json(text):
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _format_context_docs(context_docs):
    lines = []
    for idx, doc in enumerate(context_docs or [], start=1):
        lines.append(f"[context {idx}] {doc.get('text', '')}")
    return "\n".join(lines) if lines else "No context documents."


def _format_chat_history(conversation_history):
    lines = []
    for item in conversation_history or []:
        role = str(item.get("role", "user")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append("{label}: {content}".format(label=label, content=content))
    return "\n".join(lines) if lines else "No prior conversation."


def _truncate_text(value, max_chars=700):
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _compact_profile(profile):
    if not isinstance(profile, dict):
        return _truncate_text(profile, max_chars=300) if profile else None

    carried_objects = []
    for item in profile.get("carried_objects", []) or []:
        if isinstance(item, dict) and item.get("label"):
            carried_objects.append(item.get("label"))
        elif item:
            carried_objects.append(str(item))

    return {
        "appearance": profile.get("appearance"),
        "carried_objects": carried_objects[:8],
        "behavior_overview": _truncate_text(profile.get("behavior_overview"), max_chars=300),
    }


def _compact_scene_packet(event_log, track_payload, max_events=20):
    events = []
    for window in event_log or []:
        for event in window.get("events", []):
            events.append(
                {
                    "window_start": window.get("start"),
                    "window_end": window.get("end"),
                    "event": event,
                }
            )

    tracks = []
    for track_id, payload in sorted(track_payload.items(), key=lambda item: int(item[0])):
        metadata = payload.get("metadata", {})
        tracks.append(
            {
                "track_id": int(track_id),
                "display_name": metadata.get("display_name"),
                "track_ref": metadata.get("track_ref"),
                "first_seen": metadata.get("first_seen"),
                "last_seen": metadata.get("last_seen"),
                "objects": list(metadata.get("objects", [])),
                "profile": _compact_profile(payload.get("profile")),
                "summary": _truncate_text(payload.get("summary"), max_chars=700),
                "segments": list(payload.get("segments", []))[:8],
            }
        )

    return {
        "video_start": None if not event_log else min(window.get("start", 0.0) for window in event_log),
        "video_end": None if not event_log else max(window.get("end", 0.0) for window in event_log),
        "event_window_count": len(event_log or []),
        "notable_events": events[:max_events],
        "tracks": tracks,
    }


def _load_vl_model(model_id, device):
    try:
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL requires a newer transformers installation with "
                "Qwen3VLForConditionalGeneration or AutoModelForImageTextToText."
            ) from exc
        model_cls = AutoModelForImageTextToText

    model = model_cls.from_pretrained(model_id, **model_load_kwargs(device))
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model


class QwenSummarizer:
    uses_visual_inputs = False

    def __init__(self, model_id=DEFAULT_TEXT_MODEL_ID, device="auto"):
        self.model_id = model_id
        self.device = resolve_device(device)
        self.input_device = self.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_load_kwargs(self.device),
        )
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.model_dtype = next(self.model.parameters()).dtype

    def _generate(self, prompt, max_new_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: (
                value.to(device=self.input_device, dtype=self.model_dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value.to(device=self.input_device)
            )
            for key, value in inputs.items()
        }
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = out[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def build_track_profile_prompt(self, track_id, segments, track_metadata=None, allowed_objects=None):
        payload = {
            "track_id": track_id,
            "segments": segments,
            "metadata": track_metadata or {},
            "allowed_objects": list(allowed_objects or []),
        }
        payload_txt = json.dumps(payload, indent=2)
        return (
            "You are a surveillance video analyst.\n"
            "Infer a structured person profile from the evidence only.\n"
            "Each track corresponds to one human subject.\n"
            "If metadata includes display_name or track_ref, treat that as the preferred human-readable identity.\n"
            "If metadata includes memory_track_id, tracker_track_ids, or identity_history, treat those as continuity evidence "
            "that multiple raw tracker IDs may belong to the same human over time.\n"
            "Because this is a text-only backend, set appearance fields to null when they are not present in the evidence.\n"
            "Use only allowed_objects labels for carried_objects.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "track_id": int,\n'
            '  "display_name": string,\n'
            '  "appearance": {"top_color": string|null, "bottom_color": string|null, "outerwear": string|null, "helmet": boolean|null},\n'
            '  "carried_objects": [{"label": string, "source": "detector|visual|both"}],\n'
            '  "behavior_overview": string,\n'
            '  "uncertain_fields": [string]\n'
            "}\n\n"
            f"Evidence JSON:\n{payload_txt}\n\n"
            "JSON:"
        )

    def build_track_prompt(self, track_id, segments, track_metadata=None):
        payload = {
            "track_id": track_id,
            "segments": segments,
            "metadata": track_metadata or {},
        }
        payload_txt = json.dumps(payload, indent=2)
        return (
            "You are a surveillance video analyst.\n"
            "Summarize the person track from structured evidence only.\n"
            "Each track corresponds to one human subject.\n"
            "Prefer metadata.display_name or metadata.track_ref instead of referring to people only by raw track IDs.\n"
            "Use metadata.identity_history and metadata.tracker_track_ids to preserve identity continuity when the tracker "
            "fragmented the same person into multiple raw IDs.\n"
            "Focus on: when the person appears, when they are last seen, what objects are attributed to them, "
            "their actions over time, and any interactions with other tracked people.\n"
            "If clothing, helmets, or bags are not in the evidence, say they are unknown instead of guessing.\n"
            "Write 3-5 sentences.\n\n"
            f"Evidence JSON:\n{payload_txt}\n\n"
            "Summary:"
        )

    def build_scene_prompt(self, event_log, track_payload):
        packet = {
            "event_log": event_log,
            "tracks": track_payload,
        }
        packet_txt = json.dumps(packet, indent=2)
        return (
            "You are a surveillance video analyst preparing a longer scene summary for one camera.\n"
            "The input is already structured by a tracker, object attribution stage, action classifier, and "
            "5-second event windows.\n"
            "Each track corresponds to one human subject. Prefer display_name or track_ref labels such as "
            "'Person 1 (track 7)' instead of raw numeric IDs alone.\n"
            "If tracks include identity_history or linked raw tracker ids, use that memory to keep the same person "
            "consistent across the full video.\n"
            "Write a factual summary of the full scene in 2 short paragraphs.\n"
            "Prioritize entries, exits or last-seen moments, people carrying bags or other tracked objects, "
            "person-person interactions, and notable action changes over time.\n"
            "If two people have a 'close' interaction with a combined_box, treat it as a joint interaction region. "
            "If they are 'nearby', keep them as separate boxes and describe the interaction conservatively.\n"
            "Do not invent clothing, helmets, or objects that are not explicitly present in the evidence.\n\n"
            f"Structured scene packet:\n{packet_txt}\n\n"
            "Scene summary:"
        )

    def build_video_scene_prompt(self, event_log, track_payload):
        packet = _compact_scene_packet(event_log, track_payload)
        packet_txt = json.dumps(packet, indent=2)
        return (
            "You are preparing a full-video scene summary for one surveillance camera.\n"
            "Use the compact structured packet as grounding for track IDs, timestamps, actions, objects, and identity continuity.\n"
            "If sampled full-video frames are available, use them for visual layout and visible scene details.\n"
            "Write 2 short paragraphs that summarize the overall scene, main people, object interactions, and notable changes.\n"
            "Do not invent clothing, objects, or interactions that are absent from the evidence.\n\n"
            f"Compact full-video packet:\n{packet_txt}\n\n"
            "Full-video scene summary:"
        )

    def build_interval_prompt(self, interval_packet):
        packet_txt = json.dumps(interval_packet, indent=2)
        return (
            "You are summarizing one longer surveillance interval that already contains 5-second event windows.\n"
            "Each track corresponds to one human subject, and display_name / track_ref should be used when present.\n"
            "If the interval packet includes identity_history or tracker_track_ids, use that continuity evidence to avoid "
            "splitting one person into multiple identities.\n"
            "Summarize the important events in 4-6 sentences.\n"
            "Focus on entries, exits, object carrying, action changes, and interactions.\n"
            "Use human-readable labels first and raw track IDs only as grounding.\n\n"
            f"Interval packet:\n{packet_txt}\n\n"
            "Interval summary:"
        )

    def build_window_prompt(self, window_packet):
        packet_txt = json.dumps(window_packet, indent=2)
        return (
            "You are summarizing one 5-second surveillance event window.\n"
            "Each track corresponds to one human subject, and display_name / track_ref should be used when present.\n"
            "Use the structured packet only. Do not infer clothing, objects, or actions that are not present.\n"
            "Write 1-3 concise factual sentences about which people are active, what each person is doing, "
            "nearby or carried objects, and person-person interactions.\n\n"
            f"Window packet:\n{packet_txt}\n\n"
            "Window summary:"
        )

    def build_qa_prompt(self, question, context_docs, conversation_history=None, has_visual_evidence=False):
        visual_instruction = ""
        if has_visual_evidence:
            visual_instruction = (
                "The answer also has visual keyframes sampled from the retrieved windows. "
                "Use those images for visible appearance, spatial layout, and fine-grained visual details; "
                "use retrieved context for track IDs, timestamps, and structured events. "
                "If track profiles or summaries conflict with the sampled keyframes, trust the keyframes for visible details.\n"
            )
        return (
            "You answer surveillance-video questions using only the provided retrieved context.\n"
            "Each track corresponds to one human subject. Prefer labels like 'Person 1 (track 7)' when available.\n"
            "If the retrieved context includes identity_history or linked raw tracker ids, treat them as evidence that "
            "multiple tracker ids may refer to the same person.\n"
            "If the retrieved context includes stitched_track_timeline, use it to connect the same person across "
            "adjacent 5-second windows before answering specific-time questions.\n"
            f"{visual_instruction}"
            "If the context is insufficient, say so explicitly.\n"
            "Answer in 3-6 sentences and cite track IDs and time ranges when available.\n\n"
            f"Conversation history:\n{_format_chat_history(conversation_history)}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{_format_context_docs(context_docs)}\n\n"
            "Answer:"
        )

    def describe_track_profile(
        self,
        track_id,
        segments,
        track_metadata=None,
        visual_evidence=None,
        allowed_objects=None,
    ):
        prompt = self.build_track_profile_prompt(
            track_id,
            segments,
            track_metadata=track_metadata,
            allowed_objects=allowed_objects,
        )
        raw = self._generate(prompt, max_new_tokens=220)
        parsed = _extract_json(raw)
        return parsed if parsed is not None else {"raw_response": raw}

    def summarize_track(self, track_id, segments, track_metadata=None, visual_evidence=None):
        prompt = self.build_track_prompt(track_id, segments, track_metadata=track_metadata)
        return self._generate(prompt, max_new_tokens=180)

    def summarize_scene(self, event_log, track_payload, scene_images=None):
        prompt = self.build_scene_prompt(event_log, track_payload)
        return self._generate(prompt, max_new_tokens=280)

    def summarize_scene_video(self, event_log, track_payload, video_frames=None):
        prompt = self.build_video_scene_prompt(event_log, track_payload)
        return self._generate(prompt, max_new_tokens=280)

    def summarize_interval(self, interval_packet, scene_images=None):
        prompt = self.build_interval_prompt(interval_packet)
        return self._generate(prompt, max_new_tokens=220)

    def summarize_window(self, window_packet, scene_images=None):
        prompt = self.build_window_prompt(window_packet)
        return self._generate(prompt, max_new_tokens=140)

    def answer_question(self, question, context_docs, conversation_history=None, visual_evidence=None):
        prompt = self.build_qa_prompt(
            question,
            context_docs,
            conversation_history=conversation_history,
            has_visual_evidence=bool(visual_evidence),
        )
        return self._generate(prompt, max_new_tokens=220)


class QwenVLSummarizer:
    uses_visual_inputs = True

    def __init__(
        self,
        model_id=DEFAULT_QWEN_VL_MODEL_ID,
        max_track_images=4,
        max_scene_images=4,
        device="auto",
    ):
        self.model_id = model_id
        self.max_track_images = max(0, int(max_track_images))
        self.max_scene_images = max(0, int(max_scene_images))
        self.device = resolve_device(device)
        self.input_device = self.device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = _load_vl_model(model_id, self.device)
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

    def build_track_profile_prompt(self, track_id, segments, track_metadata=None, allowed_objects=None):
        payload = {
            "track_id": track_id,
            "segments": segments,
            "metadata": track_metadata or {},
            "allowed_objects": list(allowed_objects or []),
        }
        payload_txt = json.dumps(payload, indent=2)
        return (
            "You are a surveillance video analyst.\n"
            "The images are sampled crops from one tracked person over time.\n"
            "Each track corresponds to one human subject.\n"
            "If metadata includes display_name or track_ref, treat that as the preferred human-readable identity.\n"
            "If metadata includes memory_track_id, tracker_track_ids, or identity_history, treat those as continuity evidence "
            "that multiple raw tracker IDs may belong to the same human over time.\n"
            "Use the images only for visible appearance and fine-grained attributes.\n"
            "Use the structured evidence for time ranges, actions, and interactions.\n"
            "Use only allowed_objects labels for carried_objects.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "track_id": int,\n'
            '  "display_name": string,\n'
            '  "appearance": {"top_color": string|null, "bottom_color": string|null, "outerwear": string|null, "helmet": boolean|null},\n'
            '  "carried_objects": [{"label": string, "source": "detector|visual|both"}],\n'
            '  "behavior_overview": string,\n'
            '  "uncertain_fields": [string]\n'
            "}\n\n"
            f"Evidence JSON:\n{payload_txt}\n\n"
            "JSON:"
        )

    def build_track_prompt(self, track_id, segments, track_metadata=None):
        payload = {
            "track_id": track_id,
            "segments": segments,
            "metadata": track_metadata or {},
        }
        payload_txt = json.dumps(payload, indent=2)
        return (
            "You are a surveillance video analyst.\n"
            "The images are sampled crops from one tracked person over time.\n"
            "Each track corresponds to one human subject.\n"
            "Prefer metadata.display_name or metadata.track_ref instead of referring to people only by raw track IDs.\n"
            "Use metadata.identity_history and metadata.tracker_track_ids to preserve identity continuity when the tracker "
            "fragmented the same person into multiple raw IDs.\n"
            "Use the images only to describe visible appearance and carried items when they are clearly shown.\n"
            "Use the structured evidence to describe timing, actions, and interactions.\n"
            "If a detail is not visible in the images or not present in the evidence, say it is unknown.\n"
            "Write 3-5 factual sentences.\n\n"
            f"Evidence JSON:\n{payload_txt}\n\n"
            "Track summary:"
        )

    def build_scene_prompt(self, event_log, track_payload):
        packet = {
            "event_log": event_log,
            "tracks": track_payload,
        }
        packet_txt = json.dumps(packet, indent=2)
        return (
            "You are a surveillance video analyst preparing a scene summary for one camera.\n"
            "The images are sampled scene frames from across the video.\n"
            "Each track corresponds to one human subject. Prefer display_name or track_ref labels such as "
            "'Person 1 (track 7)' instead of raw numeric IDs alone.\n"
            "If tracks include identity_history or linked raw tracker ids, use that memory to keep the same person "
            "consistent across the full video.\n"
            "Use the images for visible layout and appearance cues, and use the structured evidence for chronology.\n"
            "Prioritize entries, last-seen moments, carried objects, interactions, and action changes.\n"
            "Do not invent details that are absent from both the images and structured evidence.\n"
            "Write 2 short paragraphs.\n\n"
            f"Structured scene packet:\n{packet_txt}\n\n"
            "Scene summary:"
        )

    def build_video_scene_prompt(self, event_log, track_payload):
        packet = _compact_scene_packet(event_log, track_payload)
        packet_txt = json.dumps(packet, indent=2)
        return (
            "You are preparing a full-video scene summary for one surveillance camera.\n"
            "The visual input contains frames sampled uniformly across the full video.\n"
            "Use the sampled frames for visual layout and visible scene details, and use the compact structured packet "
            "for track IDs, timestamps, actions, objects, and identity continuity.\n"
            "Write 2 short paragraphs that summarize the overall scene, main people, object interactions, and notable changes.\n"
            "Do not invent clothing, objects, or interactions that are absent from both the sampled frames and the structured evidence.\n\n"
            f"Compact full-video packet:\n{packet_txt}\n\n"
            "Full-video scene summary:"
        )

    def build_interval_prompt(self, interval_packet):
        packet_txt = json.dumps(interval_packet, indent=2)
        return (
            "You are summarizing one surveillance interval.\n"
            "The images are scene frames sampled from within the interval.\n"
            "Each track corresponds to one human subject, and display_name / track_ref should be used when present.\n"
            "If the interval packet includes identity_history or tracker_track_ids, use that continuity evidence to avoid "
            "splitting one person into multiple identities.\n"
            "Use images for visual context and the structured packet for chronology.\n"
            "Summarize the interval in 4-6 sentences with human-readable labels first and track IDs as grounding.\n\n"
            f"Interval packet:\n{packet_txt}\n\n"
            "Interval summary:"
        )

    def build_window_prompt(self, window_packet):
        packet_txt = json.dumps(window_packet, indent=2)
        return (
            "You are summarizing one 5-second surveillance event window.\n"
            "The images are scene frames sampled from within this short window, when available.\n"
            "Each track corresponds to one human subject, and display_name / track_ref should be used when present.\n"
            "Use images only for visible context. Use the structured packet for chronology, actions, object attribution, "
            "and interactions. Do not invent details absent from both sources.\n"
            "Write 1-3 concise factual sentences about which people are active and what each person is doing.\n\n"
            f"Window packet:\n{packet_txt}\n\n"
            "Window summary:"
        )

    def build_qa_prompt(self, question, context_docs, conversation_history=None, has_visual_evidence=False):
        visual_instruction = ""
        if has_visual_evidence:
            visual_instruction = (
                "The images are keyframes sampled from the retrieved windows after RAG. "
                "Use images for visible appearance, spatial layout, and fine-grained visual details; "
                "use retrieved context for track IDs, timestamps, and structured events. "
                "If track profiles or summaries conflict with the sampled keyframes, trust the keyframes for visible details.\n"
            )
        return (
            "You answer surveillance-video questions using only the provided retrieved context.\n"
            "Each track corresponds to one human subject. Prefer labels like 'Person 1 (track 7)' when available.\n"
            "If the retrieved context includes identity_history or linked raw tracker ids, treat them as evidence that "
            "multiple tracker ids may refer to the same person.\n"
            "If the retrieved context includes stitched_track_timeline, use it to connect the same person across "
            "adjacent 5-second windows before answering specific-time questions.\n"
            f"{visual_instruction}"
            "If the context is insufficient, say so explicitly.\n"
            "Answer in 3-6 sentences and cite track IDs and time ranges when available.\n\n"
            f"Conversation history:\n{_format_chat_history(conversation_history)}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{_format_context_docs(context_docs)}\n\n"
            "Answer:"
        )

    def describe_track_profile(
        self,
        track_id,
        segments,
        track_metadata=None,
        visual_evidence=None,
        allowed_objects=None,
    ):
        prompt = self.build_track_profile_prompt(
            track_id,
            segments,
            track_metadata=track_metadata,
            allowed_objects=allowed_objects,
        )
        images = list(visual_evidence or [])[: self.max_track_images]
        raw = self._generate(prompt, image_arrays=images, max_new_tokens=220)
        parsed = _extract_json(raw)
        return parsed if parsed is not None else {"raw_response": raw}

    def summarize_track(self, track_id, segments, track_metadata=None, visual_evidence=None):
        prompt = self.build_track_prompt(track_id, segments, track_metadata=track_metadata)
        images = list(visual_evidence or [])[: self.max_track_images]
        return self._generate(prompt, image_arrays=images, max_new_tokens=220)

    def summarize_scene(self, event_log, track_payload, scene_images=None):
        prompt = self.build_scene_prompt(event_log, track_payload)
        images = list(scene_images or [])[: self.max_scene_images]
        return self._generate(prompt, image_arrays=images, max_new_tokens=320)

    def summarize_scene_video(self, event_log, track_payload, video_frames=None):
        prompt = self.build_video_scene_prompt(event_log, track_payload)
        images = list(video_frames or [])[: self.max_scene_images]
        return self._generate(prompt, image_arrays=images, max_new_tokens=320)

    def summarize_interval(self, interval_packet, scene_images=None):
        prompt = self.build_interval_prompt(interval_packet)
        images = list(scene_images or [])[: self.max_scene_images]
        return self._generate(prompt, image_arrays=images, max_new_tokens=220)

    def summarize_window(self, window_packet, scene_images=None):
        prompt = self.build_window_prompt(window_packet)
        images = list(scene_images or [])[: min(1, self.max_scene_images)]
        return self._generate(prompt, image_arrays=images, max_new_tokens=140)

    def answer_question(self, question, context_docs, conversation_history=None, visual_evidence=None):
        images = list(visual_evidence or [])[: self.max_scene_images]
        prompt = self.build_qa_prompt(
            question,
            context_docs,
            conversation_history=conversation_history,
            has_visual_evidence=bool(images),
        )
        return self._generate(prompt, image_arrays=images, max_new_tokens=240)


def build_summarizer(
    backend="text",
    model_id=None,
    max_track_images=4,
    max_scene_images=4,
    device="auto",
):
    backend = str(backend or "text").strip().lower()
    resolved_model_id = _select_model_id(model_id, backend)

    if backend == "text":
        return QwenSummarizer(model_id=resolved_model_id, device=device)
    if backend == "vl":
        if _is_gemma4_model_id(resolved_model_id):
            from summarization.gemma4_summary import build_gemma4_summarizer

            return build_gemma4_summarizer(
                model_id=resolved_model_id,
                max_track_images=max_track_images,
                max_scene_images=max_scene_images,
                device=device,
            )
        return QwenVLSummarizer(
            model_id=resolved_model_id,
            max_track_images=max_track_images,
            max_scene_images=max_scene_images,
            device=device,
        )
    raise ValueError(f"Unsupported summary backend: {backend}")
