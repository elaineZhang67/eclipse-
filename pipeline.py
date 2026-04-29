from collections import defaultdict

from tqdm.auto import tqdm

from detection.sam2_detector import Sam2Detector
from detection.sam3_detector import Sam3Detector
from tracking.tracker import MultiObjectTracker
from tracking.appearance_memory import TrackMemoryBank
from preprocessing.frame_sampler import FrameSampler
from preprocessing.clip_builder import ClipBuilder
from video_encoder.videomae_encoder import VideoMAEActionModel
from temporal.aggregator import TemporalAggregator
from temporal.segmenter import EventSegmenter
from temporal.event_log import SceneEventBuilder
from temporal.interval_summary import SceneIntervalBuilder
from retrieval.event_rag import EventRAG
from summarization.qwen_summary import build_summarizer
from summarization.visual_evidence import VisualEvidenceCollector
from runtime.object_preset import describe_environment, resolve_object_labels


def _parse_csv_arg(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]


def _build_tqdm(iterable=None, total=None, desc="", unit="it", disable=False):
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
        dynamic_ncols=True,
    )


def _track_order_key(item):
    track_id, metadata = item
    first_seen = metadata.get("first_seen")
    return (first_seen is None, float(first_seen or 0.0), int(track_id))


def _annotate_person_aliases(event_log, per_track_metadata):
    alias_map = {}
    ordered_tracks = sorted(per_track_metadata.items(), key=_track_order_key)
    for idx, (track_id, metadata) in enumerate(ordered_tracks, start=1):
        display_name = "Person {idx}".format(idx=idx)
        metadata["entity_type"] = "person"
        metadata["display_name"] = display_name
        metadata["track_ref"] = "{name} (track {track_id})".format(
            name=display_name,
            track_id=track_id,
        )
        alias_map[int(track_id)] = display_name

    for metadata in per_track_metadata.values():
        for interaction in metadata.get("interactions", []):
            other_track_id = int(interaction.get("other_track_id"))
            other_name = alias_map.get(other_track_id, "Track {track_id}".format(track_id=other_track_id))
            interaction["other_display_name"] = other_name
            interaction["other_track_ref"] = "{name} (track {track_id})".format(
                name=other_name,
                track_id=other_track_id,
            )

    for window in event_log:
        window["active_track_refs"] = [
            "{name} (track {track_id})".format(
                name=alias_map.get(int(track_id), "Track {track_id}".format(track_id=track_id)),
                track_id=int(track_id),
            )
            for track_id in window.get("active_tracks", [])
        ]
        for event in window.get("events", []):
            if event.get("track_id") is not None:
                track_id = int(event["track_id"])
                event["track_display_name"] = alias_map.get(track_id, "Track {track_id}".format(track_id=track_id))
                event["track_ref"] = "{name} (track {track_id})".format(
                    name=event["track_display_name"],
                    track_id=track_id,
                )
            if event.get("track_ids"):
                track_names = []
                track_refs = []
                for track_id in event["track_ids"]:
                    tid = int(track_id)
                    name = alias_map.get(tid, "Track {track_id}".format(track_id=tid))
                    track_names.append(name)
                    track_refs.append("{name} (track {track_id})".format(name=name, track_id=tid))
                event["track_display_names"] = track_names
                event["track_refs"] = track_refs

    return alias_map


def _merge_track_memory_metadata(per_track_metadata, memory_metadata):
    for track_id, memory_payload in (memory_metadata or {}).items():
        target = per_track_metadata.setdefault(int(track_id), {})
        target.update(memory_payload)
        target.setdefault("entity_type", "person")
    for track_id, metadata in per_track_metadata.items():
        metadata.setdefault("entity_type", "person")
        metadata.setdefault("memory_track_id", int(track_id))
        metadata.setdefault("tracker_track_ids", [int(track_id)])
        metadata.setdefault(
            "identity_history",
            "memory track {track_id} linked tracker ids {track_id}".format(track_id=int(track_id)),
        )
        tracker_track_ids = metadata.get("tracker_track_ids", [])
        if tracker_track_ids and "tracker_id_count" not in metadata:
            metadata["tracker_id_count"] = len(tracker_track_ids)

    return per_track_metadata


def run_pipeline(args):
    track_labels = _parse_csv_arg(getattr(args, "track_labels", None), default=["person"])
    explicit_object_labels = _parse_csv_arg(
        getattr(args, "object_labels", None),
        default=[],
    )
    object_labels = resolve_object_labels(
        explicit_labels=explicit_object_labels,
        environment=getattr(args, "environment", "generic"),
        max_object_types=getattr(args, "max_object_types", 5),
    )
    clip_len = args.clip_len
    stride = args.stride
    if getattr(args, "clip_sec", None):
        clip_len = max(4, int(round(args.fps * args.clip_sec)))
    if getattr(args, "stride_sec", None):
        stride = max(1, int(round(args.fps * args.stride_sec)))

    # 1) modules
    sampler = FrameSampler(target_fps=args.fps)
    object_backend = str(getattr(args, "object_backend", "sam3")).strip().lower()
    if object_backend == "sam2":
        object_source = Sam2Detector(
            model_id=getattr(args, "sam2_model", "facebook/sam2.1-hiera-large"),
            yolo_weights=args.yolo,
            conf=args.min_conf,
            mask_threshold=getattr(args, "sam2_mask_threshold", 0.5),
            classes=object_labels,
            device=getattr(args, "device", "auto"),
            track_iou=getattr(args, "sam2_track_iou", 0.3),
            track_ttl=getattr(args, "sam2_track_ttl", 12),
        )
        object_source_missing = list(getattr(object_source, "missing_classes", []))
    elif object_backend == "sam3":
        object_source = Sam3Detector(
            model_id=getattr(args, "sam3_model", "facebook/sam3"),
            conf=args.min_conf,
            mask_threshold=getattr(args, "sam3_mask_threshold", 0.5),
            classes=object_labels,
            device=getattr(args, "device", "auto"),
            track_iou=getattr(args, "sam3_track_iou", 0.3),
            track_ttl=getattr(args, "sam3_track_ttl", 12),
        )
        object_source_missing = list(getattr(object_source, "missing_classes", []))
    else:
        object_source = MultiObjectTracker(
            weights=args.yolo,
            tracker_cfg=args.tracker,
            conf=args.min_conf,
            classes=object_labels,
        )
        object_source_missing = list(getattr(object_source, "missing_classes", []))
    tracker = MultiObjectTracker(
        weights=args.yolo,
        tracker_cfg=args.tracker,
        conf=args.min_conf,
        classes=track_labels,
    )
    track_memory_bank = None
    if getattr(args, "use_track_memory", False):
        track_memory_bank = TrackMemoryBank(
            similarity_threshold=getattr(args, "appearance_match_threshold", 0.82),
            mapping_ttl_sec=getattr(args, "appearance_memory_ttl_sec", 8.0),
            reassoc_gap_sec=getattr(args, "appearance_reassoc_gap_sec", 20.0),
        )
    clip_builder = ClipBuilder(clip_len=clip_len, stride=stride)
    action_model = VideoMAEActionModel(
        model_id="MCG-NJU/videomae-base-finetuned-kinetics",
        device=getattr(args, "device", "auto"),
    )
    aggregator = TemporalAggregator()
    segmenter = EventSegmenter(min_seg_len_sec=1.0)
    event_builder = SceneEventBuilder(
        window_sec=args.event_window_sec,
        combine_iou=args.interaction_combine_iou,
        combine_dist=args.interaction_combine_dist,
        nearby_dist=args.interaction_nearby_dist,
    )
    interval_builder = SceneIntervalBuilder(interval_sec=getattr(args, "long_summary_sec", 60.0))
    rag_builder = EventRAG()
    disable_progress = bool(getattr(args, "disable_progress", False))
    summarizer = None
    visual_evidence = None
    if args.use_llm:
        summarizer = build_summarizer(
            backend=getattr(args, "summary_backend", "text"),
            model_id=getattr(args, "llm_model", None),
            max_track_images=getattr(args, "vl_max_track_images", 4),
            max_scene_images=getattr(args, "vl_max_scene_images", 4),
            device=getattr(args, "device", "auto"),
        )
        if getattr(summarizer, "uses_visual_inputs", False):
            visual_evidence = VisualEvidenceCollector(
                max_track_images=getattr(args, "vl_max_track_images", 4),
                max_scene_images=getattr(args, "vl_max_scene_images", 4),
                track_gap_sec=getattr(args, "vl_track_gap_sec", 1.5),
                scene_gap_sec=getattr(args, "vl_scene_gap_sec", 3.0),
            )

    # state per track_id
    per_track_probs = defaultdict(list)   # tid -> list of (t_start, t_end, label, conf)
    sampler.open(args.video)
    clip_count = 0
    frame_progress = _build_tqdm(
        total=sampler.estimated_sampled_frames or None,
        desc="Video",
        unit="frame",
        disable=disable_progress,
    )
    try:
        while True:
            ok, frame_bgr, t_sec = sampler.read()
            if not ok:
                break

            if object_backend == "sam3":
                object_tracks = object_source.update(frame_bgr)
            else:
                object_tracks = object_source.update(frame_bgr)
            tracks = tracker.update(frame_bgr)
            person_tracks = [tr for tr in tracks if tr.get("label") == "person"]
            if track_memory_bank is not None:
                person_tracks = track_memory_bank.update(frame_bgr, person_tracks, t_sec)
            else:
                person_tracks = [
                    {
                        **tr,
                        "raw_track_id": tr.get("track_id"),
                        "memory_track_id": tr.get("track_id"),
                    }
                    for tr in person_tracks
                ]
            if visual_evidence is not None:
                visual_evidence.update_scene(frame_bgr, t_sec)
                visual_evidence.update_tracks(frame_bgr, person_tracks, t_sec)

            # limit state size (optional)
            if len(per_track_probs) > args.max_people:
                # you can implement eviction policy; simplest: do nothing for now
                pass

            clip_ready = clip_builder.push(frame_bgr, person_tracks, t_sec)

            # clip_ready: list of dicts:
            # {track_id, clip_rgb_frames(list), t_start, t_end}
            for item in clip_ready:
                tid = item["track_id"]
                clip_rgb = item["clip_rgb"]
                t_start, t_end = item["t_start"], item["t_end"]

                label, conf = action_model.predict_label(clip_rgb)
                per_track_probs[tid].append((t_start, t_end, label, float(conf)))
                clip_count += 1

            event_builder.update(t_sec, person_tracks, object_tracks)
            frame_progress.update(1)
            frame_progress.set_postfix(
                time_s=round(float(t_sec), 1),
                people=len(person_tracks),
                objs=len(object_tracks),
                clips=clip_count,
            )
    finally:
        frame_progress.close()

    # (D) temporal aggregation + segmentation + optional LLM
    event_log, per_track_metadata = event_builder.finalize()
    per_track_metadata = _merge_track_memory_metadata(
        per_track_metadata,
        track_memory_bank.export_metadata() if track_memory_bank is not None else {},
    )
    _annotate_person_aliases(event_log, per_track_metadata)
    track_outputs = {}
    all_track_payload = {}
    tracked_ids = sorted(set(per_track_metadata) | set(per_track_probs))

    track_progress = _build_tqdm(
        tracked_ids,
        desc="Tracks",
        unit="track",
        disable=disable_progress,
    )
    for tid in track_progress:
        preds = per_track_probs.get(tid, [])
        # preds: list of windows
        agg = aggregator.smooth(preds)  # list of (t_start, t_end, label, score)
        segments = segmenter.segment(agg)  # list of dict segments
        metadata = per_track_metadata.get(
            tid,
            {
                "first_seen": None,
                "last_seen": None,
                "object_counts": {},
                "objects": [],
                "interactions": [],
                "tracker_track_ids": [int(tid)],
                "memory_track_id": int(tid),
            },
        )

        out = {
            "segments": segments,
            "metadata": metadata,
        }
        if summarizer is not None:
            track_images = visual_evidence.get_track_images(tid) if visual_evidence is not None else None
            out["profile"] = summarizer.describe_track_profile(
                tid,
                segments,
                track_metadata=metadata,
                visual_evidence=track_images,
                allowed_objects=object_labels,
            )
            out["summary"] = summarizer.summarize_track(
                tid,
                segments,
                track_metadata=metadata,
                visual_evidence=track_images,
            )
        track_outputs[tid] = out
        all_track_payload[tid] = {
            "metadata": metadata,
            "segments": segments,
        }
        if "profile" in out:
            all_track_payload[tid]["profile"] = out["profile"]
        if "summary" in out:
            all_track_payload[tid]["summary"] = out["summary"]
        track_progress.set_postfix(segments=len(segments), objects=len(metadata.get("objects", [])))
    track_progress.close()

    interval_packets = interval_builder.build(event_log, all_track_payload)
    interval_outputs = []
    interval_progress = _build_tqdm(
        interval_packets,
        desc="Intervals",
        unit="chunk",
        disable=disable_progress,
    )
    for interval in interval_progress:
        out = {
            "start": interval["start"],
            "end": interval["end"],
            "active_tracks": interval["active_tracks"],
            "objects": interval["objects"],
            "interaction_count": interval["interaction_count"],
            "event_count": interval["event_count"],
        }
        if summarizer is not None:
            scene_images = (
                visual_evidence.get_scene_images(
                    start_sec=interval["start"],
                    end_sec=interval["end"],
                )
                if visual_evidence is not None
                else None
            )
            out["summary"] = summarizer.summarize_interval(interval, scene_images=scene_images)
        interval_outputs.append(out)
        interval_progress.set_postfix(
            tracks=len(interval["active_tracks"]),
            events=interval["event_count"],
        )
    interval_progress.close()

    outputs = {
        "config": {
            "track_labels": track_labels,
            "object_labels": object_labels,
            "unsupported_track_labels": tracker.missing_classes,
            "unsupported_object_labels": object_source_missing,
            "clip_len": clip_len,
            "stride": stride,
            "event_window_sec": args.event_window_sec,
            "long_summary_sec": getattr(args, "long_summary_sec", 60.0),
            "tracker": args.tracker,
            "object_backend": object_backend,
            "sam2_model": getattr(args, "sam2_model", None) if object_backend == "sam2" else None,
            "sam2_mask_threshold": getattr(args, "sam2_mask_threshold", None) if object_backend == "sam2" else None,
            "sam2_track_iou": getattr(args, "sam2_track_iou", None) if object_backend == "sam2" else None,
            "sam2_track_ttl": getattr(args, "sam2_track_ttl", None) if object_backend == "sam2" else None,
            "sam3_model": getattr(args, "sam3_model", None) if object_backend == "sam3" else None,
            "sam3_mask_threshold": getattr(args, "sam3_mask_threshold", None) if object_backend == "sam3" else None,
            "sam3_track_iou": getattr(args, "sam3_track_iou", None) if object_backend == "sam3" else None,
            "sam3_track_ttl": getattr(args, "sam3_track_ttl", None) if object_backend == "sam3" else None,
            "device": getattr(args, "device", "auto"),
            "environment": describe_environment(getattr(args, "environment", "generic")),
            "summary_backend": getattr(args, "summary_backend", "text"),
            "llm_model": getattr(summarizer, "model_id", None),
            "use_track_memory": bool(getattr(args, "use_track_memory", False)),
            "appearance_match_threshold": getattr(args, "appearance_match_threshold", 0.82),
            "appearance_memory_ttl_sec": getattr(args, "appearance_memory_ttl_sec", 8.0),
            "appearance_reassoc_gap_sec": getattr(args, "appearance_reassoc_gap_sec", 20.0),
        },
        "stats": {
            "sampled_frames": int(frame_progress.n),
            "estimated_sampled_frames": int(sampler.estimated_sampled_frames or 0),
            "native_total_frames": int(sampler.total_frames or 0),
            "video_duration_sec": round(float(sampler.duration_sec or 0.0), 2),
            "tracks": len(tracked_ids),
            "identity_tracks": 0 if track_memory_bank is None else len(track_memory_bank.export_bank()),
            "event_windows": len(event_log),
            "intervals": len(interval_outputs),
            "action_clips": int(clip_count),
        },
        "tracks": track_outputs,
        "event_log": event_log,
        "interval_summaries": interval_outputs,
        "track_memory_bank": [] if track_memory_bank is None else track_memory_bank.export_bank(),
    }
    if summarizer is not None:
        scene_images = visual_evidence.get_scene_images() if visual_evidence is not None else None
        outputs["scene_summary"] = summarizer.summarize_scene(
            event_log,
            all_track_payload,
            scene_images=scene_images,
        )

    if getattr(args, "question", None):
        documents = rag_builder.build_documents(event_log, all_track_payload, interval_outputs)
        retrieved = rag_builder.retrieve(
            getattr(args, "question", ""),
            documents,
            top_k=getattr(args, "question_top_k", 4),
        )
        qa_output = {
            "question": getattr(args, "question", ""),
            "retrieved_context": retrieved,
        }
        if summarizer is not None:
            qa_output["answer"] = summarizer.answer_question(
                getattr(args, "question", ""),
                retrieved,
            )
        outputs["qa"] = qa_output

    return outputs
