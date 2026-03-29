from collections import defaultdict

from tqdm.auto import tqdm

from detection.detector import YoloDetector
from tracking.tracker import MultiObjectTracker
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
    detector = YoloDetector(weights=args.yolo, conf=args.min_conf, classes=object_labels)
    tracker = MultiObjectTracker(
        weights=args.yolo,
        tracker_cfg=args.tracker,
        conf=args.min_conf,
        classes=track_labels,
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

            object_dets = detector.detect(frame_bgr)
            tracks = tracker.update(frame_bgr)
            person_tracks = [tr for tr in tracks if tr.get("label") == "person"]
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

            event_builder.update(t_sec, person_tracks, object_dets)
            frame_progress.update(1)
            frame_progress.set_postfix(
                time_s=round(float(t_sec), 1),
                people=len(person_tracks),
                objs=len(object_dets),
                clips=clip_count,
            )
    finally:
        frame_progress.close()

    # (D) temporal aggregation + segmentation + optional LLM
    event_log, per_track_metadata = event_builder.finalize()
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
            "unsupported_object_labels": detector.missing_classes,
            "clip_len": clip_len,
            "stride": stride,
            "event_window_sec": args.event_window_sec,
            "long_summary_sec": getattr(args, "long_summary_sec", 60.0),
            "tracker": args.tracker,
            "device": getattr(args, "device", "auto"),
            "environment": describe_environment(getattr(args, "environment", "generic")),
            "summary_backend": getattr(args, "summary_backend", "text"),
            "llm_model": getattr(summarizer, "model_id", None),
        },
        "stats": {
            "sampled_frames": int(frame_progress.n),
            "estimated_sampled_frames": int(sampler.estimated_sampled_frames or 0),
            "native_total_frames": int(sampler.total_frames or 0),
            "video_duration_sec": round(float(sampler.duration_sec or 0.0), 2),
            "tracks": len(tracked_ids),
            "event_windows": len(event_log),
            "intervals": len(interval_outputs),
            "action_clips": int(clip_count),
        },
        "tracks": track_outputs,
        "event_log": event_log,
        "interval_summaries": interval_outputs,
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
