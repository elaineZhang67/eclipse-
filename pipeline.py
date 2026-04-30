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
from summarization.debug_evidence import DebugEvidenceWriter, default_debug_evidence_dir
from runtime.object_preset import describe_environment, resolve_object_labels


def _parse_csv_arg(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]


def _normalize_label(value):
    return str(value or "").strip().lower().replace("_", " ")


def _dedupe_labels(labels):
    deduped = []
    seen = set()
    for label in labels or []:
        normalized = _normalize_label(label)
        if not normalized or normalized in seen:
            continue
        deduped.append(str(label).strip())
        seen.add(normalized)
    return deduped


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


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _overlaps(start_a, end_a, start_b, end_b):
    return max(_safe_float(start_a), _safe_float(start_b)) < min(
        _safe_float(end_a),
        _safe_float(end_b),
    )


def _format_track_ref(track_id, metadata):
    return metadata.get("track_ref") or "{name} (track {track_id})".format(
        name=metadata.get("display_name") or "Person",
        track_id=int(track_id),
    )


def _profile_hint(profile):
    if not isinstance(profile, dict):
        return ""

    parts = []
    appearance = profile.get("appearance") or {}
    if appearance.get("top_color"):
        parts.append("top color {value}".format(value=appearance["top_color"]))
    if appearance.get("bottom_color"):
        parts.append("bottom color {value}".format(value=appearance["bottom_color"]))
    if appearance.get("outerwear"):
        parts.append("outerwear {value}".format(value=appearance["outerwear"]))
    carried = [
        item.get("label")
        for item in profile.get("carried_objects", [])
        if isinstance(item, dict) and item.get("label")
    ]
    if carried:
        parts.append("carried " + ", ".join(carried))
    if profile.get("behavior_overview"):
        parts.append("profile {value}".format(value=profile["behavior_overview"]))
    return "; ".join(parts)


def _window_track_view(track_id, payload, window_start, window_end):
    metadata = payload.get("metadata", {})
    segments = [
        segment
        for segment in payload.get("segments", [])
        if _overlaps(segment.get("start"), segment.get("end"), window_start, window_end)
    ]
    object_tracks = [
        item
        for item in metadata.get("object_tracks", [])
        if item.get("first_seen") is None
        or item.get("last_seen") is None
        or _overlaps(item.get("first_seen"), item.get("last_seen"), window_start, window_end)
    ]

    return {
        "track_id": int(track_id),
        "display_name": metadata.get("display_name"),
        "track_ref": _format_track_ref(track_id, metadata),
        "first_seen": metadata.get("first_seen"),
        "last_seen": metadata.get("last_seen"),
        "objects": list(metadata.get("objects", [])),
        "object_tracks": object_tracks,
        "segments": segments,
        "profile": payload.get("profile"),
        "summary": payload.get("summary"),
    }


def _object_labels_from_event(event):
    labels = []
    for item in event.get("objects", []):
        if isinstance(item, dict):
            label = item.get("label")
            object_track_id = item.get("object_track_id")
            confidence = item.get("confidence")
            text = str(label) if label else None
            if text and object_track_id is not None:
                text = "{label} object track {object_track_id}".format(
                    label=text,
                    object_track_id=object_track_id,
                )
            if text and confidence is not None:
                text = "{text} confidence {confidence}".format(text=text, confidence=confidence)
            if text:
                labels.append(text)
        elif item:
            labels.append(str(item))
    return labels


def _format_window_event(event):
    event_type = event.get("type")
    if event_type == "enter":
        return "{track_ref} entered".format(
            track_ref=event.get("track_ref") or "track {track_id}".format(track_id=event.get("track_id")),
        )
    if event_type == "last_seen":
        return "{track_ref} last seen at {time}s".format(
            track_ref=event.get("track_ref") or "track {track_id}".format(track_id=event.get("track_id")),
            time=event.get("time"),
        )
    if event_type in {"attributed_objects", "near_objects"}:
        objects = _object_labels_from_event(event)
        if not objects:
            return ""
        verb = "associated with" if event_type == "attributed_objects" else "near"
        return "{track_ref} {verb} {objects}".format(
            track_ref=event.get("track_ref") or "track {track_id}".format(track_id=event.get("track_id")),
            verb=verb,
            objects=", ".join(objects),
        )
    if event_type == "interaction":
        refs = event.get("track_refs") or [
            "track {track_id}".format(track_id=track_id)
            for track_id in event.get("track_ids", [])
        ]
        if refs:
            return "{kind} interaction between {refs}".format(
                kind=event.get("kind", "unknown"),
                refs=" and ".join(refs),
            )
    return str(event)


def _format_window_track(track):
    ref = track.get("track_ref") or "track {track_id}".format(track_id=track.get("track_id"))
    pieces = [ref]
    hint = _profile_hint(track.get("profile"))
    if hint:
        pieces.append(hint)

    segments = []
    for segment in track.get("segments", []):
        segments.append(
            "{action} {start}-{end}s".format(
                action=segment.get("action", "unknown action"),
                start=segment.get("start"),
                end=segment.get("end"),
            )
        )
    pieces.append("actions " + (", ".join(segments) if segments else "no classified action in this window"))

    if track.get("objects"):
        pieces.append("objects " + ", ".join(track.get("objects", [])))
    return "; ".join(pieces)


def _build_window_packets(event_log, track_payload):
    packets = []
    for window in event_log:
        start = window["start"]
        end = window["end"]
        tracks = {}
        for track_id in window.get("active_tracks", []):
            payload = track_payload.get(track_id) or track_payload.get(str(track_id), {})
            tracks[int(track_id)] = _window_track_view(track_id, payload, start, end)

        packets.append(
            {
                "start": start,
                "end": end,
                "active_tracks": list(window.get("active_tracks", [])),
                "active_track_refs": list(window.get("active_track_refs", [])),
                "event_count": len(window.get("events", [])),
                "events": list(window.get("events", [])),
                "tracks": tracks,
            }
        )
    return packets


def _structured_window_summary(packet):
    start = packet.get("start")
    end = packet.get("end")
    active_track_refs = packet.get("active_track_refs", [])
    tracks = [_format_window_track(track) for track in packet.get("tracks", {}).values()]
    event_lines = [
        line
        for line in (_format_window_event(event) for event in packet.get("events", []))
        if line
    ]

    summary_parts = [
        "{start}-{end}s".format(start=start, end=end),
        "active people {tracks}".format(tracks=", ".join(active_track_refs) or "none"),
    ]
    if tracks:
        visible_tracks = tracks[:12]
        if len(tracks) > len(visible_tracks):
            visible_tracks.append("{count} additional active people omitted".format(count=len(tracks) - len(visible_tracks)))
        summary_parts.append("per-person evidence: " + " | ".join(visible_tracks))
    if event_lines:
        summary_parts.append("events: " + "; ".join(event_lines))
    return ". ".join(summary_parts)


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
    track_backend = str(getattr(args, "track_backend", "sam3")).strip().lower()
    if object_backend == "sam3" and track_backend == "sam3":
        combined_sam3_detector = Sam3Detector(
            model_id=getattr(args, "sam3_model", "facebook/sam3"),
            conf=args.min_conf,
            mask_threshold=getattr(args, "sam3_mask_threshold", 0.5),
            classes=_dedupe_labels(track_labels + object_labels),
            device=getattr(args, "device", "auto"),
            track_iou=getattr(args, "sam3_track_iou", 0.3),
            track_ttl=getattr(args, "sam3_track_ttl", 12),
        )
        object_source = combined_sam3_detector
        tracker = combined_sam3_detector
        object_source_missing = list(getattr(combined_sam3_detector, "missing_classes", []))
        tracker_missing_classes = list(getattr(combined_sam3_detector, "missing_classes", []))
    elif object_backend == "sam2":
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
    if not (object_backend == "sam3" and track_backend == "sam3"):
        if track_backend == "sam3":
            tracker = Sam3Detector(
                model_id=getattr(args, "sam3_model", "facebook/sam3"),
                conf=args.min_conf,
                mask_threshold=getattr(args, "sam3_mask_threshold", 0.5),
                classes=track_labels,
                device=getattr(args, "device", "auto"),
                track_iou=getattr(args, "sam3_track_iou", 0.3),
                track_ttl=getattr(args, "sam3_track_ttl", 12),
            )
        else:
            tracker = MultiObjectTracker(
                weights=args.yolo,
                tracker_cfg=args.tracker,
                conf=args.min_conf,
                classes=track_labels,
            )
        tracker_missing_classes = list(getattr(tracker, "missing_classes", []))
    track_label_set = {_normalize_label(label) for label in track_labels}
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
        object_min_hits=getattr(args, "object_attribution_min_hits", 3),
        object_min_confidence=getattr(args, "object_attribution_min_confidence", 0.55),
        object_near_confidence=getattr(args, "object_attribution_near_confidence", 0.28),
    )
    interval_builder = SceneIntervalBuilder(interval_sec=getattr(args, "long_summary_sec", 60.0))
    rag_builder = EventRAG()
    disable_progress = bool(getattr(args, "disable_progress", False))
    debug_evidence = None
    if bool(getattr(args, "debug_evidence", True)):
        debug_evidence = DebugEvidenceWriter(
            output_dir=getattr(args, "debug_evidence_dir", None)
            or default_debug_evidence_dir(getattr(args, "run_id", None)),
            stride_sec=getattr(args, "debug_evidence_stride_sec", 2.0),
            max_frames=getattr(args, "debug_evidence_max_frames", 80),
            enabled=True,
        )
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

            if object_source is tracker:
                combined_tracks = tracker.update(frame_bgr)
                person_tracks = [
                    tr for tr in combined_tracks if _normalize_label(tr.get("label")) in track_label_set
                ]
                object_tracks = [
                    tr for tr in combined_tracks if _normalize_label(tr.get("label")) not in track_label_set
                ]
            else:
                object_tracks = object_source.update(frame_bgr)
                tracks = tracker.update(frame_bgr)
                person_tracks = [
                    tr for tr in tracks if _normalize_label(tr.get("label")) in track_label_set
                ]
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

            assignments = event_builder.update(t_sec, person_tracks, object_tracks)
            if debug_evidence is not None:
                debug_evidence.update(
                    frame_bgr,
                    t_sec=t_sec,
                    person_tracks=person_tracks,
                    object_tracks=object_tracks,
                    assignments=assignments,
                    window_idx=int(t_sec // args.event_window_sec),
                )
            frame_progress.update(1)
            frame_progress.set_postfix(
                time_s=round(float(t_sec), 1),
                people=len(person_tracks),
                objs=len(object_tracks),
                clips=clip_count,
            )
    finally:
        frame_progress.close()
    debug_evidence_output = debug_evidence.close() if debug_evidence is not None else None

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

    window_packets = _build_window_packets(event_log, all_track_payload)
    window_outputs = []
    if bool(getattr(args, "summarize_event_windows", True)):
        window_progress = _build_tqdm(
            window_packets,
            desc="Windows",
            unit="window",
            disable=disable_progress,
        )
        for window_packet in window_progress:
            out = {
                "start": window_packet["start"],
                "end": window_packet["end"],
                "active_tracks": window_packet["active_tracks"],
                "active_track_refs": window_packet["active_track_refs"],
                "event_count": window_packet["event_count"],
                "structured_summary": _structured_window_summary(window_packet),
            }
            out["summary"] = out["structured_summary"]
            if bool(getattr(args, "llm_window_summaries", False)) and summarizer is not None:
                scene_images = (
                    visual_evidence.get_scene_images(
                        start_sec=window_packet["start"],
                        end_sec=window_packet["end"],
                    )
                    if visual_evidence is not None
                    else None
                )
                out["llm_summary"] = summarizer.summarize_window(
                    window_packet,
                    scene_images=scene_images,
                )
                out["summary"] = out["llm_summary"]
            window_outputs.append(out)
            window_progress.set_postfix(
                tracks=len(window_packet["active_tracks"]),
                events=window_packet["event_count"],
            )
        window_progress.close()

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
            "unsupported_track_labels": tracker_missing_classes,
            "unsupported_object_labels": object_source_missing,
            "clip_len": clip_len,
            "stride": stride,
            "event_window_sec": args.event_window_sec,
            "long_summary_sec": getattr(args, "long_summary_sec", 60.0),
            "summarize_event_windows": bool(getattr(args, "summarize_event_windows", True)),
            "llm_window_summaries": bool(getattr(args, "llm_window_summaries", False)),
            "tracker": args.tracker,
            "track_backend": track_backend,
            "person_tracker_model": getattr(args, "sam3_model", None) if track_backend == "sam3" else args.yolo,
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
            "object_attribution_min_hits": getattr(args, "object_attribution_min_hits", 3),
            "object_attribution_min_confidence": getattr(args, "object_attribution_min_confidence", 0.55),
            "object_attribution_near_confidence": getattr(args, "object_attribution_near_confidence", 0.28),
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
            "window_summaries": len(window_outputs),
            "intervals": len(interval_outputs),
            "action_clips": int(clip_count),
        },
        "tracks": track_outputs,
        "event_log": event_log,
        "window_summaries": window_outputs,
        "interval_summaries": interval_outputs,
        "debug_evidence": debug_evidence_output,
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
        documents = rag_builder.build_documents(event_log, all_track_payload, interval_outputs, window_outputs)
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
