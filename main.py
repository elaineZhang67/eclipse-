import json
import argparse

from pipeline import run_pipeline
from memory_store.sqlite_store import SurveillanceMemoryStore
from runtime.object_preset import available_environments

print("MAIN START")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True, help="path to input video")
    p.add_argument("--yolo", type=str, default="yolov8n.pt", help="YOLOv8 weights")
    p.add_argument("--tracker", type=str, default="botsort.yaml", help="Ultralytics tracker config")
    p.add_argument(
        "--track_backend",
        type=str,
        default="yolo",
        choices=["yolo", "sam3"],
        help="person tracking backend: sam3 uses SAM3 text-prompt person segmentation, yolo uses YOLO + tracker",
    )
    p.add_argument(
        "--object_backend",
        type=str,
        default="sam3",
        choices=["yolo", "sam2", "sam3"],
        help=(
            "object detection backend: yolo uses YOLO tracking, sam2 uses YOLO proposals + "
            "SAM2 box-prompt refinement, sam3 uses SAM3 text-prompt segmentation"
        ),
    )
    p.add_argument(
        "--sam2_model",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="SAM2 model id used when --object_backend sam2",
    )
    p.add_argument(
        "--sam2_mask_threshold",
        type=float,
        default=0.5,
        help="mask threshold used when --object_backend sam2",
    )
    p.add_argument(
        "--sam2_track_iou",
        type=float,
        default=0.3,
        help="IoU threshold for lightweight SAM2 object ID association",
    )
    p.add_argument(
        "--sam2_track_ttl",
        type=int,
        default=12,
        help="number of sampled frames to keep unmatched SAM2 object tracks alive",
    )
    p.add_argument(
        "--sam3_model",
        type=str,
        default="facebook/sam3",
        help="SAM3 model id used when --object_backend sam3",
    )
    p.add_argument(
        "--sam3_mask_threshold",
        type=float,
        default=0.5,
        help="mask threshold used when --object_backend sam3",
    )
    p.add_argument(
        "--sam3_track_iou",
        type=float,
        default=0.3,
        help="IoU threshold for lightweight SAM3 object ID association",
    )
    p.add_argument(
        "--sam3_track_ttl",
        type=int,
        default=12,
        help="number of sampled frames to keep unmatched SAM3 object tracks alive",
    )
    p.add_argument("--disable_progress", action="store_true", help="disable tqdm progress bars")
    p.add_argument(
        "--use_track_memory",
        action="store_true",
        help="augment tracking with lightweight appearance embeddings and an identity memory bank",
    )
    p.add_argument(
        "--appearance_match_threshold",
        type=float,
        default=0.82,
        help="cosine similarity threshold used to merge fragmented person tracks in the appearance memory bank",
    )
    p.add_argument(
        "--appearance_memory_ttl_sec",
        type=float,
        default=8.0,
        help="how long to keep raw tracker-to-memory bindings alive before reassociation",
    )
    p.add_argument(
        "--appearance_reassoc_gap_sec",
        type=float,
        default=20.0,
        help="maximum gap allowed when matching a new raw tracker ID to an older memory track",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cpu", "cuda", "mps"],
        help="execution device for torch models",
    )
    p.add_argument("--fps", type=float, default=4.0, help="sampling FPS (approx)")
    p.add_argument("--clip_len", type=int, default=16, help="frames per clip")
    p.add_argument("--stride", type=int, default=8, help="slide stride in frames")
    p.add_argument("--clip_sec", type=float, default=2.0, help="seconds per action clip; overrides clip_len")
    p.add_argument("--stride_sec", type=float, default=1.0, help="seconds between clips; overrides stride")
    p.add_argument("--min_conf", type=float, default=0.25, help="YOLO conf threshold")
    p.add_argument("--max_people", type=int, default=20, help="max tracked people to keep state for")
    p.add_argument("--track_labels", type=str, default="person", help="comma-separated classes to track")
    p.add_argument(
        "--environment",
        type=str,
        default="generic",
        choices=available_environments(),
        help="environment preset used to choose up to five object types when --object_labels is omitted",
    )
    p.add_argument("--max_object_types", type=int, default=5, help="maximum number of object types to monitor")
    p.add_argument(
        "--object_labels",
        type=str,
        default=None,
        help="comma-separated object labels to attribute to nearby people; overrides environment preset",
    )
    p.add_argument("--event_window_sec", type=float, default=5.0, help="event-log window size in seconds")
    p.add_argument("--long_summary_sec", type=float, default=60.0, help="longer summary chunk size in seconds")
    p.add_argument(
        "--no_window_summaries",
        action="store_false",
        dest="summarize_event_windows",
        help="disable structured 5-second window summaries",
    )
    p.set_defaults(summarize_event_windows=True)
    p.add_argument(
        "--llm_window_summaries",
        action="store_true",
        help="also run the selected summarizer on every 5-second window; this is slower and uses more GPU memory",
    )
    p.add_argument("--interaction_combine_iou", type=float, default=0.05, help="IOU threshold to merge two boxes into one interaction region")
    p.add_argument("--interaction_combine_dist", type=float, default=1.2, help="normalized center-distance threshold for a merged interaction region")
    p.add_argument("--interaction_nearby_dist", type=float, default=2.5, help="normalized center-distance threshold for nearby but separate interactions")
    p.add_argument("--use_llm", action="store_true", help="enable LLM summary")
    p.add_argument(
        "--summary_backend",
        type=str,
        default="text",
        choices=["text", "vl"],
        help="summary backend: text uses Qwen text-only prompts, vl uses Gemma4 image-conditioned prompts by default",
    )
    p.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="HF model id for the summarizer; defaults depend on --summary_backend",
    )
    p.add_argument("--vl_max_track_images", type=int, default=4, help="max sampled person crops to pass to the VL summarizer per track")
    p.add_argument("--vl_max_scene_images", type=int, default=4, help="max sampled scene frames to pass to the VL summarizer")
    p.add_argument("--vl_track_gap_sec", type=float, default=1.5, help="minimum time gap between stored track crops for VL summarization")
    p.add_argument("--vl_scene_gap_sec", type=float, default=3.0, help="minimum time gap between stored scene frames for VL summarization")
    p.add_argument("--question", type=str, default=None, help="optional question to answer from the event log and long summaries")
    p.add_argument("--question_top_k", type=int, default=4, help="number of retrieved context chunks for QA")
    p.add_argument("--camera_id", type=str, default="camera_1", help="camera identifier for persistence and chat")
    p.add_argument("--run_id", type=str, default=None, help="optional run identifier when saving to persistent memory")
    p.add_argument("--save_json", type=str, default=None, help="optional JSON file path for saving outputs")
    p.add_argument("--memory_db", type=str, default=None, help="optional sqlite memory store path")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = run_pipeline(args)

    if args.memory_db:
        store = SurveillanceMemoryStore(args.memory_db)
        persisted_run_id = store.save_run(
            results,
            camera_id=args.camera_id,
            video_path=args.video,
            run_id=args.run_id,
        )
        results["persisted_run_id"] = persisted_run_id

    if args.save_json:
        with open(args.save_json, "w") as handle:
            json.dump(results, handle, indent=2)

    print("\n================= FINAL OUTPUT =================")
    print("\nConfig:")
    print(results["config"])
    if "stats" in results:
        print("\nStats:")
        print(results["stats"])
    if "persisted_run_id" in results:
        print("\nPersisted run id:")
        print(results["persisted_run_id"])

    if results["config"]["unsupported_track_labels"]:
        print("\nUnsupported track labels:")
        print(results["config"]["unsupported_track_labels"])
    if results["config"]["unsupported_object_labels"]:
        print("\nUnsupported object labels:")
        print(results["config"]["unsupported_object_labels"])

    print("\nEvent log:")
    for window in results["event_log"]:
        print(f"  - {window}")

    if results.get("interval_summaries"):
        print("\nInterval summaries:")
        for interval in results["interval_summaries"]:
            print(f"  - {interval}")

    if results.get("window_summaries"):
        print("\n5-second window summaries:")
        for window in results["window_summaries"]:
            print(f"  - {window}")

    if "scene_summary" in results:
        print("\nScene summary:")
        print(results["scene_summary"])

    if "qa" in results:
        print("\nQuestion:")
        print(results["qa"]["question"])
        print("Retrieved context:")
        for item in results["qa"].get("retrieved_context", []):
            print(f"  - {item}")
        if "answer" in results["qa"]:
            print("Answer:")
            print(results["qa"]["answer"])

    for tid, obj in results["tracks"].items():
        display_name = obj.get("metadata", {}).get("display_name")
        if display_name:
            print(f"\n[{display_name} | track_id={tid}]")
        else:
            print(f"\n[track_id={tid}]")
        print("Metadata:")
        print(f"  {obj['metadata']}")
        if "profile" in obj:
            print("Profile:")
            print(f"  {obj['profile']}")
        print("Timeline segments:")
        for seg in obj["segments"]:
            print(f"  - {seg}")
        if "summary" in obj:
            print("\nLLM Summary:")
            print(obj["summary"])

    if results.get("track_memory_bank"):
        print("\nTrack memory bank:")
        for item in results["track_memory_bank"]:
            print(f"  - {item}")
