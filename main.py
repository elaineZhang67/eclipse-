import argparse
from pipeline import run_pipeline

print("MAIN START")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True, help="path to input video")
    p.add_argument("--yolo", type=str, default="yolov8n.pt", help="YOLOv8 weights")
    p.add_argument("--tracker", type=str, default="botsort.yaml", help="Ultralytics tracker config")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="execution device for torch models",
    )
    p.add_argument("--fps", type=float, default=12.0, help="sampling FPS (approx)")
    p.add_argument("--clip_len", type=int, default=16, help="frames per clip")
    p.add_argument("--stride", type=int, default=8, help="slide stride in frames")
    p.add_argument("--clip_sec", type=float, default=2.0, help="seconds per action clip; overrides clip_len")
    p.add_argument("--stride_sec", type=float, default=1.0, help="seconds between clips; overrides stride")
    p.add_argument("--min_conf", type=float, default=0.25, help="YOLO conf threshold")
    p.add_argument("--max_people", type=int, default=20, help="max tracked people to keep state for")
    p.add_argument("--track_labels", type=str, default="person", help="comma-separated classes to track")
    p.add_argument(
        "--object_labels",
        type=str,
        default="backpack,handbag,suitcase",
        help="comma-separated object labels to attribute to nearby people",
    )
    p.add_argument("--event_window_sec", type=float, default=5.0, help="event-log window size in seconds")
    p.add_argument("--interaction_combine_iou", type=float, default=0.05, help="IOU threshold to merge two boxes into one interaction region")
    p.add_argument("--interaction_combine_dist", type=float, default=1.2, help="normalized center-distance threshold for a merged interaction region")
    p.add_argument("--interaction_nearby_dist", type=float, default=2.5, help="normalized center-distance threshold for nearby but separate interactions")
    p.add_argument("--use_llm", action="store_true", help="enable Qwen summary")
    p.add_argument(
        "--summary_backend",
        type=str,
        default="text",
        choices=["text", "vl"],
        help="summary backend: text uses Qwen text-only prompts, vl uses image-conditioned prompts",
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
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = run_pipeline(args)

    print("\n================= FINAL OUTPUT =================")
    print("\nConfig:")
    print(results["config"])

    if results["config"]["unsupported_track_labels"]:
        print("\nUnsupported track labels:")
        print(results["config"]["unsupported_track_labels"])
    if results["config"]["unsupported_object_labels"]:
        print("\nUnsupported object labels:")
        print(results["config"]["unsupported_object_labels"])

    print("\nEvent log:")
    for window in results["event_log"]:
        print(f"  - {window}")

    if "scene_summary" in results:
        print("\nScene summary:")
        print(results["scene_summary"])

    for tid, obj in results["tracks"].items():
        print(f"\n[track_id={tid}]")
        print("Metadata:")
        print(f"  {obj['metadata']}")
        print("Timeline segments:")
        for seg in obj["segments"]:
            print(f"  - {seg}")
        if "summary" in obj:
            print("\nLLM Summary:")
            print(obj["summary"])
