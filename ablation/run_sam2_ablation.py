import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import parse_args
from pipeline import run_pipeline

from ablation._common import persist_and_report


def main():
    args = parse_args()
    args.object_backend = "sam2"
    if not getattr(args, "save_json", None):
        args.save_json = "outputs/sam2_ablation.json"
    print("SAM2 ablation: object backend set to YOLO proposals + SAM2 box-prompt refinement")
    results = run_pipeline(args)
    persist_and_report(args, results)


if __name__ == "__main__":
    main()
