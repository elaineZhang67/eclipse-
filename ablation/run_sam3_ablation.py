from main import parse_args
from pipeline import run_pipeline

from ablation._common import persist_and_report


def main():
    args = parse_args()
    args.object_backend = "sam3"
    if not getattr(args, "save_json", None):
        args.save_json = "outputs/sam3_ablation.json"
    print("SAM3 ablation: object detection backend set to SAM3")
    results = run_pipeline(args)
    persist_and_report(args, results)


if __name__ == "__main__":
    main()
