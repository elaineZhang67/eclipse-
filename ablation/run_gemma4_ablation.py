import pipeline as pipeline_module

from main import parse_args
from ablation._common import persist_and_report
from summarization.gemma4_summary import DEFAULT_GEMMA4_MODEL_ID, build_gemma4_summarizer


def _patched_build_summarizer(backend="vl", model_id=None, max_track_images=4, max_scene_images=4, device="auto"):
    return build_gemma4_summarizer(
        model_id=model_id or DEFAULT_GEMMA4_MODEL_ID,
        max_track_images=max_track_images,
        max_scene_images=max_scene_images,
        device=device,
    )


def main():
    args = parse_args()
    args.use_llm = True
    args.summary_backend = "vl"
    if not getattr(args, "llm_model", None):
        args.llm_model = DEFAULT_GEMMA4_MODEL_ID
    if not getattr(args, "save_json", None):
        args.save_json = "outputs/gemma4_ablation.json"

    print("Gemma 4 ablation: summarizer backend set to Gemma 4")
    original_builder = pipeline_module.build_summarizer
    pipeline_module.build_summarizer = _patched_build_summarizer
    try:
        results = pipeline_module.run_pipeline(args)
    finally:
        pipeline_module.build_summarizer = original_builder

    persist_and_report(args, results)


if __name__ == "__main__":
    main()
