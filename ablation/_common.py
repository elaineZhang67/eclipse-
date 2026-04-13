import json

from memory_store.sqlite_store import SurveillanceMemoryStore


def persist_and_report(args, results):
    if getattr(args, "memory_db", None):
        store = SurveillanceMemoryStore(args.memory_db)
        persisted_run_id = store.save_run(
            results,
            camera_id=args.camera_id,
            video_path=args.video,
            run_id=getattr(args, "run_id", None),
        )
        results["persisted_run_id"] = persisted_run_id

    if getattr(args, "save_json", None):
        with open(args.save_json, "w") as handle:
            json.dump(results, handle, indent=2)

    print("\n================= ABLATION OUTPUT =================")
    print("\nConfig:")
    print(results.get("config", {}))
    if "stats" in results:
        print("\nStats:")
        print(results["stats"])
    if "persisted_run_id" in results:
        print("\nPersisted run id:")
        print(results["persisted_run_id"])
    if "scene_summary" in results:
        print("\nScene summary:")
        print(results["scene_summary"])
    return results
