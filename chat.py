import argparse

from chatbot.session import SurveillanceChatSession
from memory_store.sqlite_store import SurveillanceMemoryStore
from summarization.qwen_summary import build_summarizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory_db",
        type=str,
        default="memory_store/surveillance_memory.db",
        help="sqlite memory database path",
    )
    parser.add_argument("--camera_id", type=str, default=None, help="optional camera filter")
    parser.add_argument(
        "--run_id",
        action="append",
        default=[],
        help="optional run filter; can be repeated",
    )
    parser.add_argument("--session_id", type=str, default=None, help="resume or name a chat session")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--summary_backend", type=str, default="text", choices=["text", "vl"])
    parser.add_argument("--llm_model", type=str, default=None, help="chat model id")
    parser.add_argument("--top_k", type=int, default=4, help="retrieved context chunks per question")
    parser.add_argument("--history_turns", type=int, default=8, help="chat history turns to include")
    parser.add_argument("--question", type=str, default=None, help="ask one question and exit")
    return parser.parse_args()


def _print_help():
    print("Commands:")
    print("  /help                 show commands")
    print("  /runs                 list available runs")
    print("  /camera <id|all>      set camera filter")
    print("  /use <run_id|all>     set run filter")
    print("  /history              show stored chat history")
    print("  /quit                 exit")


def main():
    args = parse_args()
    store = SurveillanceMemoryStore(args.memory_db)
    summarizer = build_summarizer(
        backend=args.summary_backend,
        model_id=args.llm_model,
        device=args.device,
    )
    session = SurveillanceChatSession(
        store=store,
        summarizer=summarizer,
        session_id=args.session_id,
        camera_id=args.camera_id,
        run_ids=args.run_id,
        top_k=args.top_k,
        history_turns=args.history_turns,
    )

    if args.question:
        result = session.ask(args.question)
        print(result["answer"])
        return

    print("Surveillance chat ready. Session:", session.session_id)
    _print_help()

    while True:
        try:
            question = input("user> ").strip()
        except EOFError:
            print()
            break

        if not question:
            continue
        if question == "/quit":
            break
        if question == "/help":
            _print_help()
            continue
        if question == "/runs":
            for run in session.list_runs(limit=20):
                print(
                    "{run_id} | camera={camera_id} | created={created_at} | video={video_path}".format(
                        run_id=run["run_id"],
                        camera_id=run["camera_id"],
                        created_at=run["created_at"],
                        video_path=run["video_path"],
                    )
                )
            continue
        if question.startswith("/camera "):
            value = question.split(" ", 1)[1].strip()
            session.set_camera_filter(None if value == "all" else value)
            if value == "all":
                session.set_run_filter([])
            print("camera filter:", session.camera_id or "all")
            continue
        if question.startswith("/use "):
            value = question.split(" ", 1)[1].strip()
            session.set_run_filter([] if value == "all" else [value])
            print("run filter:", session.run_ids or ["all"])
            continue
        if question == "/history":
            for item in session.history():
                print("{role}: {content}".format(role=item["role"], content=item["content"]))
            continue

        result = session.ask(question)
        print("assistant>", result["answer"])
        print("context>")
        for doc in result["retrieved_context"]:
            print(" -", doc["text"])


if __name__ == "__main__":
    main()
