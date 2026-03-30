import uuid

import streamlit as st

from chatbot.session import SurveillanceChatSession
from memory_store.sqlite_store import SurveillanceMemoryStore
from summarization.qwen_summary import build_summarizer


def _build_session(memory_db, device, summary_backend, llm_model, camera_id, run_ids, top_k, history_turns):
    store = SurveillanceMemoryStore(memory_db)
    summarizer = build_summarizer(
        backend=summary_backend,
        model_id=llm_model or None,
        device=device,
    )
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex[:12]
        st.session_state["session_id"] = session_id
    return SurveillanceChatSession(
        store=store,
        summarizer=summarizer,
        session_id=session_id,
        camera_id=camera_id,
        run_ids=run_ids,
        top_k=top_k,
        history_turns=history_turns,
    )


def _render_run_table(runs):
    if not runs:
        st.info("No persisted runs found.")
        return

    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run["run_id"],
                "camera_id": run["camera_id"],
                "created_at": run["created_at"],
                "video_path": run["video_path"],
                "environment": run.get("config", {}).get("environment", {}).get("name"),
                "summary_backend": run.get("config", {}).get("summary_backend"),
            }
        )
    st.dataframe(rows, use_container_width=True)


def _render_chat_history(history):
    for item in history:
        role = item["role"]
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(item["content"])


def main():
    st.set_page_config(page_title="Surveillance Chatbot", layout="wide")
    st.title("Surveillance Chatbot")
    st.caption("Query persisted surveillance runs from SQLite memory.")

    with st.sidebar:
        st.header("Settings")
        memory_db = st.text_input("Memory DB", value="memory_store/surveillance_memory.db")
        device = st.selectbox("Device", options=["auto", "cpu", "cuda", "mps"], index=0)
        summary_backend = st.selectbox("Chat backend", options=["text", "vl"], index=0)
        llm_model = st.text_input("LLM model override", value="")
        top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=4)
        history_turns = st.slider("History turns", min_value=0, max_value=20, value=8)

        store = SurveillanceMemoryStore(memory_db)
        cameras = store.list_cameras()
        camera_options = ["all"] + [item["camera_id"] for item in cameras]
        selected_camera = st.selectbox("Camera", options=camera_options, index=0)

        runs = store.list_runs(
            camera_id=None if selected_camera == "all" else selected_camera,
            limit=100,
        )
        run_labels = [
            "{run_id} | {camera_id} | {created_at}".format(
                run_id=run["run_id"],
                camera_id=run["camera_id"],
                created_at=run["created_at"],
            )
            for run in runs
        ]
        selected_run_labels = st.multiselect("Runs", options=run_labels, default=[])
        selected_run_ids = [label.split(" | ", 1)[0] for label in selected_run_labels]

        if st.button("New Session"):
            st.session_state["session_id"] = uuid.uuid4().hex[:12]
            st.rerun()

        st.caption("Session ID: {session_id}".format(session_id=st.session_state.get("session_id", "pending")))

    camera_filter = None if selected_camera == "all" else selected_camera
    session = _build_session(
        memory_db=memory_db,
        device=device,
        summary_backend=summary_backend,
        llm_model=llm_model,
        camera_id=camera_filter,
        run_ids=selected_run_ids,
        top_k=top_k,
        history_turns=history_turns,
    )

    left_col, right_col = st.columns([2, 1], gap="large")

    with right_col:
        st.subheader("Available Runs")
        _render_run_table(runs[:20])
        if cameras:
            st.subheader("Cameras")
            st.dataframe(cameras, use_container_width=True)

    with left_col:
        st.subheader("Chat")
        history = session.history()
        _render_chat_history(history)

        question = st.chat_input("Ask about entries, exits, interactions, clothing, or actions")
        if question:
            result = session.ask(question)
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                st.markdown(result["answer"])
                with st.expander("Retrieved context", expanded=False):
                    for doc in result["retrieved_context"]:
                        st.markdown(
                            "- **{doc_type}** [{start}, {end}]  \n{txt}".format(
                                doc_type=doc.get("type", "doc"),
                                start=doc.get("start"),
                                end=doc.get("end"),
                                txt=doc.get("text", ""),
                            )
                        )
            st.rerun()


if __name__ == "__main__":
    main()
