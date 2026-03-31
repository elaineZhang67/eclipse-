import uuid

import streamlit as st

from chatbot.session import SurveillanceChatSession
from memory_store.sqlite_store import SurveillanceMemoryStore
from summarization.qwen_summary import build_summarizer


@st.cache_resource(show_spinner=False)
def _get_store(memory_db):
    return SurveillanceMemoryStore(memory_db)


@st.cache_resource(show_spinner=False)
def _get_summarizer(device, summary_backend, llm_model):
    return build_summarizer(
        backend=summary_backend,
        model_id=llm_model or None,
        device=device,
    )


def _build_session(memory_db, device, summary_backend, llm_model, camera_id, run_ids, top_k, history_turns):
    store = _get_store(memory_db)
    summarizer = _get_summarizer(device, summary_backend, llm_model)
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex[:12]
        st.session_state["session_id"] = session_id
    session = SurveillanceChatSession(
        store=store,
        summarizer=summarizer,
        session_id=session_id,
        camera_id=camera_id,
        run_ids=run_ids,
        top_k=top_k,
        history_turns=history_turns,
    )
    session.last_track_ids = list(st.session_state.get("last_track_ids", []))
    return session


def _question_input():
    if hasattr(st, "chat_input"):
        return st.chat_input("Ask about entries, exits, interactions, clothing, or actions")

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Question")
        submitted = st.form_submit_button("Ask")
    if submitted and question.strip():
        return question.strip()
    return None


def _sample_questions():
    cols = st.columns(3)
    prompts = [
        "Who entered the scene and when?",
        "What was track 1 wearing and doing?",
        "Which object track stayed with the same person over time?",
    ]
    selected = None
    for idx, prompt in enumerate(prompts):
        if cols[idx].button(prompt, use_container_width=True):
            selected = prompt
    return selected


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
        _render_message("user" if role == "user" else "assistant", item["content"])


def _render_message(role, content):
    if hasattr(st, "chat_message"):
        with st.chat_message(role):
            st.markdown(content)
        return

    label = "User" if role == "user" else "Assistant"
    st.markdown("**{label}:** {content}".format(label=label, content=content))


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

        store = _get_store(memory_db)
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
            st.session_state["last_track_ids"] = []
            st.rerun()

        st.caption("Session ID: {session_id}".format(session_id=st.session_state.get("session_id", "pending")))

    camera_filter = None if selected_camera == "all" else selected_camera
    try:
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
    except Exception as exc:
        st.error("Failed to initialize chat backend: {error}".format(error=exc))
        st.stop()

    left_col, right_col = st.columns([2, 1], gap="large")

    with right_col:
        st.subheader("Available Runs")
        _render_run_table(runs[:20])
        if cameras:
            st.subheader("Cameras")
            st.dataframe(cameras, use_container_width=True)

    with left_col:
        st.subheader("Chat")
        if not runs:
            st.warning("No saved runs found. Run main.py with --memory_db first.")
        history = session.history()
        _render_chat_history(history)
        st.caption("Select a camera or specific run, then ask a question below.")
        sample_question = _sample_questions()
        question = sample_question or _question_input()
        if question:
            try:
                with st.spinner("Generating answer..."):
                    result = session.ask(question)
                st.session_state["last_track_ids"] = list(result.get("last_track_ids", []))
            except Exception as exc:
                st.error("Question answering failed: {error}".format(error=exc))
            else:
                _render_message("user", question)
                _render_message("assistant", result["answer"])
                with st.expander("Retrieved context", expanded=False):
                    if result["retrieved_context"]:
                        for doc in result["retrieved_context"]:
                            st.markdown(
                                "- **{doc_type}** [{start}, {end}]  \n{txt}".format(
                                    doc_type=doc.get("type", "doc"),
                                    start=doc.get("start"),
                                    end=doc.get("end"),
                                    txt=doc.get("text", ""),
                                )
                            )
                    else:
                        st.caption("No retrieved context.")
                st.rerun()


if __name__ == "__main__":
    main()
