import json
import os
import time
from pathlib import Path

import requests
import streamlit as st


DEFAULT_API_BASE = os.environ.get("SURVEILLANCE_API_BASE", "http://127.0.0.1:8000")
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]
PROCESSING_STATUSES = {"uploaded", "queued", "running"}


def _inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f6f8;
            color: #15191f;
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        .topbar {
            border-bottom: 1px solid #d8dde6;
            padding-bottom: 0.9rem;
            margin-bottom: 1rem;
        }
        .title {
            font-size: 1.45rem;
            font-weight: 760;
            line-height: 1.15;
        }
        .muted {
            color: #69717d;
            font-size: 0.88rem;
        }
        .panel {
            background: #ffffff;
            border: 1px solid #dfe4eb;
            border-radius: 8px;
            padding: 1rem;
        }
        .status-row {
            align-items: center;
            border: 1px solid #dfe4eb;
            border-radius: 8px;
            display: flex;
            gap: 0.75rem;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding: 0.75rem 0.9rem;
        }
        .status-pill {
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 720;
            padding: 0.2rem 0.55rem;
            text-transform: uppercase;
        }
        .status-processing {
            background: #fff1c2;
            color: #674f00;
        }
        .status-completed {
            background: #dff5e5;
            color: #126b36;
        }
        .status-failed {
            background: #ffe0df;
            color: #a12622;
        }
        .session-meta {
            color: #4d5561;
            font-size: 0.82rem;
            overflow-wrap: anywhere;
        }
        .video-shell {
            margin-top: 0.35rem;
        }
        div[data-testid="stChatInput"] {
            border-top: 1px solid #dfe4eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _query_session_id():
    try:
        value = st.query_params.get("session_id")
    except Exception:
        params = st.experimental_get_query_params()
        value = params.get("session_id", [None])[0]
    if isinstance(value, list):
        value = value[0] if value else None
    return value


def _set_active_session(session_id):
    st.session_state["active_session_id"] = session_id
    try:
        st.query_params["session_id"] = session_id
    except Exception:
        st.experimental_set_query_params(session_id=session_id)


def _clear_active_session():
    st.session_state.pop("active_session_id", None)
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


def _active_session_id():
    return st.session_state.get("active_session_id") or _query_session_id()


def _api_request(method, api_base, path, timeout=(8, 60), **kwargs):
    url = api_base.rstrip("/") + path
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError("Backend request failed: {error}".format(error=exc)) from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        raise RuntimeError(detail or "Backend returned HTTP {code}".format(code=response.status_code))
    if not response.content:
        return None
    return response.json()


def _list_sessions(api_base):
    return _api_request("GET", api_base, "/chat/sessions", params={"limit": 25}, timeout=(5, 20))


def _load_session(api_base, session_id):
    return _api_request("GET", api_base, "/chat/sessions/{session_id}".format(session_id=session_id), timeout=(5, 20))


def _load_messages(api_base, session_id):
    payload = _api_request(
        "GET",
        api_base,
        "/chat/sessions/{session_id}/messages".format(session_id=session_id),
        params={"limit": 100},
        timeout=(5, 20),
    )
    return payload.get("messages", [])


def _upload_video(
    api_base,
    uploaded_file,
    camera_id,
    label,
    pipeline_device,
    summary_backend,
    llm_model,
):
    options = {
        "device": pipeline_device,
        "summary_backend": summary_backend,
    }
    if llm_model:
        options["llm_model"] = llm_model

    uploaded_file.seek(0)
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file,
            uploaded_file.type or "application/octet-stream",
        )
    }
    data = {
        "camera_id": camera_id,
        "label": label or uploaded_file.name,
        "options_json": json.dumps(options),
    }
    return _api_request("POST", api_base, "/chat/upload", files=files, data=data, timeout=(20, 300))


def _ask(api_base, session_id, question, top_k, history_turns, answer_backend, answer_model, device):
    payload = {
        "question": question,
        "top_k": top_k,
        "history_turns": history_turns,
        "answer_backend": answer_backend,
        "answer_model": answer_model or None,
        "device": device,
    }
    return _api_request(
        "POST",
        api_base,
        "/chat/sessions/{session_id}/ask".format(session_id=session_id),
        json=payload,
        timeout=(10, 600),
    )


def _format_session_label(session):
    label = session.get("label") or session.get("video_id") or session["session_id"]
    return "{label} | {status} | {session_id}".format(
        label=label,
        status=session.get("status", "unknown"),
        session_id=session["session_id"],
    )


def _status_class(status):
    if status == "completed":
        return "status-completed"
    if status == "failed":
        return "status-failed"
    return "status-processing"


def _render_status(session):
    job = session.get("processing_job") or {}
    status = session.get("status", "unknown")
    job_status = job.get("status")
    parts = [
        "session {session_id}".format(session_id=session.get("session_id")),
        "video {video_id}".format(video_id=session.get("video_id")),
    ]
    if session.get("run_id"):
        parts.append("run {run_id}".format(run_id=session["run_id"]))
    if job_status:
        parts.append("job {job_status}".format(job_status=job_status))

    st.markdown(
        """
        <div class="status-row">
          <div>
            <span class="status-pill {klass}">{status}</span>
            <div class="session-meta">{meta}</div>
          </div>
          <div class="muted">{updated}</div>
        </div>
        """.format(
            klass=_status_class(status),
            status=status,
            meta=" / ".join(parts),
            updated=session.get("updated_at", ""),
        ),
        unsafe_allow_html=True,
    )


def _render_video(session):
    video_path = session.get("video_path")
    if not video_path:
        return
    path = Path(video_path)
    if path.exists():
        st.markdown('<div class="video-shell">', unsafe_allow_html=True)
        st.video(str(path))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("Video file is registered but is not accessible from the Streamlit process.")


def _render_messages(messages):
    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def _render_upload(api_base, pipeline_device, summary_backend, llm_model):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    with st.form("upload_form", clear_on_submit=False):
        uploaded_file = st.file_uploader("Video", type=VIDEO_TYPES)
        camera_id = st.text_input("Camera", value="camera_1")
        label = st.text_input("Label", value="")
        submitted = st.form_submit_button("Upload and process", use_container_width=True)

    if submitted:
        if uploaded_file is None:
            st.error("Choose a video first.")
        else:
            with st.spinner("Uploading and starting pipeline..."):
                session = _upload_video(
                    api_base=api_base,
                    uploaded_file=uploaded_file,
                    camera_id=camera_id.strip() or "camera_1",
                    label=label.strip(),
                    pipeline_device=pipeline_device,
                    summary_backend=summary_backend,
                    llm_model=llm_model.strip(),
                )
            _set_active_session(session["session_id"])
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Video Chat", layout="wide")
    _inject_css()

    st.markdown(
        """
        <div class="topbar">
          <div class="title">Video Chat</div>
          <div class="muted">Upload a video, wait for processing, then ask grounded questions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        api_base = st.text_input("Backend", value=DEFAULT_API_BASE)
        st.divider()
        pipeline_device = st.selectbox("Pipeline device", options=["auto", "cuda", "cpu", "mps"], index=0)
        summary_backend = st.selectbox("Pipeline summary", options=["vl", "text"], index=0)
        llm_model = st.text_input("Pipeline model", value="")
        st.divider()
        answer_backend = st.selectbox("Answer backend", options=["text", "vl"], index=0)
        answer_model = st.text_input("Answer model", value="")
        qa_device = st.selectbox("Answer device", options=["auto", "cuda", "cpu", "mps"], index=0)
        top_k = st.slider("Evidence", min_value=1, max_value=10, value=4)
        history_turns = st.slider("History", min_value=0, max_value=20, value=8)
        st.divider()

        try:
            sessions = _list_sessions(api_base)
        except Exception as exc:
            sessions = []
            st.error(exc)

        active_session_id = _active_session_id()
        if sessions:
            labels = ["New chat"] + [_format_session_label(session) for session in sessions]
            session_ids = [None] + [session["session_id"] for session in sessions]
            index = session_ids.index(active_session_id) if active_session_id in session_ids else 0
            selected = st.selectbox("Sessions", labels, index=index)
            selected_session_id = session_ids[labels.index(selected)]
            if selected_session_id != active_session_id:
                if selected_session_id:
                    _set_active_session(selected_session_id)
                else:
                    _clear_active_session()
                st.rerun()

        if st.button("New chat", use_container_width=True):
            _clear_active_session()
            st.rerun()

    session_id = _active_session_id()
    if not session_id:
        _render_upload(api_base, pipeline_device, summary_backend, llm_model)
        return

    try:
        session = _load_session(api_base, session_id)
    except Exception as exc:
        st.error(exc)
        if st.button("Start over", use_container_width=True):
            _clear_active_session()
            st.rerun()
        return

    _render_status(session)

    status = session.get("status")
    left, right = st.columns([1.35, 1], gap="large")

    with right:
        _render_video(session)
        if session.get("error"):
            st.error(session["error"])

    with left:
        messages = _load_messages(api_base, session_id)
        _render_messages(messages)

        if status in PROCESSING_STATUSES:
            st.info("Processing video. Chat will unlock when the pipeline is complete.")
            time.sleep(2)
            st.rerun()
        elif status == "failed":
            st.error("Pipeline failed. Start a new chat after checking the backend log.")
        elif status == "completed":
            question = st.chat_input("Ask about this video")
            if question:
                with st.spinner("Retrieving evidence and generating answer..."):
                    _ask(
                        api_base=api_base,
                        session_id=session_id,
                        question=question,
                        top_k=top_k,
                        history_turns=history_turns,
                        answer_backend=answer_backend,
                        answer_model=answer_model.strip(),
                        device=qa_device,
                    )
                st.rerun()
        else:
            st.warning("Unknown session status: {status}".format(status=status))


if __name__ == "__main__":
    main()
