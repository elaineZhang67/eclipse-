import json
import os
import time
import html
from pathlib import Path

import requests
import streamlit as st


DEFAULT_API_BASE = os.environ.get("SURVEILLANCE_API_BASE", "http://127.0.0.1:8000")
DEFAULT_PIPELINE_LLM_MODEL = "google/gemma-4-E4B-it"
DEFAULT_CHAT_ANSWER_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_TRACK_BACKEND = "sam3"
DEFAULT_OBJECT_BACKEND = "sam3"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]
PROCESSING_STATUSES = {"uploaded", "queued", "running"}


def _esc(value):
    return html.escape(str(value or ""))


def _inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,650;9..144,760&family=Manrope:wght@400;520;650;760;820&display=swap');
        :root {
            --bg: #f3efe6;
            --bg-2: #e9f0ea;
            --panel: rgba(255, 252, 246, 0.88);
            --panel-solid: #fffaf1;
            --ink: #18201b;
            --muted: #68736d;
            --line: rgba(36, 54, 43, 0.13);
            --line-strong: rgba(36, 54, 43, 0.24);
            --sidebar: #0e1a16;
            --sidebar-2: #13251f;
            --sidebar-ink: #edf7f0;
            --sidebar-muted: #9eb5aa;
            --green: #176f50;
            --green-soft: #dff3e9;
            --amber: #98610c;
            --amber-soft: #fff1cc;
            --red: #a23a31;
            --red-soft: #ffe1dc;
            --blue: #285d76;
            --blue-soft: #e1eef3;
            --shadow: 0 22px 70px rgba(23, 36, 29, 0.12);
            --shadow-soft: 0 12px 42px rgba(23, 36, 29, 0.08);
        }
        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 12%, rgba(243, 197, 118, 0.34), transparent 28rem),
                radial-gradient(circle at 88% 0%, rgba(91, 152, 121, 0.22), transparent 26rem),
                linear-gradient(135deg, #fbf7ed 0%, var(--bg) 42%, var(--bg-2) 100%);
            color: var(--ink);
            font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif;
        }
        .stApp::before {
            background-image:
                linear-gradient(rgba(23, 111, 80, 0.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(23, 111, 80, 0.055) 1px, transparent 1px);
            background-size: 42px 42px;
            content: "";
            inset: 0;
            mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.55), transparent 68%);
            pointer-events: none;
            position: fixed;
            z-index: 0;
        }
        .block-container {
            max-width: 1240px;
            padding-top: 1.1rem;
            padding-bottom: 2.25rem;
            position: relative;
            z-index: 1;
        }
        .topbar {
            align-items: center;
            background:
                radial-gradient(circle at 90% 5%, rgba(247, 188, 92, 0.34), transparent 17rem),
                radial-gradient(circle at 76% 82%, rgba(23, 111, 80, 0.22), transparent 15rem),
                linear-gradient(135deg, rgba(255, 252, 246, 0.94), rgba(241, 248, 241, 0.86));
            backdrop-filter: blur(18px);
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: var(--shadow);
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 1.1rem;
            overflow: hidden;
            padding: 1.25rem 1.35rem;
            position: relative;
            animation: riseIn 420ms ease-out both;
        }
        .topbar::after {
            border: 1px solid rgba(23, 111, 80, 0.18);
            border-radius: 999px;
            content: "";
            height: 9.5rem;
            position: absolute;
            right: -2.1rem;
            top: -3.8rem;
            transform: rotate(-14deg);
            width: 17rem;
        }
        .topbar > * {
            position: relative;
            z-index: 1;
        }
        .title {
            font-family: "Fraunces", Georgia, serif;
            font-size: clamp(2rem, 4vw, 3.35rem);
            font-weight: 760;
            line-height: 0.98;
            letter-spacing: -0.055em;
            max-width: 760px;
        }
        .kicker {
            color: var(--green);
            font-size: 0.74rem;
            font-weight: 820;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }
        .muted {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.55;
            max-width: 660px;
        }
        .top-actions {
            background: rgba(255, 250, 241, 0.76);
            border: 1px solid var(--line);
            border-radius: 18px;
            color: var(--ink);
            font-size: 0.8rem;
            font-weight: 700;
            line-height: 1.55;
            padding: 0.72rem 0.86rem;
            text-align: right;
            white-space: nowrap;
        }
        .section-title {
            color: var(--ink);
            font-size: 0.78rem;
            font-weight: 780;
            letter-spacing: 0.06em;
            margin: 0 0 0.55rem 0;
            text-transform: uppercase;
        }
        .panel {
            background: var(--panel);
            backdrop-filter: blur(18px);
            border: 1px solid var(--line);
            border-radius: 26px;
            box-shadow: var(--shadow-soft);
            padding: 1.12rem 1.12rem 1.18rem;
            animation: riseIn 480ms ease-out both;
        }
        .panel-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.55;
            margin: -0.2rem 0 0.85rem;
        }
        .mini-card {
            background:
                linear-gradient(135deg, rgba(255, 244, 220, 0.95), rgba(227, 243, 232, 0.92));
            border: 1px solid rgba(23, 111, 80, 0.13);
            border-radius: 18px;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
            margin-top: 0.85rem;
            padding: 0.82rem 0.9rem;
        }
        .prompt-grid {
            display: grid;
            gap: 0.62rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 0.86rem;
        }
        .prompt-card {
            background: rgba(255, 255, 255, 0.54);
            border: 1px solid rgba(36, 54, 43, 0.1);
            border-radius: 18px;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.42;
            padding: 0.82rem;
        }
        .prompt-card-title {
            color: var(--ink);
            font-size: 0.78rem;
            font-weight: 780;
            margin-bottom: 0.24rem;
        }
        .metric-strip {
            display: grid;
            gap: 0.62rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-top: 0.9rem;
        }
        .metric {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(36, 54, 43, 0.1);
            border-radius: 17px;
            min-height: 4.1rem;
            padding: 0.72rem;
        }
        .metric:last-child {
            grid-column: 1 / -1;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 780;
            overflow-wrap: anywhere;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 760;
            letter-spacing: 0.05em;
            margin-top: 0.12rem;
            text-transform: uppercase;
        }
        .upload-grid {
            display: grid;
            gap: 1rem;
            grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
        }
        @media (max-width: 760px) {
            .topbar {
                align-items: start;
                display: block;
                border-radius: 24px;
            }
            .top-actions {
                margin-top: 0.35rem;
                text-align: left;
            }
            .upload-grid {
                grid-template-columns: 1fr;
            }
            .prompt-grid {
                grid-template-columns: 1fr;
            }
        }
        .status-row {
            align-items: center;
            background: rgba(255, 252, 246, 0.88);
            backdrop-filter: blur(18px);
            border: 1px solid var(--line);
            border-radius: 24px;
            display: flex;
            gap: 0.75rem;
            justify-content: space-between;
            margin-bottom: 1.05rem;
            padding: 0.92rem 1.02rem;
            box-shadow: var(--shadow-soft);
            animation: riseIn 420ms ease-out both;
        }
        .status-left {
            min-width: 0;
            width: 100%;
        }
        .status-grid {
            display: grid;
            gap: 0.5rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin-top: 0.65rem;
        }
        .status-cell {
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(36, 54, 43, 0.1);
            border-radius: 16px;
            min-height: 3.1rem;
            padding: 0.45rem 0.55rem;
        }
        .status-cell-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .status-cell-value {
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 680;
            overflow-wrap: anywhere;
            padding-top: 0.15rem;
        }
        @media (max-width: 900px) {
            .status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        .status-pill {
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.72rem;
            font-weight: 760;
            letter-spacing: 0.04em;
            padding: 0.26rem 0.65rem;
            text-transform: uppercase;
        }
        .status-processing {
            background: var(--amber-soft);
            color: var(--amber);
        }
        .status-completed {
            background: var(--green-soft);
            color: var(--green);
        }
        .status-failed {
            background: var(--red-soft);
            color: var(--red);
        }
        .session-meta {
            color: var(--muted);
            font-size: 0.9rem;
            font-weight: 650;
            margin-top: 0.34rem;
            overflow-wrap: anywhere;
        }
        .chat-shell {
            min-height: 560px;
        }
        .chat-header {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.7rem;
        }
        .chat-status {
            background: var(--blue-soft);
            border: 1px solid rgba(40, 93, 118, 0.12);
            border-radius: 999px;
            color: var(--blue);
            font-size: 0.72rem;
            font-weight: 760;
            padding: 0.26rem 0.64rem;
        }
        .empty-chat {
            background:
                radial-gradient(circle at 18% 20%, rgba(247, 188, 92, 0.24), transparent 10rem),
                linear-gradient(135deg, rgba(255, 255, 255, 0.52), rgba(232, 244, 236, 0.66));
            border: 1px dashed var(--line-strong);
            border-radius: 22px;
            color: var(--muted);
            font-size: 0.9rem;
            margin: 0.2rem 0 1rem;
            padding: 1.05rem 1.1rem;
        }
        .preview-title {
            align-items: center;
            display: flex;
            gap: 0.55rem;
            justify-content: space-between;
            margin-bottom: 0.55rem;
        }
        .path-text {
            color: var(--muted);
            font-size: 0.74rem;
            overflow-wrap: anywhere;
        }
        .video-shell {
            background: #111418;
            border-radius: 20px;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
            overflow: hidden;
        }
        .detail-list {
            display: grid;
            gap: 0.55rem;
            margin-top: 0.75rem;
        }
        .detail-item {
            border-top: 1px solid var(--line);
            padding-top: 0.55rem;
        }
        .detail-label {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 760;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .detail-value {
            color: var(--ink);
            font-size: 0.85rem;
            overflow-wrap: anywhere;
            padding-top: 0.12rem;
        }
        .pipeline-flow {
            display: grid;
            gap: 0.58rem;
            margin-top: 0.86rem;
        }
        .flow-step {
            align-items: center;
            background: rgba(255, 255, 255, 0.54);
            border: 1px solid rgba(36, 54, 43, 0.1);
            border-radius: 17px;
            display: grid;
            gap: 0.65rem;
            grid-template-columns: 2.15rem minmax(0, 1fr);
            padding: 0.66rem 0.72rem;
        }
        .flow-number {
            align-items: center;
            background: #123d2f;
            border-radius: 999px;
            color: #fff8e9;
            display: flex;
            font-size: 0.76rem;
            font-weight: 820;
            height: 2.05rem;
            justify-content: center;
            width: 2.05rem;
        }
        .flow-title {
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 780;
        }
        .flow-copy {
            color: var(--muted);
            font-size: 0.74rem;
            line-height: 1.35;
            margin-top: 0.06rem;
        }
        div[data-testid="stChatInput"] {
            border-top: 1px solid var(--line);
            padding-top: 0.75rem;
        }
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 10% 0%, rgba(247, 188, 92, 0.16), transparent 14rem),
                linear-gradient(180deg, var(--sidebar-2), var(--sidebar) 44%, #09120f);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: var(--sidebar-muted);
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stTextInput label,
        section[data-testid="stSidebar"] .stSlider label {
            color: var(--sidebar-ink);
        }
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(255, 255, 255, 0.065);
            border-radius: 14px;
            color: var(--sidebar-ink);
            justify-content: flex-start;
            min-height: 2.15rem;
            padding-left: 0.72rem;
            text-align: left;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.16);
            color: #ffffff;
        }
        section[data-testid="stSidebar"] .stButton > button:focus {
            background: rgba(247, 188, 92, 0.16);
            border-color: rgba(247, 188, 92, 0.5);
            color: #ffffff;
        }
        section[data-testid="stSidebar"] details {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 0.8rem;
            padding-top: 0.55rem;
        }
        .sidebar-brand {
            color: var(--sidebar-ink);
            font-family: "Fraunces", Georgia, serif;
            font-size: 1.45rem;
            font-weight: 820;
            letter-spacing: -0.03em;
            margin: 0.15rem 0 0.05rem;
        }
        .sidebar-brand-sub {
            color: var(--sidebar-muted);
            font-size: 0.76rem;
            line-height: 1.4;
            margin-bottom: 0.85rem;
        }
        .sidebar-title {
            color: var(--sidebar-ink);
            font-size: 1.08rem;
            font-weight: 790;
            margin: 0.25rem 0 0.15rem;
        }
        .sidebar-caption {
            color: var(--sidebar-muted);
            font-size: 0.78rem;
            line-height: 1.4;
            margin-bottom: 0.7rem;
        }
        .sidebar-section {
            color: #f3c576;
            font-size: 0.7rem;
            font-weight: 820;
            letter-spacing: 0.08em;
            margin: 0.45rem 0 0.42rem;
            text-transform: uppercase;
        }
        .history-meta {
            color: var(--sidebar-muted);
            font-size: 0.68rem;
            margin: -0.22rem 0 0.46rem 0.72rem;
            overflow-wrap: anywhere;
        }
        .history-empty {
            background: rgba(255, 255, 255, 0.04);
            border: 1px dashed rgba(255, 255, 255, 0.18);
            border-radius: 16px;
            color: var(--sidebar-muted);
            font-size: 0.8rem;
            line-height: 1.45;
            padding: 0.78rem;
        }
        .sidebar-footer {
            color: var(--sidebar-muted);
            font-size: 0.72rem;
            line-height: 1.4;
            margin-top: 0.75rem;
        }
        .stButton > button, .stFormSubmitButton > button {
            border-radius: 16px;
            font-weight: 720;
            min-height: 2.45rem;
        }
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, #176f50, #2f8a64);
            border-color: rgba(23, 111, 80, 0.18);
            box-shadow: 0 12px 28px rgba(23, 111, 80, 0.22);
            color: white;
        }
        .stFormSubmitButton > button:hover {
            background: linear-gradient(135deg, #105d42, #287a58);
            border-color: rgba(23, 111, 80, 0.3);
            color: white;
        }
        div[data-testid="stFileUploader"] {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 0.4rem;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.68), rgba(232, 244, 236, 0.68));
            border-radius: 18px;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 18px;
            padding: 0.36rem 0.1rem;
        }
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
            background: rgba(255, 255, 255, 0.52);
            border: 1px solid rgba(36, 54, 43, 0.09);
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(23, 36, 29, 0.045);
            padding: 0.72rem 0.86rem;
        }
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p:last-child {
            margin-bottom: 0;
        }
        div[data-testid="stAlert"] {
            border-radius: 18px;
        }
        input, textarea {
            border-radius: 14px !important;
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
    return _api_request("GET", api_base, "/chat/sessions", params={"limit": 50}, timeout=(5, 20))


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
    track_backend,
    object_backend,
    summary_backend,
    llm_model,
):
    options = {
        "device": pipeline_device,
        "track_backend": track_backend,
        "object_backend": object_backend,
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
    return "{status} / {label} / {session_id}".format(
        label=label,
        status=session.get("status", "unknown"),
        session_id=session["session_id"],
    )


def _shorten(value, limit=36):
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _history_title(session):
    title = session.get("label") or session.get("video_id") or session.get("session_id")
    return _shorten(title, limit=34)


def _history_meta(session):
    status = session.get("status", "unknown")
    camera = session.get("camera_id") or session.get("video_id") or "video"
    updated = str(session.get("updated_at") or "")[:16]
    parts = [status, _shorten(camera, limit=18)]
    if updated:
        parts.append(updated)
    return " / ".join(parts)


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
    run_id = session.get("run_id") or "pending"
    job_id = session.get("job_id") or "pending"

    st.markdown(
        """
        <div class="status-row">
          <div class="status-left">
            <span class="status-pill {klass}">{status}</span>
            <div class="session-meta">{label}</div>
            <div class="status-grid">
              <div class="status-cell">
                <div class="status-cell-label">Session</div>
                <div class="status-cell-value">{session_id}</div>
              </div>
              <div class="status-cell">
                <div class="status-cell-label">Video</div>
                <div class="status-cell-value">{video_id}</div>
              </div>
              <div class="status-cell">
                <div class="status-cell-label">Job</div>
                <div class="status-cell-value">{job_status} / {job_id}</div>
              </div>
              <div class="status-cell">
                <div class="status-cell-label">Run</div>
                <div class="status-cell-value">{run_id}</div>
              </div>
            </div>
          </div>
          <div class="muted">{updated}</div>
        </div>
        """.format(
            klass=_status_class(status),
            status=_esc(status),
            label=_esc(session.get("label") or "Untitled video"),
            session_id=_esc(session.get("session_id")),
            video_id=_esc(session.get("video_id")),
            job_id=_esc(job_id),
            job_status=_esc(job_status or status),
            run_id=_esc(run_id),
            updated=_esc(session.get("updated_at", "")),
        ),
        unsafe_allow_html=True,
    )


def _render_video(session):
    video_path = session.get("video_path")
    if not video_path:
        return
    path = Path(video_path)
    st.markdown(
        """
        <div class="preview-title">
          <div class="section-title">Preview</div>
          <div class="path-text">{label}</div>
        </div>
        """.format(label=_esc(session.get("label") or path.name)),
        unsafe_allow_html=True,
    )
    if path.exists():
        st.markdown('<div class="video-shell">', unsafe_allow_html=True)
        st.video(str(path))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="detail-list">
              <div class="detail-item">
                <div class="detail-label">Source</div>
                <div class="detail-value">{path}</div>
              </div>
              <div class="detail-item">
                <div class="detail-label">Camera</div>
                <div class="detail-value">{camera}</div>
              </div>
            </div>
            """.format(
                path=_esc(video_path),
                camera=_esc(session.get("camera_id")),
            ),
            unsafe_allow_html=True,
        )
    else:
        st.warning("Video file is registered but is not accessible from the Streamlit process.")


def _render_messages(messages):
    if not messages:
        st.markdown('<div class="empty-chat">Ready for questions once processing is complete.</div>', unsafe_allow_html=True)
        return
    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def _render_upload(api_base, pipeline_device, track_backend, object_backend, summary_backend, llm_model):
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">New Video</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Drop in any surveillance clip. The backend will run the full pipeline first, then the chat unlocks with grounded answers from that processed session.</div>',
            unsafe_allow_html=True,
        )
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
                        track_backend=track_backend,
                        object_backend=object_backend,
                        summary_backend=summary_backend,
                        llm_model=llm_model.strip(),
                )
                _set_active_session(session["session_id"])
                st.rerun()
        st.markdown(
            '<div class="mini-card">The frontend will not answer while processing. Once the session is completed, questions use the indexed pipeline outputs for retrieval and response generation.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="prompt-grid">
              <div class="prompt-card">
                <div class="prompt-card-title">Ask what happened</div>
                "Summarize the main events in this clip."
              </div>
              <div class="prompt-card">
                <div class="prompt-card-title">Ask about people</div>
                "When did a person enter or leave the scene?"
              </div>
              <div class="prompt-card">
                <div class="prompt-card-title">Ask for evidence</div>
                "Which timestamps support that answer?"
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel">
              <div class="section-title">Workspace</div>
              <div class="panel-copy">Current runtime settings for the next upload.</div>
              <div class="metric-strip">
                <div class="metric">
                  <div class="metric-value">{pipeline_device}</div>
                  <div class="metric-label">Device</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{track_backend}</div>
                  <div class="metric-label">People</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{object_backend}</div>
                  <div class="metric-label">Objects</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{summary_backend}</div>
                  <div class="metric-label">Summary</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{model}</div>
                  <div class="metric-label">Model</div>
                </div>
              </div>
              <div class="pipeline-flow">
                <div class="flow-step">
                  <div class="flow-number">1</div>
                  <div>
                    <div class="flow-title">Upload video</div>
                    <div class="flow-copy">Create a dedicated session and persist the raw clip.</div>
                  </div>
                </div>
                <div class="flow-step">
                  <div class="flow-number">2</div>
                  <div>
                    <div class="flow-title">Run full pipeline</div>
                    <div class="flow-copy">SAM3 detection, tracking, summaries, memory, and retrieval indexes.</div>
                  </div>
                </div>
                <div class="flow-step">
                  <div class="flow-number">3</div>
                  <div>
                    <div class="flow-title">Chat unlocks</div>
                    <div class="flow-copy">Answers are generated only after processing completes.</div>
                  </div>
                </div>
              </div>
              <div class="detail-list">
                <div class="detail-item">
                  <div class="detail-label">Backend</div>
                  <div class="detail-value">{api_base}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Pipeline</div>
                  <div class="detail-value">{pipeline_device} / people {track_backend} / objects {object_backend} / {summary_backend}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Model</div>
                  <div class="detail-value">{model}</div>
                </div>
              </div>
            </div>
            """.format(
                api_base=_esc(api_base),
                pipeline_device=_esc(pipeline_device),
                track_backend=_esc(track_backend),
                object_backend=_esc(object_backend),
                summary_backend=_esc(summary_backend),
                model=_esc(llm_model or DEFAULT_PIPELINE_LLM_MODEL),
            ),
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Video Chat", layout="wide")
    _inject_css()

    sidebar_defaults = {
        "api_base": DEFAULT_API_BASE,
        "pipeline_device": "auto",
        "track_backend": DEFAULT_TRACK_BACKEND,
        "object_backend": DEFAULT_OBJECT_BACKEND,
        "summary_backend": "vl",
        "llm_model": DEFAULT_PIPELINE_LLM_MODEL,
        "answer_backend": "text",
        "answer_model": DEFAULT_CHAT_ANSWER_MODEL,
        "qa_device": "auto",
        "top_k": 4,
        "history_turns": 8,
    }
    for key, value in sidebar_defaults.items():
        st.session_state.setdefault(key, value)

    st.markdown(
        """
        <div class="topbar">
          <div>
            <div class="kicker">Eclipse Video Intelligence</div>
            <div class="title">Ask your video like a chat.</div>
            <div class="muted">Upload a clip, wait for the pipeline to finish, then ask grounded questions about what happened.</div>
          </div>
          <div class="top-actions">Pipeline-gated answers<br>Session-aware memory</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">Eclipse Chat</div><div class="sidebar-brand-sub">Video sessions stay here, like ChatGPT history.</div>',
            unsafe_allow_html=True,
        )

        if st.button("+ New chat", use_container_width=True):
            _clear_active_session()
            st.rerun()

        api_base = st.session_state["api_base"]
        active_session_id = _active_session_id()

        try:
            sessions = _list_sessions(api_base)
        except Exception as exc:
            sessions = []
            st.error(exc)

        st.markdown('<div class="sidebar-section">Chat history</div>', unsafe_allow_html=True)
        if sessions:
            for index, session in enumerate(sessions):
                session_id = session["session_id"]
                is_active = session_id == active_session_id
                title = _history_title(session)
                label = "Current: {title}".format(title=title) if is_active else title
                if st.button(
                    label,
                    key="history_{index}_{session_id}".format(index=index, session_id=session_id),
                    help=_format_session_label(session),
                    use_container_width=True,
                ):
                    _set_active_session(session_id)
                    st.rerun()
                st.markdown(
                    '<div class="history-meta">{meta}</div>'.format(meta=_esc(_history_meta(session))),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="history-empty">No chats yet. Upload a video to start the first session.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Advanced settings", expanded=False):
            st.markdown('<div class="sidebar-section">Connection</div>', unsafe_allow_html=True)
            api_base = st.text_input("Backend", key="api_base")
            st.markdown('<div class="sidebar-section">Processing</div>', unsafe_allow_html=True)
            pipeline_device = st.selectbox("Pipeline device", options=["auto", "cuda", "cpu", "mps"], key="pipeline_device")
            track_backend = st.selectbox("Person tracker", options=["sam3", "yolo"], key="track_backend")
            object_backend = st.selectbox("Object backend", options=["sam3", "yolo", "sam2"], key="object_backend")
            summary_backend = st.selectbox("Pipeline summary", options=["vl", "text"], key="summary_backend")
            llm_model = st.text_input("Pipeline model", key="llm_model")
            st.markdown('<div class="sidebar-section">Answers</div>', unsafe_allow_html=True)
            answer_backend = st.selectbox("Answer backend", options=["text", "vl"], key="answer_backend")
            answer_model = st.text_input("Answer model", key="answer_model")
            qa_device = st.selectbox("Answer device", options=["auto", "cuda", "cpu", "mps"], key="qa_device")
            top_k = st.slider("Evidence", min_value=1, max_value=10, key="top_k")
            history_turns = st.slider("History", min_value=0, max_value=20, key="history_turns")

        st.markdown(
            '<div class="sidebar-footer">Upload creates a new chat. Completed sessions can be reopened anytime from this list.</div>',
            unsafe_allow_html=True,
        )

    api_base = st.session_state["api_base"]
    pipeline_device = st.session_state["pipeline_device"]
    track_backend = st.session_state["track_backend"]
    object_backend = st.session_state["object_backend"]
    summary_backend = st.session_state["summary_backend"]
    llm_model = st.session_state["llm_model"]
    answer_backend = st.session_state["answer_backend"]
    answer_model = st.session_state["answer_model"]
    qa_device = st.session_state["qa_device"]
    top_k = st.session_state["top_k"]
    history_turns = st.session_state["history_turns"]

    session_id = _active_session_id()
    if not session_id:
        _render_upload(api_base, pipeline_device, track_backend, object_backend, summary_backend, llm_model)
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
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        _render_video(session)
        if session.get("error"):
            st.error(session["error"])
        st.markdown("</div>", unsafe_allow_html=True)

    with left:
        st.markdown('<div class="panel chat-shell">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="chat-header">
              <div class="section-title">Chat</div>
              <div class="chat-status">{status}</div>
            </div>
            """.format(status=_esc("Ready" if status == "completed" else status or "Unknown")),
            unsafe_allow_html=True,
        )
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
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
