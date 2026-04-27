import uuid
import html
from pathlib import Path

import streamlit as st

from backend.schemas import AskRequest
from backend.services import TimeframeQAService
from memory_store.sqlite_store import SurveillanceMemoryStore


DEFAULT_MEMORY_DB = "memory_store/surveillance_memory.db"
DEFAULT_SAMPLE_QUESTIONS = [
    "What is Person 1 wearing and doing in this selected timeframe?",
    "Who entered or left during this selected timeframe?",
    "Which people interacted closely, and what objects were associated with them?",
]


def _esc(value):
    return html.escape(str(value or ""))


@st.cache_resource(show_spinner=False)
def _get_store(memory_db):
    return SurveillanceMemoryStore(memory_db)


@st.cache_resource(show_spinner=False)
def _get_qa_service(memory_db):
    return TimeframeQAService(_get_store(memory_db))


def _inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(162, 197, 255, 0.22), transparent 32%),
                radial-gradient(circle at top right, rgba(255, 205, 142, 0.18), transparent 28%),
                linear-gradient(180deg, #f7f4ee 0%, #f1efe8 100%);
            color: #17202a;
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }
        .dashboard-shell {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(26, 71, 126, 0.10);
            border-radius: 24px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 18px 50px rgba(33, 48, 74, 0.07);
            margin-bottom: 1rem;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(12, 52, 105, 0.96), rgba(22, 95, 126, 0.92));
            color: #f8fbff;
            border-radius: 28px;
            padding: 1.5rem 1.6rem 1.3rem 1.6rem;
            box-shadow: 0 20px 45px rgba(7, 29, 56, 0.22);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.05rem;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            opacity: 0.90;
            max-width: 52rem;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        .section-caption {
            font-size: 0.92rem;
            color: #566573;
            margin-bottom: 0.85rem;
        }
        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.8rem;
        }
        .metric-card {
            background: rgba(247, 250, 255, 0.96);
            border: 1px solid rgba(26, 71, 126, 0.08);
            border-radius: 18px;
            padding: 0.8rem 0.9rem;
        }
        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #6483a0;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            font-size: 1.18rem;
            font-weight: 800;
            color: #17324d;
        }
        .info-chip {
            display: inline-block;
            background: rgba(14, 101, 129, 0.10);
            color: #0e6581;
            border-radius: 999px;
            padding: 0.24rem 0.62rem;
            margin-right: 0.38rem;
            margin-bottom: 0.38rem;
            font-size: 0.83rem;
            font-weight: 700;
        }
        .track-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(26, 71, 126, 0.09);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.65rem;
        }
        .track-title {
            font-weight: 800;
            color: #123a59;
            margin-bottom: 0.22rem;
        }
        .track-meta {
            font-size: 0.9rem;
            color: #5a6876;
        }
        .context-card {
            background: rgba(247, 248, 252, 0.98);
            border-left: 4px solid #2b6a99;
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.7rem;
        }
        .summary-box {
            background: rgba(255, 251, 243, 0.96);
            border: 1px solid rgba(188, 141, 72, 0.20);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_header(title, caption=None):
    st.markdown(
        '<div class="section-title">{title}</div>'.format(title=_esc(title)),
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(
            '<div class="section-caption">{caption}</div>'.format(caption=_esc(caption)),
            unsafe_allow_html=True,
        )


def _format_video_option(video):
    label = video.get("label") or Path(video.get("video_path", "")).name or video["video_id"]
    return "{label} | {video_id} | {camera_id}".format(
        label=label,
        video_id=video["video_id"],
        camera_id=video["camera_id"],
    )


def _format_run_option(run):
    backend = run.get("config", {}).get("summary_backend") or "unknown"
    created = run.get("created_at", "")
    return "{run_id} | {backend} | {created}".format(
        run_id=run["run_id"],
        backend=backend,
        created=created,
    )


def _safe_results(bundle):
    if not bundle:
        return {}
    return bundle.get("results", {})


def _video_duration_sec(video, run_bundle):
    run_results = _safe_results(run_bundle)
    stats = run_results.get("stats", {})
    duration = stats.get("video_duration_sec")
    if duration:
        return float(duration)
    return None


def _default_timeframe(duration_sec):
    if not duration_sec or duration_sec <= 0:
        return None
    return (0.0, min(float(duration_sec), 10.0))


def _render_metric_row(stats, run_bundle, video):
    run_results = _safe_results(run_bundle)
    config = run_results.get("config", {})
    duration = _video_duration_sec(video, run_bundle)
    tracks = stats.get("identity_tracks") or stats.get("tracks") or 0
    windows = stats.get("event_windows") or 0
    backend = config.get("object_backend") or "yolo"
    summary_backend = config.get("summary_backend") or "text"
    st.markdown(
        """
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value">{duration}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Identity Tracks</div>
            <div class="metric-value">{tracks}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Event Windows</div>
            <div class="metric-value">{windows}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Backends</div>
            <div class="metric-value">{backend} / {summary_backend}</div>
          </div>
        </div>
        """.format(
            duration="{value:.1f}s".format(value=duration) if duration else "unknown",
            tracks=_esc(tracks),
            windows=_esc(windows),
            backend=_esc(backend.upper()),
            summary_backend=_esc(summary_backend.upper()),
        ),
        unsafe_allow_html=True,
    )


def _render_video_panel(video, run_bundle):
    _section_header(
        "Video Review",
        "Inspect the selected video directly in the dashboard before asking timeframe-specific questions.",
    )
    video_path = None
    if run_bundle and run_bundle.get("video_path"):
        video_path = run_bundle["video_path"]
    elif video:
        video_path = video.get("video_path")

    if not video_path:
        st.info("No video path is available for the current selection.")
        return

    path_obj = Path(video_path)
    if not path_obj.exists():
        st.warning("Video path is registered, but the file is not accessible from this frontend: {path}".format(path=video_path))
        return

    st.video(str(path_obj))
    st.caption("Source: {path}".format(path=video_path))


def _render_scene_summary(run_bundle):
    run_results = _safe_results(run_bundle)
    summary = run_results.get("scene_summary")
    if not summary:
        st.info("No scene summary is stored for this run yet.")
        return

    st.markdown('<div class="summary-box">{summary}</div>'.format(summary=_esc(summary)), unsafe_allow_html=True)


def _person_sort_key(item):
    track_id, payload = item
    metadata = payload.get("metadata", {})
    first_seen = metadata.get("first_seen")
    return (first_seen is None, float(first_seen or 0.0), int(track_id))


def _format_behavior(payload):
    profile = payload.get("profile") or {}
    if not isinstance(profile, dict):
        return None
    behavior = profile.get("behavior_overview")
    if behavior:
        return behavior
    segments = payload.get("segments") or []
    if segments:
        return ", ".join(segment.get("action", "unknown") for segment in segments[:2])
    return None


def _format_appearance(payload):
    profile = payload.get("profile") or {}
    if not isinstance(profile, dict):
        return None
    appearance = profile.get("appearance") or {}
    parts = []
    if appearance.get("top_color"):
        parts.append("top {value}".format(value=appearance["top_color"]))
    if appearance.get("bottom_color"):
        parts.append("bottom {value}".format(value=appearance["bottom_color"]))
    if appearance.get("outerwear"):
        parts.append("outerwear {value}".format(value=appearance["outerwear"]))
    if appearance.get("helmet") is not None:
        parts.append("helmet {value}".format(value=appearance["helmet"]))
    return ", ".join(parts) if parts else None


def _render_track_cards(run_bundle):
    run_results = _safe_results(run_bundle)
    track_payload = run_results.get("tracks", {})
    if not track_payload:
        st.info("No per-person track data is stored for this run.")
        return

    for _, payload in sorted(track_payload.items(), key=_person_sort_key)[:8]:
        metadata = payload.get("metadata", {})
        title = metadata.get("display_name") or metadata.get("track_ref") or "Unknown Person"
        first_seen = metadata.get("first_seen")
        last_seen = metadata.get("last_seen")
        appearance = _format_appearance(payload)
        behavior = _format_behavior(payload)
        st.markdown(
            """
            <div class="track-card">
              <div class="track-title">{title}</div>
              <div class="track-meta">Seen {first_seen} to {last_seen}</div>
              {appearance}
              {behavior}
            </div>
            """.format(
                title=_esc(title),
                first_seen=_esc(first_seen if first_seen is not None else "unknown"),
                last_seen=_esc(last_seen if last_seen is not None else "unknown"),
                appearance=(
                    '<div class="track-meta"><strong>Appearance:</strong> {value}</div>'.format(value=_esc(appearance))
                    if appearance
                    else ""
                ),
                behavior=(
                    '<div class="track-meta"><strong>Behavior:</strong> {value}</div>'.format(value=_esc(behavior))
                    if behavior
                    else ""
                ),
            ),
            unsafe_allow_html=True,
        )


def _render_context_panel(result):
    retrieved = result.get("retrieved_context") or []
    if not retrieved:
        st.info("No context was retrieved for the latest answer.")
        return

    for doc in retrieved:
        title = "{doc_type} [{start}, {end}]".format(
            doc_type=doc.get("type", "doc").upper(),
            start=doc.get("start"),
            end=doc.get("end"),
        )
        st.markdown(
            """
            <div class="context-card">
              <div class="track-title">{title}</div>
              <div class="track-meta">{text}</div>
            </div>
            """.format(title=_esc(title), text=_esc(doc.get("text", ""))),
            unsafe_allow_html=True,
        )


def _render_chat_history(history):
    for item in history:
        role = item.get("role", "assistant")
        content = item.get("content", "")
        if hasattr(st, "chat_message"):
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(content)
        else:
            label = "User" if role == "user" else "Assistant"
            st.markdown("**{label}:** {content}".format(label=label, content=content))


def _sample_question_buttons():
    cols = st.columns(len(DEFAULT_SAMPLE_QUESTIONS))
    selected = None
    for idx, prompt in enumerate(DEFAULT_SAMPLE_QUESTIONS):
        if cols[idx].button(prompt, key="sample_{idx}".format(idx=idx), use_container_width=True):
            selected = prompt
    return selected


def _question_input():
    if hasattr(st, "chat_input"):
        return st.chat_input("Ask about clothing, actions, entries, exits, or interactions in the selected timeframe")

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Question")
        submitted = st.form_submit_button("Ask")
    if submitted and question.strip():
        return question.strip()
    return None


def _active_session_id():
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex[:12]
        st.session_state["session_id"] = session_id
    return session_id


def _build_ask_request(
    question,
    selected_video_id,
    selected_run_id,
    selected_camera_id,
    start_sec,
    end_sec,
    top_k,
    history_turns,
    answer_backend,
    answer_model,
    device,
):
    return AskRequest(
        question=question,
        video_id=selected_video_id,
        run_id=selected_run_id,
        camera_id=selected_camera_id,
        start_sec=start_sec,
        end_sec=end_sec,
        session_id=_active_session_id(),
        top_k=top_k,
        history_turns=history_turns,
        answer_backend=answer_backend,
        answer_model=answer_model or None,
        device=device,
    )


def main():
    st.set_page_config(page_title="Surveillance Monitor", layout="wide")
    _inject_css()

    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-title">Surveillance Monitor</div>
          <div class="hero-subtitle">
            Review a registered video, focus on a precise timeframe, and ask grounded questions over the
            timestamped memory built by the pipeline.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Console")
        memory_db = st.text_input("Memory DB", value=DEFAULT_MEMORY_DB)
        device = st.selectbox("QA device", options=["auto", "cpu", "cuda", "mps"], index=0)
        answer_backend = st.selectbox("QA backend", options=["text", "vl"], index=0)
        answer_model = st.text_input("QA model override", value="")
        top_k = st.slider("Top-K evidence", min_value=1, max_value=10, value=4)
        history_turns = st.slider("History turns", min_value=0, max_value=20, value=8)
        if st.button("New Session", use_container_width=True):
            st.session_state["session_id"] = uuid.uuid4().hex[:12]
            st.session_state["last_answer"] = None
            st.rerun()
        st.caption("Session: {session_id}".format(session_id=_active_session_id()))

    store = _get_store(memory_db)
    qa_service = _get_qa_service(memory_db)

    cameras = store.list_cameras()
    camera_options = ["all"] + [item["camera_id"] for item in cameras]

    shell = st.container()
    with shell:
        st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
        filter_col, stats_col = st.columns([1.6, 1], gap="large")
        with filter_col:
            _section_header(
                "Selection",
                "Choose the camera, video, and processed run you want the QA system to ground itself on.",
            )
            selected_camera = st.selectbox("Camera", options=camera_options, index=0)
        with stats_col:
            _section_header("Chat Mode", "Timeframe-aware QA is recommended for better retrieval quality.")
            st.markdown(
                '<span class="info-chip">QA backend: {backend}</span>'
                '<span class="info-chip">Top-K: {top_k}</span>'
                '<span class="info-chip">History: {history}</span>'.format(
                    backend=answer_backend.upper(),
                    top_k=top_k,
                    history=history_turns,
                ),
                unsafe_allow_html=True,
            )

        camera_filter = None if selected_camera == "all" else selected_camera
        videos = store.list_videos(camera_id=camera_filter, limit=200)
        if not videos:
            st.warning("No registered videos were found. Register or upload a video first, then refresh this page.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        video_options = [_format_video_option(video) for video in videos]
        default_video_idx = 0
        selected_video_label = st.selectbox("Video", options=video_options, index=default_video_idx)
        selected_video = videos[video_options.index(selected_video_label)]

        runs = store.list_runs(camera_id=camera_filter, video_id=selected_video["video_id"], limit=50)
        run_options = ["latest processed run"] + [_format_run_option(run) for run in runs]
        selected_run_label = st.selectbox("Run", options=run_options, index=0)
        selected_run = runs[0] if runs and selected_run_label == "latest processed run" else None
        if selected_run is None and selected_run_label != "latest processed run":
            selected_run = runs[run_options.index(selected_run_label) - 1]

        active_run_bundle = store.load_run(selected_run["run_id"]) if selected_run else None
        stats = _safe_results(active_run_bundle).get("stats", {})
        _render_metric_row(stats, active_run_bundle, selected_video)
        st.markdown("</div>", unsafe_allow_html=True)

    duration_sec = _video_duration_sec(selected_video, active_run_bundle)
    default_window = _default_timeframe(duration_sec)

    main_col, side_col = st.columns([1.5, 1], gap="large")

    with main_col:
        video_shell = st.container()
        with video_shell:
            st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
            _render_video_panel(selected_video, active_run_bundle)
            if default_window:
                event_window_sec = _safe_results(active_run_bundle).get("config", {}).get("event_window_sec", 5.0)
                start_sec, end_sec = st.slider(
                    "Inspection timeframe (seconds)",
                    min_value=0.0,
                    max_value=float(duration_sec),
                    value=default_window,
                    step=max(0.5, float(event_window_sec) / 10.0),
                )
                st.caption("Questions will be grounded only on evidence that overlaps this timeframe.")
            else:
                time_cols = st.columns(2)
                start_sec = time_cols[0].number_input("Start second", min_value=0.0, value=0.0, step=0.5)
                end_sec = time_cols[1].number_input("End second", min_value=start_sec + 0.5, value=start_sec + 10.0, step=0.5)
            st.markdown("</div>", unsafe_allow_html=True)

        chat_shell = st.container()
        with chat_shell:
            st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
            _section_header(
                "Timeframe QA",
                "Ask about clothing, actions, entries, exits, or interactions inside the selected window.",
            )
            selected_sample = _sample_question_buttons()
            history = store.load_messages(_active_session_id(), limit=history_turns)
            _render_chat_history(history)
            question = selected_sample or _question_input()
            if question:
                ask_request = _build_ask_request(
                    question=question,
                    selected_video_id=selected_video["video_id"],
                    selected_run_id=None if selected_run_label == "latest processed run" else selected_run["run_id"],
                    selected_camera_id=selected_video["camera_id"],
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    top_k=top_k,
                    history_turns=history_turns,
                    answer_backend=answer_backend,
                    answer_model=answer_model,
                    device=device,
                )
                try:
                    with st.spinner("Retrieving evidence and generating answer..."):
                        result = qa_service.ask(ask_request)
                    st.session_state["last_answer"] = result
                except Exception as exc:
                    st.error("Question answering failed: {error}".format(error=exc))
                else:
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with side_col:
        run_shell = st.container()
        with run_shell:
            st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
            _section_header(
                "Run Summary",
                "This panel helps you sanity-check what the system already thinks happened before you ask follow-up questions.",
            )
            if active_run_bundle:
                st.markdown(
                    '<span class="info-chip">Run: {run_id}</span>'
                    '<span class="info-chip">Camera: {camera}</span>'
                    '<span class="info-chip">Video: {video}</span>'.format(
                        run_id=_esc(active_run_bundle.get("run_id")),
                        camera=_esc(active_run_bundle.get("camera_id")),
                        video=_esc(selected_video["video_id"]),
                    ),
                    unsafe_allow_html=True,
                )
                _render_scene_summary(active_run_bundle)
            else:
                st.info("No processed run exists for this video yet. Process it first, then return here for QA.")
            st.markdown("</div>", unsafe_allow_html=True)

        people_shell = st.container()
        with people_shell:
            st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
            _section_header("Tracked People", "Top track summaries from the selected run.")
            _render_track_cards(active_run_bundle)
            st.markdown("</div>", unsafe_allow_html=True)

        context_shell = st.container()
        with context_shell:
            st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
            _section_header("Latest Evidence", "Retrieved context used for the most recent answer in this session.")
            last_answer = st.session_state.get("last_answer") or {}
            _render_context_panel(last_answer)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
