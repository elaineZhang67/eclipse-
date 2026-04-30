import json
import os
import time
import html
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None

import requests
import streamlit as st


DEFAULT_API_BASE = os.environ.get("SURVEILLANCE_API_BASE", "http://127.0.0.1:8000")
DEFAULT_PIPELINE_LLM_MODEL = "google/gemma-4-E4B-it"
DEFAULT_CHAT_ANSWER_MODEL = DEFAULT_PIPELINE_LLM_MODEL
DEFAULT_TRACK_BACKEND = "yolo"
DEFAULT_OBJECT_BACKEND = "sam3"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm"]
PROCESSING_STATUSES = {"uploaded", "queued", "running"}
DISPLAY_TIMEZONE = os.environ.get("ECLIPSE_DISPLAY_TZ", "America/New_York")


def _esc(value):
    return html.escape(str(value or ""))


def _display_tz():
    if ZoneInfo is not None:
        try:
            return ZoneInfo(DISPLAY_TIMEZONE)
        except Exception:
            pass
    return datetime.now().astimezone().tzinfo


def _parse_timestamp(value):
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_timestamp(value):
    parsed = _parse_timestamp(value)
    if parsed is None:
        return str(value or "")
    local_dt = parsed.astimezone(_display_tz())
    hour = local_dt.strftime("%I").lstrip("0") or "0"
    return "{month} {day}, {hour}:{minute} {ampm} {tz}".format(
        month=local_dt.strftime("%b"),
        day=local_dt.day,
        hour=hour,
        minute=local_dt.strftime("%M"),
        ampm=local_dt.strftime("%p"),
        tz=local_dt.strftime("%Z"),
    )


def _latest_activity_timestamp(session, messages=None):
    candidates = [
        session.get("updated_at"),
        session.get("created_at"),
        (session.get("processing_job") or {}).get("updated_at"),
    ]
    candidates.extend(message.get("created_at") for message in messages or [])

    latest = None
    latest_raw = None
    for candidate in candidates:
        parsed = _parse_timestamp(candidate)
        if parsed is None:
            continue
        if latest is None or parsed > latest:
            latest = parsed
            latest_raw = candidate
    return latest_raw or session.get("updated_at") or ""


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
            background:
                linear-gradient(160deg, rgba(14, 26, 22, 0.96), rgba(22, 62, 47, 0.92));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            box-shadow: 0 20px 52px rgba(14, 26, 22, 0.18);
            color: var(--ink);
            min-width: 285px;
            padding: 0.82rem;
            text-align: left;
            white-space: normal;
        }
        .hero-copy-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.48rem;
            margin-top: 0.84rem;
        }
        .hero-pill {
            background: rgba(255, 250, 241, 0.72);
            border: 1px solid rgba(36, 54, 43, 0.11);
            border-radius: 999px;
            color: #304237;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.28rem 0.62rem;
        }
        .hero-visual {
            display: grid;
            gap: 0.64rem;
        }
        .hero-screen {
            background:
                radial-gradient(circle at 18% 22%, rgba(247, 188, 92, 0.35), transparent 5.5rem),
                linear-gradient(135deg, #172a23, #07100d);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            min-height: 142px;
            overflow: hidden;
            padding: 0.72rem;
            position: relative;
        }
        .hero-screen::before {
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.055) 1px, transparent 1px);
            background-size: 18px 18px;
            content: "";
            inset: 0;
            opacity: 0.7;
            position: absolute;
        }
        .hero-screen::after {
            border: 1px solid rgba(243, 197, 118, 0.55);
            border-radius: 20px;
            content: "";
            height: 42px;
            left: 34%;
            position: absolute;
            top: 38%;
            width: 78px;
        }
        .scan-line {
            background: linear-gradient(90deg, transparent, rgba(243, 197, 118, 0.9), transparent);
            height: 2px;
            left: 0.75rem;
            position: absolute;
            right: 0.75rem;
            top: 52%;
        }
        .hero-screen-label {
            background: rgba(255, 250, 241, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            color: #fff4d8;
            display: inline-flex;
            font-size: 0.68rem;
            font-weight: 820;
            letter-spacing: 0.06em;
            padding: 0.25rem 0.52rem;
            position: relative;
            text-transform: uppercase;
            z-index: 1;
        }
        .hero-answer {
            background: rgba(255, 250, 241, 0.94);
            border-radius: 16px;
            color: #1d2b24;
            font-size: 0.76rem;
            font-weight: 720;
            line-height: 1.36;
            padding: 0.64rem 0.72rem;
        }
        .hero-answer span {
            color: var(--green);
            display: block;
            font-size: 0.62rem;
            font-weight: 820;
            letter-spacing: 0.07em;
            margin-bottom: 0.16rem;
            text-transform: uppercase;
        }
        .hero-answer-meta {
            color: var(--muted);
            display: block;
            font-size: 0.68rem;
            font-weight: 680;
            margin-top: 0.32rem;
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
        .studio-panel {
            background:
                radial-gradient(circle at 0% 0%, rgba(247, 188, 92, 0.24), transparent 17rem),
                linear-gradient(160deg, rgba(255, 252, 246, 0.92), rgba(237, 247, 239, 0.78));
            min-height: 610px;
        }
        .studio-heading {
            align-items: start;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 0.9rem;
        }
        .studio-title {
            color: var(--ink);
            font-family: "Fraunces", Georgia, serif;
            font-size: clamp(1.55rem, 2.4vw, 2.25rem);
            font-weight: 760;
            letter-spacing: -0.045em;
            line-height: 1;
            margin-bottom: 0.34rem;
        }
        .studio-badge {
            background: #123d2f;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 999px;
            color: #fff4d8;
            flex: 0 0 auto;
            font-size: 0.7rem;
            font-weight: 820;
            letter-spacing: 0.07em;
            padding: 0.34rem 0.72rem;
            text-transform: uppercase;
        }
        .upload-hero-card {
            background:
                linear-gradient(135deg, rgba(18, 61, 47, 0.96), rgba(22, 111, 80, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 24px;
            color: #fff8e9;
            margin: 0.92rem 0 1rem;
            overflow: hidden;
            padding: 1rem;
            position: relative;
        }
        .upload-hero-card::after {
            background:
                radial-gradient(circle, rgba(247, 188, 92, 0.46), transparent 7rem);
            content: "";
            height: 12rem;
            position: absolute;
            right: -4rem;
            top: -4.5rem;
            width: 12rem;
        }
        .upload-hero-top {
            align-items: center;
            display: flex;
            justify-content: space-between;
            position: relative;
            z-index: 1;
        }
        .upload-hero-title {
            font-size: 0.92rem;
            font-weight: 820;
        }
        .upload-hero-copy {
            color: rgba(255, 248, 233, 0.72);
            font-size: 0.78rem;
            line-height: 1.45;
            margin-top: 0.38rem;
            max-width: 480px;
            position: relative;
            z-index: 1;
        }
        .upload-file-types {
            display: flex;
            flex-wrap: wrap;
            gap: 0.36rem;
            margin-top: 0.8rem;
            position: relative;
            z-index: 1;
        }
        .file-chip {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 999px;
            color: #fff8e9;
            font-size: 0.68rem;
            font-weight: 760;
            padding: 0.24rem 0.5rem;
            text-transform: uppercase;
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
        .console-panel {
            background:
                linear-gradient(180deg, rgba(14, 26, 22, 0.96), rgba(17, 42, 33, 0.94));
            border-color: rgba(255, 255, 255, 0.12);
            color: #edf7f0;
            min-height: 610px;
        }
        .console-panel .section-title,
        .console-panel .flow-title {
            color: #fff8e9;
        }
        .console-panel .panel-copy,
        .console-panel .metric-label,
        .console-panel .flow-copy,
        .console-panel .detail-label,
        .console-panel .detail-value {
            color: #9eb5aa;
        }
        .console-topline {
            align-items: center;
            display: flex;
            gap: 0.45rem;
            justify-content: space-between;
            margin-bottom: 0.85rem;
        }
        .console-lights {
            display: flex;
            gap: 0.3rem;
        }
        .console-light {
            border-radius: 999px;
            height: 0.58rem;
            width: 0.58rem;
        }
        .console-light:nth-child(1) {
            background: #f3c576;
        }
        .console-light:nth-child(2) {
            background: #64b98a;
        }
        .console-light:nth-child(3) {
            background: #7ca6b8;
        }
        .console-chip {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 999px;
            color: #fff4d8;
            font-size: 0.68rem;
            font-weight: 760;
            padding: 0.24rem 0.52rem;
        }
        .console-preview {
            background:
                radial-gradient(circle at 18% 28%, rgba(247, 188, 92, 0.18), transparent 7rem),
                linear-gradient(135deg, #0a120f, #152b23);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 22px;
            min-height: 168px;
            overflow: hidden;
            padding: 0.86rem;
            position: relative;
        }
        .console-preview::before {
            background-image:
                linear-gradient(rgba(255, 255, 255, 0.045) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.045) 1px, transparent 1px);
            background-size: 22px 22px;
            content: "";
            inset: 0;
            position: absolute;
        }
        .object-box {
            border: 1px solid rgba(243, 197, 118, 0.72);
            border-radius: 18px;
            color: #fff4d8;
            font-size: 0.65rem;
            font-weight: 820;
            letter-spacing: 0.05em;
            padding: 0.34rem;
            position: absolute;
            text-transform: uppercase;
        }
        .object-person {
            height: 68px;
            left: 17%;
            top: 32%;
            width: 54px;
        }
        .object-bag {
            border-color: rgba(126, 189, 150, 0.74);
            height: 42px;
            right: 18%;
            top: 48%;
            width: 72px;
        }
        .console-caption {
            background: rgba(255, 250, 241, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            bottom: 0.86rem;
            color: #fff8e9;
            font-size: 0.72rem;
            font-weight: 760;
            left: 0.86rem;
            padding: 0.48rem 0.62rem;
            position: absolute;
        }
        .console-live-title {
            color: #fff8e9;
            font-size: 0.9rem;
            font-weight: 820;
            line-height: 1.22;
            max-width: 82%;
            position: relative;
            z-index: 1;
        }
        .console-live-meta {
            color: #9eb5aa;
            font-size: 0.72rem;
            line-height: 1.38;
            margin-top: 0.38rem;
            max-width: 78%;
            position: relative;
            z-index: 1;
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
        .console-panel .metric {
            background: rgba(255, 255, 255, 0.07);
            border-color: rgba(255, 255, 255, 0.1);
        }
        .console-panel .metric-value {
            color: #fff8e9;
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
                min-width: 0;
            }
            .upload-grid {
                grid-template-columns: 1fr;
            }
            .prompt-grid {
                grid-template-columns: 1fr;
            }
            .studio-heading {
                display: block;
            }
            .studio-badge {
                display: inline-flex;
                margin-top: 0.6rem;
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
        .chat-panel-head {
            background:
                radial-gradient(circle at 8% 0%, rgba(247, 188, 92, 0.18), transparent 13rem),
                linear-gradient(135deg, rgba(255, 252, 246, 0.92), rgba(238, 247, 241, 0.84));
            border: 1px solid var(--line);
            border-radius: 26px;
            box-shadow: var(--shadow-soft);
            margin-bottom: 0.9rem;
            padding: 1rem 1.08rem 0.92rem;
        }
        .chat-header {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.48rem;
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
                radial-gradient(circle at 18% 12%, rgba(247, 188, 92, 0.2), transparent 11rem),
                linear-gradient(135deg, rgba(255, 252, 246, 0.9), rgba(232, 244, 236, 0.78));
            border: 1px solid rgba(36, 54, 43, 0.1);
            border-radius: 26px;
            box-shadow: var(--shadow-soft);
            color: var(--muted);
            margin: 0.2rem 0 1rem;
            padding: 1.1rem;
        }
        .ask-top-shell {
            background: rgba(255, 252, 246, 0.78);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: 0 10px 30px rgba(23, 36, 29, 0.055);
            margin: 0 0 0.95rem;
            padding: 0.74rem 0.8rem 0.18rem;
        }
        .ask-top-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin: 0 0 0.44rem 0.1rem;
            text-transform: uppercase;
        }
        .ask-top-shell div[data-testid="InputInstructions"] {
            display: none;
        }
        .ask-top-shell div[data-testid="stTextInput"] {
            margin-bottom: 0.72rem;
        }
        .ask-top-shell div[data-testid="stTextInput"] input {
            min-height: 3.1rem;
            padding-right: 1rem !important;
        }
        .empty-chat-kicker {
            color: var(--green);
            font-size: 0.68rem;
            font-weight: 820;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }
        .empty-chat-title {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 820;
            letter-spacing: -0.02em;
            line-height: 1.15;
            margin-bottom: 0.42rem;
        }
        .empty-chat-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 0.85rem;
        }
        .empty-chat-grid {
            display: grid;
            gap: 0.55rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .empty-chat-chip {
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(36, 54, 43, 0.1);
            border-radius: 16px;
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 680;
            line-height: 1.35;
            padding: 0.72rem 0.78rem;
        }
        @media (max-width: 900px) {
            .empty-chat-grid {
                grid-template-columns: 1fr;
            }
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
        .console-panel .flow-step {
            background: rgba(255, 255, 255, 0.065);
            border-color: rgba(255, 255, 255, 0.1);
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
        .landing-band {
            display: grid;
            gap: 0.8rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 1rem;
        }
        .landing-card {
            background: rgba(255, 252, 246, 0.78);
            backdrop-filter: blur(16px);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: 0 10px 32px rgba(23, 36, 29, 0.055);
            padding: 0.95rem;
        }
        .landing-card-title {
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 820;
            margin-bottom: 0.28rem;
        }
        .landing-card-copy {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.48;
        }
        @media (max-width: 900px) {
            .landing-band {
                grid-template-columns: 1fr;
            }
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


def _load_run(api_base, run_id):
    return _api_request("GET", api_base, "/developer/runs/{run_id}".format(run_id=run_id), timeout=(5, 60))


def _upload_video(
    api_base,
    uploaded_file,
    camera_id,
    label,
    pipeline_device,
    track_backend,
    object_backend,
    object_labels,
    summary_backend,
    llm_model,
):
    options = {
        "device": pipeline_device,
        "track_backend": track_backend,
        "object_backend": object_backend,
        "summary_backend": summary_backend,
    }
    object_labels = str(object_labels or "").strip()
    if object_labels:
        options["object_labels"] = object_labels
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


def _latest_assistant_message(messages):
    for message in reversed(messages or []):
        if message.get("role") == "assistant" and str(message.get("content") or "").strip():
            return str(message.get("content")).strip()
    return ""


def _select_preview_session(sessions, active_session_id=None):
    sessions = list(sessions or [])
    if active_session_id:
        for session in sessions:
            if session.get("session_id") == active_session_id:
                return session
    for session in sessions:
        if session.get("status") == "completed":
            return session
    return sessions[0] if sessions else None


def _backend_preview(api_base, active_session_id=None):
    fallback = {
        "connected": False,
        "status": "Backend offline",
        "chip": "backend unavailable",
        "title": "No live backend data yet",
        "source": "Status",
        "text": "Start the backend or upload a video to populate this preview with real session evidence.",
        "meta": "Waiting for /chat/sessions",
        "session_id": "none",
        "run_id": "pending",
        "tracks": "0",
        "windows": "0",
        "model": DEFAULT_PIPELINE_LLM_MODEL,
    }
    try:
        sessions = _list_sessions(api_base)
    except Exception as exc:
        fallback["text"] = str(exc)
        return fallback

    session = _select_preview_session(sessions, active_session_id=active_session_id)
    if not session:
        fallback.update(
            {
                "connected": True,
                "status": "No sessions",
                "chip": "backend connected",
                "title": "No processed video yet",
                "text": "Upload a video first. This panel will then show the latest real scene summary or answer.",
                "meta": "Backend connected",
            }
        )
        return fallback

    messages = []
    try:
        messages = _load_messages(api_base, session["session_id"])
    except Exception:
        messages = []

    run = None
    results = {}
    if session.get("run_id"):
        try:
            run = _load_run(api_base, session["run_id"])
            results = run.get("results", {}) if run else {}
        except Exception:
            results = {}

    stats = results.get("stats", {}) or {}
    config = results.get("config", {}) or {}
    latest_answer = _latest_assistant_message(messages)
    scene_summary = str(results.get("scene_summary") or "").strip()
    if latest_answer:
        source = "Latest answer"
        text = latest_answer
    elif scene_summary:
        source = "Scene summary"
        text = scene_summary
    elif session.get("status") in PROCESSING_STATUSES:
        source = "Processing"
        text = "The selected video is still processing. This preview will update after the pipeline finishes."
    else:
        source = "Session"
        text = "This session has no answer messages yet. Ask a question to generate a preview from backend evidence."

    label = session.get("label") or session.get("video_id") or session.get("session_id")
    updated = _format_timestamp(_latest_activity_timestamp(session, messages=messages))
    return {
        "connected": True,
        "status": str(session.get("status") or "unknown").replace("_", " "),
        "chip": "live backend",
        "title": _shorten(label, limit=44),
        "source": source,
        "text": _shorten(text, limit=190),
        "meta": "{updated} / {camera}".format(
            updated=updated or "no timestamp",
            camera=session.get("camera_id") or "camera",
        ),
        "session_id": session.get("session_id") or "none",
        "run_id": session.get("run_id") or "pending",
        "tracks": str(stats.get("identity_tracks") or stats.get("tracks") or 0),
        "windows": str(stats.get("vl_window_captions") or stats.get("window_summaries") or 0),
        "model": _shorten(config.get("llm_model") or DEFAULT_PIPELINE_LLM_MODEL, limit=28),
    }


def _render_topbar(preview):
    st.markdown(
        """
        <div class="topbar">
          <div>
            <div class="kicker">Eclipse Video Intelligence</div>
            <div class="title">Turn surveillance video into a searchable conversation.</div>
            <div class="muted">Upload a clip, wait for the pipeline to finish, then ask grounded questions about what happened.</div>
            <div class="hero-copy-row">
              <div class="hero-pill">{tracks} tracks</div>
              <div class="hero-pill">{windows} VLM windows</div>
              <div class="hero-pill">{status}</div>
            </div>
          </div>
          <div class="top-actions">
            <div class="hero-visual">
              <div class="hero-screen">
                <div class="hero-screen-label">{screen_label}</div>
                <div class="scan-line"></div>
              </div>
              <div class="hero-answer">
                <span>{source}</span>{text}
                <span class="hero-answer-meta">{title} / {meta}</span>
              </div>
            </div>
          </div>
        </div>
        """.format(
            tracks=_esc(preview.get("tracks", "0")),
            windows=_esc(preview.get("windows", "0")),
            status=_esc(preview.get("status", "unknown")),
            screen_label=_esc(preview.get("status", "unknown")),
            source=_esc(preview.get("source", "Preview")),
            text=_esc(preview.get("text", "")),
            title=_esc(preview.get("title", "")),
            meta=_esc(preview.get("meta", "")),
        ),
        unsafe_allow_html=True,
    )


def _history_title(session):
    title = session.get("label") or session.get("video_id") or session.get("session_id")
    return _shorten(title, limit=34)


def _history_meta(session):
    status = session.get("status", "unknown")
    camera = session.get("camera_id") or session.get("video_id") or "video"
    updated = _format_timestamp(session.get("updated_at"))
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


def _render_status(session, messages=None):
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
            updated=_esc(_format_timestamp(_latest_activity_timestamp(session, messages=messages))),
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
        st.markdown(
            """
            <div class="empty-chat">
              <div class="empty-chat-kicker">Evidence is ready</div>
              <div class="empty-chat-title">Ask this video like a chat.</div>
              <div class="empty-chat-copy">
                The backend has already processed the clip. Questions now retrieve the relevant 5-second VLM captions,
                track profiles, keyframes, and timeline evidence before Gemma4 answers.
              </div>
              <div class="empty-chat-grid">
                <div class="empty-chat-chip">What is the woman in the green shirt doing?</div>
                <div class="empty-chat-chip">Who is near the stacked white dishes?</div>
                <div class="empty-chat-chip">What happens around 30 seconds?</div>
                <div class="empty-chat-chip">Which people interact with Person 2?</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    for group in _message_groups_latest_first(messages):
        for message in group:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            with st.chat_message(role):
                st.markdown(content)


def _message_groups_latest_first(messages):
    groups = []
    current_group = []
    for message in messages or []:
        if message.get("role") == "user" and current_group:
            groups.append(current_group)
            current_group = [message]
        else:
            current_group.append(message)
    if current_group:
        groups.append(current_group)
    return list(reversed(groups))


def _submit_question(api_base, session_id, question, top_k, history_turns, answer_backend, answer_model, qa_device):
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


def _render_upload(api_base, pipeline_device, track_backend, object_backend, summary_backend, llm_model, preview):
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="panel studio-panel">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="studio-heading">
              <div>
                <div class="section-title">Upload Studio</div>
                <div class="studio-title">Start a grounded video chat.</div>
                <div class="panel-copy">Bring in any surveillance clip. Eclipse will process the full video first, then unlock a chat that answers from the session memory instead of guessing.</div>
              </div>
              <div class="studio-badge">Pipeline first</div>
            </div>
            <div class="upload-hero-card">
              <div class="upload-hero-top">
                <div class="upload-hero-title">Video intake</div>
                <div class="file-chip">session bound</div>
              </div>
              <div class="upload-hero-copy">Each upload creates a clean chat history, stores the source clip, runs detection and summarization, then connects your questions to the processed evidence.</div>
              <div class="upload-file-types">
                <div class="file-chip">mp4</div>
                <div class="file-chip">mov</div>
                <div class="file-chip">avi</div>
                <div class="file-chip">mkv</div>
                <div class="file-chip">webm</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-title">Clip details</div>',
            unsafe_allow_html=True,
        )
        with st.form("upload_form", clear_on_submit=False):
            uploaded_file = st.file_uploader("Video", type=VIDEO_TYPES)
            camera_id = st.text_input("Camera", value="camera_1")
            label = st.text_input("Label", value="")
            object_labels = st.text_area(
                "Objects to monitor",
                value=st.session_state.get("object_labels", ""),
                placeholder="hard hat, safety vest, forklift, pallet, box",
                help="Comma-separated scene-specific objects. Leave blank to use the selected environment preset.",
            )
            st.session_state["object_labels"] = object_labels
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
                        object_labels=object_labels,
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
            <div class="panel console-panel">
              <div class="console-topline">
                <div class="console-lights">
                  <div class="console-light"></div>
                  <div class="console-light"></div>
                  <div class="console-light"></div>
                </div>
                <div class="console-chip">{chip}</div>
              </div>
              <div class="section-title">Pipeline Console</div>
              <div class="panel-copy">Live backend preview from the latest available video session.</div>
              <div class="console-preview">
                <div class="console-live-title">{preview_title}</div>
                <div class="console-live-meta">{preview_meta}</div>
                <div class="console-caption">{preview_source}: {preview_text}</div>
              </div>
              <div class="metric-strip">
                <div class="metric">
                  <div class="metric-value">{tracks}</div>
                  <div class="metric-label">Tracks</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{windows}</div>
                  <div class="metric-label">VLM windows</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{session_id}</div>
                  <div class="metric-label">Session</div>
                </div>
                <div class="metric">
                  <div class="metric-value">{run_id}</div>
                  <div class="metric-label">Run</div>
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
                  <div class="detail-label">Next upload config</div>
                  <div class="detail-value">{pipeline_device} / people {track_backend} / objects {object_backend} / {summary_backend}</div>
                </div>
                <div class="detail-item">
                  <div class="detail-label">Model</div>
                  <div class="detail-value">{model}</div>
                </div>
              </div>
            </div>
            """.format(
                chip=_esc(preview.get("chip", "backend")),
                preview_title=_esc(preview.get("title", "No session")),
                preview_meta=_esc(preview.get("meta", "")),
                preview_source=_esc(preview.get("source", "Preview")),
                preview_text=_esc(preview.get("text", "")),
                tracks=_esc(preview.get("tracks", "0")),
                windows=_esc(preview.get("windows", "0")),
                session_id=_esc(_shorten(preview.get("session_id", "none"), limit=14)),
                run_id=_esc(_shorten(preview.get("run_id", "pending"), limit=14)),
                api_base=_esc(api_base),
                pipeline_device=_esc(pipeline_device),
                track_backend=_esc(track_backend),
                object_backend=_esc(object_backend),
                summary_backend=_esc(summary_backend),
                model=_esc(preview.get("model") or llm_model or DEFAULT_PIPELINE_LLM_MODEL),
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="landing-band">
          <div class="landing-card">
            <div class="landing-card-title">No premature answers</div>
            <div class="landing-card-copy">The chat stays locked while the backend is processing, so responses are based on completed pipeline artifacts.</div>
          </div>
          <div class="landing-card">
            <div class="landing-card-title">Session memory</div>
            <div class="landing-card-copy">Every video becomes a reusable chat session in the left sidebar, with its own history and indexed evidence.</div>
          </div>
          <div class="landing-card">
            <div class="landing-card-title">Evidence-first QA</div>
            <div class="landing-card-copy">Questions retrieve relevant events and summaries from the processed video before the answer model responds.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Video Chat", layout="wide")
    _inject_css()

    sidebar_defaults = {
        "api_base": DEFAULT_API_BASE,
        "pipeline_device": "cuda",
        "track_backend": DEFAULT_TRACK_BACKEND,
        "object_backend": DEFAULT_OBJECT_BACKEND,
        "object_labels": "",
        "summary_backend": "vl",
        "llm_model": DEFAULT_PIPELINE_LLM_MODEL,
        "answer_backend": "vl",
        "answer_model": DEFAULT_CHAT_ANSWER_MODEL,
        "qa_device": "cuda",
        "top_k": 4,
        "history_turns": 8,
    }
    for key, value in sidebar_defaults.items():
        st.session_state.setdefault(key, value)

    api_base = st.session_state["api_base"]
    active_session_id = _active_session_id()
    backend_preview = _backend_preview(api_base, active_session_id=active_session_id)
    _render_topbar(backend_preview)

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
        _render_upload(api_base, pipeline_device, track_backend, object_backend, summary_backend, llm_model, backend_preview)
        return

    try:
        session = _load_session(api_base, session_id)
    except Exception as exc:
        st.error(exc)
        if st.button("Start over", use_container_width=True):
            _clear_active_session()
            st.rerun()
        return

    messages = _load_messages(api_base, session_id)
    _render_status(session, messages=messages)

    status = session.get("status")
    left, right = st.columns([1.35, 1], gap="large")

    with right:
        _render_video(session)
        if session.get("error"):
            st.error(session["error"])

    with left:
        st.markdown(
            """
            <div class="chat-panel-head">
              <div class="chat-header">
                <div class="section-title">Chat</div>
                <div class="chat-status">{status}</div>
              </div>
              <div class="panel-copy">Ask after processing is complete. Answers are grounded in retrieved windows, track evidence, and sampled keyframes.</div>
            </div>
            """.format(status=_esc("Ready" if status == "completed" else status or "Unknown")),
            unsafe_allow_html=True,
        )
        if status in PROCESSING_STATUSES:
            _render_messages(messages)
            st.info("Processing video. Chat will unlock when the pipeline is complete.")
            time.sleep(2)
            st.rerun()
        elif status == "failed":
            _render_messages(messages)
            st.error("Pipeline failed. Start a new chat after checking the backend log.")
        elif status == "completed":
            st.markdown(
                '<div class="ask-top-shell"><div class="ask-top-label">Ask question</div>',
                unsafe_allow_html=True,
            )
            with st.form("ask_form_{session_id}".format(session_id=session_id), clear_on_submit=True):
                question = st.text_input(
                    "Question",
                    label_visibility="collapsed",
                    placeholder="Ask about this video, for example: what is the woman in green doing?",
                )
                submitted = st.form_submit_button("Ask", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if submitted and question.strip():
                _submit_question(
                    api_base=api_base,
                    session_id=session_id,
                    question=question.strip(),
                    top_k=top_k,
                    history_turns=history_turns,
                    answer_backend=answer_backend,
                    answer_model=answer_model,
                    qa_device=qa_device,
                )
            elif submitted:
                st.warning("Type a question first.")
            _render_messages(messages)
        else:
            _render_messages(messages)
            st.warning("Unknown session status: {status}".format(status=status))


if __name__ == "__main__":
    main()
