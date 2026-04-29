#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${DEMO_REPO_ROOT:-$SCRIPT_DIR}"
COMMAND="${1:-start}"

source_env_file() {
    local env_file="$1"
    if [[ ! -f "$env_file" ]]; then
        return
    fi
    set -a
    # shellcheck source=/dev/null
    source "$env_file"
    set +a
}

source_env_file "$REPO_ROOT/.env.local"

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
STREAMLIT_MAX_UPLOAD_MB="${STREAMLIT_MAX_UPLOAD_MB:-4096}"

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-eclipse}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"
DEMO_SKIP_CONDA="${DEMO_SKIP_CONDA:-}"

choose_demo_root() {
    if [[ -n "${DEMO_ROOT:-}" ]]; then
        printf '%s\n' "$DEMO_ROOT"
        return
    fi
    if [[ -d "/mnt/data/$USER" && -w "/mnt/data/$USER" ]]; then
        printf '/mnt/data/%s/demo_root\n' "$USER"
        return
    fi
    printf '%s/.demo\n' "$REPO_ROOT"
}

DEMO_ROOT="$(choose_demo_root)"
RUN_DIR="$DEMO_ROOT/run"
LOG_DIR="$DEMO_ROOT/logs"
CACHE_ROOT="$DEMO_ROOT/cache"
HOME_DIR="$DEMO_ROOT/home"
TMP_FALLBACK="$DEMO_ROOT/tmp"
MEMORY_DB="$DEMO_ROOT/memory_store/surveillance_memory.db"
VIDEO_STORAGE_DIR="$DEMO_ROOT/backend_storage/videos"

if [[ -n "${TMPDIR:-}" ]]; then
    APP_TMPDIR="$TMPDIR"
elif [[ -d /dev/shm && -w /dev/shm ]]; then
    APP_TMPDIR="/dev/shm"
else
    APP_TMPDIR="$TMP_FALLBACK"
fi

mkdir -p \
    "$RUN_DIR" \
    "$LOG_DIR" \
    "$CACHE_ROOT" \
    "$HOME_DIR" \
    "$TMP_FALLBACK" \
    "$DEMO_ROOT/memory_store" \
    "$VIDEO_STORAGE_DIR"

source_env_file "$DEMO_ROOT/.env.local"
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

build_pythonpath() {
    local joined=""
    local extra
    local -a parts

    parts=("$REPO_ROOT")
    if [[ -n "$EXTRA_PYTHONPATH" ]]; then
        IFS=':' read -r -a extra_parts <<< "$EXTRA_PYTHONPATH"
        parts=("${extra_parts[@]}" "${parts[@]}")
    fi
    if [[ -d "/mnt/data/$USER/eclipse_pylibs" ]]; then
        parts=("/mnt/data/$USER/eclipse_pylibs" "${parts[@]}")
    fi
    if [[ -n "${PYTHONPATH:-}" ]]; then
        IFS=':' read -r -a current_parts <<< "$PYTHONPATH"
        parts=("${parts[@]}" "${current_parts[@]}")
    fi

    for extra in "${parts[@]}"; do
        [[ -z "$extra" ]] && continue
        if [[ -z "$joined" ]]; then
            joined="$extra"
        else
            joined="$joined:$extra"
        fi
    done
    printf '%s\n' "$joined"
}

APP_PYTHONPATH="$(build_pythonpath)"

activate_env() {
    if [[ -n "$DEMO_SKIP_CONDA" ]]; then
        return
    fi
    if [[ ! -f "$CONDA_SH" ]]; then
        return
    fi
    # shellcheck source=/dev/null
    source "$CONDA_SH"
    if ! conda activate "$CONDA_ENV_NAME" >/dev/null 2>&1; then
        printf 'warning: could not activate conda env %s, using current shell Python\n' "$CONDA_ENV_NAME" >&2
    fi
}

activate_env
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

usage() {
    printf '%s\n' \
        "Usage: ./run_demo.sh [start|stop|restart|status]" \
        "" \
        "Environment overrides:" \
        "  DEMO_REPO_ROOT        Repo root to run against when the script lives elsewhere." \
        "  DEMO_ROOT             Demo workspace root. Default: /mnt/data/\$USER/demo_root when available." \
        "  BACKEND_PORT          FastAPI port. Default: 8000" \
        "  FRONTEND_PORT         Streamlit port. Default: 8501" \
        "  CONDA_ENV_NAME        Conda env to activate. Default: eclipse" \
        "  EXTRA_PYTHONPATH      Extra import paths to prepend." \
        "  PYTHON_BIN            Python executable to use after activation." \
        "" \
        "Local secrets:" \
        "  .env.local files in the repo root or DEMO_ROOT are sourced if present."
}

pid_path() {
    local name="$1"
    printf '%s/%s.pid\n' "$RUN_DIR" "$name"
}

is_running() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
    local path="$1"
    if [[ -f "$path" ]]; then
        tr -d '[:space:]' < "$path"
    fi
}

start_backend() {
    local pid_file
    local pid
    pid_file="$(pid_path backend)"
    pid="$(read_pid "$pid_file")"
    if is_running "$pid"; then
        printf 'backend already running (pid %s)\n' "$pid"
        return
    fi

    (
        cd "$REPO_ROOT"
        export PYTHONPATH="$APP_PYTHONPATH"
        export TMPDIR="$APP_TMPDIR"
        export HOME="$HOME_DIR"
        export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
        export HF_HOME="$CACHE_ROOT/hf"
        export MPLCONFIGDIR="$CACHE_ROOT/mpl"
        export SURVEILLANCE_MEMORY_DB="$MEMORY_DB"
        export SURVEILLANCE_VIDEO_STORAGE_DIR="$VIDEO_STORAGE_DIR"
        export SURVEILLANCE_WORKER_LOG_DIR="$LOG_DIR/workers"
        nohup "$PYTHON_BIN" -m uvicorn backend.app:app \
            --host "$BACKEND_HOST" \
            --port "$BACKEND_PORT" \
            >> "$LOG_DIR/backend.log" 2>&1 &
        echo "$!" > "$pid_file"
    )
}

start_frontend() {
    local pid_file
    local pid
    pid_file="$(pid_path frontend)"
    pid="$(read_pid "$pid_file")"
    if is_running "$pid"; then
        printf 'frontend already running (pid %s)\n' "$pid"
        return
    fi

    (
        cd "$DEMO_ROOT"
        export PYTHONPATH="$APP_PYTHONPATH"
        export TMPDIR="$APP_TMPDIR"
        export HOME="$HOME_DIR"
        export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
        export HF_HOME="$CACHE_ROOT/hf"
        export MPLCONFIGDIR="$CACHE_ROOT/mpl"
        export SURVEILLANCE_API_BASE="http://127.0.0.1:${BACKEND_PORT}"
        nohup "$PYTHON_BIN" -m streamlit run "$REPO_ROOT/chat_frontend.py" \
            --global.developmentMode false \
            --server.address "$FRONTEND_HOST" \
            --server.port "$FRONTEND_PORT" \
            --server.headless true \
            --server.maxUploadSize "$STREAMLIT_MAX_UPLOAD_MB" \
            >> "$LOG_DIR/frontend.log" 2>&1 &
        echo "$!" > "$pid_file"
    )
}

wait_for_url() {
    local label="$1"
    local url="$2"
    local attempt

    for attempt in $(seq 1 30); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            printf '%s is ready: %s\n' "$label" "$url"
            return 0
        fi
        sleep 1
    done

    printf '%s did not become ready in time. Check %s\n' "$label" "$LOG_DIR" >&2
    return 1
}

stop_service() {
    local name="$1"
    local pid_file
    local pid
    pid_file="$(pid_path "$name")"
    pid="$(read_pid "$pid_file")"

    if ! is_running "$pid"; then
        rm -f "$pid_file"
        printf '%s is not running\n' "$name"
        return
    fi

    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        if ! is_running "$pid"; then
            break
        fi
        sleep 1
    done
    if is_running "$pid"; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
    printf '%s stopped\n' "$name"
}

status_service() {
    local name="$1"
    local pid_file
    local pid
    pid_file="$(pid_path "$name")"
    pid="$(read_pid "$pid_file")"
    if is_running "$pid"; then
        printf '%s: running (pid %s)\n' "$name" "$pid"
    else
        printf '%s: stopped\n' "$name"
    fi
}

start_all() {
    start_backend
    start_frontend
    wait_for_url "backend" "http://127.0.0.1:${BACKEND_PORT}/health"
    wait_for_url "frontend" "http://127.0.0.1:${FRONTEND_PORT}/"
    printf '%s\n' \
        "demo root: $DEMO_ROOT" \
        "memory db: $MEMORY_DB" \
        "backend docs: http://127.0.0.1:${BACKEND_PORT}/docs" \
        "frontend: http://127.0.0.1:${FRONTEND_PORT}" \
        "logs:" \
        "  $LOG_DIR/backend.log" \
        "  $LOG_DIR/frontend.log"
}

case "$COMMAND" in
    start)
        start_all
        ;;
    stop)
        stop_service frontend
        stop_service backend
        ;;
    restart)
        stop_service frontend
        stop_service backend
        start_all
        ;;
    status)
        status_service backend
        status_service frontend
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
