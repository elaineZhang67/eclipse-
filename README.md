# Eclipse Video Chat

Eclipse is a surveillance video question-answering system. It lets a user upload a video, waits for the backend to process the video into structured memory, and then opens a ChatGPT-style interface for grounded questions about people, actions, objects, and time ranges.

The current system is **pipeline-first**, not real-time: a full video is processed before chat answers are enabled. The design is intentionally organized around 5-second windows, so it can be extended to near-real-time monitoring by committing completed windows incrementally.

## What This Project Does

- Uploads a video through a Streamlit frontend.
- Creates a persistent chat session for each video.
- Runs a background FastAPI worker that processes the video.
- Tracks people and objects over time.
- Builds 5-second event windows with timestamps, active tracks, object associations, and interactions.
- Uses Gemma4 as the visual-language model for track summaries, short window captions, full-video scene summaries, and visual QA.
- Stores processed runs, chat sessions, messages, and metadata in SQLite.
- Answers user questions using RAG over time-stamped memory plus sampled keyframes from retrieved windows.

## Current Final Pipeline

Default demo configuration:

- Person tracking: YOLO + BoT-SORT
- Object backend: SAM3
- Frame sampling: 4 FPS
- Event window size: 5 seconds
- Track memory: enabled
- VLM model: `google/gemma-4-E4B-it`
- VLM window captions: enabled
- Full-video scene summary: 32 sampled frames
- VideoMAE: disabled
- Long interval summaries: disabled by default
- QA backend: VLM answer generation with retrieved keyframes

## Repository Structure

```text
backend/                 FastAPI app, chat API, developer API, worker service
memory_store/            SQLite persistence for runs, videos, jobs, chat sessions, messages
retrieval/               RAG document construction, keyword/semantic retrieval, stitched timelines
summarization/           Gemma4/Qwen summarizers, visual evidence, QA keyframe sampling
detection/               YOLO, SAM2, SAM3 object/person detection helpers
tracking/                YOLO/BoT-SORT wrapper and appearance memory bank
temporal/                Event windows, interval summaries, temporal aggregation
preprocessing/           Frame sampler and clip helpers
ablation/                Scripts for SAM2, SAM3, and Gemma4 ablations
chat_frontend.py         Streamlit ChatGPT-style frontend
main.py                  One-shot CLI pipeline entrypoint
pipeline.py              Main batch video processing pipeline
run_demo.sh              One-command backend + frontend launcher
requirements.txt         Python dependencies
```

## Setup

Python 3.10+ is recommended. GPU/CUDA is strongly recommended for Gemma4, SAM3, and faster demo runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some models are hosted on Hugging Face and may require access approval. Put local secrets in `.env.local`; do not commit this file.

```bash
cat > .env.local <<'EOF'
HF_TOKEN=your_huggingface_token_here
EOF
```

`run_demo.sh` sources `.env.local` from both the repo root and the demo root.

## Quick Start: Full Demo

Start or restart both services:

```bash
./run_demo.sh restart
```

Default URLs:

- Frontend: `http://127.0.0.1:8501`
- Backend health: `http://127.0.0.1:8000/health`
- Backend docs: `http://127.0.0.1:8000/docs`

Check status:

```bash
./run_demo.sh status
```

Stop services:

```bash
./run_demo.sh stop
```

The script creates a demo workspace. On the remote server it defaults to:

```text
/mnt/data/$USER/demo_root
```

Inside that workspace:

```text
memory_store/surveillance_memory.db    SQLite memory database
backend_storage/videos/                Uploaded videos
logs/backend.log                       Backend log
logs/frontend.log                      Frontend log
logs/workers/                          Per-session worker logs
cache/                                 HF and model caches
```

## Frontend Flow

1. Open the Streamlit app.
2. Upload a video.
3. Wait for the pipeline status to become `completed`.
4. Ask questions in the chat panel.
5. Reopen old chat sessions from the left sidebar.

The frontend intentionally blocks answers while processing is incomplete. This prevents premature answers that are not grounded in processed evidence.

## Backend APIs

Health:

```bash
curl http://127.0.0.1:8000/health
```

Chat upload:

```bash
curl -X POST http://127.0.0.1:8000/chat/upload \
  -F "file=@store-aisle-detection.mp4" \
  -F "camera_id=store_demo" \
  -F "label=store-aisle-detection"
```

List chat sessions:

```bash
curl http://127.0.0.1:8000/chat/sessions
```

Ask a completed session:

```bash
curl -X POST http://127.0.0.1:8000/chat/sessions/<session_id>/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the woman in the green shirt doing?",
    "top_k": 4,
    "history_turns": 8,
    "answer_backend": "vl",
    "answer_model": "google/gemma-4-E4B-it",
    "device": "cuda"
  }'
```

Developer endpoints:

```text
POST /developer/videos
POST /developer/videos/upload
POST /developer/process
POST /developer/process-sync
GET  /developer/jobs
GET  /developer/jobs/{job_id}
GET  /developer/runs
GET  /developer/runs/{run_id}
```

User QA endpoint for already-processed runs:

```text
POST /user/ask
```

## One-Shot CLI Pipeline

The CLI is useful for debugging without the web frontend.

```bash
mkdir -p outputs runtime

python main.py \
  --video store-aisle-detection.mp4 \
  --camera_id store_demo \
  --run_id local_debug_run \
  --use_llm \
  --summary_backend vl \
  --llm_model google/gemma-4-E4B-it \
  --track_backend yolo \
  --object_backend sam3 \
  --use_track_memory \
  --save_json outputs/store_debug.json \
  --memory_db runtime/surveillance_memory.db
```

Useful options:

```text
--event_window_sec 5
--fps 4
--vl_max_track_images 4
--vl_max_window_images 3
--scene_summary_video_frames 32
--no_vl_window_captions
--enable_interval_summaries
--track_backend yolo|sam3
--object_backend yolo|sam2|sam3
--device auto|cpu|cuda|mps
```

## Stored Outputs

Each processed run stores:

- `config`: pipeline configuration used for the run
- `stats`: sampled frames, video duration, number of tracks, windows, captions, etc.
- `tracks`: per-track metadata, profile, and summary
- `event_log`: 5-second temporal event windows
- `window_summaries`: structured summaries and Gemma4 action captions
- `scene_summary`: full-video Gemma4 summary
- `debug_evidence`: sampled annotated evidence frames
- `track_memory_bank`: identity memory records

Chat sessions store:

- `session_id`
- `video_id`
- `job_id`
- `run_id`
- status: `uploaded`, `queued`, `running`, `completed`, or `failed`
- raw video path
- chat messages with retrieval metadata

## Question Answering Design

When the user asks a question:

1. The backend checks that the session is completed.
2. It builds RAG documents from tracks, event windows, window captions, and summaries.
3. It retrieves relevant evidence using keyword overlap, optional embedding RAG, structural boosts, and time filtering.
4. It stitches adjacent 5-second windows for the same track when a question refers to a person over time.
5. It samples keyframes from retrieved windows.
6. Gemma4 answers using retrieved text context plus keyframes.

Embedding-based RAG is optional and can be enabled with:

```bash
export ECLIPSE_ENABLE_EMBEDDING_RAG=1
```

Answer style is user-facing. The model is instructed to say natural descriptions like:

```text
the woman in the green shirt
the person near the dishware
she / he / they
```

It should avoid exposing raw engineering IDs such as `Person 2` or `track 2`, unless the user explicitly asks for IDs.

## Ablations and Lessons Learned

Several ablations were tested to understand which components were useful.

### VideoMAE action classification

Result: removed from the final pipeline.

VideoMAE produced fixed action labels that were too rigid for open-world surveillance. In the store aisle demo, it could introduce incorrect labels such as unrelated activities. These labels polluted retrieval and QA, so the final pipeline uses Gemma4 5-second VLM captions instead.

### Long interval summaries

Result: disabled by default.

60-second interval summaries were too coarse for person-level QA. Short questions like "what is this person doing around 15 seconds?" need local evidence. The final pipeline uses 5-second windows and stitched timelines instead.

### Whole-video VLM summary

Result: kept only as scene-level memory.

Passing sampled frames from the whole video to Gemma4 works for global scene summaries, but it does not replace tracking, timestamps, or retrieval for specific person/action questions. It can also increase GPU memory pressure if too many frames are used.

### Qwen3-VL vs Gemma4

Result: Gemma4 is the final VLM.

Qwen3-VL was used as an earlier baseline. The final demo standardizes on Gemma4 for track profiles, window captions, scene summary, and visual QA.

### SAM2 / SAM3 / YOLO

Result: final reproducible default uses YOLO + BoT-SORT for people and SAM3 for objects.

SAM3 gives richer object grounding, but it is heavier and requires correct Hugging Face access. SAM3 person tracking is supported as an experimental option, while YOLO + BoT-SORT is currently the more stable demo default for people.

### Text-only QA vs VLM keyframe QA

Result: VLM keyframe QA is the final answer path.

Text-only QA is faster but cannot verify visual appearance, clothing, or layout. The final pipeline retrieves relevant windows, samples keyframes, and lets Gemma4 answer with visual grounding.

## What Worked

- Pipeline-first processing avoids ungrounded early answers.
- 5-second event windows preserve local temporal detail.
- Gemma4 window captions are more useful than fixed action classification labels.
- Track memory improves identity continuity across fragmented tracker IDs.
- Stitched timelines help answer questions about the same person over multiple windows.
- Retrieved keyframes improve visual specificity for QA.
- SQLite persistence makes runs, sessions, and chat history reproducible.
- The Streamlit UI gives a usable ChatGPT-style demo with old chat sessions.

## What Did Not Work Well

- VideoMAE labels were unreliable for open-world actions.
- Long interval summaries lost too much detail.
- Whole-video VLM prompting alone was not enough for person-specific QA.
- Too many VLM frames can trigger CUDA out-of-memory errors.
- SAM3 setup can fail if model access or processor files are unavailable.
- Track profiles can be imperfect when only a few crops are available.
- The current system is not real-time; it processes the whole video before answering.

## Future Work

The next step is near-real-time video chat.

Current:

```text
Upload full video -> process full pipeline -> ask questions
```

Future:

```text
Live video stream -> process every 5-second window -> update memory continuously -> ask about processed content so far
```

Needed changes:

- Refactor `run_pipeline()` into a stateful streaming pipeline.
- Commit each completed 5-second event window to SQLite immediately.
- Run Gemma4 window captioning asynchronously so VLM inference does not block tracking.
- Let QA retrieve only completed windows and answer "based on the video processed so far."
- Add frontend live status such as `processed up to 35s`.
- Use fast tracking continuously and run heavier SAM3/Gemma4 tasks selectively.

## Troubleshooting

### Hugging Face access errors

If SAM3 or Gemma4 fails to load, check:

- The model id is correct.
- Your Hugging Face account has access to the gated model.
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is set in `.env.local`.

### CUDA out of memory

Reduce visual input sizes:

```text
vl_max_track_images
vl_max_window_images
scene_summary_video_frames
SURVEILLANCE_QA_MAX_FRAMES
```

You can also switch to a smaller backend or CPU for debugging, but full VLM demo quality is best on GPU.

### Frontend cannot reach backend

Check:

```bash
curl http://127.0.0.1:8000/health
./run_demo.sh status
```

If the frontend runs on another host, set:

```bash
SURVEILLANCE_API_BASE=http://<backend-host>:8000
```

### Worker failed

Check worker logs in the demo root:

```text
logs/workers/
logs/backend.log
```

The frontend stores failed session status and the backend error message, so old failed sessions can be inspected from the sidebar.

## Reproducibility Checklist

- Code committed to GitHub.
- Demo starts with `./run_demo.sh restart`.
- Backend and frontend health checks pass.
- Uploaded videos are stored under the demo root.
- SQLite stores runs, sessions, messages, and retrieval metadata.
- Final pipeline config is saved inside each run's `config`.
- Ablation results and failed approaches are documented above.
