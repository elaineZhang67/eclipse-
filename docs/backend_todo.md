# Backend Redesign TODO

## Goal

Move the project from an offline script workflow into a persistent monitoring backend:

```text
Developer backend:
  register video -> process video as backend job -> persist timestamped memory

User backend:
  select video/run/timeframe -> ask question anytime -> retrieve timestamped context -> answer
```

The user should not need to manually run `main.py`, inspect JSON, then open QA. That workflow still exists for debugging, but the backend is now the primary system shape.

## Completed Tasks

- [x] Add video registry in SQLite so videos have stable `video_id`, `camera_id`, path, label, and metadata.
- [x] Add processing job table so backend runs can be queued, marked running, completed, or failed.
- [x] Link processed `runs` back to `video_id`.
- [x] Add developer backend routes for registering videos, launching processing jobs, inspecting jobs, and inspecting runs.
- [x] Add developer upload endpoint that stores videos in local backend storage and registers them.
- [x] Add user backend routes for listing videos/runs and asking questions.
- [x] Add timeframe-aware QA so a user can ask about `start_sec` to `end_sec` within a video.
- [x] Keep existing CLI pipeline available for direct experiments and ablations.
- [x] Add `.gitignore` rules for datasets, generated outputs, runtime DBs, duplicate repo copies, and large local media.

## Current Backend Shape

### Developer API

- `POST /developer/videos`
- `POST /developer/videos/upload`
- `GET /developer/videos`
- `POST /developer/process`
- `POST /developer/process-sync`
- `GET /developer/jobs`
- `GET /developer/jobs/{job_id}`
- `GET /developer/runs`
- `GET /developer/runs/{run_id}`

### User API

- `GET /user/videos`
- `GET /user/runs`
- `POST /user/ask`

The user QA request supports:

- `video_id`
- `run_id`
- `camera_id`
- `start_sec`
- `end_sec`
- `question`

## Remaining Research/Engineering TODOs

- [ ] Replace the current in-process FastAPI background task with a real worker queue for long videos.
- [ ] Add websocket or polling updates for live job progress.
- [ ] Add live camera ingestion that processes rolling chunks while preserving tracker and memory-bank state.
- [ ] Add vector embeddings for RAG documents instead of lexical retrieval only.
- [ ] Add a stronger learned person re-identification model to replace the lightweight handcrafted appearance embedding.
- [ ] Add interaction graph memory so multi-person interactions are represented as persistent group/edge states, not only pairwise proximity events.
- [ ] Add frontend controls for selecting `video_id`, timeframe, and backend job status.
- [ ] Add authentication and role separation before exposing this outside a trusted research environment.

## Run Backend

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Use Swagger docs at:

```text
http://localhost:8000/docs
```

## Example: Developer Registers And Processes A Video

```bash
curl -X POST http://localhost:8000/developer/videos \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test_video",
    "camera_id": "camera_test",
    "video_path": "test.mp4",
    "label": "test clip"
  }'
```

Or upload a local video file directly:

```bash
curl -X POST http://localhost:8000/developer/videos/upload \
  -F "file=@test.mp4" \
  -F "video_id=test_video" \
  -F "camera_id=camera_test" \
  -F "label=test clip"
```

```bash
curl -X POST http://localhost:8000/developer/process \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test_video",
    "options": {
      "device": "cuda",
      "environment": "generic",
      "use_track_memory": true,
      "use_llm": true,
      "summary_backend": "vl",
      "llm_model": "google/gemma-4-E4B-it"
    }
  }'
```

## Example: User Asks About A Timeframe

```bash
curl -X POST http://localhost:8000/user/ask \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test_video",
    "start_sec": 0,
    "end_sec": 10,
    "question": "What are the people wearing and doing?",
    "answer_backend": "text",
    "answer_model": "Qwen/Qwen2.5-14B-Instruct",
    "device": "cuda"
  }'
```
