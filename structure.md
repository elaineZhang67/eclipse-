project/
│
├── backend/
│   ├── app.py                 # FastAPI app: developer + user backend
│   ├── developer_api.py       # register/upload/process videos, inspect jobs/runs
│   ├── user_api.py            # ask QA over a video/run/timeframe
│   ├── schemas.py             # API request/response models
│   ├── services.py            # pipeline job service and timeframe QA service
│   └── settings.py            # backend env/default paths
│
├── ablation/
│   ├── run_sam2_ablation.py   # YOLO proposals + SAM2 refinement
│   ├── run_sam3_ablation.py   # SAM3 object backend
│   └── run_gemma4_ablation.py # Gemma4 summarizer backend
│
├── detection/
│   ├── detector.py            # YOLO detector
│   ├── sam2_detector.py       # SAM2 proposal-refinement detector
│   └── sam3_detector.py       # SAM3 detector
│
├── tracking/
│   ├── tracker.py             # YOLO + BoT-SORT tracking
│   └── appearance_memory.py   # track identity memory bank
│
├── preprocessing/
│   ├── frame_sampler.py
│   └── clip_builder.py
│
├── video_encoder/
│   └── videomae_encoder.py    # action recognition
│
├── temporal/
│   ├── event_log.py           # timestamped 5s events
│   ├── interval_summary.py    # longer interval packets
│   ├── aggregator.py
│   └── segmenter.py
│
├── retrieval/
│   └── event_rag.py           # timestamped retrieval docs
│
├── memory_store/
│   └── sqlite_store.py        # videos, jobs, runs, chat messages
│
├── summarization/
│   ├── qwen_summary.py
│   ├── gemma4_summary.py
│   └── visual_evidence.py
│
├── docs/
│   ├── backend_todo.md
│   └── meeting_plan.md
│
├── chat_frontend.py           # Streamlit frontend over saved memory
├── chat.py                    # terminal QA over saved memory
├── pipeline.py                # core video pipeline
└── main.py                    # direct CLI pipeline entrypoint

Generated/local-only paths should stay out of Git:

- backend_storage/
- outputs/
- memory_store/*.db
- local datasets
- local video files
- duplicate nested repo copies
