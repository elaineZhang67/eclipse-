project/
│
├── configs/
│   ├── model.yaml
│   ├── inference.yaml
│
├── detection/
│   └── yolo_detector.py
│
├── tracking/
│   └── tracker.py
│
├── preprocessing/
│   ├── frame_sampler.py
│   ├── clip_builder.py
│
├── video_encoder/
│   ├── encoder.py
│   ├── videomae.py
│   ├── internvideo.py
│
├── action_head/
│   └── classifier.py
│
├── temporal/
│   ├── aggregator.py
│   ├── segmenter.py
│
├── summarization/
│   └── llm_summary.py
│
├── pipeline.py
└── main.py
