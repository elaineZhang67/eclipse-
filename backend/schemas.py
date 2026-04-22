from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineOptions(BaseModel):
    yolo: str = "yolov8n.pt"
    tracker: str = "botsort.yaml"
    object_backend: str = "yolo"
    sam2_model: str = "facebook/sam2.1-hiera-large"
    sam2_mask_threshold: float = 0.5
    sam2_track_iou: float = 0.3
    sam2_track_ttl: int = 12
    sam3_model: str = "facebook/sam3"
    sam3_mask_threshold: float = 0.5
    sam3_track_iou: float = 0.3
    sam3_track_ttl: int = 12
    device: str = "auto"
    fps: float = 12.0
    clip_len: int = 16
    stride: int = 8
    clip_sec: float = 2.0
    stride_sec: float = 1.0
    min_conf: float = 0.25
    max_people: int = 20
    track_labels: str = "person"
    environment: str = "generic"
    max_object_types: int = 5
    object_labels: Optional[str] = None
    event_window_sec: float = 5.0
    long_summary_sec: float = 60.0
    interaction_combine_iou: float = 0.05
    interaction_combine_dist: float = 1.2
    interaction_nearby_dist: float = 2.5
    use_track_memory: bool = True
    appearance_match_threshold: float = 0.82
    appearance_memory_ttl_sec: float = 8.0
    appearance_reassoc_gap_sec: float = 20.0
    use_llm: bool = True
    summary_backend: str = "vl"
    llm_model: Optional[str] = "Qwen/Qwen3-VL-4B-Instruct"
    vl_max_track_images: int = 4
    vl_max_scene_images: int = 4
    vl_track_gap_sec: float = 1.5
    vl_scene_gap_sec: float = 3.0
    disable_progress: bool = True


class RegisterVideoRequest(BaseModel):
    video_path: str
    camera_id: str = "camera_1"
    video_id: Optional[str] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessVideoRequest(BaseModel):
    video_id: Optional[str] = None
    video_path: Optional[str] = None
    camera_id: Optional[str] = None
    run_id: Optional[str] = None
    job_id: Optional[str] = None
    options: PipelineOptions = Field(default_factory=PipelineOptions)


class AskRequest(BaseModel):
    question: str
    video_id: Optional[str] = None
    run_id: Optional[str] = None
    camera_id: Optional[str] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    session_id: Optional[str] = None
    top_k: int = 4
    history_turns: int = 8
    answer_backend: str = "text"
    answer_model: Optional[str] = None
    device: str = "auto"


class AskResponse(BaseModel):
    session_id: str
    answer: str
    resolved_question: str
    retrieved_context: List[Dict[str, Any]]
    run_ids: List[str]
    video_id: Optional[str] = None
    timeframe: Optional[Dict[str, Optional[float]]] = None
