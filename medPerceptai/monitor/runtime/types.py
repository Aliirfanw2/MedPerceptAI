from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FrameJob:
    frame_id: int
    frame: np.ndarray
    config: Dict[str, Any]
    timestamp: float


@dataclass
class CropJob:
    frame_id: int
    frame: np.ndarray
    crop: np.ndarray
    primary_box: Tuple[int, int, int, int]
    config: Dict[str, Any]
    detection_hint: str
    confidence: float
    detections: List[Dict[str, Any]]


@dataclass
class ObjectResult:
    frame_id: int
    detections: List[Dict[str, Any]] = field(default_factory=list)
    primary_box: Optional[Tuple[int, int, int, int]] = None
    crop: Optional[np.ndarray] = None
    detection_hint: str = "no patient detected"
    confidence: float = 0.0


@dataclass
class RoleResult:
    frame_id: int
    role_hint: str = "unknown"
    role_confidence: float = 0.0
    role_detections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PoseResult:
    frame_id: int
    pose_landmarks: Any = None
    pose_summary: Dict[str, Any] = field(default_factory=lambda: {"available": False})
    pose_detections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReasoningContext:
    frame_id: int
    frame: np.ndarray
    config: Dict[str, Any]
    scene: Dict[str, Any]
    object_result: ObjectResult
    role_result: RoleResult
    pose_result: PoseResult
    crop: np.ndarray
    detection_hint: str
    role_hint: str
    pose_landmarks: Any
    pose_summary: Dict[str, Any]
    primary_box: Optional[Tuple[int, int, int, int]]
    confidence: float


@dataclass
class ReasoningResult:
    frame_id: int
    intent: str
    alert_triggered: bool
    bbox: Optional[Dict[str, Any]]
    pose_summary: Dict[str, Any]
    model_status: Dict[str, str]
    use_reasoning: bool
    latency_ms: int
    role_hint: str = "unknown"
    detection_list: List[Dict[str, Any]] = field(default_factory=list)
    primary_box: Optional[Tuple[int, int, int, int]] = None
    pose_landmarks: Any = None
    confidence_scores: Dict[str, Any] = field(default_factory=dict)
    monitor_status: str = "idle"
    risk_score: Optional[float] = None
    risk_level: str = "not provided"
    reasoning_trace: List[Dict[str, str]] = field(default_factory=list)
