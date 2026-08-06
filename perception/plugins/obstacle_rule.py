#!/usr/bin/env python3
"""CPU-only rule-based obstacle-distance plugin for dry-run deployments.

Public protocol compatibility with ``plugins/obstacle.py``:

* MCP tool/prefix: ``obstacle``;
* actions: ``start``, ``stop``, ``info`` and ``config``;
* input: ``sensor_msgs/CompressedImage``;
* output topic: ``<input_topic>/obstacle``;
* output payload: ``{"pred_distance": <float metres>}``.

No model runtime is imported or initialized.  The default is a lightweight
OpenCV heuristic based on central-ROI edges, local contrast, connected
components and perspective position.  ``fixed_distance_m`` may still be set to
a number for a deterministic connectivity-only dry run.  The heuristic is only
a pipeline fallback and is not a replacement for metric monocular-depth
inference.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import re
import threading
import time
from typing import Any, Mapping, Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    _ROS2_AVAILABLE = True
    _ROS2_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:
    rclpy = None  # type: ignore[assignment]
    CompressedImage = None  # type: ignore[assignment,misc]
    String = None  # type: ignore[assignment,misc]
    _ROS2_AVAILABLE = False
    _ROS2_IMPORT_ERROR = exc

    class Node:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any):
            raise RuntimeError(
                "ROS2 packages are required to start obstacle_rule"
            ) from _ROS2_IMPORT_ERROR


log = logging.getLogger(__name__)

if _ROS2_AVAILABLE:
    _LOW_LAT_QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )
    _PUB_QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
else:
    _LOW_LAT_QOS = None
    _PUB_QOS = None


TOOLS = [
    {
        "name": "obstacle",
        "type": "processor",
        "multiInstance": True,
        "description": "CPU-only rule-based obstacle distance for dry runs",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 compressed-image topic (required for start)",
                },
            },
            "required": ["action"],
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["rule"], "scope": "shared"},
                "fixed_distance_m": {
                    "type": ["number", "null"],
                    "description": "Fixed dry-run result; null enables the image heuristic",
                    "scope": "instance",
                },
                "no_obstacle_distance": {"type": "number", "scope": "instance"},
                "near_distance_m": {"type": "number", "scope": "instance"},
                "far_distance_m": {"type": "number", "scope": "instance"},
                "processing_width": {"type": "integer", "scope": "shared"},
                "processing_height": {"type": "integer", "scope": "shared"},
                "center_width_ratio": {"type": "number", "scope": "instance"},
                "roi_top_ratio": {"type": "number", "scope": "instance"},
                "roi_bottom_ratio": {"type": "number", "scope": "instance"},
                "min_component_area_ratio": {"type": "number", "scope": "instance"},
                "contrast_threshold": {"type": "integer", "scope": "instance"},
                "score_threshold": {"type": "number", "scope": "instance"},
            },
            "required": ["provider"],
        },
        "topic_in": [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json", "desc": "obstacle distance result"}],
    }
]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class RuleBasedDistanceAdapter:
    """Decode one frame and return a deterministic or heuristic distance."""

    def __init__(self, cfg: Mapping[str, Any]):
        provider = str(cfg.get("provider", "rule")).strip().lower()
        if provider != "rule":
            raise ValueError("obstacleRule provider must be 'rule'")

        # Image-dependent inference is the normal rule-based mode. A numeric
        # value is an explicit opt-in test stub and intentionally returns the
        # same distance for every non-empty image.
        fixed = cfg.get("fixed_distance_m", None)
        self._fixed_distance = None if fixed is None or fixed == "" else float(fixed)
        if self._fixed_distance is not None and self._fixed_distance < 0.0:
            raise ValueError("fixed_distance_m must be non-negative or null")

        self.fallback_distance = max(0.0, float(cfg.get("no_obstacle_distance", 10.0)))
        self._near_distance = max(0.05, float(cfg.get("near_distance_m", 0.5)))
        self._far_distance = max(
            self._near_distance, float(cfg.get("far_distance_m", 5.0))
        )
        self._width = int(_clamp(float(cfg.get("processing_width", 320)), 64, 1920))
        self._height = int(_clamp(float(cfg.get("processing_height", 240)), 64, 1080))
        self._center_width_ratio = _clamp(
            float(cfg.get("center_width_ratio", 1.0 / 3.0)), 0.1, 1.0
        )
        self._roi_top_ratio = _clamp(float(cfg.get("roi_top_ratio", 0.0)), 0.0, 0.9)
        self._roi_bottom_ratio = _clamp(
            float(cfg.get("roi_bottom_ratio", 0.625)),
            self._roi_top_ratio + 0.05,
            1.0,
        )
        self._min_area_ratio = _clamp(
            float(cfg.get("min_component_area_ratio", 0.002)), 0.0001, 0.25
        )
        self._contrast_threshold = int(
            _clamp(float(cfg.get("contrast_threshold", 18)), 1, 255)
        )
        self._score_threshold = _clamp(float(cfg.get("score_threshold", 0.18)), 0.0, 1.0)
        self.mode = "fixed" if self._fixed_distance is not None else "heuristic"

        log.info(
            "[obstacle-rule] initialized: mode=%s fixed_distance_m=%s model_loaded=false gpu_memory_mb=0",
            self.mode,
            self._fixed_distance,
        )

    @staticmethod
    def _decode(image_bytes: bytes) -> Any:
        import cv2
        import numpy as np

        if not image_bytes:
            raise ValueError("empty compressed image")
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("unable to decode compressed image")
        return frame

    def _heuristic_distance(self, frame: Any) -> tuple[float, float, int]:
        import cv2
        import numpy as np

        resized = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)

        median = float(np.median(smooth))
        canny_low = int(_clamp(0.66 * median, 10, 180))
        canny_high = int(_clamp(1.33 * median, canny_low + 1, 255))
        edges = cv2.Canny(smooth, canny_low, canny_high)

        background = cv2.GaussianBlur(smooth, (0, 0), 7.0)
        local_contrast = cv2.absdiff(smooth, background)
        _, contrast_mask = cv2.threshold(
            local_contrast, self._contrast_threshold, 255, cv2.THRESH_BINARY
        )
        mask = cv2.bitwise_or(edges, contrast_mask)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1
        )
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

        center = self._width / 2.0
        roi_width = max(1, int(self._width * self._center_width_ratio))
        x0 = max(0, int(center - roi_width / 2.0))
        x1 = min(self._width, x0 + roi_width)
        y0 = int(self._height * self._roi_top_ratio)
        y1 = max(y0 + 1, int(self._height * self._roi_bottom_ratio))
        roi = mask[y0:y1, x0:x1]

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)
        roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
        min_area = roi_area * self._min_area_ratio
        best_score = -1.0
        candidates = 0

        for index in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[index])
            if area < min_area or area > roi_area * 0.85 or width < 3 or height < 3:
                continue
            candidates += 1
            component_bottom = (y + height) / float(max(1, roi.shape[0]))
            area_closeness = _clamp(math.sqrt(area / (roi_area * 0.20)), 0.0, 1.0)
            cx = float(centroids[index][0])
            center_closeness = 1.0 - _clamp(
                abs(cx - roi.shape[1] / 2.0) / max(1.0, roi.shape[1] / 2.0), 0.0, 1.0
            )
            score = 0.55 * component_bottom + 0.30 * area_closeness + 0.15 * center_closeness
            best_score = max(best_score, score)

        if best_score < self._score_threshold:
            return self.fallback_distance, 0.0, candidates

        closeness = _clamp(best_score, 0.0, 1.0)
        if self._far_distance == self._near_distance:
            distance = self._near_distance
        else:
            # Log interpolation gives more resolution in the safety-critical near range.
            distance = math.exp(
                math.log(self._far_distance)
                + closeness * (math.log(self._near_distance) - math.log(self._far_distance))
            )
        return float(distance), closeness, candidates

    def estimate(self, image_bytes: bytes) -> dict:
        started = time.perf_counter()
        if not image_bytes:
            raise ValueError("empty compressed image")
        if self._fixed_distance is not None:
            distance, confidence, candidates = self._fixed_distance, 1.0, 0
        else:
            frame = self._decode(image_bytes)
            distance, confidence, candidates = self._heuristic_distance(frame)
        return {
            "pred_distance": float(distance),
            "confidence": float(confidence),
            "mode": self.mode,
            "candidate_count": int(candidates),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


class _ObstacleRuleNode(Node):
    def __init__(self, input_topic: str, adapter: RuleBasedDistanceAdapter, node_suffix: str):
        super().__init__(f"obstacle_rule_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/obstacle"
        self._adapter = adapter
        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._detect_count = 0
        self._failure_count = 0
        self._dropped_count = 0
        self._last_latency_ms = 0.0
        self.state = "idle"

    def start(self) -> dict:
        if self._sub is not None:
            self.state = "running"
            return {"state": self.state, "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"obstacle_rule_worker_{self._input_topic}",
        )
        self._worker.start()
        self.state = "running"
        log.info("[obstacle-rule] started: %s -> %s", self._input_topic, self._output_topic)
        return {"state": self.state, "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        self.state = "idle"
        log.info("[obstacle-rule] stopped: %s", self._input_topic)
        return {"state": self.state, "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage) -> None:
        frame = bytes(msg.data)
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
                self._dropped_count += 1
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                self._dropped_count += 1

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                image_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                result = self._adapter.estimate(image_bytes)
                self._last_latency_ms = float(result.get("latency_ms", 0.0))
            except Exception as exc:
                self._failure_count += 1
                log.error("[obstacle-rule] processing error: %s", exc, exc_info=True)
                result = {"pred_distance": self._adapter.fallback_distance}
            self._publish_result(result)

    def _publish_result(self, result: Mapping[str, Any]) -> None:
        self._detect_count += 1
        distance = float(result.get("pred_distance", self._adapter.fallback_distance))
        if not math.isfinite(distance):
            self._failure_count += 1
            distance = self._adapter.fallback_distance
        msg = String()
        msg.data = json.dumps({"pred_distance": distance}, ensure_ascii=False)
        self._pub.publish(msg)


class ObstacleRuleDistancePlugin:
    """CPU-only rule-based implementation exposed as ``obstacle``."""

    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        if not _ROS2_AVAILABLE:
            raise RuntimeError("obstacle_rule requires rclpy, sensor_msgs and std_msgs") from _ROS2_IMPORT_ERROR
        self._executor = executor
        self._base_cfg = dict(plugin_cfg)
        self._base_cfg.setdefault("provider", "rule")
        self._adapter = RuleBasedDistanceAdapter(self._base_cfg)
        self._nodes: dict[str, _ObstacleRuleNode] = {}
        self._instance_configs: dict[str, dict] = {}

    def get_tools(self) -> list:
        return TOOLS

    def _adapter_for(self, node_key: str) -> RuleBasedDistanceAdapter:
        merged = dict(self._base_cfg)
        merged.update(self._instance_configs.get(node_key, {}))
        return RuleBasedDistanceAdapter(merged)

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            instances = {
                key: {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "detect_count": node._detect_count,
                    "failure_count": node._failure_count,
                    "dropped_count": node._dropped_count,
                    "last_latency_ms": node._last_latency_ms,
                }
                for key, node in self._nodes.items()
            }
            input_topic = args.get("input_topic", "")
            topics_list = args.get("input_topics") or []
            if not input_topic and topics_list:
                input_topic = topics_list[0]
            if instance_id and instance_id in self._nodes:
                input_topic = self._nodes[instance_id]._input_topic
            elif not input_topic and self._nodes:
                input_topic = next(iter(self._nodes.values()))._input_topic
            return {
                "name": "ObstacleDistance",
                "manufacture": "Embodied",
                "model": "rule-based-dry-run",
                "state": "running" if instances else "idle",
                "instances": instances,
                "topic_in": ([{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []),
                "topic_out": (
                    [{"topic": f"{input_topic}/obstacle", "format": "data/json"}]
                    if input_topic
                    else []
                ),
                "desc": "CPU-only rule-based obstacle distance dry run",
            }

        if action == "start":
            input_topic = args.get("input_topic")
            topics_list = args.get("input_topics") or []
            if not input_topic and topics_list:
                input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                suffix = re.sub(r"[^a-zA-Z0-9_]", "_", node_key).strip("_") or "default"
                node = _ObstacleRuleNode(input_topic, self._adapter_for(node_key), suffix)
                self._executor.add_node(node)
                self._nodes[node_key] = node
                node.start()
            return self._nodes[node_key].start()

        if action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes.pop(instance_id)
                result = node.stop()
                self._executor.remove_node(node)
                return result
            if not instance_id and self._nodes:
                stopped = []
                for key in list(self._nodes):
                    node = self._nodes.pop(key)
                    node.stop()
                    self._executor.remove_node(node)
                    stopped.append(key)
                return {"state": "idle", "stopped_instances": stopped}
            return {"state": "idle"}

        if action == "config":
            cfg = {
                key: value
                for key, value in args.items()
                if key not in {"action", "instance_id"} and value != ""
            }
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes.pop(instance_id)
                    node.stop()
                    self._executor.remove_node(node)
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            self._base_cfg.update(cfg)
            self._adapter = RuleBasedDistanceAdapter(self._base_cfg)
            return {"status": "configured", "config": cfg}

        return None

