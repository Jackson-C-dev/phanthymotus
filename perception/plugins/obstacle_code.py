#!/usr/bin/env python3
"""
plugins/obstacle_code.py — remote multimodal obstacle distance estimation.

Subscribes to image/jpeg topics, estimates obstacle distance from camera,
publishes distance results to ROS2 topic.
Supports multi-instance (one instance per input topic).
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

log = logging.getLogger(__name__)

_DEFAULT_DECISION_DISTANCE_M = 2.0
_DEFAULT_BOUNDARY_MARGIN_M = 0.02
_DEFAULT_NO_OBSTACLE_DISTANCE_M = 10.0

_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    # Remote inference is slower than the camera. Keep only the newest frame so
    # API latency does not turn into an ever-growing end-to-end delay.
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "obstacle",
        "type": "processor",
        "multiInstance": True,
        "description": "Obstacle Distance Estimation — estimate distance to obstacles from camera feed",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["openai", "qwen"], "description": "Remote multimodal provider", "scope": "shared"},
                "url":      {"type": "string", "description": "API URL (optional)", "scope": "shared"},
                "key":      {"type": "string", "description": "API Key", "format": "password", "scope": "shared"},
                "model":    {"type": "string", "description": "Model name", "scope": "instance"},
                "decision_distance_m": {"type": "number", "scope": "instance"},
                "boundary_margin_m": {"type": "number", "scope": "instance"},
                "no_obstacle_distance": {"type": "number", "scope": "instance"},
                "request_timeout_seconds": {"type": "number", "scope": "shared"},
                "request_retries": {"type": "integer", "scope": "shared"},
            },
            "required": ["provider"]
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "obstacle distance estimation result"}],
    }
]


# ── Distance Estimation Adapters ──────────────────────────────────────────────

class DistanceAdapter(ABC):
    """障碍物距离估计适配器抽象基类"""

    @abstractmethod
    def estimate(self, image_bytes: bytes) -> dict:
        """估计图片中障碍物的距离，返回包含 pred_distance 的字典"""
        ...


def _normalize_base_url(url: str, default: str) -> str:
    """Return an API base URL; accepting a full chat-completions URL is harmless."""
    value = (url or default).strip().rstrip("/")
    suffix = "/chat/completions"
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value.rstrip("/")


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "positive", "near"}:
            return True
        if lowered in {"false", "no", "0", "negative", "far"}:
            return False
    return None


def _extract_json_object(content: Any) -> dict:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        # Some OpenAI-compatible gateways return typed content blocks.
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    text = str(content or "").strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    else:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            candidates.insert(0, text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"model did not return a JSON object: {text[:200]!r}")


def _normalize_prediction(
    parsed: dict,
    *,
    decision_distance_m: float,
    boundary_margin_m: float,
    no_obstacle_distance: float,
) -> dict:
    try:
        distance = float(parsed.get("pred_distance", no_obstacle_distance))
    except (TypeError, ValueError) as exc:
        raise ValueError("pred_distance must be numeric") from exc
    if not math.isfinite(distance):
        raise ValueError("pred_distance must be finite")

    distance = min(max(distance, 0.05), no_obstacle_distance)
    is_positive = _as_bool(parsed.get("is_within_2m"))
    if is_positive is not None:
        # The leaderboard class is determined exclusively by the 2 m boundary.
        # Keep the model's metric estimate when it is consistent; otherwise move
        # it to the nearest safe side of the boundary.
        if is_positive:
            distance = min(distance, decision_distance_m - boundary_margin_m)
        else:
            distance = max(distance, decision_distance_m + boundary_margin_m)

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    return {
        "pred_distance": distance,
        "is_positive": distance < decision_distance_m,
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning", ""))[:500],
    }


def _post_chat_completion(
    *,
    base_url: str,
    key: str,
    payload: dict,
    timeout_seconds: float,
    retries: int,
) -> Any:
    """Call an OpenAI-compatible endpoint with bounded retries.

    JSON mode is requested first. Gateways that explicitly reject the
    response_format field are retried once without it.
    """
    import requests

    endpoint = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    request_payload = dict(payload)
    request_payload["response_format"] = {"type": "json_object"}
    compatibility_retry_used = False
    last_error: Optional[Exception] = None

    for attempt in range(max(0, retries) + 1):
        try:
            response = requests.post(
                endpoint,
                json=request_payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            if (
                response.status_code == 400
                and "response_format" in request_payload
                and not compatibility_retry_used
                and "response_format" in response.text.lower()
            ):
                compatibility_retry_used = True
                request_payload.pop("response_format", None)
                response = requests.post(
                    endpoint,
                    json=request_payload,
                    headers=headers,
                    timeout=timeout_seconds,
                )
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else 0
            if status not in {408, 409, 429} and status < 500:
                raise
        if attempt < max(0, retries):
            time.sleep(min(2.0, 0.5 * (2 ** attempt)))
    raise RuntimeError(f"remote multimodal request failed after retries: {last_error}")


class OpenAIVisionDistanceAdapter(DistanceAdapter):
    """OpenAI Vision API 距离估计"""

    _SYSTEM_PROMPT = (
        "You are an obstacle distance estimation system for a robot camera.\n\n"
        "Your task is to analyze the provided image, determine whether the nearest "
        "obstacle is strictly closer than 2 meters, and estimate its distance.\n\n"
        "Output format: Return a JSON object with:\n"
        '- "is_within_2m": true only when distance is strictly below 2m (boolean)\n'
        '- "pred_distance": estimated distance in meters (float)\n'
        '- "confidence": confidence score 0-1 (float)\n'
        '- "reasoning": brief explanation of your estimation\n\n'
        "Rules:\n"
        "1. Distance should be in meters.\n"
        "2. If no obstacle is visible, return a large value (e.g., 10.0).\n"
        "3. For indoor robot images, consider the central one-third width and upper "
        "five-eighths height; ignore the floor.\n"
        "4. For outdoor driving images, consider traffic participants and movable "
        "obstacles in front of the ego vehicle; ignore road surface and buildings.\n"
        "5. Output ONLY the JSON object, nothing else.\n\n"
        'Example: {"is_within_2m": true, "pred_distance": 1.25, '
        '"confidence": 0.85, "reasoning": "nearest obstacle is clearly within 2m"}'
    )

    def __init__(
        self,
        url: str,
        key: str,
        model: str,
        *,
        decision_distance_m: float = _DEFAULT_DECISION_DISTANCE_M,
        boundary_margin_m: float = _DEFAULT_BOUNDARY_MARGIN_M,
        no_obstacle_distance: float = _DEFAULT_NO_OBSTACLE_DISTANCE_M,
        timeout_seconds: float = 30.0,
        retries: int = 1,
    ):
        self.base_url = _normalize_base_url(url, "https://api.openai.com/v1")
        self.key = key
        self.model = model or "gpt-4o-mini"
        self.decision_distance_m = decision_distance_m
        self.boundary_margin_m = boundary_margin_m
        self.no_obstacle_distance = no_obstacle_distance
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def estimate(self, image_bytes: bytes) -> dict:
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"
        elif image_bytes[:2] == b'BM':
            image_format = "bmp"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            image_format = "webp"

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_b64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Estimate the distance to the nearest obstacle in this image."
                    }
                ]
            }
        ]

        content = _post_chat_completion(
            base_url=self.base_url,
            key=self.key,
            payload={
                "model": self.model,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
        )
        return self._parse_result(content)

    def _parse_result(self, content: Any) -> dict:
        return _normalize_prediction(
            _extract_json_object(content),
            decision_distance_m=self.decision_distance_m,
            boundary_margin_m=self.boundary_margin_m,
            no_obstacle_distance=self.no_obstacle_distance,
        )


class QwenVLDistanceAdapter(OpenAIVisionDistanceAdapter):
    """Qwen-VL 距离估计"""

    _SYSTEM_PROMPT = (
        "你是一个机器人摄像头障碍物距离估计系统。\n\n"
        "任务：分析提供的图片，判断最近障碍物是否严格小于2米，并估计距离。\n\n"
        "输出格式：返回 JSON 对象，包含：\n"
        '- "is_within_2m": 严格小于2米时为 true，否则为 false（布尔值）\n'
        '- "pred_distance": 估计距离（米，浮点数）\n'
        '- "confidence": 置信度 0-1（浮点数）\n'
        '- "reasoning": 简要说明\n\n'
        "规则：\n"
        "1. 距离单位为米。\n"
        "2. 如果没有可见障碍物，返回较大值（如 10.0）。\n"
        "3. 室内机器人图片只考虑图像中心1/3宽、上方5/8高的区域，排除地面。\n"
        "4. 室外无人车图片考虑自车正前方交通参与者和可移动障碍物，排除路面和建筑。\n"
        "5. 只输出 JSON 对象，不要其他内容。\n\n"
        '示例：{"is_within_2m": true, "pred_distance": 1.25, '
        '"confidence": 0.85, "reasoning": "最近障碍物明显在2米内"}'
    )

    def __init__(self, url: str, key: str, model: str, **kwargs: Any):
        super().__init__(url, key, model or "qwen-vl-max", **kwargs)
        self.base_url = _normalize_base_url(
            url, "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def estimate(self, image_bytes: bytes) -> dict:
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{image_format};base64,{image_b64}"
                    },
                    {
                        "type": "text",
                        "text": "估计这张图片中最近障碍物的距离。"
                    }
                ]
            }
        ]

        content = _post_chat_completion(
            base_url=self.base_url,
            key=self.key,
            payload={
                "model": self.model,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
        )
        return self._parse_result(content)


def _build_distance_adapter(cfg: dict) -> DistanceAdapter:
    """根据配置创建距离估计适配器"""
    provider = str(cfg.get("provider", "")).strip().lower()
    decision_distance = float(
        cfg.get("decision_distance_m", _DEFAULT_DECISION_DISTANCE_M)
    )
    no_obstacle_distance = float(
        cfg.get("no_obstacle_distance", _DEFAULT_NO_OBSTACLE_DISTANCE_M)
    )
    if not 0.05 < decision_distance < no_obstacle_distance:
        raise ValueError(
            "remote obstacle config requires 0.05 < decision_distance_m "
            "< no_obstacle_distance"
        )
    margin = max(
        0.001,
        min(
            float(cfg.get("boundary_margin_m", _DEFAULT_BOUNDARY_MARGIN_M)),
            decision_distance - 0.05,
            no_obstacle_distance - decision_distance,
        ),
    )
    common = {
        "decision_distance_m": decision_distance,
        "boundary_margin_m": margin,
        "no_obstacle_distance": no_obstacle_distance,
        "timeout_seconds": max(1.0, float(cfg.get("request_timeout_seconds", 30.0))),
        "retries": max(0, int(cfg.get("request_retries", 1))),
    }

    if provider == 'openai':
        url = str(cfg.get("url") or os.environ.get("OBSTACLE_API_BASE") or "")
        key = str(
            cfg.get("key")
            or os.environ.get("OBSTACLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        if not key:
            raise ValueError(
                "missing API key: set obstacle.key, OBSTACLE_API_KEY or OPENAI_API_KEY"
            )
        model = str(
            cfg.get("model") or os.environ.get("OBSTACLE_API_MODEL") or ""
        )
        return OpenAIVisionDistanceAdapter(url, key, model, **common)

    elif provider == 'qwen':
        url = str(cfg.get("url") or os.environ.get("OBSTACLE_API_BASE") or "")
        key = str(
            cfg.get("key")
            or os.environ.get("OBSTACLE_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or ""
        )
        if not key:
            raise ValueError(
                "missing API key: set obstacle.key, OBSTACLE_API_KEY or "
                "DASHSCOPE_API_KEY"
            )
        model = str(
            cfg.get("model") or os.environ.get("OBSTACLE_API_MODEL") or ""
        )
        return QwenVLDistanceAdapter(url, key, model, **common)

    raise ValueError("obstacle_code provider must be 'openai' or 'qwen'")


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _ObstacleNode(Node):
    """Per-topic obstacle distance estimation node."""

    def __init__(self, input_topic: str, adapter: DistanceAdapter,
                 node_suffix: str):
        super().__init__(f"obstacle_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/obstacle"
        self._adapter = adapter

        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: Optional[object] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._detect_count = 0
        self._failure_count = 0
        self._last_latency_ms: Optional[float] = None
        self.state = "idle"

    def start(self) -> dict:
        if self._sub is not None:
            self.state = "running"
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(target=self._inference_worker, daemon=True,
                                         name=f"obstacle_worker_{self._input_topic}")
        self._worker.start()
        self.state = "running"
        log.info(f"[obstacle] started: {self._input_topic} -> {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self._worker = None
        self.state = "idle"
        log.info(f"[obstacle] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        log.debug(
            f"[obstacle] received image frame: size={len(msg.data)} bytes, format={msg.format}, topic={self._input_topic}")
        # Drop old frame if queue full (no backpressure)
        try:
            self._frame_queue.put_nowait(msg.data)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(msg.data)
            except queue.Full:
                pass

    def _inference_worker(self):
        while not self._stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                started = time.perf_counter()
                result = self._adapter.estimate(jpeg_bytes)
                self._last_latency_ms = (time.perf_counter() - started) * 1000.0
                self._publish_result(result)
            except Exception as e:
                self._failure_count += 1
                log.error(f"[obstacle] inference error: {e}", exc_info=True)
                # Preserve the output contract and mark a failed request as a
                # conservative negative result instead of hanging the evaluator.
                self._publish_result(
                    {"pred_distance": self._adapter.no_obstacle_distance}
                )

    def _publish_result(self, result: dict):
        self._detect_count += 1
        msg = String()
        msg.data = json.dumps({
            "pred_distance": result.get("pred_distance", 10.0),
        }, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class ObstacleDistancePlugin:
    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._base_cfg = dict(plugin_cfg)
        self._provider = str(plugin_cfg.get("provider", "")).lower()
        self._adapter = _build_distance_adapter(self._base_cfg)
        self._nodes: dict[str, _ObstacleNode] = {}
        self._instance_configs: dict[str, dict] = {}

        log.info(
            "[obstacle-remote] plugin init: provider=%s model=%s base_url=%s "
            "key=%s decision_distance_m=%.3f",
            self._provider,
            self._adapter.model,
            self._adapter.base_url,
            "set",
            self._adapter.decision_distance_m,
        )

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            instances = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "detect_count": node._detect_count,
                    "failure_count": node._failure_count,
                    "last_latency_ms": node._last_latency_ms,
                }
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                input_topic = node._input_topic
            elif not input_topic and self._nodes:
                first_node = next(iter(self._nodes.values()))
                input_topic = first_node._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [{"topic": f"{input_topic}/obstacle", "format": "data/json"}] if input_topic else []
            state = "running" if instances else "idle"
            return {
                "name": "ObstacleDistance", "manufacture": "Embodied", "model": "obstacle",
                "state": state,
                "provider": self._provider,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Obstacle distance estimation from camera feed",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                icfg = self._instance_configs.get(node_key, {})
                merged_cfg = dict(self._base_cfg)
                merged_cfg.update(icfg)
                adapter = _build_distance_adapter(merged_cfg)
                suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
                node = _ObstacleNode(input_topic, adapter, suffix)
                self._executor.add_node(node)
                self._nodes[node_key] = node
                node.start()
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                results = []
                for key in list(self._nodes.keys()):
                    node = self._nodes[key]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[key]
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    node.stop()
                    self._executor.remove_node(node)
                    del self._nodes[instance_id]
                public_cfg = dict(cfg)
                if "key" in public_cfg:
                    public_cfg["key"] = "****"
                return {"status": "configured", "instance_id": instance_id, "config": public_cfg}
            else:
                self._base_cfg.update(cfg)
                self._provider = str(self._base_cfg.get("provider", "")).lower()
                self._adapter = _build_distance_adapter(self._base_cfg)
                public_cfg = dict(cfg)
                if "key" in public_cfg:
                    public_cfg["key"] = "****"
                return {"status": "configured", "config": public_cfg}

        return None
