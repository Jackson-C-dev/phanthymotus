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
                "enable_thinking": {"type": "boolean", "scope": "shared"},
                "image_min_pixels": {"type": "integer", "scope": "shared"},
                "image_max_pixels": {"type": "integer", "scope": "shared"},
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


_DISTANCE_BAND_BOUNDS = {
    "lt_0_5m": (0.05, 0.5),
    "0_5_to_1m": (0.5, 1.0),
    "1_to_1_5m": (1.0, 1.5),
    "1_5_to_2m": (1.5, 2.0),
    "2_to_3m": (2.0, 3.0),
    "3_to_5m": (3.0, 5.0),
    "gt_5m": (5.0, math.inf),
}


def _normalize_distance_band(value: Any) -> str:
    """Normalize the small set of bands requested in the prompt."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text).replace(".", "_")
    aliases = {
        "<0_5m": "lt_0_5m",
        "under_0_5m": "lt_0_5m",
        "0_5_1m": "0_5_to_1m",
        "1_1_5m": "1_to_1_5m",
        "1_5_2m": "1_5_to_2m",
        "2_3m": "2_to_3m",
        "3_5m": "3_to_5m",
        ">5m": "gt_5m",
        "over_5m": "gt_5m",
        "no_obstacle": "none",
        "not_visible": "none",
    }
    return aliases.get(text, text)


def _build_distance_prompt(
    decision_distance_m: float,
    no_obstacle_distance: float,
    *,
    language: str,
) -> str:
    """Build a classification-first prompt instead of asking for a blind metric guess."""
    threshold = f"{decision_distance_m:g}"
    fallback = f"{no_obstacle_distance:g}"
    bands = (
        "lt_0_5m, 0_5_to_1m, 1_to_1_5m, 1_5_to_2m, "
        "2_to_3m, 3_to_5m, gt_5m, none"
    )
    if language == "zh":
        return f"""你是机器人前向相机的障碍物风险判定器。主目标不是伪造精确深度，而是正确判断最近的可碰撞障碍物是否严格小于 {threshold} 米。

请在内部按以下顺序检查，但最终只输出 JSON：
1. 找出与机器人前进行驶走廊相交的、最近的实体障碍物。完整查看画面，重点关注中间约 70% 宽度和下方约 85% 高度；不要忽略贴近画面底部的低矮或被截断障碍物。
2. 地面/道路本身、阴影、天空、天花板，以及行驶走廊外的墙面和建筑不是障碍物；但走廊内的箱子、桌椅腿、路沿、车辆、行人和突出物属于障碍物。
3. 综合使用障碍物在画面中的占比、底部接地点、透视关系、遮挡/截断程度和常见物体尺度。接地点越靠近画面底部、物体越大或被近距离截断，通常越近。不要仅凭物体类别猜距离。
4. 先选择距离档位，再判断是否小于 {threshold} 米，最后给出档位内的代表距离。临界不确定时基于视觉证据选择一侧，不要习惯性全部判远或全部判近。

distance_band 只能是以下之一：{bands}。
返回字段：obstacle_present（布尔）、nearest_obstacle（字符串）、distance_band（字符串）、is_within_threshold（布尔）、pred_distance（米，数字）、confidence（0到1）、visual_evidence（简短字符串）。
若无相关障碍物：obstacle_present=false、distance_band="none"、is_within_threshold=false、pred_distance={fallback}。
字段必须互相一致：小于 {threshold} 米时 is_within_threshold=true，否则为 false。只输出一个 JSON 对象，不要 Markdown，不要额外文字。"""

    return f"""You are a collision-risk judge for a robot's forward camera. The primary goal is not false metric precision; it is correctly deciding whether the nearest collidable obstacle is strictly closer than {threshold} meters.

Inspect internally in this order, but return only JSON:
1. Locate the nearest solid obstacle intersecting the robot's forward travel corridor. Inspect the full frame, prioritizing roughly the central 70% width and lower 85% height. Do not ignore low or truncated objects near the bottom edge.
2. Floor/road surface, shadows, sky, ceiling, and walls/buildings outside the travel corridor are not obstacles. Boxes, furniture legs, curbs, vehicles, people, and protrusions inside the corridor are obstacles.
3. Combine image occupancy, ground-contact position, perspective, occlusion/cropping, and familiar object scale. A lower contact point, larger apparent size, or near-frame truncation usually means closer. Do not infer distance from object class alone.
4. Select a distance band first, decide the {threshold} m class second, then provide a representative distance inside that band. For ambiguous boundary cases, choose from visual evidence instead of defaulting every case to far or near.

distance_band must be one of: {bands}.
Return: obstacle_present (boolean), nearest_obstacle (string), distance_band (string), is_within_threshold (boolean), pred_distance (meters, number), confidence (0..1), visual_evidence (short string).
If no relevant obstacle exists, use obstacle_present=false, distance_band="none", is_within_threshold=false, pred_distance={fallback}.
Fields must agree: is_within_threshold is true exactly when distance is below {threshold} m. Output one JSON object only, without Markdown or extra text."""


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
    obstacle_present = _as_bool(parsed.get("obstacle_present"))
    band = _normalize_distance_band(parsed.get("distance_band"))
    bounds = _DISTANCE_BAND_BOUNDS.get(band)

    raw_distance: Optional[float]
    try:
        raw_distance = float(parsed["pred_distance"])
        if not math.isfinite(raw_distance):
            raw_distance = None
    except (KeyError, TypeError, ValueError):
        raw_distance = None

    stated_positive = _as_bool(parsed.get("is_within_threshold"))
    if stated_positive is None:
        # Backward compatibility with the first prompt/output contract.
        stated_positive = _as_bool(parsed.get("is_within_2m"))

    vote_values: list[bool] = []
    if obstacle_present is False or band == "none":
        distance = no_obstacle_distance
        final_positive = False
    else:
        band_distance: Optional[float] = None
        band_positive: Optional[bool] = None
        if bounds is not None:
            low, high = bounds
            clipped_high = min(high, no_obstacle_distance)
            band_distance = (low + clipped_high) / 2.0
            if high <= decision_distance_m:
                band_positive = True
            elif low >= decision_distance_m:
                band_positive = False

        distance = raw_distance if raw_distance is not None else band_distance
        if distance is None:
            raise ValueError(
                "response requires a numeric pred_distance or valid distance_band"
            )
        distance = min(max(distance, 0.05), no_obstacle_distance)
        distance_positive = distance < decision_distance_m
        vote_values = [distance_positive]
        if band_positive is not None:
            vote_values.append(band_positive)
        if stated_positive is not None:
            vote_values.append(stated_positive)

        positive_votes = sum(1 for vote in vote_values if vote)
        negative_votes = len(vote_values) - positive_votes
        # Metric distance is the least ambiguous fallback in a two-way tie.
        final_positive = (
            distance_positive
            if positive_votes == negative_votes
            else positive_votes > negative_votes
        )

        # Keep the continuous output consistent with the majority class. Prefer
        # the band's midpoint over collapsing every conflict to 1.98 or 2.02,
        # which previously created an artificial boundary spike.
        if final_positive and distance >= decision_distance_m:
            if band_distance is not None and band_distance < decision_distance_m:
                distance = band_distance
            else:
                distance = decision_distance_m - boundary_margin_m
        elif not final_positive and distance < decision_distance_m:
            if band_distance is not None and band_distance >= decision_distance_m:
                distance = band_distance
            else:
                distance = decision_distance_m + boundary_margin_m

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    if len(set(vote_values)) > 1:
        confidence *= 0.65
    return {
        "pred_distance": distance,
        "is_positive": final_positive,
        "confidence": confidence,
        "distance_band": band,
        "reasoning": str(
            parsed.get("visual_evidence") or parsed.get("reasoning", "")
        )[:500],
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
            if (
                response.status_code == 400
                and "enable_thinking" in request_payload
                and "enable_thinking" in response.text.lower()
            ):
                # Third-party OpenAI-compatible Qwen gateways may not expose the
                # DashScope-specific switch. Retry without losing compatibility.
                request_payload.pop("enable_thinking", None)
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
        self._system_prompt = _build_distance_prompt(
            decision_distance_m, no_obstacle_distance, language="en"
        )

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
            {"role": "system", "content": self._system_prompt},
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
                        "text": "Analyze the forward collision corridor and return the requested JSON."
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

    def __init__(self, url: str, key: str, model: str, **kwargs: Any):
        thinking = _as_bool(kwargs.pop("enable_thinking", True))
        self.enable_thinking = True if thinking is None else thinking
        self.image_min_pixels = max(
            65536, int(kwargs.pop("image_min_pixels", 262144))
        )
        self.image_max_pixels = max(
            self.image_min_pixels,
            int(kwargs.pop("image_max_pixels", 1048576)),
        )
        super().__init__(url, key, model or "qwen-vl-max", **kwargs)
        self.base_url = _normalize_base_url(
            url, "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._system_prompt = _build_distance_prompt(
            self.decision_distance_m,
            self.no_obstacle_distance,
            language="zh",
        )

    def estimate(self, image_bytes: bytes) -> dict:
        import base64

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_format = "jpeg"
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            image_format = "png"

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_b64}"
                        },
                        "min_pixels": self.image_min_pixels,
                        "max_pixels": self.image_max_pixels,
                    },
                    {
                        "type": "text",
                        "text": "分析机器人正前方可碰撞区域，并严格按要求返回 JSON。"
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
                "temperature": 0,
                # Make Qwen3.5-27B's reasoning mode explicit. Some self-hosted
                # gateways default it off even though DashScope defaults it on.
                "enable_thinking": self.enable_thinking,
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
        return QwenVLDistanceAdapter(
            url,
            key,
            model,
            enable_thinking=_as_bool(cfg.get("enable_thinking")) is not False,
            image_min_pixels=int(cfg.get("image_min_pixels", 262144)),
            image_max_pixels=int(cfg.get("image_max_pixels", 1048576)),
            **common,
        )

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
        self._positive_count = 0
        self._negative_count = 0
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
        distance = float(result.get("pred_distance", 10.0))
        is_positive = distance < self._adapter.decision_distance_m
        if is_positive:
            self._positive_count += 1
        else:
            self._negative_count += 1
        log.info(
            "[obstacle-remote] result distance=%.3fm positive=%s band=%s "
            "confidence=%.3f totals=%d/%d failures=%d",
            distance,
            is_positive,
            result.get("distance_band", "unknown"),
            float(result.get("confidence", 0.0)),
            self._positive_count,
            self._negative_count,
            self._failure_count,
        )
        msg = String()
        msg.data = json.dumps({
            "pred_distance": distance,
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
                    "positive_count": node._positive_count,
                    "negative_count": node._negative_count,
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

