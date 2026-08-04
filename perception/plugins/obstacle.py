#!/usr/bin/env python3
"""
Drop-in ROS2 obstacle-distance plugin using a task-aligned local model.

The public plugin protocol intentionally matches ``obstacle.py``:

* actions: ``start``, ``stop``, ``info`` and ``config``;
* input: ``sensor_msgs/CompressedImage``;
* output topic: ``<input_topic>/obstacle``;
* output payload: ``{"pred_distance": <float metres>}``.

Unlike the sample implementation, this module never returns a random distance.
It loads a small ONNX or TorchScript model trained with the following output
contract (dictionary/ONNX output names are preferred):

Required (at least one):
    distance_map      [1, 1, H, W], metric task distance in metres; or
    global_distance   [1, 1], direct metric distance fallback.

Recommended:
    obstacle_logits   [1, 1, H, W], task-specific obstacle logits.
    scene_logits      [1, 2], [indoor, outdoor] logits.
    near_logit        [1, 1], auxiliary logit for distance < 1 metre.
    global_residual   [1, 1], bounded log-scale correction for distance_map.
    uncertainty       [1, 1, H, W], per-cell log uncertainty.

For indoor scenes the postprocessor uses the specification ROI (middle third
of columns, top five-eighths of rows) and the P1 metric-Z percentile.  For
outdoor scenes ``distance_map`` is expected to contain front-bumper-to-OBB
surface distances only for classes included by the benchmark; the obstacle
head suppresses static objects and the postprocessor uses a robust component
minimum.

Deployment assumptions:

* fixed 320x240 or 384x288 model input;
* model artifact below 30 MiB (enforced by default);
* TensorRT/CUDA/CPU provider selection through ONNX Runtime, or TorchScript;
* one shared model session across ROS instances;
* latest-frame queueing to keep end-to-end latency below the 3 second limit.

The trained model artifact is deliberately not embedded in this source file.
Configure ``model_path`` or ``OBSTACLE_MODEL_PATH`` before starting the plugin.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

# Keep the model adapter importable on a workstation that does not have ROS2.
# ROS2 is required only when the plugin/node layer is instantiated.
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
        """Import-time placeholder; construction gives an actionable error."""

        def __init__(self, *_args: Any, **_kwargs: Any):
            raise RuntimeError(
                "ROS2 Python packages are unavailable. Local model testing does not "
                "need ROS2; use local_test_obstacle.py. The ROS plugin requires rclpy, "
                "sensor_msgs and std_msgs."
            ) from _ROS2_IMPORT_ERROR

log = logging.getLogger(__name__)

_MIB = 1024 * 1024
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
        "description": "Task-aligned nearest-obstacle distance estimation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform",
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
                # provider/url/key/model are retained for sample compatibility.
                "provider": {
                    "type": "string",
                    "enum": ["local"],
                    "description": "Only deterministic local inference is supported",
                    "scope": "shared",
                },
                "url": {"type": "string", "description": "Reserved", "scope": "shared"},
                "key": {
                    "type": "string",
                    "description": "Reserved",
                    "format": "password",
                    "scope": "shared",
                },
                "model": {
                    "type": "string",
                    "description": "Model name or artifact path (compatibility field)",
                    "scope": "instance",
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to a task-aligned .onnx/.pt model",
                    "scope": "shared",
                },
                "runtime": {
                    "type": "string",
                    "enum": ["auto", "onnx", "torchscript"],
                    "scope": "shared",
                },
                "scene_mode": {
                    "type": "string",
                    "enum": ["auto", "indoor", "outdoor"],
                    "scope": "instance",
                },
                "input_width": {"type": "integer", "scope": "shared"},
                "input_height": {"type": "integer", "scope": "shared"},
                "input_normalization": {
                    "type": "string",
                    "enum": ["imagenet", "zero_one"],
                    "scope": "shared",
                },
                "use_first_output_as_distance": {"type": "boolean", "scope": "shared"},
                "gpu_memory_limit_mb": {"type": "integer", "scope": "shared"},
                "use_dla": {"type": "boolean", "scope": "shared"},
                "dla_core": {"type": "integer", "scope": "shared"},
                "model_size_limit_mb": {"type": "number", "scope": "shared"},
                "max_inference_seconds": {"type": "number", "scope": "shared"},
                "no_obstacle_distance": {"type": "number", "scope": "instance"},
                "obstacle_threshold": {"type": "number", "scope": "instance"},
            },
            "required": ["provider"],
        },
        "topic_in": [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json", "desc": "obstacle distance result"}],
    }
]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value, dtype=np.float32)
    array = np.clip(array, -30.0, 30.0)
    result = 1.0 / (1.0 + np.exp(-array))
    return float(result) if result.ndim == 0 else result


def _to_probability(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        return array
    return np.asarray(_sigmoid(array), dtype=np.float32)


def _quantile(values: np.ndarray, q: float) -> float:
    """NumPy-version-compatible quantile helper."""
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, q, interpolation="linear"))


def _image_kind(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "jpeg"
    return "unknown"


def _squeeze_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must reduce to HxW, got shape={array.shape}")
    return array.astype(np.float32, copy=False)


def _find_output(outputs: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): value for key, value in outputs.items()}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    for key, value in lowered.items():
        if any(alias in key for alias in aliases):
            return value
    return None


class DistanceAdapter(ABC):
    """Interface retained from the sample plugin."""

    fallback_distance = 10.0

    @abstractmethod
    def estimate(self, image_bytes: bytes) -> dict:
        """Return at least ``pred_distance`` in metres."""
        raise NotImplementedError


class _ModelRunner(ABC):
    @abstractmethod
    def infer(self, tensor: np.ndarray) -> Mapping[str, Any]:
        raise NotImplementedError


class _OnnxRunner(_ModelRunner):
    def __init__(self, model_path: Path, cfg: Mapping[str, Any]):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for .onnx models; install the Jetson-compatible build"
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = int(cfg.get("cpu_threads", 2))
        options.inter_op_num_threads = 1

        available = set(ort.get_available_providers())
        providers: list[Any] = []
        gpu_limit = int(cfg.get("gpu_memory_limit_mb", 4608)) * _MIB
        cache_dir = str(cfg.get("engine_cache_path", "/tmp/obstacle_trt_cache"))

        if "TensorrtExecutionProvider" in available:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            trt_options = {
                "trt_fp16_enable": "1",
                "trt_engine_cache_enable": "1",
                "trt_engine_cache_path": cache_dir,
                "trt_max_workspace_size": str(min(gpu_limit // 2, 2 * 1024**3)),
            }
            if _as_bool(cfg.get("use_dla"), True):
                trt_options.update(
                    {
                        "trt_dla_enable": "1",
                        "trt_dla_core": str(int(cfg.get("dla_core", 0))),
                    }
                )
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    trt_options,
                )
            )
        if "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": "0",
                        "gpu_mem_limit": str(gpu_limit),
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": "1",
                    },
                )
            )
        providers.append("CPUExecutionProvider")

        try:
            self._session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=providers
            )
        except Exception as exc:
            detail = str(exc)
            if "opset" in detail.lower() and "official support" in detail.lower():
                raise RuntimeError(
                    "ONNX model opset is newer than this ONNX Runtime supports. "
                    "Re-export the model with a compatible opset, for example: "
                    "yolo export model=yolo26n-depth.pt format=onnx imgsz=320 "
                    "opset=20 simplify=True device=cpu"
                ) from exc
            raise RuntimeError(f"failed to load ONNX model {model_path}: {detail}") from exc
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"model must have exactly one image input, found {len(inputs)}")
        self._input_name = inputs[0].name
        self._input_type = inputs[0].type
        self._output_names = [item.name for item in self._session.get_outputs()]
        log.info("[obstacle] ONNX providers=%s", self._session.get_providers())

    def infer(self, tensor: np.ndarray) -> Mapping[str, Any]:
        if "float16" in self._input_type:
            tensor = tensor.astype(np.float16, copy=False)
        else:
            tensor = tensor.astype(np.float32, copy=False)
        values = self._session.run(self._output_names, {self._input_name: tensor})
        return dict(zip(self._output_names, values))


class _TorchScriptRunner(_ModelRunner):
    _ORDERED_NAMES = (
        "distance_map",
        "obstacle_logits",
        "scene_logits",
        "near_logit",
        "global_residual",
        "uncertainty",
    )

    def __init__(self, model_path: Path, cfg: Mapping[str, Any]):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for TorchScript models") from exc

        self._torch = torch
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and not _as_bool(cfg.get("force_cpu")) else "cpu"
        )
        self._fp16 = self._device.type == "cuda" and _as_bool(cfg.get("fp16"), True)
        if self._device.type == "cuda":
            gpu_limit = int(cfg.get("gpu_memory_limit_mb", 4608)) * _MIB
            try:
                total_memory = int(torch.cuda.get_device_properties(self._device).total_memory)
                fraction = min(0.95, max(0.05, gpu_limit / total_memory))
                torch.cuda.set_per_process_memory_fraction(fraction, self._device)
            except (AttributeError, RuntimeError) as exc:
                log.warning("[obstacle] unable to set PyTorch CUDA memory fraction: %s", exc)
        self._model = torch.jit.load(str(model_path), map_location=self._device).eval()
        if self._fp16:
            self._model.half()
        log.info("[obstacle] TorchScript device=%s fp16=%s", self._device, self._fp16)

    def infer(self, tensor: np.ndarray) -> Mapping[str, Any]:
        torch = self._torch
        value = torch.from_numpy(tensor).to(self._device, non_blocking=True)
        value = value.half() if self._fp16 else value.float()
        with torch.inference_mode():
            output = self._model(value)
        if isinstance(output, dict):
            return {str(key): val.detach().float().cpu().numpy() for key, val in output.items()}
        if not isinstance(output, (tuple, list)):
            output = (output,)
        return {
            name: val.detach().float().cpu().numpy()
            for name, val in zip(self._ORDERED_NAMES, output)
        }


class TaskAlignedLocalDistanceAdapter(DistanceAdapter):
    """Lightweight metric-distance inference and benchmark-specific geometry."""

    def __init__(self, cfg: Mapping[str, Any]):
        model_value = (
            cfg.get("model_path")
            or os.environ.get("OBSTACLE_MODEL_PATH")
            or (cfg.get("model") if str(cfg.get("model", "")).endswith((".onnx", ".pt", ".pth")) else "")
        )
        if not model_value:
            raise ValueError("model_path (or OBSTACLE_MODEL_PATH) is required for local inference")

        model_path = Path(str(model_value)).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"obstacle model not found: {model_path}")
        size_limit_mb = float(cfg.get("model_size_limit_mb", 30.0))
        model_size = model_path.stat().st_size
        if model_size > size_limit_mb * _MIB:
            raise ValueError(
                f"model artifact is {model_size / _MIB:.2f} MiB; limit is {size_limit_mb:.2f} MiB"
            )

        self._cfg = dict(cfg)
        self._input_width = int(cfg.get("input_width", 320))
        self._input_height = int(cfg.get("input_height", 240))
        if self._input_width <= 0 or self._input_height <= 0:
            raise ValueError("input_width/input_height must be positive")
        self._input_normalization = str(cfg.get("input_normalization", "imagenet")).lower()
        if self._input_normalization not in {"imagenet", "zero_one"}:
            raise ValueError("input_normalization must be imagenet or zero_one")
        self._use_first_output_as_distance = _as_bool(
            cfg.get("use_first_output_as_distance"), False
        )
        self._scene_mode = str(cfg.get("scene_mode", "auto")).lower()
        if self._scene_mode not in {"auto", "indoor", "outdoor"}:
            raise ValueError("scene_mode must be auto, indoor or outdoor")

        self._obstacle_threshold = float(cfg.get("obstacle_threshold", 0.5))
        self._indoor_quantile = float(cfg.get("indoor_quantile", 0.01))
        self._outdoor_quantile = float(cfg.get("outdoor_quantile", 0.05))
        self._min_distance = float(cfg.get("min_distance", 0.05))
        self._max_distance = float(cfg.get("max_distance", 80.0))
        self.fallback_distance = float(cfg.get("no_obstacle_distance", 10.0))
        self._distance_scale = float(cfg.get("distance_scale", 1.0))
        self._distance_bias = float(cfg.get("distance_bias", 0.0))
        self._max_inference_seconds = float(cfg.get("max_inference_seconds", 3.0))
        self._lock = threading.Lock()

        runtime = str(cfg.get("runtime", "auto")).lower()
        if runtime == "auto":
            runtime = "onnx" if model_path.suffix.lower() == ".onnx" else "torchscript"
        if runtime == "onnx":
            self._runner: _ModelRunner = _OnnxRunner(model_path, cfg)
        elif runtime == "torchscript":
            self._runner = _TorchScriptRunner(model_path, cfg)
        else:
            raise ValueError("runtime must be auto, onnx or torchscript")

        self._warmup()
        log.info(
            "[obstacle] model=%s size=%.2fMiB input=%dx%d scene=%s",
            model_path,
            model_size / _MIB,
            self._input_width,
            self._input_height,
            self._scene_mode,
        )

    def _warmup(self) -> None:
        warmup = np.zeros((1, 3, self._input_height, self._input_width), dtype=np.float32)
        try:
            with self._lock:
                outputs = self._runner.infer(warmup)
        except Exception as exc:
            raise RuntimeError(f"model warm-up failed: {exc}") from exc
        has_map = self._find_distance_output(outputs) is not None
        has_scalar = _find_output(
            outputs,
            ("global_distance", "pred_distance", "distance_scalar"),
        ) is not None
        if not has_map and not has_scalar:
            raise ValueError(
                "model output contract mismatch: expected distance_map or global_distance; "
                f"found {sorted(str(key) for key in outputs)}"
            )

    def _decode_and_preprocess(self, image_bytes: bytes) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to decode CompressedImage") from exc

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("failed to decode compressed image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_AREA,
        )
        tensor = image.astype(np.float32) / 255.0
        if self._input_normalization == "imagenet":
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = (tensor - mean) / std
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

    def _find_distance_output(self, outputs: Mapping[str, Any]) -> Any:
        raw = _find_output(
            outputs,
            ("distance_map", "metric_distance", "metric_depth", "depth"),
        )
        if raw is None and self._use_first_output_as_distance and len(outputs) == 1:
            raw = next(iter(outputs.values()))
        return raw

    def _resolve_scene(self, outputs: Mapping[str, Any], image_bytes: bytes) -> str:
        if self._scene_mode != "auto":
            return self._scene_mode
        scene = _find_output(outputs, ("scene_logits", "scene", "domain_logits"))
        if scene is not None:
            logits = np.asarray(scene).reshape(-1)
            if logits.size >= 2 and np.all(np.isfinite(logits[:2])):
                return "outdoor" if int(np.argmax(logits[:2])) == 1 else "indoor"
        # The benchmark document associates PNG/29 mm and JPEG/33 mm inputs.
        # This is only a fallback; an explicit scene_mode or scene head is safer.
        return "indoor" if _image_kind(image_bytes) == "png" else "outdoor"

    def _valid_distance_map(self, outputs: Mapping[str, Any]) -> Optional[np.ndarray]:
        raw = self._find_distance_output(outputs)
        if raw is None:
            return None
        distance = _squeeze_map(raw, "distance_map")
        return distance

    def _probability_map(
        self, outputs: Mapping[str, Any], target_shape: tuple[int, int]
    ) -> Optional[np.ndarray]:
        raw = _find_output(
            outputs,
            ("obstacle_logits", "hazard_logits", "obstacle_mask", "hazard_mask"),
        )
        if raw is None:
            return None
        probability = _to_probability(_squeeze_map(raw, "obstacle_logits"))
        if probability.shape != target_shape:
            try:
                import cv2

                probability = cv2.resize(
                    probability,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            except ImportError as exc:
                raise RuntimeError("opencv-python is required to align output maps") from exc
        return probability

    def _uncertainty_mask(
        self, outputs: Mapping[str, Any], target_shape: tuple[int, int]
    ) -> Optional[np.ndarray]:
        raw = _find_output(outputs, ("uncertainty", "log_variance", "log_sigma"))
        if raw is None:
            return None
        uncertainty = _squeeze_map(raw, "uncertainty")
        if uncertainty.shape != target_shape:
            import cv2

            uncertainty = cv2.resize(
                uncertainty,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        finite = uncertainty[np.isfinite(uncertainty)]
        if finite.size < 4:
            return None
        return uncertainty <= _quantile(finite, 0.90)

    def _base_valid_mask(
        self, distance: np.ndarray, probability: Optional[np.ndarray], outputs: Mapping[str, Any]
    ) -> np.ndarray:
        valid = np.isfinite(distance) & (distance >= self._min_distance) & (distance <= self._max_distance)
        if probability is not None:
            valid &= probability >= self._obstacle_threshold
        uncertainty_valid = self._uncertainty_mask(outputs, distance.shape)
        if uncertainty_valid is not None:
            valid &= uncertainty_valid
        return valid

    def _aggregate_indoor(
        self, distance: np.ndarray, probability: Optional[np.ndarray], outputs: Mapping[str, Any]
    ) -> Optional[float]:
        height, width = distance.shape
        x0, x1 = int(math.floor(width / 3.0)), int(math.ceil(2.0 * width / 3.0))
        y1 = int(math.ceil(5.0 * height / 8.0))
        valid = self._base_valid_mask(distance, probability, outputs)
        roi_mask = np.zeros_like(valid, dtype=bool)
        roi_mask[:y1, x0:x1] = True
        values = distance[valid & roi_mask]

        # If the mask head is temporarily over-conservative, retain the exact
        # benchmark ROI and use metric depth rather than returning no result.
        if values.size < 4 and probability is not None:
            raw_valid = np.isfinite(distance) & (distance >= self._min_distance) & (distance <= self._max_distance)
            values = distance[raw_valid & roi_mask]
        if values.size == 0:
            return None
        return _quantile(values, self._indoor_quantile)

    def _aggregate_outdoor(
        self, distance: np.ndarray, probability: Optional[np.ndarray], outputs: Mapping[str, Any]
    ) -> Optional[float]:
        valid = self._base_valid_mask(distance, probability, outputs)
        if not np.any(valid):
            return None
        if probability is None:
            return _quantile(distance[valid], self._outdoor_quantile)

        # Remove isolated false-positive cells before taking the nearest valid
        # OBB distance.  Each accepted component represents an included object.
        try:
            import cv2

            count, labels = cv2.connectedComponents(valid.astype(np.uint8), connectivity=8)
            min_area = max(2, int(round(valid.size * 0.0005)))
            candidates: list[float] = []
            for label in range(1, count):
                values = distance[labels == label]
                values = values[np.isfinite(values)]
                if values.size >= min_area:
                    candidates.append(_quantile(values, self._outdoor_quantile))
            return min(candidates) if candidates else None
        except ImportError:
            return _quantile(distance[valid], self._outdoor_quantile)

    def _global_value(self, outputs: Mapping[str, Any], aliases: tuple[str, ...]) -> Optional[float]:
        raw = _find_output(outputs, aliases)
        if raw is None:
            return None
        values = np.asarray(raw, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        return float(values[0]) if values.size else None

    def _postprocess(self, outputs: Mapping[str, Any], image_bytes: bytes) -> tuple[float, str, float]:
        scene = self._resolve_scene(outputs, image_bytes)
        distance_map = self._valid_distance_map(outputs)
        distance: Optional[float] = None
        mask_confidence = 0.0

        if distance_map is not None:
            probability = self._probability_map(outputs, distance_map.shape)
            if probability is not None:
                finite = probability[np.isfinite(probability)]
                mask_confidence = float(finite.max()) if finite.size else 0.0
            if scene == "indoor":
                distance = self._aggregate_indoor(distance_map, probability, outputs)
            else:
                distance = self._aggregate_outdoor(distance_map, probability, outputs)

        if distance is None:
            distance = self._global_value(
                outputs, ("global_distance", "pred_distance", "distance_scalar")
            )
        if distance is None:
            distance = self.fallback_distance

        residual = self._global_value(outputs, ("global_residual", "distance_residual"))
        if residual is not None and distance != self.fallback_distance:
            distance *= math.exp(float(np.clip(residual, -0.30, 0.30)))
        distance = distance * self._distance_scale + self._distance_bias
        distance = float(np.clip(distance, self._min_distance, self._max_distance))

        near_logit = self._global_value(outputs, ("near_logit", "risk_logit", "near_probability"))
        if near_logit is not None:
            near_probability = near_logit if 0.0 <= near_logit <= 1.0 else float(_sigmoid(near_logit))
            confidence = max(mask_confidence, near_probability if distance < 1.0 else 1.0 - near_probability)
        else:
            confidence = mask_confidence
        return distance, scene, float(np.clip(confidence, 0.0, 1.0))

    def estimate(self, image_bytes: bytes) -> dict:
        started = time.perf_counter()
        tensor = self._decode_and_preprocess(image_bytes)
        # Multi-instance nodes share one session.  Serializing enqueue prevents
        # transient GPU/DLA memory spikes on the 16 GB Orin.
        with self._lock:
            outputs = self._runner.infer(tensor)
        distance, scene, confidence = self._postprocess(outputs, image_bytes)
        elapsed = time.perf_counter() - started
        if elapsed > self._max_inference_seconds:
            log.warning(
                "[obstacle] inference exceeded %.3fs budget: %.3fs",
                self._max_inference_seconds,
                elapsed,
            )
        return {
            "pred_distance": distance,
            "confidence": confidence,
            "scene": scene,
            "latency_ms": elapsed * 1000.0,
        }


def _build_distance_adapter(cfg: Mapping[str, Any]) -> Optional[DistanceAdapter]:
    provider = str(cfg.get("provider", "local")).lower()
    if provider != "local":
        log.error("[obstacle] provider=%s is unsupported; use provider=local", provider)
        return None
    try:
        return TaskAlignedLocalDistanceAdapter(cfg)
    except Exception as exc:
        log.error("[obstacle] local adapter initialization failed: %s", exc, exc_info=True)
        return None


class _ObstacleNode(Node):
    """One ROS2 node per subscribed camera topic."""

    def __init__(self, input_topic: str, adapter: DistanceAdapter, node_suffix: str):
        super().__init__(f"obstacle_{node_suffix}")
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
            return {"state": "running", "input": self._input_topic, "output": self._output_topic}
        self._stop_event.clear()
        self._sub = self.create_subscription(
            CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
        )
        self._worker = threading.Thread(
            target=self._inference_worker,
            daemon=True,
            name=f"obstacle_worker_{self._input_topic}",
        )
        self._worker.start()
        self.state = "running"
        log.info("[obstacle] started: %s -> %s", self._input_topic, self._output_topic)
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
        log.info("[obstacle] stopped: %s", self._input_topic)
        return {"state": "idle", "input": self._input_topic}

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

    def _inference_worker(self) -> None:
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
                log.error("[obstacle] inference error: %s", exc, exc_info=True)
                result = {"pred_distance": float(self._adapter.fallback_distance)}
            self._publish_result(result)

    def _publish_result(self, result: Mapping[str, Any]) -> None:
        self._detect_count += 1
        distance = float(result.get("pred_distance", self._adapter.fallback_distance))
        if not math.isfinite(distance):
            self._failure_count += 1
            distance = float(self._adapter.fallback_distance)
        msg = String()
        # Keep the sample's exact public data structure.
        msg.data = json.dumps({"pred_distance": distance}, ensure_ascii=False)
        self._pub.publish(msg)


class ObstacleDistancePlugin:
    """Drop-in plugin class retaining the sample name and dispatch protocol."""

    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._base_cfg = dict(plugin_cfg)
        self._base_cfg.setdefault("provider", "local")
        self._adapter = _build_distance_adapter(self._base_cfg)
        self._nodes: dict[str, _ObstacleNode] = {}
        self._instance_configs: dict[str, dict] = {}
        if self._adapter is None:
            log.warning(
                "[obstacle] model is not ready; configure a valid model_path before action=start"
            )

    def get_tools(self) -> list:
        return TOOLS

    def _adapter_for(self, node_key: str) -> DistanceAdapter:
        instance_cfg = self._instance_configs.get(node_key)
        if instance_cfg:
            merged = dict(self._base_cfg)
            merged.update(instance_cfg)
            adapter = _build_distance_adapter(merged)
        else:
            adapter = self._adapter
        if adapter is None:
            raise RuntimeError(
                "obstacle model is not configured; set provider=local and model_path=<artifact>"
            )
        return adapter

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
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in self._nodes:
                input_topic = self._nodes[instance_id]._input_topic
            elif not input_topic and self._nodes:
                input_topic = next(iter(self._nodes.values()))._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = (
                [{"topic": f"{input_topic}/obstacle", "format": "data/json"}]
                if input_topic
                else []
            )
            return {
                "name": "ObstacleDistance",
                "manufacture": "Embodied",
                "model": "obstacle",
                "state": "running" if instances else "idle",
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Task-aligned obstacle distance estimation from camera feed",
            }

        if action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            if node_key not in self._nodes:
                adapter = self._adapter_for(node_key)
                suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
                node = _ObstacleNode(input_topic, adapter, suffix)
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
                if key not in {"action", "instance_id"} and value is not None and value != ""
            }
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes.pop(instance_id)
                    node.stop()
                    self._executor.remove_node(node)
                return {"status": "configured", "instance_id": instance_id, "config": cfg}

            # Reconfiguration is applied to future nodes. Existing nodes keep
            # their current session until stopped, avoiding an unsafe live swap.
            self._base_cfg.update(cfg)
            self._adapter = _build_distance_adapter(self._base_cfg)
            return {"status": "configured", "config": cfg}

        return None
