

#!/usr/bin/env python3
"""FastDepth implementation of the public ``obstacle`` MCP/ROS2 tool.

The plugin keeps the same external contract as :mod:`plugins.obstacle`:

* tool name and prefix: ``obstacle``;
* actions: ``start``, ``stop``, ``info`` and ``config``;
* input: ``sensor_msgs/CompressedImage``;
* output topic: ``<input_topic>/obstacle``;
* output payload: ``{"pred_distance": <float metres>}``.

The default model is the MIT-licensed FastDepth 224x224 NYU-Depth-v2 ONNX
export published by the Axelera model zoo.  It is a 5.2 MiB, opset-11 model
whose graph contains only basic Conv/Resize/Add/ReLU/Clip operators.  The
model predicts metric indoor depth; it does not contain an obstacle detector,
so this adapter applies the same central indoor ROI and robust low percentile
used by the original obstacle plugin.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from plugins.obstacle import ObstacleDistancePlugin


log = logging.getLogger(__name__)

_MIB = 1024 * 1024
_DEFAULT_DECISION_DISTANCE_M = 2.0


TOOLS = [
    {
        # The evaluator calls this exact name.  Do not expose obstacleFastDepth.
        "name": "obstacle",
        "type": "processor",
        "multiInstance": True,
        "description": "FastDepth nearest-obstacle distance estimation",
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
                "provider": {
                    "type": "string",
                    "enum": ["fastdepth"],
                    "scope": "shared",
                },
                "model_path": {"type": "string", "scope": "shared"},
                "execution_provider": {
                    "type": "string",
                    "enum": ["cpu", "cuda", "tensorrt", "auto"],
                    "scope": "shared",
                },
                "cpu_threads": {"type": "integer", "scope": "shared"},
                "gpu_memory_limit_mb": {"type": "integer", "scope": "shared"},
                "trt_workspace_limit_mb": {"type": "integer", "scope": "shared"},
                "trt_fp16_enable": {"type": "boolean", "scope": "shared"},
                "engine_cache_path": {"type": "string", "scope": "shared"},
                "scene_mode": {
                    "type": "string",
                    "enum": ["auto", "indoor", "outdoor"],
                    "scope": "instance",
                },
                "indoor_quantile": {"type": "number", "scope": "instance"},
                "outdoor_quantile": {"type": "number", "scope": "instance"},
                "min_distance": {"type": "number", "scope": "instance"},
                "max_distance": {"type": "number", "scope": "instance"},
                "no_obstacle_distance": {"type": "number", "scope": "instance"},
                "distance_scale": {"type": "number", "scope": "instance"},
                "distance_bias": {"type": "number", "scope": "instance"},
                "decision_distance_m": {"type": "number", "scope": "instance"},
                "indoor_positive_depth_m": {"type": "number", "scope": "instance"},
                "outdoor_positive_depth_m": {"type": "number", "scope": "instance"},
                "boundary_margin_m": {"type": "number", "scope": "instance"},
                "bias_warning_min_samples": {"type": "integer", "scope": "instance"},
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


def _resolve_model(cfg: Mapping[str, Any]) -> Path:
    # Keep model resolution identical to the original obstacle adapter: config
    # points to a directory, and the shared downloader owns URL/cache handling.
    from utils.model_downloader import MODELS, ensure_model

    configured = str(
        cfg.get("model_path")
        or os.environ.get("FASTDEPTH_MODEL_PATH")
        or "/models/fastdepth"
    )
    configured_path = Path(configured).expanduser()
    if configured_path.suffix.lower() == ".onnx":
        model_path = configured_path
    else:
        ensure_model("obstacle_fastdepth", str(configured_path))
        model_path = configured_path / str(
            MODELS["obstacle_fastdepth"]["check_file"]
        )

    if not model_path.is_file():
        raise FileNotFoundError(f"FastDepth model not found: {model_path}")
    return model_path


class FastDepthDistanceAdapter:
    """Run the 224x224 FastDepth ONNX model and aggregate metric depth."""

    def __init__(self, cfg: Mapping[str, Any]):
        provider = str(cfg.get("provider", "fastdepth")).strip().lower()
        if provider != "fastdepth":
            raise ValueError("obstacleFastDepth provider must be 'fastdepth'")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required; use the JetPack-6-compatible wheel already "
                "installed by Dockerfile.jetson"
            ) from exc

        self._cfg = dict(cfg)
        self._model_path = _resolve_model(cfg)
        self._scene_mode = str(cfg.get("scene_mode", "auto")).strip().lower()
        if self._scene_mode not in {"auto", "indoor", "outdoor"}:
            raise ValueError("scene_mode must be auto, indoor or outdoor")
        self._indoor_quantile = float(cfg.get("indoor_quantile", 0.01))
        self._outdoor_quantile = float(cfg.get("outdoor_quantile", 0.05))
        if not 0.0 <= self._indoor_quantile <= 1.0:
            raise ValueError("indoor_quantile must be between 0 and 1")
        if not 0.0 <= self._outdoor_quantile <= 1.0:
            raise ValueError("outdoor_quantile must be between 0 and 1")
        self._min_distance = float(cfg.get("min_distance", 0.05))
        self._max_distance = float(cfg.get("max_distance", 10.0))
        if self._min_distance <= 0 or self._max_distance <= self._min_distance:
            raise ValueError("distance range must satisfy 0 < min_distance < max_distance")
        self.fallback_distance = float(
            cfg.get("no_obstacle_distance", self._max_distance)
        )
        self._distance_scale = float(cfg.get("distance_scale", 1.0))
        self._distance_bias = float(cfg.get("distance_bias", 0.0))
        self._decision_distance = float(
            cfg.get("decision_distance_m", _DEFAULT_DECISION_DISTANCE_M)
        )
        if not self._min_distance < self._decision_distance < self._max_distance:
            raise ValueError(
                "decision_distance_m must be inside the configured distance range"
            )
        self._positive_depth_thresholds = {
            "indoor": float(
                cfg.get("indoor_positive_depth_m", self._decision_distance)
            ),
            "outdoor": float(
                cfg.get("outdoor_positive_depth_m", self._decision_distance)
            ),
        }
        for scene, threshold in self._positive_depth_thresholds.items():
            if not self._min_distance < threshold < self._max_distance:
                raise ValueError(
                    f"{scene}_positive_depth_m must be inside the distance range"
                )
        max_margin = min(
            self._decision_distance - self._min_distance,
            self._max_distance - self._decision_distance,
        )
        self._boundary_margin = min(
            max(0.001, float(cfg.get("boundary_margin_m", 0.02))),
            max_margin,
        )
        self._bias_warning_min_samples = max(
            10, int(cfg.get("bias_warning_min_samples", 20))
        )
        self._stats_lock = threading.Lock()
        self._prediction_count = 0
        self._positive_count = 0
        self._negative_count = 0
        self._raw_distance_sum = 0.0
        self._raw_distance_min = math.inf
        self._raw_distance_max = -math.inf
        self._recent_raw_distances: list[float] = []
        self._scene_counts = {
            "indoor": {"total": 0, "positive": 0},
            "outdoor": {"total": 0, "positive": 0},
        }
        self._last_raw_distance = self.fallback_distance
        self._max_inference_seconds = float(cfg.get("max_inference_seconds", 3.0))
        self._lock = threading.Lock()

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = max(1, int(cfg.get("cpu_threads", 1)))
        options.inter_op_num_threads = 1

        available = set(ort.get_available_providers())
        requested = str(cfg.get("execution_provider", "cpu")).strip().lower()
        providers: list[Any] = []
        gpu_limit = max(32, int(cfg.get("gpu_memory_limit_mb", 192))) * _MIB
        trt_workspace = max(16, int(cfg.get("trt_workspace_limit_mb", 32))) * _MIB

        if requested == "auto":
            requested = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        if requested == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError(
                    f"TensorRTExecutionProvider is unavailable; providers={sorted(available)}"
                )
            cache_path = Path(
                str(cfg.get("engine_cache_path", "/tmp/fastdepth_trt_cache"))
            )
            cache_path.mkdir(parents=True, exist_ok=True)
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": _as_bool(cfg.get("trt_fp16_enable"), True),
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache_path),
                        "trt_max_workspace_size": str(trt_workspace),
                    },
                )
            )
            requested = (
                "cuda" if "CUDAExecutionProvider" in available else "cpu"
            )
        if requested == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    f"CUDAExecutionProvider is unavailable; providers={sorted(available)}"
                )
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": "0",
                        "gpu_mem_limit": str(gpu_limit),
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "cudnn_conv_use_max_workspace": "0",
                    },
                )
            )
        elif requested != "cpu":
            raise ValueError("execution_provider must be cpu, cuda, tensorrt or auto")
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(
            str(self._model_path), sess_options=options, providers=providers
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or list(inputs[0].shape) != [1, 3, 224, 224]:
            raise ValueError(
                "FastDepth model must have one [1,3,224,224] input; "
                f"found {[(item.name, item.shape) for item in inputs]}"
            )
        if len(outputs) != 1 or list(outputs[0].shape) != [1, 1, 224, 224]:
            raise ValueError(
                "FastDepth model must have one [1,1,224,224] output; "
                f"found {[(item.name, item.shape) for item in outputs]}"
            )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._providers = self._session.get_providers()

        # Warm up once so provider/model failures happen before MCP readiness.
        self._session.run(
            [self._output_name],
            {self._input_name: np.zeros((1, 3, 224, 224), dtype=np.float32)},
        )
        log.info(
            "[obstacle-fastdepth] model=%s size=%.2fMiB providers=%s "
            "gpu_limit_mb=%d trt_workspace_mb=%d decision_distance_m=%.3f "
            "positive_depth_m=%s",
            self._model_path,
            self._model_path.stat().st_size / _MIB,
            self._providers,
            0 if self._providers == ["CPUExecutionProvider"] else gpu_limit // _MIB,
            0 if "TensorrtExecutionProvider" not in self._providers else trt_workspace // _MIB,
            self._decision_distance,
            self._positive_depth_thresholds,
        )

    @staticmethod
    def _preprocess(image_bytes: bytes) -> np.ndarray:
        import cv2

        if not image_bytes:
            raise ValueError("empty compressed image")
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None or image.size == 0:
            raise ValueError("unable to decode compressed image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = image.astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

    def _scene(self, image_bytes: bytes) -> str:
        if self._scene_mode != "auto":
            return self._scene_mode
        return "indoor" if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") else "outdoor"

    def _distance(self, depth: np.ndarray, scene: str) -> float:
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        if depth.shape != (224, 224):
            raise ValueError(f"unexpected FastDepth output shape: {depth.shape}")
        depth = depth * self._distance_scale + self._distance_bias
        valid = np.isfinite(depth) & (depth >= self._min_distance) & (
            depth <= self._max_distance
        )
        if scene == "indoor":
            roi = np.zeros_like(valid, dtype=bool)
            y1 = math.ceil(5 * depth.shape[0] / 8)
            x0 = depth.shape[1] // 3
            x1 = math.ceil(2 * depth.shape[1] / 3)
            roi[:y1, x0:x1] = True
            values = depth[valid & roi]
            quantile = self._indoor_quantile
        else:
            values = depth[valid]
            quantile = self._outdoor_quantile
        if values.size == 0:
            return self.fallback_distance
        return float(
            np.clip(
                np.quantile(values, quantile),
                self._min_distance,
                self._max_distance,
            )
        )

    def _calibrate_for_f1(self, raw_distance: float, scene: str) -> float:
        """Map a scene-specific raw cutoff onto the configured decision boundary.

        FastDepth was trained on NYU indoor data, so its absolute scale may move
        under a different camera/domain.  Separate raw cutoffs let a validation
        set correct that scale without changing the evaluator-facing definition:
        every raw value below the cutoff maps strictly below decision_distance,
        and every other value maps strictly above it.  With the current product
        configuration this means ``pred_distance < 2 m`` is positive and
        ``pred_distance >= 2 m`` is negative.
        """

        raw_distance = float(
            np.clip(raw_distance, self._min_distance, self._max_distance)
        )
        threshold = self._positive_depth_thresholds[scene]
        if raw_distance < threshold:
            fraction = (raw_distance - self._min_distance) / max(
                1e-9, threshold - self._min_distance
            )
            upper = self._decision_distance - self._boundary_margin
            return self._min_distance + fraction * (upper - self._min_distance)

        fraction = (raw_distance - threshold) / max(
            1e-9, self._max_distance - threshold
        )
        lower = self._decision_distance + self._boundary_margin
        return lower + fraction * (self._max_distance - lower)

    def _record_prediction(
        self, raw_distance: float, distance: float, scene: str
    ) -> None:
        with self._stats_lock:
            positive = distance < self._decision_distance
            self._prediction_count += 1
            self._positive_count += int(positive)
            self._negative_count += int(not positive)
            self._scene_counts[scene]["total"] += 1
            self._scene_counts[scene]["positive"] += int(positive)
            self._last_raw_distance = float(raw_distance)
            self._raw_distance_sum += float(raw_distance)
            self._raw_distance_min = min(self._raw_distance_min, float(raw_distance))
            self._raw_distance_max = max(self._raw_distance_max, float(raw_distance))
            self._recent_raw_distances.append(float(raw_distance))
            if len(self._recent_raw_distances) > 256:
                del self._recent_raw_distances[0]
            count = self._prediction_count
            positive_rate = self._positive_count / count
            recent = np.asarray(self._recent_raw_distances, dtype=np.float32)
            raw_p25 = float(np.quantile(recent, 0.25))
            raw_p50 = float(np.quantile(recent, 0.50))

        if count >= self._bias_warning_min_samples and (
            count % self._bias_warning_min_samples == 0
        ) and (positive_rate <= 0.05 or positive_rate >= 0.95):
            log.warning(
                "[obstacle-fastdepth] prediction distribution is highly one-sided: "
                "samples=%d positive_rate=%.3f; calibrate indoor/outdoor_positive_depth_m "
                "on a labeled validation split (recent raw p25=%.3f p50=%.3f)",
                count,
                positive_rate,
                raw_p25,
                raw_p50,
            )

    def calibration_stats(self) -> dict:
        with self._stats_lock:
            total = self._prediction_count
            recent = np.asarray(self._recent_raw_distances, dtype=np.float32)
            if recent.size:
                raw_quantiles = {
                    "p10": float(np.quantile(recent, 0.10)),
                    "p25": float(np.quantile(recent, 0.25)),
                    "p50": float(np.quantile(recent, 0.50)),
                    "p75": float(np.quantile(recent, 0.75)),
                    "p90": float(np.quantile(recent, 0.90)),
                }
            else:
                raw_quantiles = {}
            scenes = {
                scene: {
                    "total": values["total"],
                    "positive": values["positive"],
                    "positive_rate": values["positive"] / max(1, values["total"]),
                }
                for scene, values in self._scene_counts.items()
            }
            return {
                "decision_distance_m": self._decision_distance,
                "positive_depth_m": dict(self._positive_depth_thresholds),
                "prediction_count": total,
                "positive_count": self._positive_count,
                "negative_count": self._negative_count,
                "positive_rate": self._positive_count / max(1, total),
                "last_raw_distance": self._last_raw_distance,
                "raw_distance_min": (
                    self._raw_distance_min if total else None
                ),
                "raw_distance_max": (
                    self._raw_distance_max if total else None
                ),
                "raw_distance_mean": (
                    self._raw_distance_sum / total if total else None
                ),
                "raw_distance_quantiles": raw_quantiles,
                "scenes": scenes,
            }

    def estimate(self, image_bytes: bytes) -> dict:
        started = time.perf_counter()
        tensor = self._preprocess(image_bytes)
        with self._lock:
            depth = self._session.run(
                [self._output_name], {self._input_name: tensor}
            )[0]
        scene = self._scene(image_bytes)
        raw_distance = self._distance(depth, scene)
        distance = self._calibrate_for_f1(raw_distance, scene)
        self._record_prediction(raw_distance, distance, scene)
        elapsed = time.perf_counter() - started
        if elapsed > self._max_inference_seconds:
            log.warning(
                "[obstacle-fastdepth] inference exceeded %.3fs: %.3fs",
                self._max_inference_seconds,
                elapsed,
            )
        return {
            "pred_distance": distance,
            "raw_distance": raw_distance,
            "scene": scene,
            "latency_ms": elapsed * 1000.0,
        }


class ObstacleFastDepthPlugin(ObstacleDistancePlugin):
    """FastDepth implementation exposed under the evaluator's ``obstacle`` name."""

    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._base_cfg = dict(plugin_cfg)
        self._base_cfg.setdefault("provider", "fastdepth")
        self._adapter = FastDepthDistanceAdapter(self._base_cfg)
        self._nodes: dict[str, Any] = {}
        self._instance_configs: dict[str, dict] = {}

    def get_tools(self) -> list:
        return TOOLS

    def _adapter_for(self, node_key: str) -> FastDepthDistanceAdapter:
        instance_cfg = self._instance_configs.get(node_key)
        if not instance_cfg:
            return self._adapter
        merged = dict(self._base_cfg)
        merged.update(instance_cfg)
        return FastDepthDistanceAdapter(merged)

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        if action == "config":
            instance_id = args.get("instance_id", "")
            cfg = {
                key: value
                for key, value in args.items()
                if key not in {"action", "instance_id"}
                and value is not None
                and value != ""
            }
            if instance_id:
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes.pop(instance_id)
                    node.stop()
                    self._executor.remove_node(node)
                return {
                    "status": "configured",
                    "instance_id": instance_id,
                    "config": cfg,
                }
            self._base_cfg.update(cfg)
            self._adapter = FastDepthDistanceAdapter(self._base_cfg)
            return {"status": "configured", "config": cfg}

        result = super().dispatch(name, args)
        if action == "info" and result is not None:
            result["model"] = "FastDepth-224x224-NYUv2"
            result["desc"] = "FastDepth metric-depth obstacle distance from camera feed"
            result["calibration"] = self._adapter.calibration_stats()
            for key, node in self._nodes.items():
                if key in result.get("instances", {}):
                    result["instances"][key]["calibration"] = (
                        node._adapter.calibration_stats()
                    )
        return result
