"""
utils/model_downloader.py — Auto-download sherpa-onnx models from COS if missing.
"""

from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import zipfile
from urllib.request import urlretrieve

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"
JUICEFS_BASE = "http://172.28.4.81:34567/chenhanqing/obstacle"

MODELS = {
    "asr": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-paraformer-bilingual-zh-en.zip",
        "check_file": "tokens.txt",
    },
    "tts": {
        "url": f"{COS_BASE}/matcha-icefall-zh-en.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    "tts_vocoder": {
        "url": f"{COS_BASE}/vocos-16khz-univ.onnx",
        "check_file": "vocos-16khz-univ.onnx",
        "single_file": True,
    },
    "kws": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2",
        "check_file": "tokens.txt",
    },
    "vad": {
        "url": f"{COS_BASE}/silero_vad.onnx",
        "check_file": "silero_vad.onnx",
        "single_file": True,  # Not an archive, just a single file download
    },
    "obstacle": {
        "url": f"{JUICEFS_BASE}/obstacle-distance.tar.bz2",
        "check_file": "yolo26n-depth.onnx",
    },}


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download from COS if missing."""
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_path = os.path.join(model_dir, info["check_file"])
    if os.path.exists(check_path):
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        return

    url = info["url"]
    os.makedirs(model_dir, exist_ok=True)
    log.info(f"[model_downloader] {name}: downloading from {url} ...")

    if info.get("single_file"):
        # Direct file download (not an archive)
        dest = os.path.join(model_dir, info["check_file"])
        urlretrieve(url, dest)
        log.info(f"[model_downloader] {name}: done.")
        return

    # Determine suffix from URL
    if url.endswith(".zip"):
        suffix = ".zip"
    else:
        suffix = ".tar.bz2"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path)
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")

        if suffix == ".zip":
            _extract_zip(tmp_path, model_dir)
        else:
            _extract_tar(tmp_path, model_dir)

        log.info(f"[model_downloader] {name}: done.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Verify
    if not os.path.exists(check_path):
        raise RuntimeError(
            f"[model_downloader] {name}: download completed but {info['check_file']} "
            f"not found in {model_dir}"
        )


def _extract_zip(zip_path: str, model_dir: str) -> None:
    """Extract zip, stripping common top-level directory prefix."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Filter out __MACOSX and directory entries
        names = [n for n in zf.namelist()
                 if not n.endswith('/') and not n.startswith('__MACOSX')]
        if not names:
            raise RuntimeError(f"Empty archive: {zip_path}")

        prefix = _common_prefix_from_names(names)
        for name in names:
            stripped = name[len(prefix):] if prefix else name
            if not stripped:
                continue
            dest = os.path.join(model_dir, stripped)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())


def _extract_tar(tar_path: str, model_dir: str) -> None:
    """Extract tar.bz2, stripping common top-level directory prefix."""
    with tarfile.open(tar_path, "r:bz2") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"Empty archive: {tar_path}")

        names = [m.name for m in members if not m.isdir()]
        prefix = _common_prefix_from_names(names)
        for m in members:
            if m.isdir():
                continue
            if prefix:
                m.name = m.name[len(prefix):]
            if not m.name:
                continue
            m.name = m.name.lstrip("/")
            tf.extract(m, model_dir)


def _common_prefix_from_names(names: list[str]) -> str:
    """Find common top-level directory prefix from file name list."""
    dirs_with_slash = [n.split("/", 1) for n in names if "/" in n]
    if not dirs_with_slash:
        return ""
    first_parts = set(parts[0] for parts in dirs_with_slash)
    if len(first_parts) == 1:
        return first_parts.pop() + "/"
    return ""
