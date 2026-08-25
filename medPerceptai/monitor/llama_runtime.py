"""
Llama / llama-cpp hardware detection — lazy import only (no startup requirement).
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:
    torch = None  # type: ignore

_LLAMA_IMPORT_TRIED = False
_LLAMA_IMPORT_OK = False
_LlamaClass: Any = None


def is_llama_cpp_installed() -> bool:
    """Check package presence without importing (safe at Django startup)."""
    return importlib.util.find_spec("llama_cpp") is not None


def ensure_llama_cpp_imported() -> bool:
    """Import llama_cpp on first reasoning attempt; never at module import."""
    global _LLAMA_IMPORT_TRIED, _LLAMA_IMPORT_OK, _LlamaClass
    if _LLAMA_IMPORT_TRIED:
        return _LLAMA_IMPORT_OK
    _LLAMA_IMPORT_TRIED = True
    try:
        from llama_cpp import Llama

        _LlamaClass = Llama
        _LLAMA_IMPORT_OK = True
    except Exception as exc:
        _LLAMA_IMPORT_OK = False
        _LlamaClass = None
        logger.info(
            "[LLaMA_DEVICE] llama-cpp-python not installed, reasoning fallback active"
        )
        logger.debug("llama_cpp import failed: %s", exc)
    return _LLAMA_IMPORT_OK


def get_llama_class() -> Any:
    if ensure_llama_cpp_imported():
        return _LlamaClass
    return None


def llama_cpp_available() -> bool:
    """True after a successful lazy import (not used at Django startup)."""
    return _LLAMA_IMPORT_OK


def _llama_gpu_offload_supported() -> bool:
    if not ensure_llama_cpp_imported():
        return False
    fn = getattr(__import__("llama_cpp"), "llama_supports_gpu_offload", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


@dataclass
class ReasoningHardwareProfile:
    cuda_available: bool = False
    cuda_device_name: str = ""
    gpu_vram_mb: Optional[int] = None
    llama_cpp_installed: bool = False
    llama_gpu_offload_supported: bool = False
    n_gpu_layers: int = 0
    backend: str = "cpu"  # cpu | gpu | unavailable
    selection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "gpu_vram_mb": self.gpu_vram_mb,
            "llama_cpp_installed": self.llama_cpp_installed,
            "llama_gpu_offload_supported": self.llama_gpu_offload_supported,
            "n_gpu_layers": self.n_gpu_layers,
            "backend": self.backend,
            "selection_reason": self.selection_reason,
        }


def _cuda_info() -> tuple[bool, str, Optional[int]]:
    if torch is None or not getattr(torch, "cuda", None):
        return False, "", None
    try:
        if not torch.cuda.is_available():
            return False, "", None
        name = torch.cuda.get_device_name(0) or "CUDA GPU"
        vram_mb = None
        try:
            props = torch.cuda.get_device_properties(0)
            vram_mb = int(props.total_memory // (1024 * 1024))
        except Exception:
            pass
        return True, name, vram_mb
    except Exception as exc:
        logger.debug("CUDA detection failed: %s", exc)
        return False, "", None


def _use_cuda_enabled() -> bool:
    raw = os.environ.get("USE_CUDA", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _llama_device_env() -> str:
    return (os.environ.get("LLAMA_DEVICE", "auto").strip().lower() or "auto")


def _llama_wants_cuda() -> bool:
    device = _llama_device_env()
    if device == "cpu":
        return False
    return _use_cuda_enabled()


def is_llama_cuda_error(exc: BaseException) -> bool:
    """True when an exception looks like a llama-cpp CUDA/ggml GPU failure."""
    return is_llama_cuda_error_text(f"{type(exc).__name__}: {exc}")


def is_llama_cuda_error_text(text: str) -> bool:
    lower = str(text or "").lower()
    markers = (
        "cuda error",
        "ggml-cuda",
        "ggml_cuda",
        "ggml cuda",
        "cublas",
        "cudnn",
        "device-side assert",
    )
    return any(marker in lower for marker in markers)


def _parse_requested_gpu_layers() -> Tuple[Optional[int], str]:
    """Resolve layer count from LLAMA_N_GPU_LAYERS / LLAMA_N_GPU_LAYERS_AUTO."""
    raw = os.environ.get("LLAMA_N_GPU_LAYERS", "auto").strip().lower()
    if raw in ("", "auto"):
        auto_raw = os.environ.get("LLAMA_N_GPU_LAYERS_AUTO", "20").strip()
        try:
            return int(auto_raw), "LLAMA_N_GPU_LAYERS_AUTO"
        except ValueError:
            return 20, "LLAMA_N_GPU_LAYERS_AUTO(default)"
    try:
        return int(raw), "LLAMA_N_GPU_LAYERS"
    except ValueError:
        return None, "auto"


def _python_env_label() -> str:
    exe = Path(sys.executable)
    parts = exe.parts
    for index, part in enumerate(parts):
        if part.lower() in {"env", "env311", "venv", ".venv"} and index + 1 < len(parts):
            if parts[index + 1].lower() in {"scripts", "bin"}:
                return part
    return str(exe)


def _llama_device_mode(n_gpu_layers: int) -> str:
    return "cuda" if int(n_gpu_layers) != 0 else "cpu"


def resolve_llm_backend_mode(profile: ReasoningHardwareProfile) -> str:
    """cpu | cuda | unavailable — for startup logs without importing llama_cpp."""
    if not profile.llama_cpp_installed:
        return "unavailable"
    if profile.n_gpu_layers != 0 and profile.backend == "gpu":
        return "cuda"
    return "cpu"


def log_llama_device(profile: ReasoningHardwareProfile) -> None:
    mode = resolve_llm_backend_mode(profile)
    logger.info(
        "[LLaMA_DEVICE] mode=%s n_gpu_layers=%s",
        mode,
        profile.n_gpu_layers,
    )


def log_runtime_environment(
    *,
    yolo_object: str,
    yolo_role: str,
    yolo_pose: str,
) -> None:
    logger.info("[PYTHON_ENV] %s", _python_env_label())
    cuda_ok, device_name, _ = _cuda_info()
    logger.info("[CUDA] torch.cuda.is_available=%s", cuda_ok)
    if cuda_ok and device_name:
        logger.info("[CUDA] device_name=%s", device_name)
    logger.info("[YOLO_DEVICE] object=%s", yolo_object)
    logger.info("[YOLO_DEVICE] role=%s", yolo_role)
    logger.info("[YOLO_DEVICE] pose=%s", yolo_pose)
    profile = detect_hardware(probe_llama=False)
    log_llama_device(profile)
    logger.info(
        "[LLM_BACKEND] llama_cpp_available=%s llama_gpu_offload=%s",
        profile.llama_cpp_installed,
        profile.llama_gpu_offload_supported,
    )


def _apply_gpu_layer_request(
    profile: ReasoningHardwareProfile,
    requested: int,
    *,
    source: str,
    cuda_available: bool,
    llama_gpu_offload_supported: bool,
    strict_probe: bool,
) -> None:
    if requested == -1:
        can_use_gpu = cuda_available and (
            llama_gpu_offload_supported or not strict_probe
        )
        if can_use_gpu:
            profile.n_gpu_layers = -1
            profile.backend = "gpu"
            suffix = "" if llama_gpu_offload_supported else " (planned — verified at load)"
            profile.selection_reason = f"{source}=-1 (all layers on GPU){suffix}"
            return
        else:
            profile.n_gpu_layers = 0
            profile.backend = "cpu"
            profile.selection_reason = f"{source}=-1 but GPU offload unavailable; using CPU"
            logger.info(
                "[LLaMA_DEVICE] %s=-1 requested but GPU offload unavailable "
                "(cuda=%s llama_gpu_offload=%s) — using CPU",
                source,
                cuda_available,
                llama_gpu_offload_supported,
            )
        return

    if requested > 0:
        can_use_gpu = cuda_available and (
            llama_gpu_offload_supported or not strict_probe
        )
        if can_use_gpu:
            profile.n_gpu_layers = requested
            profile.backend = "gpu"
            suffix = "" if llama_gpu_offload_supported else " (planned — verified at load)"
            profile.selection_reason = f"{source}={requested}{suffix}"
            return
        profile.n_gpu_layers = 0
        profile.backend = "cpu"
        profile.selection_reason = (
            f"{source}={requested} but GPU offload unavailable; using CPU"
        )
        logger.info(
            "[LLaMA_DEVICE] %s=%s requested but GPU offload unavailable "
            "(cuda=%s llama_gpu_offload=%s) — using CPU",
            source,
            requested,
            cuda_available,
            llama_gpu_offload_supported,
        )
        return

    profile.n_gpu_layers = 0
    profile.backend = "cpu"
    profile.selection_reason = f"{source}=0 — LLM CPU only"


def resolve_n_gpu_layers(
    *,
    cuda_available: bool,
    llama_gpu_offload_supported: bool,
    gpu_vram_mb: Optional[int],
    llama_cpp_installed: bool,
    strict_probe: bool = True,
) -> ReasoningHardwareProfile:
    """Pick GPU layer count from env; fallback to CPU when offload unavailable."""
    profile = ReasoningHardwareProfile(
        cuda_available=cuda_available,
        llama_cpp_installed=llama_cpp_installed,
        llama_gpu_offload_supported=llama_gpu_offload_supported,
    )
    if cuda_available:
        _, profile.cuda_device_name, profile.gpu_vram_mb = _cuda_info()

    if not _use_cuda_enabled() or _llama_device_env() == "cpu":
        profile.n_gpu_layers = 0
        profile.backend = "cpu"
        profile.selection_reason = (
            "LLAMA_DEVICE=cpu" if _llama_device_env() == "cpu" else "USE_CUDA=0 — CPU inference"
        )
        return profile

    requested, source = _parse_requested_gpu_layers()
    if requested is not None:
        _apply_gpu_layer_request(
            profile,
            requested,
            source=source,
            cuda_available=cuda_available,
            llama_gpu_offload_supported=llama_gpu_offload_supported,
            strict_probe=strict_probe,
        )
        return profile

    if cuda_available and (llama_gpu_offload_supported or not strict_probe):
        auto_layers, auto_source = _parse_requested_gpu_layers()
        layers = auto_layers if auto_layers is not None else 20
        profile.n_gpu_layers = layers
        profile.backend = "gpu"
        profile.selection_reason = f"{auto_source}={layers} (LLAMA_DEVICE=auto)"
        if strict_probe and not llama_gpu_offload_supported:
            profile.n_gpu_layers = 0
            profile.backend = "cpu"
            profile.selection_reason = "LLAMA_DEVICE=auto but GPU offload unavailable; using CPU"
            logger.info(
                "[LLaMA_DEVICE] LLAMA_DEVICE=auto but GPU offload unavailable — using CPU"
            )
        return profile

    profile.n_gpu_layers = 0
    profile.backend = "cpu"
    if not llama_cpp_installed:
        profile.backend = "unavailable"
        profile.selection_reason = "llama-cpp-python not installed"
    elif not cuda_available:
        profile.selection_reason = "LLAMA_DEVICE=auto but no CUDA — CPU inference"
    else:
        profile.selection_reason = "LLAMA_DEVICE=auto but GPU offload unavailable — CPU inference"
        logger.info(
            "[LLaMA_DEVICE] llama-cpp GPU offload unavailable — using CPU inference "
            "(install CUDA prebuilt wheel; do not build from source)"
        )
    return profile


def detect_hardware(*, probe_llama: bool = False) -> ReasoningHardwareProfile:
    cuda_ok, _, vram = _cuda_info()
    installed = is_llama_cpp_installed()
    llama_gpu = False
    if probe_llama and installed:
        llama_gpu = _llama_gpu_offload_supported()
    return resolve_n_gpu_layers(
        cuda_available=cuda_ok,
        llama_gpu_offload_supported=llama_gpu,
        gpu_vram_mb=vram,
        llama_cpp_installed=installed,
        strict_probe=probe_llama,
    )


def resolve_gguf_path(model_path: Path) -> Optional[Path]:
    if not model_path.exists():
        return None
    if model_path.is_dir():
        gguf_files = sorted(model_path.glob("*.gguf"))
        return gguf_files[0] if gguf_files else None
    return model_path


def build_health(
    *,
    model_path: Path,
    profile: ReasoningHardwareProfile,
    model_loaded: bool = False,
    load_error: Optional[str] = None,
    loading: bool = False,
) -> Dict[str, Any]:
    gguf = resolve_gguf_path(model_path)
    status = "loading" if loading else ("ready" if model_loaded else "pending")
    if not profile.llama_cpp_installed:
        status = "unavailable"
        load_error = load_error or "llama-cpp-python not installed"
    elif gguf is None:
        status = "unavailable"
        load_error = load_error or f"GGUF not found at {model_path}"
    elif load_error:
        status = "error"

    display_mode = "llama_unavailable"
    if loading:
        display_mode = "loading"
    elif model_loaded:
        display_mode = "llama"

    return {
        "status": status,
        "display_mode": display_mode,
        "model_loaded": model_loaded,
        "loading": loading,
        "gguf_path": str(gguf) if gguf else None,
        "load_error": load_error,
        "hardware": profile.to_dict(),
    }


def _env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def llama_use_mmap(model_path: Path) -> bool:
    """
    Windows and cloud-synced paths (OneDrive) often raise WinError 1 with mmap.
    Default mmap off on Windows; override with LLAMA_USE_MMAP=1.
    """
    override = _env_flag("LLAMA_USE_MMAP")
    if override is not None:
        return override
    if sys.platform == "win32":
        path_text = str(model_path).lower()
        if "onedrive" in path_text:
            logger.info(
                "[LLAMA] OneDrive model path detected — use_mmap=False (avoids WinError 1)"
            )
        return False
    return True


def llama_load_attempts(
    *,
    profile: ReasoningHardwareProfile,
    model_path: Path,
) -> List[Tuple[int, bool]]:
    """Ordered (n_gpu_layers, use_mmap) attempts for resilient GGUF loading."""
    layer_values: List[int] = []
    if profile.n_gpu_layers != 0 and profile.llama_gpu_offload_supported:
        layer_values.append(profile.n_gpu_layers)
    elif profile.n_gpu_layers != 0 and not profile.llama_gpu_offload_supported:
        logger.info(
            "[LLaMA_DEVICE] Skipping GPU load attempts — llama GPU offload not available; using CPU only"
        )
    if 0 not in layer_values:
        layer_values.append(0)

    mmap_values: List[bool] = []
    preferred_mmap = llama_use_mmap(model_path)
    mmap_values.append(preferred_mmap)
    if preferred_mmap:
        mmap_values.append(False)

    attempts: List[Tuple[int, bool]] = []
    seen: set[Tuple[int, bool]] = set()
    for layers in layer_values:
        for use_mmap in mmap_values:
            key = (layers, use_mmap)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(key)
    return attempts


def log_startup_verification(model_path: Path, health: Dict[str, Any]) -> None:
    hw = health.get("hardware") or {}
    logger.info("=== Llama reasoning startup verification ===")
    logger.info("GGUF path configured: %s", model_path)
    logger.info("GGUF resolved: %s", health.get("gguf_path"))
    logger.info("llama_cpp installed: %s", hw.get("llama_cpp_installed"))
    logger.info("CUDA detected: %s (%s)", hw.get("cuda_available"), hw.get("cuda_device_name") or "n/a")
    logger.info("GPU VRAM MB: %s", hw.get("gpu_vram_mb"))
    logger.info("llama GPU offload supported: %s", hw.get("llama_gpu_offload_supported"))
    logger.info("n_gpu_layers selected: %s", hw.get("n_gpu_layers"))
    logger.info("backend: %s — %s", hw.get("backend"), hw.get("selection_reason"))
    logger.info("model loaded: %s", health.get("model_loaded"))
    logger.info("reasoning display mode: %s", health.get("display_mode"))
    if health.get("load_error"):
        logger.info("Llama pending: %s", health.get("load_error"))
    log_llama_device(
        ReasoningHardwareProfile(
            cuda_available=bool(hw.get("cuda_available")),
            cuda_device_name=str(hw.get("cuda_device_name") or ""),
            gpu_vram_mb=hw.get("gpu_vram_mb"),
            llama_cpp_installed=bool(hw.get("llama_cpp_installed")),
            llama_gpu_offload_supported=bool(hw.get("llama_gpu_offload_supported")),
            n_gpu_layers=int(hw.get("n_gpu_layers") or 0),
            backend=str(hw.get("backend") or "cpu"),
            selection_reason=str(hw.get("selection_reason") or ""),
        )
    )
    logger.info("=== end Llama verification ===")
