"""Best-effort hardware/runtime capability probing for training preflight."""

from __future__ import annotations

from typing import Any, Dict, List


def _probe_gpu_devices(torch_module: Any, gpu_count: int, notes: List[str]) -> List[Dict[str, Any]]:
    """Probe GPU names and compute capability when CUDA APIs are available."""
    devices: List[Dict[str, Any]] = []
    get_name = getattr(torch_module.cuda, "get_device_name", None)
    get_capability = getattr(torch_module.cuda, "get_device_capability", None)

    for index in range(gpu_count):
        item: Dict[str, Any] = {"index": index, "name": None, "compute_capability": None}

        if callable(get_name):
            try:
                item["name"] = str(get_name(index))
            except Exception as exc:  # pragma: no cover - defensive branch
                notes.append(f"failed to read GPU name for index={index}: {exc}")

        if callable(get_capability):
            try:
                major, minor = get_capability(index)
                item["compute_capability"] = f"{major}.{minor}"
            except Exception as exc:  # pragma: no cover - defensive branch
                notes.append(f"failed to read GPU capability for index={index}: {exc}")

        devices.append(item)

    return devices


def _build_diagnostics_summary(capabilities: Dict[str, Any]) -> str:
    if not capabilities.get("torch_installed"):
        return "torch is not installed"
    if not capabilities.get("cuda_available"):
        return "torch installed, CUDA unavailable"
    return (
        f"torch+CUDA detected (gpus={capabilities.get('gpu_count', 0)}, "
        f"bf16={capabilities.get('bf16_supported', False)}, fp16={capabilities.get('fp16_supported', False)})"
    )


def probe_training_capabilities() -> Dict[str, Any]:
    """Return a structured runtime capability snapshot without hard dependencies."""
    capabilities: Dict[str, Any] = {
        "torch_installed": False,
        "cuda_available": False,
        "gpu_count": 0,
        "bf16_supported": False,
        "fp16_supported": False,
        "cuda_version": None,
        "gpu_devices": [],
        "diagnostics": "",
        "notes": [],
    }

    try:
        import torch  # type: ignore

        capabilities["torch_installed"] = True

        cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())
        capabilities["cuda_available"] = cuda_available

        gpu_count = int(getattr(torch.cuda, "device_count", lambda: 0)() or 0)
        capabilities["gpu_count"] = gpu_count

        capabilities["cuda_version"] = getattr(torch.version, "cuda", None)

        if cuda_available:
            # fp16 is generally supported on CUDA GPUs for modern training stacks.
            capabilities["fp16_supported"] = True
            capabilities["gpu_devices"] = _probe_gpu_devices(torch, gpu_count, capabilities["notes"])

            bf16_supported_fn = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(bf16_supported_fn):
                capabilities["bf16_supported"] = bool(bf16_supported_fn())
            else:
                capabilities["notes"].append("torch.cuda.is_bf16_supported is unavailable; bf16 support unknown.")
        else:
            capabilities["notes"].append("CUDA is not available.")

    except ImportError:
        capabilities["notes"].append("torch is not installed.")
    except Exception as exc:  # pragma: no cover - defensive branch
        capabilities["notes"].append(f"hardware probe warning: {exc}")

    capabilities["diagnostics"] = _build_diagnostics_summary(capabilities)
    return capabilities

