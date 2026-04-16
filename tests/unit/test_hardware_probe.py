from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llama_vllm.utils.hardware import probe_training_capabilities


def test_probe_training_capabilities_has_expected_keys() -> None:
    data = probe_training_capabilities()
    expected = {
        "torch_installed",
        "cuda_available",
        "gpu_count",
        "bf16_supported",
        "fp16_supported",
        "cuda_version",
        "gpu_devices",
        "diagnostics",
        "notes",
    }
    assert expected.issubset(data.keys())
    assert isinstance(data["notes"], list)


def test_probe_training_capabilities_with_mocked_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        is_bf16_supported=lambda: True,
        get_device_name=lambda _idx: "Fake GPU",
        get_device_capability=lambda _idx: (8, 0),
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda, version=types.SimpleNamespace(cuda="12.1"))

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    data = probe_training_capabilities()
    assert data["torch_installed"] is True
    assert data["cuda_available"] is True
    assert data["gpu_count"] == 1
    assert data["bf16_supported"] is True
    assert data["fp16_supported"] is True
    assert data["gpu_devices"][0]["name"] == "Fake GPU"
    assert data["gpu_devices"][0]["compute_capability"] == "8.0"


