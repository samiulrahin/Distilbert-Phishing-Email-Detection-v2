from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def capture() -> dict[str, object]:
    details: dict[str, object] = {
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "process_architecture": platform.architecture()[0],
        "environment_controls": {
            "random_seed": 42,
            "tokenizer_max_length": 256,
            "loopback_api": "127.0.0.1:8765",
        },
    }
    try:
        import torch

        details["torch"] = torch.__version__
        details["cuda_available"] = torch.cuda.is_available()
        details["cuda_version"] = torch.version.cuda
        details["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            details["gpu_memory_gb"] = round(properties.total_memory / (1024**3), 2)
    except ImportError:
        details["torch"] = None
        details["cuda_available"] = False

    return details


if __name__ == "__main__":
    output = PROJECT_ROOT / "evaluation" / "results" / "machine_environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = capture()
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))

