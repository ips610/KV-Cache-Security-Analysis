"""Per-pass GPU/timing/energy metrics and environment snapshots.

Additive instrumentation for experiment runners. Everything degrades
gracefully on CPU-only machines: torch.cuda guarded, pynvml optional.
No KV-cache or model logic lives here.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_VERSION = 1


class EnergySampler:
    """Samples GPU power via NVML on a background thread.

    If pynvml is missing or NVML init fails, the sampler is inert and
    reports None values.
    """

    def __init__(self, interval_s: float = 0.1, device_index: int = 0):
        self._interval = interval_s
        self._device_index = device_index
        self._samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pynvml = None
        self._handle = None

    def start(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except Exception:
            self._pynvml = None
            self._handle = None
            return
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                milliwatts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append(milliwatts / 1000.0)
            except Exception:
                break
            self._stop.wait(self._interval)

    def stop(self, duration_s: float) -> Dict[str, Optional[float]]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        if not self._samples:
            return {"avg_power_w": None, "energy_joules": None}
        avg_power = sum(self._samples) / len(self._samples)
        return {"avg_power_w": avg_power, "energy_joules": avg_power * duration_s}


def estimate_kv_bytes_per_token(llm_name: str) -> Optional[int]:
    """KV-cache bytes per token: 2 (K+V) * layers * kv_heads * head_dim * 2 (bf16).

    Uses the locally cached HF config; returns None when unavailable.
    """
    try:
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(llm_name, local_files_only=True)
        except Exception:
            config = AutoConfig.from_pretrained(llm_name)
        num_layers = int(config.num_hidden_layers)
        num_kv_heads = int(
            getattr(config, "num_key_value_heads", None)
            or config.num_attention_heads
        )
        head_dim = int(
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        return 2 * num_layers * num_kv_heads * head_dim * 2
    except Exception:
        return None


class MetricsCollector:
    def __init__(
        self,
        metrics_path: Union[str, Path],
        *,
        llm_name: str,
        mode: str,
        enable_energy: bool = True,
    ):
        self.metrics_path = Path(metrics_path)
        self.llm_name = llm_name
        self.mode = mode
        self.passes: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None
        self._sampler = EnergySampler() if enable_energy else None

    def start_pass(self, pass_name: str) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] WARNING: could not reset GPU peak-memory stats: {exc}")
        self._current = {"pass_name": pass_name, "start": perf_counter()}
        if self._sampler is not None:
            self._sampler.start()

    def end_pass(self, task_count: int) -> None:
        if self._current is None:
            raise RuntimeError("end_pass() called without a matching start_pass().")
        duration = perf_counter() - self._current["start"]
        peak_mem: Optional[int] = 0
        try:
            import torch

            if torch.cuda.is_available():
                peak_mem = int(torch.cuda.max_memory_allocated())
        except Exception as exc:  # noqa: BLE001
            peak_mem = None
            print(f"[metrics] WARNING: could not read GPU peak memory: {exc}")
        entry: Dict[str, Any] = {
            "pass_name": self._current["pass_name"],
            "duration_s": duration,
            "gpu_peak_mem_bytes": peak_mem,
            "task_count": task_count,
            "tasks_per_minute": (task_count / (duration / 60.0)) if duration > 0 else None,
        }
        if self._sampler is not None:
            entry.update(self._sampler.stop(duration))
        else:
            entry.update({"avg_power_w": None, "energy_joules": None})
        self.passes.append(entry)
        self._current = None

    def write(self, extra: Optional[Dict[str, Any]] = None) -> Path:
        payload: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "llm_name": self.llm_name,
            "mode": self.mode,
            "kv_bytes_per_token": estimate_kv_bytes_per_token(self.llm_name),
            "passes": self.passes,
        }
        if extra:
            payload.update(extra)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.metrics_path


def collect_env(cli_args: Dict[str, Any]) -> Dict[str, Any]:
    gpu_name = None
    cuda_version = None
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    transformers_version = None
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        pass
    git_commit = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        pass
    return {
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "git_commit": git_commit,
        "seed": os.getenv("SEED"),
        "cli_args": cli_args,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_env(path: Union[str, Path], cli_args: Dict[str, Any]) -> Path:
    payload = {"schema_version": _SCHEMA_VERSION, **collect_env(cli_args)}
    env_path = Path(path)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return env_path
