"""Sweep spec loading, validation, and cell expansion.

Pure module: no torch, no model code. A *cell* is one runner invocation:
one model x one mode x one kv-parameter point x one repetition.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class SpecError(ValueError):
    """Raised when a sweep spec file is invalid."""


_ALLOWED_MODES = ("baseline", "kvcomm")

_DEFAULTS: Dict[str, Any] = {
    "modes": list(_ALLOWED_MODES),
    "repetitions": 3,
    "seeds": None,  # derived from repetitions when absent
    "tasks_path": "datasets/secure_code/coding_tasks.jsonl",
    "limit": None,
    "max_tokens": 512,
    "kv_params": {
        "kv_threshold": [0.3],
        "kv_max_anchor_num": [20],
        "kv_window_size": [5],
    },
    "cell_timeout_minutes": 120,
    "output_root": "results",
    # Record a SECOND measurement-only dense baseline on every anchor hit, so the
    # dense-vs-dense noise floor is measured rather than assumed. Costs one extra
    # dense generation per hit.
    "noise_floor_baseline": False,
    # Per task: dense reference first, then the natural pass.
    "dense_reference_always": False,
    # Generator never uses KV reuse: the optimization under study is the
    # validator's alone, and the code under review stays identical.
    "generator_dense": False,
    # Roles that decode with sampling (comma-separated) and their temperature.
    # Empty = fully greedy, which makes seeds inert and runs byte-identical.
    "sample_roles": "",
    "sample_temperature": 0.7,
}

_REQUIRED_KEYS = ("sweep_name", "models")
_ALLOWED_KEYS = set(_REQUIRED_KEYS) | set(_DEFAULTS)
_KV_PARAM_KEYS = ("kv_threshold", "kv_max_anchor_num", "kv_window_size")


@dataclass(frozen=True)
class Cell:
    model: str
    mode: str
    kv_point: Dict[str, Any]
    rep_index: int
    seed: int

    @property
    def model_slug(self) -> str:
        return self.model.replace("/", "_")

    @property
    def kv_slug(self) -> str:
        return (
            f"kv_t{self.kv_point['kv_threshold']}"
            f"_a{self.kv_point['kv_max_anchor_num']}"
            f"_w{self.kv_point['kv_window_size']}"
        )

    @property
    def cell_dir(self) -> str:
        return f"{self.model_slug}/{self.mode}/{self.kv_slug}/rep_{self.rep_index}"

    @property
    def cell_id(self) -> str:
        return self.cell_dir


@dataclass
class SweepSpec:
    sweep_name: str
    models: List[str]
    modes: List[str]
    repetitions: int
    seeds: List[int]
    tasks_path: str
    limit: Optional[int]
    max_tokens: int
    kv_params: Dict[str, List[Any]]
    cell_timeout_minutes: int
    output_root: str
    noise_floor_baseline: bool = False
    dense_reference_always: bool = False
    generator_dense: bool = False
    sample_roles: str = ""
    sample_temperature: float = 0.7

    @property
    def sweep_dir(self) -> Path:
        return Path(self.output_root) / self.sweep_name


def load_sweep_spec(path: Union[str, Path]) -> SweepSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SpecError(f"Spec file {path} must contain a YAML mapping.")

    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise SpecError(f"Spec contains unknown keys: {sorted(unknown)}")
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise SpecError(f"Spec is missing required keys: {missing}")

    merged = {**_DEFAULTS, **raw}

    models = merged["models"]
    if not isinstance(models, list) or not models:
        raise SpecError("'models' must be a non-empty list of model names.")

    modes = merged["modes"]
    bad_modes = [m for m in modes if m not in _ALLOWED_MODES]
    if bad_modes:
        raise SpecError(f"'modes' contains invalid entries {bad_modes}; allowed: {list(_ALLOWED_MODES)}")

    repetitions = int(merged["repetitions"])
    seeds = merged["seeds"]
    if seeds is None:
        seeds = [42 + i for i in range(repetitions)]
    if len(seeds) != repetitions:
        raise SpecError(f"'seeds' must have exactly 'repetitions' ({repetitions}) entries, got {len(seeds)}.")

    kv_params = dict(_DEFAULTS["kv_params"])
    kv_params.update(merged["kv_params"] or {})
    bad_kv = set(kv_params) - set(_KV_PARAM_KEYS)
    if bad_kv:
        raise SpecError(f"'kv_params' contains unknown keys: {sorted(bad_kv)}")
    for key, values in kv_params.items():
        if not isinstance(values, list) or not values:
            raise SpecError(f"'kv_params.{key}' must be a non-empty list.")

    return SweepSpec(
        sweep_name=str(merged["sweep_name"]),
        models=[str(m) for m in models],
        modes=list(modes),
        repetitions=repetitions,
        seeds=[int(s) for s in seeds],
        tasks_path=str(merged["tasks_path"]),
        limit=None if merged["limit"] is None else int(merged["limit"]),
        max_tokens=int(merged["max_tokens"]),
        kv_params=kv_params,
        cell_timeout_minutes=int(merged["cell_timeout_minutes"]),
        output_root=str(merged["output_root"]),
        noise_floor_baseline=bool(merged["noise_floor_baseline"]),
        dense_reference_always=bool(merged["dense_reference_always"]),
        generator_dense=bool(merged["generator_dense"]),
        sample_roles=str(merged["sample_roles"]),
        sample_temperature=float(merged["sample_temperature"]),
    )


def expand_cells(spec: SweepSpec) -> List[Cell]:
    """models x modes x cartesian(kv_params) x repetitions, grouped by model."""
    kv_points = [
        dict(zip(_KV_PARAM_KEYS, combo))
        for combo in itertools.product(*(spec.kv_params[k] for k in _KV_PARAM_KEYS))
    ]
    cells: List[Cell] = []
    for model in spec.models:
        for mode in spec.modes:
            for kv_point in kv_points:
                for rep_index in range(spec.repetitions):
                    cells.append(Cell(
                        model=model, mode=mode, kv_point=kv_point,
                        rep_index=rep_index, seed=spec.seeds[rep_index],
                    ))
    return cells
