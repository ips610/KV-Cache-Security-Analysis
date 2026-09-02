"""Resumable cell-state tracking for sweeps.

The manifest is the single source of truth for which cells already ran.
Every mutation is written to disk immediately so a killed sweep loses at
most the in-flight cell (which is detected as stale on the next load).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from experiments.harness.spec import Cell

_SCHEMA_VERSION = 1


class Manifest:
    def __init__(self, sweep_dir: Path, cells: Dict[str, Dict[str, Any]]):
        self._sweep_dir = Path(sweep_dir)
        self._cells = cells

    @property
    def path(self) -> Path:
        return self._sweep_dir / "manifest.json"

    @classmethod
    def load(cls, sweep_dir: Path) -> "Manifest":
        sweep_dir = Path(sweep_dir)
        manifest_path = sweep_dir / "manifest.json"
        cells: Dict[str, Dict[str, Any]] = {}
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            cells = data.get("cells", {})
        manifest = cls(sweep_dir, cells)
        # A cell still marked running was interrupted mid-flight.
        stale = [cid for cid, e in cells.items() if e.get("status") == "running"]
        for cell_id in stale:
            cells[cell_id]["status"] = "failed"
            cells[cell_id]["stderr_tail"] = "sweep interrupted while cell was running"
        if stale:
            manifest._save()
        return manifest

    def _save(self) -> None:
        self._sweep_dir.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": _SCHEMA_VERSION, "cells": self._cells}
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_path, self.path)

    def entries(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._cells)

    def status(self, cell: Cell) -> str:
        entry = self._cells.get(cell.cell_id)
        return entry["status"] if entry else "pending"

    def _base_entry(self, cell: Cell) -> Dict[str, Any]:
        return {
            "model": cell.model,
            "mode": cell.mode,
            "kv_point": cell.kv_point,
            "rep_index": cell.rep_index,
            "seed": cell.seed,
            "cell_dir": cell.cell_dir,
        }

    def mark_running(self, cell: Cell) -> None:
        entry = self._base_entry(cell)
        entry["status"] = "running"
        entry["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        entry["stderr_tail"] = None
        self._cells[cell.cell_id] = entry
        self._save()

    def mark_done(self, cell: Cell) -> None:
        entry = self._cells.setdefault(cell.cell_id, self._base_entry(cell))
        entry["status"] = "done"
        entry["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save()

    def mark_failed(self, cell: Cell, stderr_tail: str) -> None:
        entry = self._cells.setdefault(cell.cell_id, self._base_entry(cell))
        entry["status"] = "failed"
        entry["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        entry["stderr_tail"] = (stderr_tail or "")[-2000:]
        self._save()

    def cells_to_run(self, cells: List[Cell], retry_failed: bool) -> List[Cell]:
        skip = {"done"} if retry_failed else {"done", "failed"}
        return [c for c in cells if self.status(c) not in skip]
