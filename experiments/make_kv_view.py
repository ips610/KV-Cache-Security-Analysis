"""Build a single-kv-point view of a sweep tree so paper_metrics can run on it.

``paper_metrics.collapse_to_unique`` groups cases by ``(model, task)`` only
(``experiments/paper_metrics.py:103-105``); the kv point is parsed out of the
path but never used as part of the key. On a sweep that varies the kv point,
every kv cell for a task therefore collapses together as if the cells were
repetitions of each other, and any difference *caused* by the kv point is
counted as run-to-run irreproducibility.

That is what happened to ``results/MarginThresholdRun``: 533 paired folders
became "205 unique cases" at 72.2% reproducibility, and the pooled fidelity
numbers in its ``report/paper_metrics.md`` describe no configuration that was
actually run. The paper's methodology section flags this; nothing fixed it.

This builds a view containing exactly one kv point, so the existing analysis
runs **unmodified** and its grouping assumption (reps of one configuration)
holds. Files are hardlinked where the filesystem allows, so a view costs
essentially nothing.

Usage::

    python experiments/make_kv_view.py results/MarginThresholdRun --kv kv_t0.3_a20_w5
    python experiments/make_kv_view.py results/MarginThresholdRun --all
    python experiments/paper_metrics.py results/MarginThresholdRun__view_kv_t0.3_a20_w5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Files at the sweep root that the analysis scripts read.
_ROOT_FILES = ("spec.yaml", "METHODOLOGY.md", "provenance.json")

# Depth of a cell dir relative to the sweep root: <model>/<mode>/<kv>/rep_<n>
_KV_DEPTH = 2


def _kv_of(cell_dir: str) -> Optional[str]:
    parts = Path(cell_dir).as_posix().split("/")
    return parts[_KV_DEPTH] if len(parts) > _KV_DEPTH else None


def _link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink a file into the view, falling back to a copy.

    Hardlinks fail across volumes and on some filesystems; symlinks additionally
    need elevated privileges on Windows, so they are not attempted.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skipped"
    try:
        os.link(src, dst)
        return "linked"
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return "copied"


def _mirror_tree(src_root: Path, dst_root: Path) -> Dict[str, int]:
    counts = {"linked": 0, "copied": 0, "skipped": 0}
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        dst = dst_root / src.relative_to(src_root)
        counts[_link_or_copy(src, dst)] += 1
    return counts


def build_view(sweep_root: Path, kv_slug: str, out_root: Optional[Path] = None) -> Path:
    manifest_path = sweep_root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest.json under {sweep_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells: Dict[str, dict] = manifest.get("cells", {})

    selected = {
        cell_id: entry
        for cell_id, entry in cells.items()
        if _kv_of(entry.get("cell_dir", cell_id)) == kv_slug
    }
    if not selected:
        available = sorted(
            {kv for e in cells.values() if (kv := _kv_of(e.get("cell_dir", ""))) }
        )
        raise SystemExit(
            f"no cells with kv point '{kv_slug}' in {sweep_root}. Available: {available}"
        )

    view_root = out_root or sweep_root.parent / f"{sweep_root.name}__view_{kv_slug}"
    view_root.mkdir(parents=True, exist_ok=True)

    totals = {"linked": 0, "copied": 0, "skipped": 0}
    for entry in selected.values():
        cell_dir = entry.get("cell_dir")
        src = sweep_root / cell_dir
        if not src.is_dir():
            print(f"  warning: cell dir missing, skipped: {cell_dir}")
            continue
        for key, value in _mirror_tree(src, view_root / cell_dir).items():
            totals[key] += value

    for name in _ROOT_FILES:
        src = sweep_root / name
        if src.is_file():
            _link_or_copy(src, view_root / name)

    (view_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": manifest.get("schema_version", 1),
                "cells": selected,
                "view_of": str(sweep_root),
                "view_kv_point": kv_slug,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"{kv_slug}: {len(selected)} cells -> {view_root}  "
        f"(linked={totals['linked']} copied={totals['copied']} existing={totals['skipped']})"
    )
    return view_root


def discover_kv_points(sweep_root: Path) -> List[str]:
    manifest_path = sweep_root / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    found = {
        kv
        for entry in manifest.get("cells", {}).values()
        if (kv := _kv_of(entry.get("cell_dir", "")))
    }
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--kv", help="kv point slug to isolate, e.g. kv_t0.3_a20_w5")
    parser.add_argument(
        "--all", action="store_true", help="build one view per kv point present"
    )
    parser.add_argument(
        "--out-root", type=Path, default=None, help="explicit output directory"
    )
    args = parser.parse_args()

    if not args.sweep_root.is_dir():
        raise SystemExit(f"sweep root not found: {args.sweep_root}")

    kv_points = discover_kv_points(args.sweep_root)
    if args.all:
        if args.out_root:
            raise SystemExit("--out-root cannot be combined with --all")
        if not kv_points:
            raise SystemExit(f"no kv points discovered in {args.sweep_root}")
        for kv_slug in kv_points:
            build_view(args.sweep_root, kv_slug)
        print(
            f"\nRun the analysis against each view separately; pooling them "
            f"reintroduces the collapse this works around."
        )
        return 0

    if not args.kv:
        raise SystemExit(
            f"--kv or --all required. kv points present: {kv_points or '(none)'}"
        )
    build_view(args.sweep_root, args.kv, args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
