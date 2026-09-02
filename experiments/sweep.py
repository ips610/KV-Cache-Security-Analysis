"""Config-driven sweep orchestrator: cells -> fresh subprocesses -> manifest.

Each cell runs the existing runner in a NEW subprocess so every model gets
a clean CUDA context and class-level anchor state. A failing cell is
recorded and the sweep continues.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from repo_paths import bootstrap  # noqa: E402
bootstrap()

from experiments.harness.manifest import Manifest
from experiments.harness.spec import Cell, SweepSpec, expand_cells, load_sweep_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = str(Path(__file__).parent / "run_secure_code.py")


def build_runner_cmd(cell: Cell, spec: SweepSpec, sweep_dir: Path, runner: str) -> List[str]:
    cell_parent = sweep_dir / cell.model_slug / cell.mode / cell.kv_slug
    cmd = [
        sys.executable, runner,
        "--mode", cell.mode,
        "--llm_name", cell.model,
        "--tasks_path", spec.tasks_path,
        "--max_tokens", str(spec.max_tokens),
        "--output_dir", str(cell_parent),
        "--run_name", f"rep_{cell.rep_index}",
        "--kv-threshold", str(cell.kv_point["kv_threshold"]),
        "--kv-max-anchor-num", str(cell.kv_point["kv_max_anchor_num"]),
        "--kv-window-size", str(cell.kv_point["kv_window_size"]),
    ]
    if spec.limit is not None:
        cmd += ["--limit", str(spec.limit)]
    if spec.noise_floor_baseline:
        cmd += ["--noise-floor-baseline", "true"]
    if spec.dense_reference_always:
        cmd += ["--dense-reference-always", "true"]
    if spec.generator_dense:
        cmd += ["--generator-dense", "true"]
    if spec.sample_roles:
        cmd += ["--sample-roles", spec.sample_roles,
                "--sample-temperature", str(spec.sample_temperature)]
    return cmd


def run_cell(cell: Cell, spec: SweepSpec, sweep_dir: Path, runner: str, manifest: Manifest) -> bool:
    cmd = build_runner_cmd(cell, spec, sweep_dir, runner)
    env = dict(os.environ, SEED=str(cell.seed))
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (cell.cell_id.replace("/", "__") + ".log")

    manifest.mark_running(cell)
    print(f"[sweep] RUN  {cell.cell_id}")
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=spec.cell_timeout_minutes * 60,
        )
    except subprocess.TimeoutExpired as exc:
        def _as_text(v):
            if v is None:
                return ""
            return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
        tail = (f"TIMEOUT after {spec.cell_timeout_minutes} min\n"
                f"--- stdout ---\n{_as_text(exc.stdout)}\n"
                f"--- stderr ---\n{_as_text(exc.stderr)}")
        manifest.mark_failed(cell, tail)
        log_path.write_text(tail, encoding="utf-8")
        print(f"[sweep] FAIL {cell.cell_id} (timeout)")
        return False
    except Exception as exc:  # noqa: BLE001 — one cell must never abort the sweep
        manifest.mark_failed(cell, f"sweep-side error launching cell: {exc!r}")
        log_path.write_text(f"SWEEP-SIDE ERROR\n{exc!r}", encoding="utf-8")
        print(f"[sweep] FAIL {cell.cell_id} (sweep-side error: {exc!r})")
        return False

    log_path.write_text(
        f"--- cmd ---\n{' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        manifest.mark_failed(cell, proc.stderr)
        print(f"[sweep] FAIL {cell.cell_id} (exit {proc.returncode})")
        return False
    manifest.mark_done(cell)
    print(f"[sweep] DONE {cell.cell_id}")
    return True


ANALYSIS_STEPS: List[Tuple[str, str]] = [
    ("report.py", "timing/throughput report + figures + LaTeX tables"),
    ("compare_validation_reports.py", "raw per-case verdict diffing"),
    ("paper_metrics.py", "unique-case fidelity, silencing rate, noise floor"),
    ("timing_metrics.py", "TTFT vs end-to-end deflation (T2/F2)"),
    ("methodology.py", "METHODOLOGY.md + provenance.json"),
]


def _safe_print(text: str) -> None:
    """Print text that may contain characters the console encoding cannot represent.

    Analysis subprocesses echo model output and paths; on a cp1252 Windows
    console an un-encodable character would otherwise raise and abort the whole
    analysis run after the work was already done.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def run_analysis(sweep_dir: Path) -> int:
    """Run every offline analysis step against a finished sweep.

    Invoked automatically at the end of a sweep so the results directory is
    complete with no follow-up commands. Each step runs in its own subprocess
    and a failure in one is reported but never prevents the others — a broken
    figure should not cost you the metrics.
    """
    print(f"\n[sweep] running analysis over {sweep_dir}")
    failures = 0
    for script, description in ANALYSIS_STEPS:
        script_path = Path(__file__).parent / script
        if not script_path.is_file():
            print(f"[analysis] SKIP {script} (not found)")
            continue
        print(f"[analysis] {script} — {description}")
        proc = subprocess.run(
            [sys.executable, str(script_path), str(sweep_dir)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        log_dir = sweep_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"analysis__{script}.log").write_text(
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}", encoding="utf-8"
        )
        if proc.returncode != 0:
            failures += 1
            print(f"[analysis] FAIL {script} (exit {proc.returncode}) — see logs/analysis__{script}.log")
        else:
            for line in proc.stdout.strip().splitlines()[-6:]:
                _safe_print(f"    {line}")
    if failures:
        print(f"[analysis] {failures} analysis step(s) failed; results are still in {sweep_dir}")
    else:
        print(f"[analysis] complete — see {sweep_dir / 'report'} and {sweep_dir / 'METHODOLOGY.md'}")
    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a multi-model KVCOMM sweep from a YAML spec.")
    parser.add_argument("--spec", required=True, help="Path to the sweep YAML spec.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Also re-run cells previously marked failed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the cells that would run, then exit.")
    parser.add_argument("--runner", default=DEFAULT_RUNNER,
                        help="Runner script (tests substitute a stub).")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip the automatic post-sweep analysis.")
    parser.add_argument("--analyze-on-failure", action="store_true",
                        help="Run the analysis even if some cells failed.")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Skip execution; just (re)generate all reports for this sweep.")
    args = parser.parse_args(argv)

    spec = load_sweep_spec(args.spec)
    cells = expand_cells(spec)
    sweep_dir = Path(spec.sweep_dir).expanduser().resolve()

    if args.dry_run:
        for cell in cells:
            print(cell.cell_id)
        print(f"[sweep] {len(cells)} cells total (dry run, nothing executed).")
        return 0

    if args.analysis_only:
        return run_analysis(sweep_dir)

    stored_spec = sweep_dir / "spec.yaml"
    current_text = Path(args.spec).read_text(encoding="utf-8")
    if stored_spec.exists():
        if stored_spec.read_text(encoding="utf-8") != current_text:
            print(f"[sweep] ERROR: spec drift — {args.spec} differs from the spec this sweep "
                  f"was started with ({stored_spec}). Use a new sweep_name, or delete the "
                  f"stored copy to accept the change.")
            return 1
    else:
        sweep_dir.mkdir(parents=True, exist_ok=True)
        stored_spec.write_text(current_text, encoding="utf-8")

    manifest = Manifest.load(sweep_dir)
    todo = manifest.cells_to_run(cells, retry_failed=args.retry_failed)
    skipped = len(cells) - len(todo)
    print(f"[sweep] {len(cells)} cells; {skipped} already done/failed-skipped; {len(todo)} to run.")

    failures = 0
    for cell in todo:
        if not run_cell(cell, spec, sweep_dir, args.runner, manifest):
            failures += 1
    print(f"[sweep] finished: {len(todo) - failures} ok, {failures} failed. Manifest: {manifest.path}")

    if args.no_analysis:
        print("[sweep] --no-analysis given; skipping report generation.")
        return failures
    if failures and not args.analyze_on_failure:
        print(f"[sweep] {failures} cell(s) failed — skipping analysis so partial results are not "
              f"mistaken for complete ones. Re-run with --retry-failed, or force with "
              f"--analyze-on-failure.")
        return failures
    run_analysis(sweep_dir)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
