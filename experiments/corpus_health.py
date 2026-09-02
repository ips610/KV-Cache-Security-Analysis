"""Report how much of the secure-code corpus is actually usable, per model.

Three independent defects shrink the corpus, and they overlap only partly, so
the usable set is smaller than any one of them suggests:

* **truncated**  - the generator hit ``max_tokens``, so the code under review is
  cut off mid-output and the validator judged an incomplete program.
* **kv-drifted** - the generator's output is not identical across kv points even
  though it ran dense and greedy, so there is no single canonical code artifact
  for that task.
* **not token-exact** - ``generated_code.md`` stores decoded text, so
  re-encoding it does not always reproduce the ids the generator emitted; the
  rebuilt prompt then differs from the one KVCOMM actually validated.

The first two are properties of the recorded run and affect the existing KVCOMM
results as much as any new arm. The third only affects work that replays the
prompt on another stack.

Because KVCOMM's fidelity numbers are conditioned on anchor hits while a
replayed arm covers every task, the comparable set is ``usable AND hit`` - which
this reports per kv point, since the hit set changes with the entropy gate.

Usage::

    python experiments/corpus_health.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FROZEN_DIR = _REPO_ROOT / "datasets" / "secure_code" / "frozen_validator_inputs"
_SPEC_DIR = _REPO_ROOT / "datasets" / "secure_code" / "validator_prompt_spec"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _token_exact_by_index(spec_dir: Path, model_slug: str) -> Dict[int, bool]:
    """Map task_index -> whether the rebuilt prompt matched the recorded run.

    Returns an empty mapping when the prompt spec has not been exported, so the
    corpus report still works from the frozen inputs alone.
    """
    spec_path = spec_dir / f"{model_slug}_prompt_spec.json"
    if not spec_path.exists():
        return {}
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return {t["task_index"]: bool(t["token_count_matches_run"]) for t in spec["tasks"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=_FROZEN_DIR)
    parser.add_argument("--spec-dir", type=Path, default=_SPEC_DIR)
    args = parser.parse_args()

    frozen_files = sorted(args.frozen_dir.glob("*.jsonl"))
    if not frozen_files:
        raise SystemExit(
            f"no frozen inputs under {args.frozen_dir}. "
            "Run experiments/export_frozen_inputs.py first."
        )

    print(f"{'model':<34} {'n':>4} {'trunc':>6} {'drift':>6} {'clean':>6} "
          f"{'exact':>6} {'usable':>7}")
    print("-" * 76)

    summary: Dict[str, Dict[str, Any]] = {}
    for path in frozen_files:
        records = _load_jsonl(path)
        model_slug = path.stem
        exact_map = _token_exact_by_index(args.spec_dir, model_slug)
        have_spec = bool(exact_map)

        truncated = {r["task_index"] for r in records if r["generator_truncated"]}
        drifted = {r["task_index"] for r in records if not r["code_identical_across_kv"]}
        clean = {r["task_index"] for r in records} - truncated - drifted
        exact = {i for i, ok in exact_map.items() if ok}
        usable = clean & exact if have_spec else clean

        summary[model_slug] = {"records": records, "usable": usable}
        print(
            f"{model_slug:<34} {len(records):>4} {len(truncated):>6} {len(drifted):>6} "
            f"{len(clean):>6} {len(exact) if have_spec else '-':>6} {len(usable):>7}"
        )

    if not all(_token_exact_by_index(args.spec_dir, p.stem) for p in frozen_files):
        print(
            "\nNote: 'exact' is blank for models with no exported prompt spec; "
            "'usable' then means clean only. Run "
            "experiments/export_validator_prompt_spec.py for the full picture."
        )

    # The comparable set: usable AND an anchor hit, per kv point.
    print("\nComparable set (usable AND KVCOMM anchor hit), per kv point:")
    print(f"{'model':<34} {'kv point':<18} {'hits':>6} {'usable&hit':>11}")
    print("-" * 73)
    for model_slug, data in summary.items():
        kv_points = sorted(
            {kv for r in data["records"] for kv in r["kvcomm_natural_mode_by_kv"]}
        )
        for kv in kv_points:
            hits = {
                r["task_index"]
                for r in data["records"]
                if r["kvcomm_natural_mode_by_kv"].get(kv) == "kv_reuse"
            }
            print(
                f"{model_slug:<34} {kv:<18} {len(hits):>6} "
                f"{len(hits & data['usable']):>11}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
