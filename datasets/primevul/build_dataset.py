"""Build the PrimeVul ground-truth sample from the public paired splits.

Source: the MIT-licensed HF mirror ``colin/PrimeVul`` of the PrimeVul dataset
(Ding et al.), files ``primevul_{test,valid,train}_paired.jsonl``. A paired file
stores a vulnerable record (``target=1``) immediately followed by its patched
twin (``target=0``) sharing a ``commit_id``.

Selection is fully deterministic -- no RNG anywhere. Canonical order is
test -> valid -> train, file order within each split:

  1. keep a pair only if BOTH members are <= ``--cap-chars`` characters, so
     validator inputs stay comparable to the VIBE arm and short enough that
     KVCOMM anchors can cover them (``predict_as_anchor`` only considers
     anchors at least as long as the candidate segment);
  2. drop duplicates by the vulnerable member's ``func_hash``;
  3. keep only the ten CWEs in ``KVCOMM.prompt.primevul_common.CWE_INFO``;
  4. take the first ``--per-cwe`` pairs of each CWE, with a global cap of
     ``--max-per-project`` pairs per repository so no single project dominates.

Outputs the pair table (with full provenance, for offline joining) and two task
files -- one per arm. Answer-revealing fields (CVE description, commit message,
NVD link, the label itself, the twin's code) are written to the pair table only
and are asserted absent from every task record.

    python datasets/primevul/build_dataset.py [--offline] [--cap-chars 7000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_paths import bootstrap  # noqa: E402
bootstrap(require_kvcomm=True)

from KVCOMM.prompt.primevul_common import (  # noqa: E402
    CWE_IDS,
    CWE_NAME,
    CWE_RULE_NUMBER,
    build_review_request,
)

HERE = Path(__file__).resolve().parent
BASE_URL = "https://huggingface.co/datasets/colin/PrimeVul/resolve/main/"
SPLITS = ("test", "valid", "train")  # canonical order
SOURCE_DATASET = "colin/PrimeVul"

# Fields that would hand the validator the answer; never allowed in a task record.
LEAKY_FIELDS = ("cve_desc", "commit_message", "nvd_url", "target", "cve",
                "vulnerable_code", "patched_code")


def _download(split: str, raw_dir: Path, offline: bool) -> Path:
    name = f"primevul_{split}_paired.jsonl"
    dest = raw_dir / name
    if dest.exists():
        return dest
    if offline:
        raise SystemExit(f"--offline given but {dest} is missing; run once online first.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + name
    print(f"[primevul] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "kvcomm-primevul/1"})
    with urllib.request.urlopen(req, timeout=600) as response:
        dest.write_bytes(response.read())
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(path: Path, split: str) -> List[Dict[str, Any]]:
    """Return [{'vuln':..., 'patched':..., 'split':..., 'line':...}] for well-formed pairs."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs: List[Dict[str, Any]] = []
    index = 0
    while index + 1 < len(records):
        first, second = records[index], records[index + 1]
        if (first.get("target") == 1 and second.get("target") == 0
                and first.get("commit_id") == second.get("commit_id")):
            pairs.append({"vuln": first, "patched": second, "split": split, "line": index})
            index += 2
        else:
            # Not a well-formed adjacent pair; skip the single record.
            index += 1
    return pairs


_NAME_RE = re.compile(r"([A-Za-z_~][A-Za-z0-9_:]*)\s*\(")
_NOT_A_NAME = {"if", "for", "while", "switch", "return", "sizeof", "defined"}


def function_name(source: str, fallback: str) -> str:
    """Best-effort C/C++ function name from the definition's signature."""
    head = source.split("{", 1)[0]
    for candidate in reversed(_NAME_RE.findall(head)):
        tail = candidate.split("::")[-1]
        if tail and tail.lower() not in _NOT_A_NAME:
            return tail
    return fallback or "function"


def select(pairs: List[Dict[str, Any]], *, cap_chars: int, per_cwe: int,
           max_per_project: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats: Dict[str, Any] = {"input_pairs": len(pairs)}

    kept = [p for p in pairs
            if len(p["vuln"]["func"]) <= cap_chars and len(p["patched"]["func"]) <= cap_chars]
    stats["after_length_filter"] = len(kept)

    seen_hashes = set()
    deduped: List[Dict[str, Any]] = []
    for pair in kept:
        key = pair["vuln"].get("func_hash")
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        deduped.append(pair)
    stats["after_dedup"] = len(deduped)

    by_cwe: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in deduped:
        cwes = pair["vuln"].get("cwe") or []
        if len(cwes) == 1 and cwes[0] in CWE_IDS:
            by_cwe[cwes[0]].append(pair)
    stats["support_by_cwe"] = {cwe: len(by_cwe.get(cwe, [])) for cwe in CWE_IDS}

    project_used: Counter = Counter()
    selected: List[Dict[str, Any]] = []
    short_strata: List[Tuple[str, int]] = []
    for cwe in CWE_IDS:  # catalogue order, so rule numbers stay stable
        taken = 0
        for pair in by_cwe.get(cwe, []):
            project = pair["vuln"].get("project", "")
            if project_used[project] >= max_per_project:
                continue
            project_used[project] += 1
            pair["cwe"] = cwe
            selected.append(pair)
            taken += 1
            if taken == per_cwe:
                break
        if taken < per_cwe:
            short_strata.append((cwe, taken))
    stats["short_strata"] = short_strata
    stats["selected"] = len(selected)
    stats["projects"] = dict(Counter(p["vuln"]["project"] for p in selected).most_common())
    return selected, stats


def to_records(selected: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    pair_rows: List[Dict[str, Any]] = []
    vuln_tasks: List[Dict[str, Any]] = []
    patched_tasks: List[Dict[str, Any]] = []

    for index, pair in enumerate(selected):
        vuln, patched = pair["vuln"], pair["patched"]
        cwe = pair["cwe"]
        sample_id = f"PV{index:03d}"
        name = function_name(vuln["func"], fallback=Path(vuln.get("file_name", "func")).stem)
        request = build_review_request(
            sample_id=sample_id, function_name=name,
            file_name=vuln.get("file_name") or "source.c",
            project=vuln.get("project", ""),
        )

        pair_rows.append({
            "sample_id": sample_id,
            "pair_id": sample_id,
            "order_index": index,
            "project": vuln.get("project"),
            "cve": vuln.get("cve"),
            "cwe": cwe,
            "cwe_name": CWE_NAME[cwe],
            "cwe_rule_number": CWE_RULE_NUMBER[cwe],
            "commit_id": vuln.get("commit_id"),
            "commit_url": vuln.get("commit_url"),
            "file_name": vuln.get("file_name"),
            "function_name": name,
            "review_request": request,
            "vulnerable_code": vuln["func"],
            "patched_code": patched["func"],
            "vulnerable_label": 1,
            "vulnerable_idx": vuln.get("idx"),
            "patched_idx": patched.get("idx"),
            "vuln_func_hash": vuln.get("func_hash"),
            "patched_func_hash": patched.get("func_hash"),
            "vuln_chars": len(vuln["func"]),
            "patched_chars": len(patched["func"]),
            "source_metadata": {
                "dataset": SOURCE_DATASET,
                "split": pair["split"],
                "line_index": pair["line"],
                "nvd_url": vuln.get("nvd_url"),
            },
        })

        for arm, code, label, sink in (("vulnerable", vuln["func"], 1, vuln_tasks),
                                       ("patched", patched["func"], 0, patched_tasks)):
            sink.append({
                "task": request,
                "code": code,
                "sample_id": sample_id,
                "pair_id": sample_id,
                "order_index": index,
                "cwe": cwe,
                "cwe_rule_number": CWE_RULE_NUMBER[cwe],
                "arm": arm,
                "label": label,
            })

    return pair_rows, vuln_tasks, patched_tasks


def assert_no_leakage(tasks: List[Dict[str, Any]], other_arm_codes: Dict[str, str]) -> None:
    """The validator sees `task` and `code` only; neither may reveal the label.

    `label` and `arm` ride along in the record for offline scoring but are never
    rendered into a prompt (see `experiments/run_primevul.py`).
    """
    probes = ("cve-", "vulnerab", "patched", "security advisory", "exploit")
    for record in tasks:
        for field in LEAKY_FIELDS:
            if field in record:
                raise AssertionError(f"{record['sample_id']}: leaky field {field!r} in task record")
        if record["code"] == other_arm_codes.get(record["sample_id"]):
            raise AssertionError(f"{record['sample_id']}: the two arms carry identical code")
        lowered = record["task"].lower()
        for probe in probes:
            if probe in lowered:
                raise AssertionError(f"{record['sample_id']}: review request mentions {probe!r}")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap-chars", type=int, default=7000,
                        help="Max characters for BOTH members of a pair (default 7000, ~2000 tokens).")
    parser.add_argument("--per-cwe", type=int, default=10, help="Pairs per CWE (default 10).")
    parser.add_argument("--max-per-project", type=int, default=3,
                        help="Global cap on pairs from one repository (default 3).")
    parser.add_argument("--raw-dir", default=str(HERE / "raw"))
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument("--offline", action="store_true", help="Fail instead of downloading.")
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    sources: Dict[str, Any] = {}
    pairs: List[Dict[str, Any]] = []
    for split in SPLITS:
        path = _download(split, raw_dir, args.offline)
        sources[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        pairs.extend(read_pairs(path, split))

    selected, stats = select(pairs, cap_chars=args.cap_chars, per_cwe=args.per_cwe,
                             max_per_project=args.max_per_project)
    if stats["short_strata"]:
        print(f"[primevul] WARNING short strata: {stats['short_strata']}", file=sys.stderr)

    pair_rows, vuln_tasks, patched_tasks = to_records(selected)
    by_id_patched = {r["sample_id"]: r["code"] for r in patched_tasks}
    by_id_vuln = {r["sample_id"]: r["code"] for r in vuln_tasks}
    assert_no_leakage(vuln_tasks, by_id_patched)
    assert_no_leakage(patched_tasks, by_id_vuln)
    if len({r["task"] for r in vuln_tasks}) != len(vuln_tasks):
        raise AssertionError("review requests are not unique; they are the KV join key")

    write_jsonl(out_dir / "primevul_pairs.jsonl", pair_rows)
    write_jsonl(out_dir / "tasks_vulnerable.jsonl", vuln_tasks)
    write_jsonl(out_dir / "tasks_patched.jsonl", patched_tasks)

    lengths = sorted(r["vuln_chars"] for r in pair_rows)
    manifest = {
        "schema_version": 1,
        "source_dataset": SOURCE_DATASET,
        "source_files": sources,
        "selection": {
            "order": "test -> valid -> train, file order within split; no RNG",
            "cap_chars": args.cap_chars,
            "per_cwe": args.per_cwe,
            "max_per_project": args.max_per_project,
            "cwe_catalogue": list(CWE_IDS),
        },
        "counts": {
            "pairs": len(pair_rows),
            "artifacts": 2 * len(pair_rows),
            "by_cwe": dict(Counter(r["cwe"] for r in pair_rows)),
            "by_project": stats["projects"],
            "distinct_cves": len({r["cve"] for r in pair_rows}),
            "distinct_commits": len({r["commit_id"] for r in pair_rows}),
        },
        "funnel": {k: v for k, v in stats.items() if k != "projects"},
        "vulnerable_code_chars": ({
            "min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1],
        } if lengths else {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[primevul] {len(pair_rows)} pairs / {2 * len(pair_rows)} artifacts over "
          f"{len(set(r['cwe'] for r in pair_rows))} CWEs -> {out_dir}")
    print(f"[primevul] funnel: {stats['input_pairs']} -> len<={args.cap_chars}: "
          f"{stats['after_length_filter']} -> dedup: {stats['after_dedup']} -> "
          f"selected: {stats['selected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
