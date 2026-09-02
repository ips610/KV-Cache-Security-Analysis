"""Where the harness and its third-party checkouts live.

Every entry point (experiment runners, analysis scripts, dataset builders, the
test suite) calls :func:`bootstrap` first. It puts two directories at the front
of ``sys.path``:

* the repository root, so ``experiments``, ``datasets`` and ``cacheblend`` resolve
  to this repository. The root must win over site-packages because a HuggingFace
  ``datasets`` install would otherwise shadow the local ``datasets`` package;
* ``KVCOMM_ROOT``, the upstream KVCOMM checkout that ``kvcomm_patch/setup_kvcomm.sh``
  creates (default ``external/KVCOMM``), so ``import KVCOMM`` works.

Neither KVCOMM nor CacheBlend is distributed with this repository; see
``external/README.md``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXTERNAL_DIR = REPO_ROOT / "external"
KVCOMM_ROOT = Path(os.environ.get("KVCOMM_ROOT") or EXTERNAL_DIR / "KVCOMM").resolve()
CACHEBLEND_ROOT = Path(os.environ.get("CACHEBLEND_ROOT") or EXTERNAL_DIR / "CacheBlend").resolve()
CACHEBLEND_BENCH = REPO_ROOT / "cacheblend" / "bench"

KVCOMM_SETUP_HINT = (
    "run `bash kvcomm_patch/setup_kvcomm.sh`, or set KVCOMM_ROOT to an existing "
    "checkout that has the harness patch applied"
)


def kvcomm_available() -> bool:
    """True when the patched upstream package can be imported from KVCOMM_ROOT."""
    return (KVCOMM_ROOT / "KVCOMM" / "__init__.py").is_file()


def bootstrap(require_kvcomm: bool = False) -> Path:
    """Put REPO_ROOT then KVCOMM_ROOT at the front of sys.path (root first).

    With ``require_kvcomm=True`` the call exits with an actionable message when
    the upstream checkout is missing, instead of a bare ``ModuleNotFoundError``
    several imports later.
    """
    for path in (KVCOMM_ROOT, REPO_ROOT):  # REPO_ROOT inserted last => index 0
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    if require_kvcomm and not kvcomm_available():
        raise SystemExit(
            f"KVCOMM package not found at {KVCOMM_ROOT}. Please {KVCOMM_SETUP_HINT}."
        )
    return REPO_ROOT
