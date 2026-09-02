#!/usr/bin/env bash
# Fetch upstream KVCOMM at the pinned commit and apply this harness's changes.
#
# KVCOMM (https://github.com/FastMAS/KVCOMM) is NOT distributed with this
# repository: upstream ships no license file, so nothing of it is redistributed
# here. This script clones it into external/KVCOMM (gitignored), checks out the
# exact commit the experiments were run against, applies the patch to the five
# upstream files the harness modifies, and copies in the eight new agent and
# prompt-set files from overlay/.
#
#   bash kvcomm_patch/setup_kvcomm.sh [install_dir] [--reset] [--skip-import-check]
#
#   install_dir           default: $KVCOMM_ROOT if set, else <repo>/external/KVCOMM
#   --reset               discard a previous or partial application under KVCOMM/
#                         and apply again
#   --skip-import-check   do not import the patched package at the end (use when
#                         the Python dependencies are not installed yet)
#
# Safe to re-run: a state file inside the checkout records what was applied, and
# a second run with identical patch/overlay contents is a no-op.
#
# Windows: run under Git Bash or WSL. The clone is created with
# core.autocrlf=false so the LF patch applies; never open the patch in an editor
# that rewrites line endings.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
UPSTREAM_URL="https://github.com/FastMAS/KVCOMM.git"
UPSTREAM_SHA="48ca0b376c7f4fbf1c24042c1709a6fe4148c959"
# git tree hash of the KVCOMM/ package directory at that commit; a cheap check
# that the checkout really is the pristine upstream package.
UPSTREAM_PKG_TREE="a4d838a28707815602f8e9fad0badd1b4c4032ae"
PATCH="$HERE/patches/0001-security-review-hooks.patch"
OVERLAY="$HERE/overlay"
MARKER_NAME=".harness_state.txt"
PYTHON="${PYTHON:-python}"

INSTALL_DIR=""
RESET=0
IMPORT_CHECK=1
for arg in "$@"; do
  case "$arg" in
    --reset) RESET=1 ;;
    --skip-import-check) IMPORT_CHECK=0 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 1 ;;
    *) INSTALL_DIR="$arg" ;;
  esac
done
INSTALL_DIR="${INSTALL_DIR:-${KVCOMM_ROOT:-$REPO_ROOT/external/KVCOMM}}"

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# Everything the applied state depends on, as one comparable text block.
expected_state() {
  echo "upstream_sha=$UPSTREAM_SHA"
  echo "patch=$(sha256_of "$PATCH")"
  while IFS= read -r f; do
    echo "overlay:${f#./}=$(sha256_of "$OVERLAY/$f")"
  done < <(cd "$OVERLAY" && find . -type f | LC_ALL=C sort)
}

echo "==> install dir : $INSTALL_DIR"

# 1. Clone or reuse.
if [ ! -d "$INSTALL_DIR/.git" ]; then
  if [ -e "$INSTALL_DIR" ] && [ -n "$(ls -A "$INSTALL_DIR")" ]; then
    echo "FATAL: $INSTALL_DIR exists, is not a git checkout, and is not empty." >&2
    exit 1
  fi
  echo "==> cloning $UPSTREAM_URL"
  git -c core.autocrlf=false clone --quiet "$UPSTREAM_URL" "$INSTALL_DIR"
  git -C "$INSTALL_DIR" config core.autocrlf false
  git -C "$INSTALL_DIR" checkout --quiet --detach "$UPSTREAM_SHA"
else
  echo "==> reusing existing checkout"
fi

HEAD_SHA="$(git -C "$INSTALL_DIR" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$UPSTREAM_SHA" ]; then
  echo "FATAL: checkout is at $HEAD_SHA, expected $UPSTREAM_SHA." >&2
  echo "       git -C \"$INSTALL_DIR\" fetch origin && git -C \"$INSTALL_DIR\" checkout --detach $UPSTREAM_SHA" >&2
  exit 1
fi
PKG_TREE="$(git -C "$INSTALL_DIR" rev-parse HEAD:KVCOMM)"
if [ "$PKG_TREE" != "$UPSTREAM_PKG_TREE" ]; then
  echo "FATAL: KVCOMM/ tree hash is $PKG_TREE, expected $UPSTREAM_PKG_TREE." >&2
  exit 1
fi
echo "==> upstream commit verified: $UPSTREAM_SHA"

# Status of KVCOMM/ ignoring Python bytecode, which importing the package
# creates and which is not a modification of upstream code.
pkg_status() {
  git -C "$INSTALL_DIR" status --porcelain -- KVCOMM/ | grep -v '__pycache__' || true
}

restore_pristine() {
  git -C "$INSTALL_DIR" checkout --quiet -- KVCOMM/
  git -C "$INSTALL_DIR" clean --quiet -fdx -- KVCOMM/
}

# The five files the patch modifies, from the patch itself.
patched_files() { sed -n 's#^+++ b/##p' "$PATCH"; }

# Hashes of the patched files as they are right after a successful apply. They
# are appended to the state file so later runs can detect any local edit.
patched_state() {
  while IFS= read -r f; do
    echo "patched:$f=$(sha256_of "$INSTALL_DIR/$f")"
  done < <(patched_files)
}

# True when the checkout carries exactly the patch + overlay and nothing else.
# With a state file given, the patched files must also hash to what was
# recorded when they were applied.
verify_applied() {
  local marker="${1:-}"
  git -C "$INSTALL_DIR" apply --check -R "$PATCH" 2>/dev/null || return 1
  while IFS= read -r f; do
    [ -f "$INSTALL_DIR/$f" ] || return 1
    [ "$(sha256_of "$OVERLAY/$f")" = "$(sha256_of "$INSTALL_DIR/$f")" ] || return 1
  done < <(cd "$OVERLAY" && find . -type f)
  if [ -n "$marker" ]; then
    [ "$(grep '^patched:' "$marker")" = "$(patched_state)" ] || return 1
  fi
  local status n_mod n_new n_other
  status="$(pkg_status)"
  n_mod="$(printf '%s\n' "$status" | grep -c '^ M ' || true)"
  n_new="$(printf '%s\n' "$status" | grep -c '^?? ' || true)"
  n_other="$(printf '%s\n' "$status" | grep -vc -e '^ M ' -e '^?? ' -e '^$' || true)"
  [ "$n_mod" = 5 ] && [ "$n_new" = 8 ] && [ "$n_other" = 0 ]
}

# 2. Decide whether to apply.
MARKER="$INSTALL_DIR/$MARKER_NAME"
WANT="$(expected_state)"
APPLY=1
if [ "$RESET" = 1 ]; then
  echo "==> --reset: restoring pristine KVCOMM/"
  restore_pristine
  rm -f "$MARKER"
elif [ -f "$MARKER" ]; then
  WANT_N="$(printf '%s
' "$WANT" | wc -l)"
  if [ "$(head -n "$WANT_N" "$MARKER")" != "$WANT" ]; then
    echo "FATAL: $MARKER exists but does not match the current kvcomm_patch/ contents." >&2
    echo "       The patch or overlay changed since it was applied. Re-run with --reset." >&2
    exit 1
  fi
  if verify_applied "$MARKER"; then
    echo "==> harness changes already applied and verified; nothing to apply"
    APPLY=0
  else
    echo "FATAL: $MARKER says the harness changes are applied, but KVCOMM/ differs" >&2
    echo "       from patch + overlay (local edits?). Current status:" >&2
    pkg_status >&2
    echo "       Re-run with --reset to discard local changes and re-apply." >&2
    exit 1
  fi
fi

if [ "$APPLY" = 1 ]; then
  # 3. The package directory must be pristine.
  DIRTY="$(pkg_status)"
  if [ -n "$DIRTY" ]; then
    echo "FATAL: KVCOMM/ in $INSTALL_DIR has local changes:" >&2
    echo "$DIRTY" >&2
    echo "       Re-run with --reset to discard them, or point at a clean checkout." >&2
    exit 1
  fi

  # 4. Apply.
  echo "==> applying $(basename "$PATCH")"
  git -C "$INSTALL_DIR" apply --check "$PATCH"
  git -C "$INSTALL_DIR" apply "$PATCH"
  echo "==> copying overlay files"
  cp -R "$OVERLAY/KVCOMM/." "$INSTALL_DIR/KVCOMM/"

  # 5. Verify and record.
  if ! verify_applied; then
    echo "FATAL: verification after apply failed. Status of KVCOMM/:" >&2
    pkg_status >&2
    exit 1
  fi
  echo "==> KVCOMM/: 5 modified, 8 new files (verified)"
  { printf '%s\n' "$WANT"; patched_state; } > "$MARKER"
  echo "==> state recorded in $MARKER"
fi

# 6. Import smoke test.
if [ "$IMPORT_CHECK" = 1 ]; then
  echo "==> import smoke test"
  if ! KVCOMM_ROOT="$INSTALL_DIR" "$PYTHON" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["KVCOMM_ROOT"])
import KVCOMM.agents, KVCOMM.prompt  # noqa: F401
from KVCOMM.agents import CodeGenerator, CodeProvider, SecurityValidator, CweValidator  # noqa: F401
from KVCOMM.prompt import SecureCodePromptSet, PrimeVulPromptSet  # noqa: F401
from KVCOMM.llm.kvcomm_engine import KVCOMMEngine, BareCodeKVPackage  # noqa: F401
from KVCOMM.llm.llm import LLM
assert LLM.DEFAULT_MAX_TOKENS == 6000, LLM.DEFAULT_MAX_TOKENS
print("    KVCOMM + harness hooks import OK")
PY
  then
    echo "FATAL: import check failed." >&2
    echo "       For a ModuleNotFoundError: pip install -r requirements.txt" >&2
    echo "       (or re-run with --skip-import-check)" >&2
    exit 1
  fi
fi

echo
echo "==> done."
if [ "$INSTALL_DIR" != "$REPO_ROOT/external/KVCOMM" ]; then
  echo "    non-default location; export it before running anything:"
  echo "    export KVCOMM_ROOT=\"$INSTALL_DIR\""
fi
