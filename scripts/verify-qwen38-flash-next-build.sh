#!/usr/bin/env bash
# Verify that a Qwen3.8-Flash-Next llama.cpp/qwen4exp source tree includes
# ggml-org/llama.cpp PR #27941 before reproducing compatibility claims.
set -euo pipefail

readonly PR27941_MERGE_REVISION="36b10154383b60eb15baac2c7a40d2a5f784faa7"
readonly PR27941_URL="https://github.com/ggml-org/llama.cpp/pull/27941"

usage() {
  cat <<'USAGE'
Usage: verify-qwen38-flash-next-build.sh --source-dir PATH

Checks that PATH is a clean git checkout whose HEAD is PR #27941's merge commit
or a descendant. This is a source-provenance gate only: it does not establish
runtime correctness or performance.

Before claiming a configuration, retain this command's output and run its gate:
  multi-lane  : simultaneous completions in every configured slot; correct,
                request-distinct outputs; no server assert/error/restart.
  vision      : an image request through the native --mmproj path; expected
                semantic result; no cross-request contamination.
  native-262K: an actual 262144-token prompt plus completion at the configured
                context; correct sentinel output; no CUDA/server abort.
USAGE
}

source_dir=""
while (($#)); do
  case "$1" in
    --source-dir)
      (($# >= 2)) || { echo "error: --source-dir needs a path" >&2; exit 2; }
      source_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$source_dir" ]] || { echo "error: --source-dir is required" >&2; usage >&2; exit 2; }
git -C "$source_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "error: not a git work tree: $source_dir" >&2
  exit 1
}

# A dirty source tree is not an auditable build input.
if [[ -n "$(git -C "$source_dir" status --porcelain)" ]]; then
  echo "error: source tree is dirty; commit/stash changes before verification" >&2
  exit 1
fi

if ! git -C "$source_dir" cat-file -e "${PR27941_MERGE_REVISION}^{commit}" 2>/dev/null; then
  echo "error: PR #27941 revision is absent; fetch ggml-org/llama.cpp history first" >&2
  exit 1
fi

head_revision="$(git -C "$source_dir" rev-parse HEAD)"
if ! git -C "$source_dir" merge-base --is-ancestor "$PR27941_MERGE_REVISION" "$head_revision"; then
  echo "error: $head_revision does not descend from PR #27941 ($PR27941_MERGE_REVISION)" >&2
  exit 1
fi

printf 'PASS qwen3.8-flash-next source gate\n'
printf 'source_dir=%s\n' "$(cd "$source_dir" && pwd -P)"
printf 'origin=%s\n' "$(git -C "$source_dir" remote get-url origin 2>/dev/null || printf '<none>')"
printf 'head_revision=%s\n' "$head_revision"
printf 'minimum_revision=%s\n' "$PR27941_MERGE_REVISION"
printf 'minimum_revision_url=%s\n' "$PR27941_URL"
printf '%s\n' 'Next: run and retain the applicable multi-lane, vision, and/or native-262K regression gate documented above.'
