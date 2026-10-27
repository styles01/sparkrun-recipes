---
name: bash-scripting
description: Writing shell scripts that fail loudly and handle paths safely - set -euo pipefail, quoting, arrays, functions, arguments, temp files with trap cleanup, and testing the error paths.
---
# Bash scripting

Rules:
- Header: `#!/usr/bin/env bash` and `set -euo pipefail`. `IFS=$'\n\t'` when splitting lines.
- Quote every expansion: `"$var"`, `"$@"`, `"$(cmd)"`. Arrays for lists: `files=(*.txt); for f in "${files[@]}"`.
- Paths from `$0`: `here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`. Never `cd` without checking; prefer absolute paths.
- Functions with local variables; `main "$@"` at the end. Usage text and exit 2 on bad arguments.
- Temp files with `mktemp` and `trap 'rm -rf "$tmp"' EXIT`.
- Check the tools you call exist (`command -v jq >/dev/null || { echo "jq missing" >&2; exit 1; }`). Run `shellcheck` if available.

What goes wrong:
- `set -e` does not fire inside `if cmd`, `cmd || true`, or a pipeline's left side without `pipefail`.
- Unquoted `$var` with spaces or globs splits into several arguments.
- `[ $x == y ]` with empty `$x` is a syntax error; use `[[ "$x" == y ]]`.

Check: run the script with valid input, with a missing argument, with a path containing a space, and with a failing inner command; read `echo $?` after each.

Done when:
- [ ] Four runs in the result (valid, missing arg, space in path, inner failure) with their exit codes.
- [ ] `shellcheck script.sh` output (or "shellcheck not installed") in the result.
- [ ] Temp files are gone after the run (`ls "$tmp"` fails).

## Reference

```bash
#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { echo "usage: $0 <input-dir> [--dry-run]" >&2; exit 2; }
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }
main() {
  [[ $# -ge 1 ]] || usage
  local dir="$1" dry=0; [[ "${2:-}" == "--dry-run" ]] && dry=1
  [[ -d "$dir" ]] || { log "not a directory: $dir"; exit 1; }
  command -v jq >/dev/null || { log "jq missing"; exit 1; }
  local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  local files=(); while IFS= read -r -d '' f; do files+=("$f"); done < <(find "$dir" -name '*.json' -print0)
  log "found ${#files[@]} files"
  local f; for f in "${files[@]}"; do
    if (( dry )); then echo "would process: $f"; else jq -c . "$f" > "$tmp/$(basename "$f")"; fi
  done
  log "done"
}
main "$@"
```
Patterns: default value `"${VAR:-default}"`; require `"${VAR:?VAR is required}"`; read a file line by line `while IFS= read -r line; do ...; done < "$file"`; capture and test `if out="$(cmd 2>&1)"; then ...; else log "$out"; fi`; retry `for i in 1 2 3; do cmd && break; sleep $((i*2)); done`.
Portable: `sed -i.bak 's/a/b/' f && rm f.bak`; dates in UTC `date -u +%FT%TZ`.
Probe: `bash script.sh "/tmp/dir with space"; echo $?` and `bash script.sh; echo $?`.

More pitfalls:
- `cd` into a missing directory, then `rm -rf *` in the wrong place. Always `cd dir || exit 1`.
- macOS vs Linux tools differ (`sed -i`, `date -d`, `readlink -f`); test on the target.
