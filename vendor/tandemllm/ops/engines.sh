# Which processes are engines. Sourced by start.sh, stop.sh, hold.sh, watchdog.sh and gate.sh.
#
# They used to ask `pgrep -f "server/app.py ..."`, which matches any process whose command line
# merely CONTAINS the text: on 2026-09-23 at 22:47 a lock holder, `flock -o LOCK bash -c "<script
# text naming server/app.py>"`, was taken for the engine -- a hold's restore refused to restart
# :8000 because of it, and the next hold's stop.sh sent it SIGTERM, which freed the box lock while
# that hold's command ran. A flock, an ssh session's `bash -c`, a grep or an editor all match.
#
# An engine is a process whose EXECUTABLE is a Python interpreter (/proc/<pid>/exe) and whose
# argument vector runs the script server/app.py: `python [-u ...] server/app.py ...`, the path
# relative or ending in /server/app.py -- what start.sh, the systemd unit and row3 all launch.
# `python -c "..."` and `python tools/x.py --server server/app.py` are not engines. The port is the
# `--port` argument, app.py's default 8000 when there is none. No pid file: a service started by
# an older start.sh, by hand or by row3 is still found.
#
# ENGINE_PROC exists so tests/test_ops_scripts.py can hand the scripts a process table of its own
# (and never the box's real one, which holds the operator's engine).
ENGINE_PROC="${ENGINE_PROC:-/proc}"

# engine_pids [PORT] -- the pid of every engine, one a line; only those on PORT when given.
engine_pids() {
    local want="${1:-}" d exe arg i n port
    local -a argv
    for d in "$ENGINE_PROC"/[0-9]*; do
        argv=()
        # builtins only until the argument vector says "engine": this runs over every process
        while IFS= read -r -d '' arg; do argv+=("$arg"); done 2>/dev/null < "$d/cmdline"
        n=${#argv[@]}
        i=1
        while [ "$i" -lt "$n" ]; do
            case "${argv[$i]}" in
                -c|-m) i=$n ;;                       # a command or a module, not a script
                -*) i=$((i + 1)) ;;                  # an interpreter option (-u)
                *) break ;;
            esac
        done
        [ "$i" -lt "$n" ] || continue
        case "${argv[$i]}" in server/app.py|*/server/app.py) ;; *) continue ;; esac
        exe="$(readlink "$d/exe" 2>/dev/null)" || continue
        case "${exe##*/}" in python*) ;; *) continue ;; esac
        if [ -n "$want" ]; then
            port=8000
            while [ "$i" -lt "$n" ]; do
                case "${argv[$i]}" in
                    --port) port="${argv[$((i + 1))]:-}" ;;
                    --port=*) port="${argv[$i]#--port=}" ;;
                esac
                i=$((i + 1))
            done
            [ "$port" = "$want" ] || continue
        fi
        echo "${d##*/}"
    done
}
