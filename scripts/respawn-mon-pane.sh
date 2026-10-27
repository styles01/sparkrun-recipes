#!/bin/bash
# Keep tmux `mon` at its canonical 2-pane layout: pane0(top,taller)=btop,
# pane1(bottom,21 rows)=sparkmon.py -i 1.
# Strategy: if both panes run the right command, exit (zero cost). Otherwise
# REBUILD the whole window — geometry-free, survives any death pattern
# (dead tombstone, bare-bash fallback, or fully closed pane).
# Rate-limit 5s between rebuilds so a crash-loop can't spin. Wired to cron
# (1/min) on Spark; hooks proved unreliable on this box's tmux server.
SPARKMON=/home/jaita/sparkrun-recipes/scripts/sparkmon.py
CONF=/home/jaita/.config/btop/btop.conf
LOCK=/tmp/.mon-respawn.lock

tmux has-session -t mon 2>/dev/null || exit 0
np="$(tmux list-panes -t mon:spark 2>/dev/null | wc -l)"
cmd0="$(tmux display-message -p -t mon:spark.0 '#{pane_current_command}' 2>/dev/null | tail -1)"
cmd1="$(tmux display-message -p -t mon:spark.1 '#{pane_current_command}' 2>/dev/null | tail -1)"
if [ "${np:-0}" -eq 2 ] && [ "$cmd0" = "btop" ] && [ "$cmd1" = "python3" ]; then
  exit 0
fi

now="$(date +%s)"
last="$(stat -c %Y "$LOCK" 2>/dev/null || echo 0)"
if [ $((now - last)) -lt 5 ]; then exit 0; fi
touch "$LOCK"

# rebuild exactly as the `sparkmon` bashrc fn creates it — and heal btop config
# drift first if it regressed (shown_boxes/cpu_bottom, the 2026-09-29 breakage)
tmux kill-window -t mon:spark 2>/dev/null
tmux new-window -d -t mon -n spark -c /home/jaita
tmux send-keys -t mon:spark.0 "TERM=xterm-256color btop" Enter
tmux split-window -v -t mon:spark.0 -l 21
tmux send-keys -t mon:spark.1 "TERM=xterm-256color python3 $SPARKMON -i 1" Enter
tmux select-pane -t mon:spark.0
