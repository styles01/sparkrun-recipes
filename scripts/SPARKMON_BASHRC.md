# sparkmon bashrc function (v2, 2026-09-28)

sparkmon() {
  tmux has-session -t mon 2>/dev/null || {
    tmux new-session -d -s mon -n spark -x 200 -y 50
    tmux send-keys -t mon "TERM=xterm-256color btop" Enter
    tmux split-window -v -t mon:spark.0 -l 21
    tmux send-keys -t mon:spark.1 "TERM=xterm-256color python3 ~/sparkrun-recipes/scripts/sparkmon.py -i 1" Enter
    tmux select-pane -t mon:spark.0
  }
  tmux attach-session -t mon
}

Also in bashrc: big/bigmem (>=N% memory process filter).
