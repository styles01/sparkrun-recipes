#!/bin/bash
# sparkDash :5555 ownership — ONE docker container ("sparkDash",
# restart: always, network_mode: host). This is the canonical restart.
# Host-side node processes on :5555 are strays: never start them.
cd /home/jaita/sparkDash || exit 1
mkdir -p run
if docker ps --format '{{.Names}}' | grep -qx sparkDash; then
  docker compose restart sparkdash
else
  docker compose up -d sparkdash
fi
PID=$(docker inspect -f '{{.State.Pid}}' sparkDash 2>/dev/null)
echo "$PID" > run/sparkdash.pid
# Verify exactly one listener on :5555
OWN=$(ss -ltnp 2>/dev/null | grep ':5555 ' | grep -oP 'pid=\K[0-9]+' | sort -u)
echo "container main pid (host ns): $PID (pidfile: run/sparkdash.pid)"
echo ":5555 owned by pid(s): $OWN — must match the container process"