cd ~/imagination-studio || exit 1
nohup setsid bash -c 'HOST=0.0.0.0 bash scripts/app.sh' > ~/studio-app.log 2>&1 < /dev/null &
sleep 30
echo "=== listener ==="
ss -ltn | grep 7870 || echo NO-LISTENER
echo "=== log ==="
tail -5 ~/studio-app.log
