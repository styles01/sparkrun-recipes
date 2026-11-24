cd ~/imagination-studio || exit 1
echo "=== where is the web UI served from? ==="
ls -d static web ui dist frontend 2>/dev/null
grep -n "StaticFiles\|FileResponse\|mount(" app/server.py | head -10
echo
echo "=== find any built js referencing Connections ==="
find . -maxdepth 4 -name "*.js" -newermt "-60 days" 2>/dev/null | grep -v node_modules | head -10
