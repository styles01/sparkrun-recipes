cd ~/imagination-studio || exit 1
echo "=== local_models: the actual HTTP probe ==="
grep -n "def local_models\|def endpoint_of" -A 22 app/slate.py | head -50
echo
echo "=== what does endpoint_of do to an mDNS name ==="
~/.local/opt/tensorfold-studio/venv/bin/python -c "
import sys; sys.path.insert(0,'/home/jaita/imagination-studio/app')
import slate
for e in ['http://larryspark.local:8000/v1','http://192.168.2.185:8000/v1','http://10.200.0.1:8000/v1']:
    try:
        print(e, '-> endpoint_of:', slate.endpoint_of(e), '| models:', slate.local_models(slate.endpoint_of(e)))
    except Exception as ex:
        print(e, 'ERR', type(ex).__name__, ex)
"
