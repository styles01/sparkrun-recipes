cd ~/imagination-studio || exit 1
echo "=== studio_home NAME / OLD constants ==="
head -40 app/studio_home.py | grep -nE "^NAME|^OLD|NAME =|OLD =" 
echo
echo "=== what studio_home resolves to right now ==="
~/.local/opt/tensorfold-studio/venv/bin/python -c "
import sys; sys.path.insert(0,'/home/jaita/imagination-studio/app')
from studio_home import studio_home
print('studio_home ->', studio_home())
"
echo
echo "=== where do the agent settings actually live for the running server? ==="
~/.local/opt/tensorfold-studio/venv/bin/python -c "
import sys; sys.path.insert(0,'/home/jaita/imagination-studio/app')
import agents
from studio_home import studio_home
agents.HOME = studio_home()
print('agents.HOME ->', agents.HOME)
print('folder() ->', agents.folder())
import json,os
p = agents.folder()/'settings.json'
print('settings path:', p, 'exists:', p.exists())
if p.exists(): print(json.load(open(p)))
"
