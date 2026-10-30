#!/usr/bin/env python3
"""Give every profile a working vision route: model=vision, provider=larryspark.

Tonight's gateway moved the lane to :8001 but only 4 profiles named the vision
model; the other 16 had an empty auxiliary.vision and fell back to a text model
that rejects images ("this model does not support image input").
"""
import datetime
import glob
import os
import shutil
import yaml

stamp = datetime.date.today().strftime('%Y%m%d')
changed, kept = [], []

for cfg in sorted(glob.glob(os.path.expanduser('~/.hermes/profiles/*/config.yaml'))):
    prof = cfg.split('/profiles/')[1].split('/')[0]
    d = yaml.safe_load(open(cfg))
    lp = (d.get('providers') or {}).get('larryspark') or {}
    if ':8001' not in str(lp.get('api', '')):
        kept.append((prof, 'no gateway route'))
        continue

    aux = d.get('auxiliary') or {}
    vis = aux.get('vision') or {}

    if vis.get('model') == 'vision' and vis.get('provider') == 'larryspark':
        kept.append((prof, 'already set'))
        continue

    shutil.copy2(cfg, cfg + f'.bak-visionroute-{stamp}')
    vis['model'] = 'vision'
    vis['provider'] = 'larryspark'
    vis['timeout'] = 600
    aux['vision'] = vis
    d['auxiliary'] = aux
    yaml.safe_dump(d, open(cfg, 'w'), sort_keys=False, default_flow_style=False)
    changed.append(prof)

print("VISION ROUTE SET:", len(changed))
print(" ", changed)
print("UNTOUCHED:")
for k in kept:
    print("  ", k)
