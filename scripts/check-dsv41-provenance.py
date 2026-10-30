#!/usr/bin/env python3
"""Provenance check: does anything upstream behind our dsv41 ablit lane have a newer version?"""
import json, urllib.request, datetime

def g(u, raw=False):
    r = urllib.request.Request(u, headers={'User-Agent': 'oracle/1', 'Accept': 'application/vnd.github+json'})
    b = urllib.request.urlopen(r, timeout=30).read()
    return b.decode() if raw else json.loads(b)

print("=== 1. OUR published repo (deploy source of truth) ===")
d = g('https://api.github.com/repos/styles01/deepseek-v41-1017-exl3-2x-dgx-sparks')
print("  last push:", d['pushed_at'], "| branch:", d['default_branch'])
c = g('https://api.github.com/repos/styles01/deepseek-v41-1017-exl3-2x-dgx-sparks/commits?per_page=1')[0]
print("  HEAD:", c['sha'][:10], c['commit']['committer']['date'][:10], c['commit']['message'].split('\n')[0][:70])

print("\n=== 2. The weights pack we load ===")
p = g('https://huggingface.co/api/models/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED')
print("  dealignai UNCENSORED 1017:", p['lastModified'][:10], "| files", len(p.get('siblings', [])))

print("\n=== 3. The engine base we forked (Sudosu) ===")
# find sudoingX repo used by the recipe
for cand in ['sudoingX/dgx-spark-deepseek-v41', 'sudoingx/dgx-spark-deepseek-v41']:
    try:
        r = g('https://api.github.com/repos/' + cand)
        print(f"  {r['full_name']}: pushed {r['pushed_at']} | stars {r['stargazers_count']}")
        cc = g(f"https://api.github.com/repos/{cand}/commits?per_page=2")
        for x in cc:
            print("     ", x['commit']['committer']['date'][:10], x['commit']['message'].split('\n')[0][:70])
        break
    except Exception as e:
        print("  ", cand, "ERR", getattr(e, 'code', e))

print("\n=== 4. Upstream deepseek (base model) ===")
u = g('https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017')
print("  deepseek-ai/DeepSeek-V4.1-Flash-1017:", u['lastModified'][:10])
recent = g('https://huggingface.co/api/models?author=deepseek-ai&limit=5&sort=lastModified')
print("  newest deepseek-ai repos:", [(m['id'].split('/')[-1], m['lastModified'][:10]) for m in recent[:3]])
