#!/usr/bin/env python3
"""Running spend of the EXL3 program on HF Jobs, from the Jobs API: finished jobs at their
running seconds x the flavor's hourly price, running jobs at their worst case (the --timeout),
so the number printed is what the cap must cover, not a hope. Jobs are selected by the label
program=exl3 that jobs.sh sets. Prices come from `hf jobs hardware` (2026-10-08); unknown flavors
are priced as h200.

  spend.py            table + totals
  spend.py --json     machine-readable
  spend.py --cap 70   exit 1 when committed spend exceeds the cap
"""
import argparse, json, os, sys, time, urllib.request

PRICE = {"h200": 5.00, "a100-large": 2.50, "rtx-pro-6000": 2.75, "cpu-basic": 0.01, "cpu-upgrade": 0.03,
         "l40sx1": 1.80, "a10g-large": 1.50, "t4-small": 0.40}
NS = "0xBakeer"


def token():
    t = os.environ.get("HF_TOKEN")
    if t: return t
    return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()


def get(path):
    req = urllib.request.Request("https://huggingface.co" + path)
    req.add_header("Authorization", "Bearer " + token())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def seconds(job, now):
    d = job.get("durations") or {}
    for k in ("runningSecs", "running_secs"):
        if d.get(k) is not None:
            return float(d[k])
    st, fin = job.get("startedAt") or job.get("started_at"), job.get("finishedAt") or job.get("finished_at")
    if st:
        a = time.mktime(time.strptime(st[:19], "%Y-%m-%dT%H:%M:%S"))
        b = time.mktime(time.strptime(fin[:19], "%Y-%m-%dT%H:%M:%S")) if fin else now
        return max(0.0, b - a)
    return 0.0


def timeout_secs(job):
    t = job.get("timeout") or job.get("timeoutSeconds") or (job.get("spec") or {}).get("timeoutSeconds")
    return float(t) if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--label", default="program=exl3")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    now = time.mktime(time.gmtime())
    jobs = get(f"/api/jobs/{NS}")
    jobs = jobs.get("jobs", jobs) if isinstance(jobs, dict) else jobs
    k, v = a.label.split("=", 1)
    rows, finished, committed = [], 0.0, 0.0
    for j in jobs:
        labels = j.get("labels") or (j.get("metadata") or {}).get("labels") or {}
        if labels.get(k) != v: continue
        flavor = j.get("flavor") or (j.get("spec") or {}).get("flavor") or "h200"
        price = PRICE.get(flavor, PRICE["h200"])
        stage = (j.get("status") or {}).get("stage") or j.get("stage") or "?"
        secs = seconds(j, now)
        cost = secs / 3600 * price
        tmo = timeout_secs(j)
        worst = (tmo / 3600 * price) if (stage in ("RUNNING", "SCHEDULING", "SCHEDULED", "PENDING") and tmo) else cost
        rows.append({"id": j.get("id"), "name": labels.get("name") or j.get("name"), "flavor": flavor, "stage": stage,
                     "minutes": round(secs / 60, 1), "cost_usd": round(cost, 2), "worst_usd": round(worst, 2)})
        if stage in ("COMPLETED", "ERROR", "CANCELED", "CANCELLED", "FAILED", "TIMEOUT", "TIMED_OUT"):
            finished += cost
        committed += worst
    out = {"finished_usd": round(finished, 2), "committed_usd": round(committed, 2), "jobs": rows}
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        for r in rows:
            print(f"{r['id']}  {str(r['name']):34.34}  {r['flavor']:12}  {r['stage']:10}  {r['minutes']:7.1f} min  ${r['cost_usd']:6.2f}  worst ${r['worst_usd']:6.2f}")
        print(f"finished ${finished:.2f}   committed (finished + running at their timeouts) ${committed:.2f}")
    if a.cap is not None and committed > a.cap:
        print(f"over the cap of ${a.cap:.2f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
