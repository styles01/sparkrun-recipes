"""The files in ops/monitoring/, checked where they can be checked without the cluster.

  * every qse_* name the alert rules and the Grafana dashboard query is a metric this server
    registers (a renamed metric breaks a panel silently otherwise); every dgx_* name is one the
    exporter publishes (with the hold patch);
  * the PrometheusRule CR carries exactly the plain file's groups; promtool's own rule tests pass
    (where promtool is installed -- the Mac has it, the box does not);
  * the ServiceMonitor scrapes the two proxied metrics paths, with the token on the metrics one only
    (the reverse-proxy config that maps them to the engine's routes is site-specific and not in this
    repository);
  * the dashboard JSON is what tools/grafana_dashboard.py generates; the Secret carries no value.

Run: python tests/test_monitoring.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server import metrics  # noqa: E402
from tools import grafana_dashboard  # noqa: E402

MON = os.path.join(ROOT, "ops", "monitoring")
EXPORTER = {"dgx_qse_hold_active", "dgx_qse_hold_file_age_seconds",
            "dgx_gpu_temperature_celsius", "dgx_gpu_power_watts", "dgx_gpu_sm_clock_mhz"}


def _read(name: str) -> str:
    return open(os.path.join(MON, name)).read()


def _names(text: str) -> set[str]:
    return set(re.findall(r"\b((?:qse|dgx)_[a-z0-9_]+)\b", text))


def test_every_queried_series_exists():
    exprs = [t["expr"] for p in grafana_dashboard.dashboard()["panels"]
             for t in p.get("targets", [])]
    exprs += [a["expr"] for a in grafana_dashboard.dashboard()["annotations"]["list"]
              if "expr" in a]
    bad = unknown(_names("\n".join(exprs) + _read("qse-engine-rules-plain.yaml")))
    assert not bad, f"queried but never published: {sorted(bad)}"
    assert len(exprs) >= 30
    assert unknown({"qse_bogus_total", "qse_verify_seconds_bucket", "dgx_nope"}) == \
        {"qse_bogus_total", "dgx_nope"}, "the check itself works"


def unknown(names: set[str]) -> set[str]:
    registered = {m.name for m in metrics.REGISTRY._metrics}
    hist = {m.name for m in metrics.REGISTRY._metrics if isinstance(m, metrics.Histogram)}
    bad = set()
    for name in names:
        if name.startswith("dgx_"):
            if name not in EXPORTER:
                bad.add(name)
            continue
        base = re.sub(r"_(bucket|sum|count)$", "", name)
        if name not in registered and not (base in hist and base != name):
            bad.add(name)
    return bad


def test_the_rule_cr_is_the_plain_file():
    plain = _read("qse-engine-rules-plain.yaml").split("groups:\n", 1)[1]
    cr = _read("qse-engine-rules.yaml").split("  groups:\n", 1)[1]
    unindented = "".join(l[2:] if l.startswith("  ") else l for l in cr.splitlines(True))
    assert unindented == plain
    assert "kind: PrometheusRule" in _read("qse-engine-rules.yaml")
    assert "release: kps" in _read("qse-engine-rules.yaml")


def test_promtool():
    tool = shutil.which("promtool")
    if tool is None:
        print("    (promtool not installed here: skipped)")
        return
    r = subprocess.run([tool, "check", "rules", "qse-engine-rules-plain.yaml"], cwd=MON,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([tool, "test", "rules", "qse-engine-rules.test.yaml"], cwd=MON,
                       capture_output=True, text=True)
    assert r.returncode == 0 and "SUCCESS" in r.stdout, r.stdout + r.stderr


def test_the_scrape_paths():
    scrape = _read("qse-engine-scrape.yaml")
    paths = re.findall(r"^\s+path:\s*(\S+)", scrape, re.M)
    assert sorted(paths) == ["/metrics/qse-engine", "/metrics/qse-engine-up"]
    # the metrics endpoint carries the token Secret, the liveness endpoint none
    blocks = scrape.split("- port: metrics")[1:]
    auth = {re.search(r"path:\s*(\S+)", b).group(1): "authorization:" in b for b in blocks}
    assert auth == {"/metrics/qse-engine-up": False, "/metrics/qse-engine": True}, auth
    assert "name: qse-metrics-token" in scrape
    jobs = re.findall(r"targetLabel: job\s*\n\s*replacement: (\S+)", scrape)
    assert sorted(jobs) == ["qse-engine", "qse-engine-health"], "never job=vllm (the roster)"


def test_the_dashboard_json_is_generated():
    want = json.dumps(grafana_dashboard.dashboard(), indent=2) + "\n"
    assert _read("qse-engine-dashboard.json") == want, "run tools/grafana_dashboard.py"
    cm = _read("qse-engine-dashboard-configmap.yaml")
    assert cm.startswith("# GENERATED") and '"grafana_dashboard": "1"' in cm
    ids = [p["id"] for p in grafana_dashboard.dashboard()["panels"]]
    assert len(ids) == len(set(ids))


def test_the_secret_carries_no_value():
    s = _read("qse-metrics-token-secret.example.yaml")
    assert re.search(r'token: "<QSE_METRICS_TOKEN', s)
    assert not re.search(r"\b[0-9a-f]{40,}\b", s)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
