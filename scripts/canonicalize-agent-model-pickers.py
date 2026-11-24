#!/usr/bin/env python3
"""Audit + update each profile's DGX-Spark model picker to canonical names.

Does not alter the user's primary provider or model defaults, except Loca:
her model.default is intentionally empty to let the daily-driver lane decide.
"""
import glob, os, shutil, yaml, datetime

PROFILES = os.path.expanduser("~/.hermes/profiles")
CANON = ["DeepSeek-4.1-Flash", "GLM-5.3-Flash-EXL3", "Qwen3.8-Flash-Next"]

for path in sorted(glob.glob(os.path.join(PROFILES, "*", "config.yaml"))):
    profile = os.path.basename(os.path.dirname(path))
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    provs = cfg.setdefault("providers", {})
    spark = provs.get("larryspark")
    if not isinstance(spark, dict):
        print(f"SKIP {profile}: no larryspark provider")
        continue
    old = spark.get("models", [])
    spark["models"] = list(CANON)
    spark["default_model"] = "DeepSeek-4.1-Flash"
    # Keep Loca's selector unpinned; her provider.default_model must remain empty.
    if profile == "loca":
        (cfg.setdefault("model", {}))["default"] = ""
        (cfg["model"])["provider"] = "larryspark"
        cfg["model"]["base_url"] = spark.get("api", "http://larryspark.local:8001/v1")
        spark["default_model"] = ""
    backup = path + ".bak-picker-" + datetime.date.today().strftime("%Y%m%d")
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, width=120)
    # verify the saved structure
    check = yaml.safe_load(open(path))
    got = check["providers"]["larryspark"]["models"]
    assert got == CANON, (profile, got)
    if profile == "loca":
        assert check["model"]["default"] == "" and check["model"]["provider"] == "larryspark"
        assert check["providers"]["larryspark"]["default_model"] == ""
    print(f"UPDATED {profile}: {old} -> {got}; default={check.get('model',{}).get('default')!r}/{check.get('model',{}).get('provider')!r}")
