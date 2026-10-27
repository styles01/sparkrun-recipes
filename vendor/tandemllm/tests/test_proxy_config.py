"""The proxy's config database, the env overlay and the keys at rest, on a CPU.

Run: python tests/test_proxy_config.py
"""

from __future__ import annotations

import base64
import os
import sqlite3
import stat
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import FIXTURE_KEYS, run  # noqa: E402

from proxy import config as C  # noqa: E402
from proxy import secrets as V  # noqa: E402

MASTER = base64.b64encode(b"k" * 32).decode()


def _db(env=None, path=None):
    d = tempfile.mkdtemp(prefix="pxcfg-")
    env = dict(env or {})
    env.setdefault(V.ENV, MASTER)
    return C.ConfigDB(path or os.path.join(d, "config.sqlite3"), env=env,
                      use_keychain=False).open()


def _grep(path: str, needle: str) -> bool:
    for suffix in ("", "-wal", "-shm"):
        try:
            with open(path + suffix, "rb") as f:
                if needle.encode() in f.read():
                    return True
        except OSError:
            pass
    return False


def test_fresh_file_is_version_2_at_0600():
    db = _db()
    con = sqlite3.connect(db.path)
    assert con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("2",)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"backends", "backend_secrets", "models", "model_features", "routing_rules", "users",
            "settings", "audit", "meta"} <= tables
    assert stat.S_IMODE(os.stat(db.path).st_mode) == 0o600
    assert db.schema_was == 0 and db.empty()
    db.close()
    db2 = C.ConfigDB(db.path, env={V.ENV: MASTER}, use_keychain=False).open()
    assert db2.schema_was == 2, "a second open migrates nothing"
    db2.close()


def test_an_older_v1_file_gets_the_added_columns_in_place():
    d = tempfile.mkdtemp(prefix="pxold-")
    path = os.path.join(d, "config.sqlite3")
    con = sqlite3.connect(path)
    con.executescript(C.DDL.replace("price_offpeak TEXT, ", ""))
    con.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    con.execute("INSERT INTO backends(id, kind, url, created_at, updated_at) VALUES "
                "('a', 'generic', 'http://h:1/v1', 1, 1)")
    con.commit()
    con.close()
    db = C.ConfigDB(path, env={V.ENV: MASTER}, use_keychain=False).open()
    assert db.schema_was == 1 and db.backend("a")["url"] == "http://h:1/v1"
    db.update_model("a", "m", {"price": {"input": 1, "output": 2, "offpeak": {"input": 0.5,
                                         "cached_input": None, "cache_write": None,
                                         "output": 1}}}, create=True)
    assert db.models("a")[0]["price"]["offpeak"]["output"] == 1
    db.close()


def test_inflight_no_limit_by_default_and_the_old_default_64_migrates_to_none():
    """PL1 (2026-10-06): the proxy adds no in-flight cap unless the user set one. A schema-1 file's
    64 (the default every backend got) becomes none; another number the user set stays."""
    d = tempfile.mkdtemp(prefix="pxinf-")
    path = os.path.join(d, "config.sqlite3")
    con = sqlite3.connect(path)
    con.executescript(C.DDL.replace("DEFAULT 0, timeouts", "DEFAULT 64, timeouts"))
    con.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    for bid, cap in (("old", 64), ("mine", 7)):
        con.execute("INSERT INTO backends(id, kind, url, max_inflight, created_at, updated_at) "
                    "VALUES (?, 'vllm', 'http://h:1/v1', ?, 1, 1)", (bid, cap))
    con.commit()
    con.close()
    db = C.ConfigDB(path, env={V.ENV: MASTER}, use_keychain=False).open()
    assert db.schema_was == 1
    assert db.backend("old")["max_inflight"] is None and db.backend("mine")["max_inflight"] == 7
    # a new backend, from the GUI without the field, from the environment without the variable
    db.create_backend({"id": "oa", "kind": "openai", "url": "https://api.openai.com/v1"})
    assert db.backend("oa")["max_inflight"] is None
    db.update_backend("oa", {"max_inflight": 3})
    assert db.backend("oa")["max_inflight"] == 3
    db.update_backend("oa", {"max_inflight": None})
    assert db.backend("oa")["max_inflight"] is None
    db.close()
    for v, want in (("none", None), ("", None), ("12", 12)):
        out = C.parse_env({"TANDEM_BACKENDS": "x", "TANDEM_BACKEND_X_KIND": "sglang",
                           "TANDEM_BACKEND_X_URL": "http://h:1",
                           "TANDEM_BACKEND_X_MAX_INFLIGHT": v})
        got = out["backends"]["x"].get("max_inflight", "unset")
        assert got == want or (v == "" and got in (None, "unset")), (v, got)


def test_env_creates_and_locks_and_unlocks_keeping_the_value():
    env = {"TANDEM_BACKENDS": "ds", "TANDEM_BACKEND_DS_KIND": "deepseek",
           "TANDEM_BACKEND_DS_KEY": FIXTURE_KEYS[2], "TANDEM_BACKEND_DS_MAX_INFLIGHT": "7",
           "TANDEM_ROUTE_1": "deepseek-*=ds",
           "TANDEM_PRICE_1": "ds/deepseek-chat=0.27,0.07,1.10",
           "TANDEM_PRICE_1_OFFPEAK": "0.135,0.035,0.55"}
    db = _db(env)
    db.apply_env()
    b = db.backend("ds")
    assert b["url"] == "https://api.deepseek.com/v1" and b["max_inflight"] == 7 and b["cloud"]
    assert db.backend_locks("ds") == {"kind": "TANDEM_BACKEND_DS_KIND",
                                      "max_inflight": "TANDEM_BACKEND_DS_MAX_INFLIGHT",
                                      "key": "TANDEM_BACKEND_DS_KEY"}
    assert db.key_for("ds") == FIXTURE_KEYS[2]
    assert db.key_hint("ds")["source"] == "env" and db.key_hint("ds")["last4"] == "5555"
    assert not _grep(db.path, FIXTURE_KEYS[2]), "an env key is never written"
    try:
        db.update_backend("ds", {"max_inflight": 9})
        raise AssertionError("a locked field saved")
    except C.ConfigError as exc:
        assert exc.var == "TANDEM_BACKEND_DS_MAX_INFLIGHT" and exc.status == 422
    db.update_backend("ds", {"label": "DeepSeek"})            # an unlocked field saves
    m = [x for x in db.models("ds") if x["model"] == "deepseek-chat"][0]
    assert m["price"]["output"] == 1.10 and m["locks"] == {"price": "TANDEM_PRICE_1"}
    assert m["price"]["offpeak"]["output"] == 0.55
    assert db.rules()[0]["variable"] == "TANDEM_ROUTE_1"
    path = db.path
    db.close()
    # the variable goes away: the field unlocks and keeps its last value
    env2 = {"TANDEM_BACKENDS": "ds", "TANDEM_BACKEND_DS_KIND": "deepseek"}
    db = _db(env2, path=path)
    db.apply_env()
    assert db.backend("ds")["max_inflight"] == 7 and "max_inflight" not in db.backend_locks("ds")
    assert db.key_for("ds") is None, "the env key was never stored"
    assert db.rules()[0]["source"] == "gui", "the rule stays, unlocked"
    db.update_backend("ds", {"max_inflight": 9})
    assert db.backend("ds")["max_inflight"] == 9
    db.close()


def test_env_errors_name_the_variable():
    for env, var in (({"TANDEM_BACKENDS": "x", "TANDEM_BACKEND_X_KIND": "nope"},
                      "TANDEM_BACKEND_X_KIND"),
                     ({"TANDEM_BACKENDS": "x", "TANDEM_BACKEND_X_KIND": "generic"},
                      "TANDEM_BACKEND_X_URL"),
                     ({"TANDEM_BACKENDS": "x", "TANDEM_BACKEND_X_KIND": "generic",
                       "TANDEM_BACKEND_X_URL": "http://h:1", "TANDEM_BACKEND_X_COLOR": "red"},
                      "TANDEM_BACKEND_X_COLOR"),
                     ({"TANDEM_BACKEND_Y_KIND": "generic"}, "TANDEM_BACKEND_Y_KIND"),
                     ({"TANDEM_BACKENDS": "x", "TANDEM_BACKEND_X_KIND": "generic",
                       "TANDEM_BACKEND_X_URL": "http://h:1",
                       "TANDEM_BACKEND_X_MAX_INFLIGHT": "0"}, "TANDEM_BACKEND_X_MAX_INFLIGHT"),
                     ({"TANDEM_ROUTE_1": "no-equals"}, "TANDEM_ROUTE_1"),
                     ({"TANDEM_BACKENDS": "Bad Id"}, "TANDEM_BACKENDS")):
        try:
            C.parse_env(env)
            raise AssertionError(f"accepted {env}")
        except C.ConfigError as exc:
            assert exc.var == var and var in exc.message, (exc.var, exc.message)


def test_gui_errors_name_the_field_and_keys_are_encrypted():
    db = _db()
    for doc, field in (({"id": "a", "kind": "generic", "url": "ftp://x"}, "url"),
                       ({"id": "a", "kind": "generic", "url": "http://u:p@h/v1"}, "url"),
                       ({"id": "A!", "kind": "generic", "url": "http://h"}, "id"),
                       ({"id": "a", "kind": "generic", "url": "http://h", "x": 1}, "x")):
        try:
            db.create_backend(doc)
            raise AssertionError(f"accepted {doc}")
        except C.ConfigError as exc:
            assert exc.field == field and exc.status == 422, (exc.field, exc.message)
    b = db.create_backend({"id": "oa", "kind": "openai", "key": FIXTURE_KEYS[0]})
    assert b["url"] == "https://api.openai.com/v1"
    assert db.key_for("oa") == FIXTURE_KEYS[0]
    assert db.key_hint("oa") == {"key_set": True, "last4": "1234",
                                 "sha256_6": V.hint(FIXTURE_KEYS[0])["sha256_6"],
                                 "source": "database", "readable": True}
    db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert not _grep(db.path, FIXTURE_KEYS[0]), "the key is stored as ciphertext only"
    assert not any(FIXTURE_KEYS[0] in str(a) for a in db.audit())
    db.update_backend("oa", {"key": None})
    assert db.key_for("oa") is None and not db.key_hint("oa")["key_set"]
    try:
        db.create_backend({"id": "oa", "kind": "openai"})
        raise AssertionError("a duplicate id saved")
    except C.ConfigError as exc:
        assert exc.status == 409
    db.close()


def test_users_key_shown_once_and_hashed():
    db = _db()
    assert not db.users_active()
    user, key = db.create_user({"label": "laptop"})
    assert key.startswith("tdm-") and len(key) > 30 and db.users_active()
    assert db.user_for_key(key) == "laptop" and db.user_for_key(key + "x") is None
    assert not _grep(db.path, key) and all("key" not in u or u.get("key") is None
                                           for u in db.users())
    db.update_user(user["id"], {"enabled": False})
    assert db.user_for_key(key) is None and not db.users_active()
    db.close()
    db = _db({"TANDEM_API_KEYS": "ci:tdm-0123456789abcdef"})
    db.apply_env()
    assert db.users_active() and db.user_for_key("tdm-0123456789abcdef") == "ci"
    assert db.users()[-1]["source"] == "env"
    db.close()


def test_rules_env_rule_cannot_change_and_unknown_backend_refused():
    db = _db({"TANDEM_BACKENDS": "a", "TANDEM_BACKEND_A_KIND": "generic",
              "TANDEM_BACKEND_A_URL": "http://h:1", "TANDEM_ROUTE_3": "x-*=a"})
    db.apply_env()
    env_rule = db.rules()[0]
    got = db.set_rules([{"pattern": "y-*", "backend": "a"}, env_rule])
    assert [r["pattern"] for r in got] == ["y-*", "x-*"]
    for rules in ([{"pattern": "y-*", "backend": "a"}],
                  [dict(env_rule, pattern="z-*")],
                  [env_rule, {"pattern": "q", "backend": "nope"}]):
        try:
            db.set_rules(rules)
            raise AssertionError(f"accepted {rules}")
        except C.ConfigError:
            pass
    assert C.glob("gpt-*", "gpt-5.4-mini") and not C.glob("gpt-*", "o4")
    db.close()


def test_secrets_sources_and_fail_closed():
    d = tempfile.mkdtemp(prefix="pxsec-")
    v = V.Vault.load(d, env={}, use_keychain=False)
    assert v.source == "file"
    kf = os.path.join(d, V.KEY_FILE)
    assert stat.S_IMODE(os.stat(kf).st_mode) == 0o600
    ct, nonce, kid = v.encrypt("b1", FIXTURE_KEYS[1])
    assert FIXTURE_KEYS[1].encode() not in ct
    assert V.Vault.load(d, env={}, use_keychain=False).decrypt("b1", ct, nonce, kid) == \
        FIXTURE_KEYS[1], "the key file is read back, not made again"
    try:
        v.decrypt("b2", ct, nonce, kid)
        raise AssertionError("a ciphertext moved to another row decrypted")
    except V.SecretError:
        pass
    other = V.Vault.load(d, env={V.ENV: MASTER}, use_keychain=False)
    assert other.source == "env"
    try:
        other.decrypt("b1", ct, nonce, kid)
        raise AssertionError("a wrong master key decrypted")
    except V.SecretError as exc:
        assert "another master key" in str(exc)
    fake_kc = [sys.executable, "-c", f"print('{MASTER}')"]
    kc = V.Vault.load(d, env={}, keychain_cmd=fake_kc)
    assert kc.source == "keychain" and kc.key_id == other.key_id
    missing = [sys.executable, "-c", "import sys; sys.exit(44)"]
    assert V.Vault.load(d, env={}, keychain_cmd=missing).source == "file"
    try:
        V.Vault.load(d, env={V.ENV: "not base64!"}, use_keychain=False)
        raise AssertionError("a bad env key loaded")
    except V.SecretError:
        pass
    assert V.hint(None) == {"key_set": False, "last4": None, "sha256_6": None}


if __name__ == "__main__":
    run(globals())
