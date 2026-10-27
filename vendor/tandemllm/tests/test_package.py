"""P5, the installable package: pyproject.toml against the tree, the version mapping, the page in
an installed layout, and the `tandem-dash` commands besides `serve`. The wheel itself is built and
installed into a clean venv by tools/package_check.sh (CI and the night pass).

    python tests/test_package.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import shutil
import stat
import sys
import tempfile
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from proxy import __main__ as cli_main  # noqa: E402
from proxy import _version  # noqa: E402

KEY = "a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="
# packages of the wheel that hold data alone (no .py): the built page and the shipped texts
DATA_ONLY = ("proxy.dashboard", "proxy.packages")


def _toml() -> dict:
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def test_pyproject_matches_the_tree():
    t = _toml()
    p, st = t["project"], t["tool"]["setuptools"]
    assert p["dependencies"] == ["cryptography>=42"], "the dash needs cryptography alone"
    assert not any(d.split(">")[0].split("=")[0].strip().lower() in ("torch", "numpy",
                                                                    "transformers")
                   for d in p["dependencies"] + p["optional-dependencies"]["host"])
    assert p["scripts"]["tandem-dash"] == "proxy.__main__:main"
    assert callable(cli_main.main)
    dirs = st.get("package-dir", {})
    for pkg in st["packages"]:
        d = dirs.get(pkg, pkg.replace(".", "/"))
        assert os.path.isdir(os.path.join(ROOT, d)), (pkg, d)
        if pkg.startswith(("proxy", "server")) and not pkg.startswith(DATA_ONLY):
            assert any(f.endswith(".py") for f in os.listdir(os.path.join(ROOT, d))), pkg
    # every Python package under proxy/ and server/ is listed, so nothing the code imports is lost
    for top in ("proxy", "server"):
        for d, subs, files in os.walk(os.path.join(ROOT, top)):
            subs[:] = [s for s in subs if s != "__pycache__"]
            if any(f.endswith(".py") for f in files):
                pkg = os.path.relpath(d, ROOT).replace(os.sep, ".")
                assert pkg in st["packages"], f"{pkg} is not in pyproject.toml's packages"
    for f in p["license-files"] + [p["readme"]]:
        assert os.path.exists(os.path.join(ROOT, f)), f
    assert os.path.exists(os.path.join(ROOT, "dashboard", "dist", "index.html"))


def test_the_shipped_texts_are_in_the_wheel():
    """The standalone proxy runs the conductor (P4), whose schema reads packages/seed.json at
    import: every file under packages/ must be package data of the wheel (2026-10-04 night pass:
    `tandem-dash serve` died on a missing seed.json)."""
    import fnmatch
    st = _toml()["tool"]["setuptools"]
    assert "proxy.packages" in st["packages"]
    assert st["package-dir"]["proxy.packages"] == "packages"
    globs = st["package-data"]["proxy.packages"]
    base = os.path.join(ROOT, "packages")
    files = [os.path.relpath(os.path.join(d, f), base).replace(os.sep, "/")
             for d, _, fs in os.walk(base) for f in fs]
    assert "seed.json" in files and len(files) > 10
    def covered(rel):
        # setuptools: `*` within one directory, `**/` any depth below
        return any(fnmatch.fnmatch(rel, g) and (g.count("/") == rel.count("/") or "**" in g)
                   for g in globs)
    missing = [f for f in files if not covered(f)]
    assert not missing, f"not in the wheel: {missing[:5]}"


def test_the_shipped_texts_in_an_installed_layout():
    """A wheel puts packages/ at proxy/packages; a checkout keeps it at the root and wins."""
    from server.conductor import packages
    d = tempfile.mkdtemp(prefix="pkg-texts-")
    try:
        assert packages.shipped_root(d) == d, "nothing anywhere: the root, as before"
        os.makedirs(os.path.join(d, "proxy", "packages"))
        with open(os.path.join(d, "proxy", "packages", "seed.json"), "w") as f:
            f.write("{}")
        assert packages.shipped_root(d) == os.path.join(d, "proxy")
        os.makedirs(os.path.join(d, "packages"))
        with open(os.path.join(d, "packages", "seed.json"), "w") as f:
            f.write("{}")
        assert packages.shipped_root(d) == d, "a checkout wins"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert packages.SHIPPED_ROOT == ROOT, "this checkout reads its own packages/"
    assert packages.Library()._dirs({"skill_dirs": ["{repo}/packages/skills"]}) == [
        os.path.join(ROOT, "packages", "skills")]


def test_version_mapping():
    assert _version.pep440("0.3.29-test") == "0.3.29.dev0"
    assert _version.pep440("0.4.0") == "0.4.0"
    assert _version.pep440("0.4.0-rc2") == "0.4.0rc2"
    assert _version.pep440("1.0.0-b1") == "1.0.0b1"
    try:
        _version.pep440("next")
        raise AssertionError("a bad VERSION passed")
    except ValueError:
        pass
    with open(os.path.join(ROOT, "VERSION")) as f:
        assert _version.release() == f.read().strip()
    assert _version.PEP440 == _version.pep440(_version.release())


def test_the_page_in_an_installed_layout():
    d = tempfile.mkdtemp(prefix="pkg-")
    try:
        os.makedirs(os.path.join(d, "server"))
        os.makedirs(os.path.join(d, "proxy", "dashboard"))
        shutil.copy(os.path.join(ROOT, "server", "static.py"), os.path.join(d, "server"))
        spec = importlib.util.spec_from_file_location("static_installed",
                                                      os.path.join(d, "server", "static.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.default_dir() == os.path.join(d, "dashboard", "dist"), "no page: the old path"
        with open(os.path.join(d, "proxy", "dashboard", "index.html"), "w") as f:
            f.write("<html></html>")
        assert mod.default_dir() == os.path.join(d, "proxy", "dashboard")
        os.makedirs(os.path.join(d, "dashboard", "dist"))
        assert mod.default_dir() == os.path.join(d, "dashboard", "dist"), "a checkout wins"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _run(argv, env) -> tuple[int, str]:
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            try:
                cli_main.main(argv)
                code = 0
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return code, buf.getvalue()


def test_check_secrets_and_backends_commands():
    d = tempfile.mkdtemp(prefix="pkg-cli-")
    try:
        db = os.path.join(d, "conf", "config.sqlite3")
        base = {"HOME": d, "PATH": os.environ.get("PATH", "")}
        envf = os.path.join(d, "dash.env")
        with open(envf, "w") as f:
            f.write("TANDEM_BACKENDS=ds,v1\nTANDEM_BACKEND_DS_KIND=deepseek\n"
                    "TANDEM_BACKEND_DS_KEY=sk-fixture-CLI-0001\n"
                    "TANDEM_BACKEND_V1_KIND=generic\nTANDEM_BACKEND_V1_URL=http://127.0.0.1:9/v1\n"
                    "TANDEM_BACKEND_V1_HOST=collector\n"
                    "TANDEM_BACKEND_V1_HOST_URL=http://127.0.0.1:9108\n"
                    f"TANDEM_SECRETS_KEY={KEY}\n")
        code, out = _run(["check", "--env-file", envf, "--config-db", db], base)
        assert code == 0, out
        assert "backends 2" in out and "host collector http://127.0.0.1:9108" in out, out
        assert "master key from env" in out and "OK" in out.splitlines()[-1]
        assert "sk-fixture-CLI-0001" not in out
        assert not os.path.exists(db), "check writes nothing"
        code, out = _run(["check", "--env-file", envf, "--config-db", db, "--host", "0.0.0.0"],
                         base)
        assert code == 0 and "WARN    bound beyond loopback" in out, out
        with open(envf, "a") as f:
            f.write("TANDEM_BACKEND_V1_HOST_DCGM=x\n")
        code, out = _run(["check", "--env-file", envf, "--config-db", db], base)
        assert code == 1 and "TANDEM_BACKEND_V1_HOST_DCGM" in out, out
        # secrets: init writes the key file once, at 0600; status reads it
        code, out = _run(["secrets", "status", "--config-db", db], base)
        assert code == 0 and "none yet" in out, out
        code, out = _run(["secrets", "init", "--config-db", db, "--no-keychain"], base)
        kf = os.path.join(d, "conf", "secrets.key")
        assert code == 0 and os.path.exists(kf), out
        assert stat.S_IMODE(os.stat(kf).st_mode) == 0o600
        first = open(kf).read()
        code, out = _run(["secrets", "init", "--config-db", db, "--no-keychain"], base)
        assert code == 0 and "nothing written" in out and open(kf).read() == first
        # backends list on an existing database, read through a copy
        from proxy import config as config_mod
        cfg = config_mod.ConfigDB(db, env={}, use_keychain=False).open()
        cfg.create_backend({"id": "local", "kind": "generic", "url": "http://127.0.0.1:8100/v1",
                            "key": "sk-fixture-CLI-0002"})
        cfg.close()
        before = os.path.getmtime(db)
        code, out = _run(["backends", "list", "--config-db", db], base)
        assert code == 0 and out.startswith("local ") and "key …0002 (database)" in out, out
        assert "sk-fixture-CLI-0002" not in out and os.path.getmtime(db) == before
        code, out = _run(["version"], base)
        assert code == 0 and out.startswith("tandemllm "), out
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    from proxy_util import run
    run(globals())
