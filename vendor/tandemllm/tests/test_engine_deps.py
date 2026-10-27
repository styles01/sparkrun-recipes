"""The image path's packages are in install.sh's lock, and the image processor's backend is pinned.

A user installed TandemLLM in a plain PyTorch container and had to add pillow and torchvision by
hand (2026-10-05): server/images.py decodes with PIL, and transformers 5.12 builds Qwen3.8's image
processor on torchvision.transforms.v2 when torchvision is there. Neither was in the lock. These
tests fail when the image path imports a package the lock does not list:

    the static imports of server/images.py           (no third-party package needed to run)
    what a decode and one processor call load        (a subprocess: decode a PNG, build the
                                                      processor on the pinned backend from a
                                                      preprocessor_config.json like Qwen3.8's,
                                                      run it; the processor module's hard
                                                      imports, and every loaded module whose
                                                      distribution the image packages require)

and check `load_processor`'s pin: the backend asked for or `BackendMissing`, never transformers'
silent fallback. CPU only; the subprocess part needs transformers (skipped without it).

Run: python tests/test_engine_deps.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# a module's top-level name -> its distribution, where the two differ
DIST_OF = {"PIL": "pillow", "yaml": "pyyaml", "google": "protobuf"}

# Qwen3.8's preprocessor_config.json (Qwen/Qwen3.8-27B-FP8), as shipped
QWEN38_PRE = {"size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16,
              "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5], "processor_class": "Qwen3VLProcessor",
              "image_processor_type": "Qwen2VLImageProcessorFast"}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock() -> dict:
    """install.sh's LOCK: {normalized name: version}."""
    src = open(os.path.join(ROOT, "install.sh")).read()
    m = re.search(r"LOCK=\"\$\(cat <<'EOF'\n(.*?)\nEOF\n\)\"", src, re.S)
    assert m, "install.sh has no LOCK heredoc"
    out = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            n, _, v = line.partition("==")
            out[norm(n)] = v
    return out


def pyproject_deps() -> set:
    """The wheel's own dependencies (the engine venv installs the repo with `pip install -e .`)."""
    src = open(os.path.join(ROOT, "pyproject.toml")).read()
    m = re.search(r"^dependencies = \[(.*?)\]", src, re.S | re.M)
    return {norm(re.split(r"[<>=!~\[; ]", d.strip().strip('"'))[0])
            for d in (m.group(1).split(",") if m else []) if d.strip()}


def _repo_module(name: str) -> bool:
    return os.path.exists(os.path.join(ROOT, name)) or os.path.exists(
        os.path.join(ROOT, name + ".py"))


def test_lock_has_the_image_packages():
    lk = lock()
    assert lk.get("pillow") == "11.3.0", lk.get("pillow")
    assert lk.get("torchvision") == "0.28.0+cu130", lk.get("torchvision")
    # torchvision's build must match torch's (both from the cu130 index)
    assert lk["torch"].split("+")[1] == lk["torchvision"].split("+")[1], (lk["torch"],
                                                                         lk["torchvision"])
    return f"pillow {lk['pillow']}, torchvision {lk['torchvision']} next to torch {lk['torch']}"


def test_readme_names_the_image_packages():
    src = open(os.path.join(ROOT, "README.md")).read()
    lk = lock()
    assert f"pillow=={lk['pillow']} torchvision=={lk['torchvision']}" in src
    assert "torchvision 0.28" in src and "Pillow 11.3" in src
    return "the Quick start's requirements and the container pip line match the lock"


def test_install_check_imports_the_image_packages():
    src = open(os.path.join(ROOT, "install.sh")).read()
    assert "import torch, triton, transformers, PIL, torchvision" in src
    assert "image processor {name} (torchvision backend)" in src
    return "the post-install check imports PIL and torchvision and prints the processor class"


def test_images_py_static_imports_are_locked():
    """Every import in server/images.py (function-level ones too) is the standard library, this
    repo, or a package of the lock."""
    tree = ast.parse(open(os.path.join(ROOT, "server", "images.py")).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    lk, third = lock(), []
    for n in sorted(names):
        if n in sys.stdlib_module_names or n == "__future__" or _repo_module(n):
            continue
        third.append(n)
        assert norm(DIST_OF.get(n, n)) in lk, f"server/images.py imports {n}, not in the lock"
    assert "PIL" in third and "transformers" in third, third
    return f"third-party: {', '.join(third)}"


_PROBE = r"""
import io, json, sys
before = set(sys.modules)
sys.path.insert(0, sys.argv[1])
from server import images as I
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (333, 1000), (10, 200, 30)).save(buf, format="PNG")
pil = I.decode(buf.getvalue(), "p", I.Limits())
proc = I.load_processor(sys.argv[2], sys.argv[3])
out = proc(images=[pil], return_tensors="pt")
grid = [int(x) for x in out["image_grid_thw"][0].tolist()]
loaded = sorted({m.split(".")[0] for m in set(sys.modules) - before})
mods = [sys.modules[type(proc).__module__].__file__]
for extra in ("transformers.image_processing_backends", "transformers.image_utils"):
    if extra in sys.modules:
        mods.append(sys.modules[extra].__file__)
print(json.dumps({"loaded": loaded, "cls": type(proc).__name__, "grid": grid, "files": mods}))
"""


def _probe(backend: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "preprocessor_config.json"), "w") as f:
            json.dump(QWEN38_PRE, f)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_OFFLINE="1")
        r = subprocess.run([sys.executable, "-c", _PROBE, ROOT, d, backend], env=env,
                           capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _have_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _hard_imports(path: str) -> set:
    """The top-level names a module imports unconditionally (in its body, not under an `if` or a
    `try`): what it cannot load without."""
    out = set()
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def _closure(roots) -> set:
    """The installed distributions `roots` require, transitively (their extras left out)."""
    from importlib.metadata import PackageNotFoundError, requires
    try:
        from packaging.requirements import Requirement
    except ImportError:               # pragma: no cover - packaging is in the lock
        return set(roots)
    seen, todo = set(), list(roots)
    while todo:
        d = norm(todo.pop())
        if d in seen:
            continue
        seen.add(d)
        try:
            reqs = requires(d) or []
        except PackageNotFoundError:
            continue
        for r in reqs:
            try:
                q = Requirement(r)
            except Exception:                                      # noqa: BLE001
                continue
            if q.marker is not None and not q.marker.evaluate({"extra": ""}):
                continue
            todo.append(q.name)
    return seen


def test_what_the_image_path_loads_is_locked():
    """Decode a PNG and run Qwen3.8's processor on the pinned backend in a fresh interpreter.
    Fails when (1) the processor's own module (and transformers' backend and image utils) imports
    a package unconditionally that the lock does not list, or (2) a loaded module belongs to a
    distribution that the image path's packages require (pillow, torchvision, transformers,
    torch, transitively) and the lock does not list. A package transformers or torch imports only
    because it happens to be installed (accelerate, scipy, torchaudio in the box's venv) is
    optional: reported, not a failure."""
    if not _have_transformers():
        return "skipped (no transformers here)"
    from importlib.metadata import packages_distributions
    dists = packages_distributions()
    got = _probe("torchvision")
    assert got["cls"] == "Qwen2VLImageProcessor", got["cls"]
    assert got["grid"] == [1, 62, 20], got["grid"]
    allowed = set(lock()) | pyproject_deps()
    hard = set()
    for f in got["files"]:
        hard |= _hard_imports(f)
    bad = [n for n in sorted(hard) if not (n in sys.stdlib_module_names or n == "__future__"
                                           or _repo_module(n))
           and not any(norm(d) in allowed for d in dists.get(n, [DIST_OF.get(n, n)]))]
    assert not bad, f"the image processor's modules import packages the lock does not list: {bad}"
    assert "torchvision" in hard, sorted(hard)
    need = _closure(["pillow", "torchvision", "transformers", "torch"])
    missing, optional, seen = [], set(), set()
    for m in got["loaded"]:
        if m in sys.stdlib_module_names or _repo_module(m) or m.startswith("_"):
            continue
        for dist in dists.get(m, []):
            d = norm(dist)
            seen.add(d)
            if d in allowed:
                continue
            if d in need:
                missing.append(f"{m} ({dist})")
            else:
                optional.add(d)
    assert not missing, f"the image path needs packages the lock does not list: {missing}"
    assert {"pillow", "torchvision", "transformers", "torch"} <= seen, sorted(seen)
    return (f"{len(seen - optional)} required distributions loaded, all locked; the processor's "
            f"hard imports: {', '.join(sorted(n for n in hard if n not in sys.stdlib_module_names))}"
            + (f"; optional, installed here only: {', '.join(sorted(optional))}" if optional else ""))


def test_pin_pil_backend_resolves_pil():
    if not _have_transformers():
        return "skipped (no transformers here)"
    got = _probe("pil")
    assert got["cls"] == "Qwen2VLImageProcessorPil", got["cls"]
    return got["cls"]


def test_pin_refuses_a_silent_fallback():
    """transformers returns the other backend's class with only a warning when the asked one is
    missing; load_processor raises BackendMissing naming the package instead."""
    from server import images as I

    class Qwen2VLImageProcessorPil:
        pass

    class Qwen2VLImageProcessor:
        pass

    class Auto:
        ret = Qwen2VLImageProcessorPil

        @classmethod
        def from_pretrained(cls, snap, backend=None):
            cls.asked = backend
            return cls.ret()

    import types
    fake = types.ModuleType("transformers")
    fake.AutoImageProcessor = Auto
    real = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    try:
        try:
            I.load_processor("/nowhere", "torchvision")
            raise AssertionError("no BackendMissing")
        except I.BackendMissing as e:
            assert "torchvision==0.28.0+cu130" in str(e) and "--image-backend pil" in str(e), e
        assert Auto.asked == "torchvision"
        Auto.ret = Qwen2VLImageProcessor
        assert I.backend_of(I.load_processor("/nowhere", "torchvision")) == "torchvision"
        Auto.ret = Qwen2VLImageProcessorPil
        assert I.backend_of(I.load_processor("/nowhere", "pil")) == "pil"
        try:
            I.load_processor("/nowhere", "fast")
            raise AssertionError("no ValueError")
        except ValueError:
            pass
    finally:
        if real is not None:
            sys.modules["transformers"] = real
        else:
            del sys.modules["transformers"]
    return "torchvision asked, PIL returned: BackendMissing with the install hint"


def test_server_flag_defaults_to_torchvision():
    src = open(os.path.join(ROOT, "server", "app.py")).read()
    assert re.search(r'"--image-backend", default=os\.environ\.get\("QSE_IMAGE_BACKEND", '
                     r'"torchvision"\)', src)
    assert 'backend=getattr(a, "image_backend", "torchvision")' in src
    return "--image-backend torchvision (QSE_IMAGE_BACKEND)"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}", flush=True)
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
