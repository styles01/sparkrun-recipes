---
name: pytest-practice
description: Writing and running Python tests with pytest - test layout, fixtures, parametrize, tmp_path, monkeypatch and mocking, testing a CLI or API, and reading the summary line.
---
# pytest practice

Rules:
- `tests/test_<module>.py`, one test per behavior, named `test_<what>_<when>_<expected>`. Arrange, act, assert; one assert topic per test.
- Fixtures for setup (`tmp_path`, a client, a database); `conftest.py` for shared ones. No setup in module scope.
- `@pytest.mark.parametrize` for the same check over many inputs, including the edges: empty, one, max, negative, unicode.
- Mock only the boundary (network, time, randomness) with `monkeypatch` or `unittest.mock`; never mock the function under test.

What goes wrong:
- Tests pass only in one order (shared state); run `pytest -p no:randomly` or `--lf` to see; isolate with fixtures.
- Asserting on log text or printed output instead of the return value; use `capsys` only when stdout is the product.
- A test that writes into the repo; use `tmp_path`.

Check: the full suite in one call with the summary line visible (`N passed, 0 failed`), plus the failing-then-passing run for any bug fix.

Done when:
- [ ] Summary line of the full run in the result: 0 failed, count not lower than before the change.
- [ ] For a fix: the output of the test failing before and passing after.
- [ ] New code paths covered by at least one normal, one edge and one error test (names listed).

## Reference

```python
# tests/test_count.py
import pytest
from mytool.cli import count, main
@pytest.mark.parametrize("text,lines,words", [("", 0, 0), ("a\n", 1, 1), ("a b\nc\n", 2, 3), ("ü ö\n", 1, 2)])
def test_count(text, lines, words):
    assert count(text) == {"lines": lines, "words": words}
def test_main_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.txt")]) == 1
    assert "file not found" in capsys.readouterr().err
def test_main_reads_file(tmp_path, capsys):
    p = tmp_path / "in.txt"; p.write_text("a b\n", encoding="utf-8")
    assert main([str(p), "--json"]) == 0
    assert '"words": 2' in capsys.readouterr().out
```
Fixtures and mocking:
```python
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
def test_create(client):
    r = client.post("/items", json={"text": "a"}); assert r.status_code == 201 and r.json()["id"] == 1
def test_no_network(monkeypatch):
    monkeypatch.setattr("mytool.fetcher.httpx.get", lambda *a, **k: FakeResp(200, b"ok"))
    monkeypatch.setenv("API_KEY", "test")
```
Commands: `pytest -q`, `pytest -q -x --lf`, `pytest -q -k "count and not slow"`, `pytest -q --durations=5`, coverage: `pytest -q --cov=mytool --cov-report=term-missing`.
Expected exception: `with pytest.raises(ValueError, match="negative"): f(-1)`.

More rules:
- Run `pytest -q` for the whole suite in one call; `-x` to stop at the first failure while fixing; `-k name` to focus, then the full run again.
- A bug fix starts with a failing test that reproduces it; show it fail, then pass.

More pitfalls:
- `ModuleNotFoundError`: the package is not installed editable, or `tests/` has an `__init__.py` clash.
