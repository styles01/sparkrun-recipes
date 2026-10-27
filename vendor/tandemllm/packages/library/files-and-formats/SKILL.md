---
name: files-and-formats
description: Reading and writing files and data formats correctly - text encodings (UTF-8, BOM, latin-1), newlines, binary versus text, JSON, YAML, CSV with quoting, large files streamed, atomic writes, and paths with pathlib.
---
# Files, encodings and formats

Rules:
- Always name the encoding: `open(p, encoding="utf-8")`. Unknown origin: try `utf-8-sig` (strips a BOM), then detect or ask; never `errors="ignore"` silently.
- Paths with `pathlib.Path`, resolved, joined with `/`. Say which path failed.
- JSON: `json.load`/`json.dump(..., ensure_ascii=False, indent=2)`; dates as ISO strings; `Decimal` for money via a custom default.
- CSV: the `csv` module (or pandas), `newline=""` on open, explicit `delimiter`, quoting handled by the module. Never `split(",")`.

What goes wrong:
- `UnicodeDecodeError` at byte 0: a BOM or a latin-1 file; `UnicodeEncodeError` on print: the terminal is not UTF-8 (`PYTHONIOENCODING=utf-8`).
- Windows newlines (`\r\n`) left in values; `.rstrip("\r\n")` or universal newlines.
- A CSV field with a comma or a newline inside quotes broken by hand-splitting.

Check: round-trip a sample (write, read back, compare) with a non-ASCII string, a quoted comma field and an empty field; print the byte count and `file` output.

Done when:
- [ ] Round-trip equality shown for a sample with unicode, a quoted comma and an empty value.
- [ ] `file path` (or the first bytes in hex) and the byte size of the written file in the result.
- [ ] A read of a wrong-encoding or missing file produces one clear error line, not a traceback.

## Reference

```python
import csv, json, os, yaml
from pathlib import Path
def read_text(p: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try: return p.read_text(encoding=enc)
        except UnicodeDecodeError: continue
    raise ValueError(f"cannot decode {p}")
def write_atomic(p: Path, data: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
def read_csv(p: Path):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
def write_csv(p: Path, rows, fields):
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
rows = [{"name": "Zoë", "note": 'a, "b"', "n": ""}]
write_csv(Path("/tmp/t.csv"), rows, ["name", "note", "n"]); assert read_csv(Path("/tmp/t.csv")) == rows
cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
```
Large file by lines: `with open(p, encoding="utf-8") as f: for line in f: ...`; by chunks: `while chunk := f.read(1 << 20): ...` with `"rb"`.
JSON lines: one object per line, `for line in f: obj = json.loads(line)`.
Inspect bytes: `head -c 16 file | xxd`; `file -I path`; `iconv -f cp1252 -t utf-8 in > out`.
Node: `fs.readFileSync(p, 'utf8')`, `JSON.parse`, `csv-parse` for CSV, `fs.writeFileSync(tmp); fs.renameSync(tmp, p)`.

More rules:
- YAML: `yaml.safe_load` only. Beware `no`, `yes`, `on`, `1e3`, leading zeros becoming non-strings; quote them.
- Write atomically: write to `path.with_suffix(".tmp")`, `fsync`, `os.replace`. Stream large files line by line or in chunks; never `read()` a 2 GB file.

More pitfalls:
- A crash during write leaves a half file; the atomic pattern avoids it.
- Binary opened in text mode; corrupted bytes.
