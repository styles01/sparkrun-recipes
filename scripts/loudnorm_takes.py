#!/usr/bin/env python3
"""Two-pass EBU R128 loudness matching for Larry music A/B batches.

Policy (card t_64b86f82, 2026-10-14): the yue2 lane and the MiniMax Music 3
lane emit RAW engine output — no lane-side gain, so renders stay debuggable
and seed-replay stays byte-identical. Loudness matching is a DELIVERY step
run here on the Mac: every arm, both lanes, gets the SAME pure linear gain
gain = min(target_I - measured_I, target_TP - measured_TP), so dynamics are
never compressed and duration/format stay untouched. Arms already riding the
ceiling become TP-limited: they land quieter than target and get flagged in
the manifest for QC (their clipping is a take-quality signal — yue2's
decode() clamps to [-1,1] BEFORE the lane sees the audio, so no gain step
can undo it; the remedy is a seed-swap re-render, not processing). Raw
masters remain the authoritative files; normalized copies go to the
delivery dir.

Usage:
  loudnorm_takes.py --out DIR [--target -14] [--tp -1.0] \
      [--exclude GLOB]... [DIR|FILE.wav ...]

DIR inputs normalize every *.wav inside. Exit 0 only if every file verified
(post I within 0.5 LU of target unless TP-limited, post TP <= --tp,
post clipped samples <= pre). Writes manifest.tsv into --out.

Env: FFMPEG overrides the ffmpeg binary (default: bundled tools path, then PATH).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_FF = (
    Path.home() / ".hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffmpeg"
)


def ffmpeg_bin() -> str:
    import os

    cand = os.environ.get("FFMPEG") or (
        str(DEFAULT_FF) if DEFAULT_FF.exists() else shutil.which("ffmpeg")
    )
    if not cand:
        sys.exit("ffmpeg not found (set $FFMPEG)")
    return cand


import math


def measure(ff: str, path: Path) -> dict:
    """JSON from the loudnorm print pass + true peak/clips decoded off the
    RAW pipe. NOTE: loudnorm's pass-1 *output stream* is dynamically
    normalized garbage — only its stderr JSON is valid there — so peaks and
    clip counts must come from a separate raw decode (this was a real bug:
    peaks read off the pass-1 stream came out 9 dB low)."""
    p = subprocess.run(
        [ff, "-hide_banner", "-i", str(path),
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"measure failed for {path}:\n{p.stderr[-800:]}")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        raise RuntimeError(f"no loudnorm JSON for {path}:\n{p.stderr[-800:]}")
    j = json.loads(m.group(0))

    raw = subprocess.run(
        [ff, "-hide_banner", "-i", str(path), "-f", "f64le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if raw.returncode != 0:
        raise RuntimeError(f"raw decode failed for {path}")
    import array

    buf = array.array("d")
    clips = 0
    peak = 0.0
    data = raw.stdout
    step = 1 << 20
    for off in range(0, len(data), step):
        buf.frombytes(data[off:off + step])
        for v in buf:
            a = abs(v)
            if a > peak:
                peak = a
            if a >= 0.9999:  # full-scale riding: s16 ±3 LSB / s24 ±84 LSB
                clips += 1
        buf = array.array("d")
    return {
        "i": float(j["input_i"]),
        "tp": float(j["input_tp"]),
        "lra": float(j["input_lra"]),
        "thresh": float(j["input_thresh"]),
        "clips": clips,
        "peak_db": 20 * math.log10(peak) if peak > 0 else -120.0,
    }


def normalize(ff: str, src: Path, dst: Path, tgt: dict, pre: dict) -> float:
    """Pure linear gain = min(gain-to-I, gain-to-TP), applied with volume.
    Deliberately NOT loudnorm apply-mode: with linear=true ffmpeg silently
    falls back to DYNAMIC compression when the linear gain can't reach the
    TP ceiling (observed on s311: LRA 5.9->4.4, +0.65 LU off target). The
    clamped-linear approach keeps dynamics untouched, is deterministic, and
    simply reports TP-limited arms that land quieter than target."""
    gain_i = tgt["i"] - pre["i"]
    gain_tp = tgt["tp"] - pre["tp"]
    gain = min(gain_i, gain_tp)
    p = subprocess.run(
        [ff, "-hide_banner", "-y", "-i", str(src),
         "-af", f"volume={gain:.4f}dB",
         "-c:a", "pcm_s24le", str(dst)],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"apply failed for {src}:\n{p.stderr[-800:]}")
    return gain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="*", type=Path,
                    help="batch dirs and/or .wav files")
    ap.add_argument("--out", type=Path, required=True, help="delivery dir")
    ap.add_argument("--target", type=float, default=-14.0,
                    help="integrated LUFS target (default -14)")
    ap.add_argument("--tp", type=float, default=-1.0,
                    help="true-peak ceiling dBTP (default -1.0)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob to skip inside dirs (repeatable)")
    args = ap.parse_args()

    import fnmatch

    wavs: list[Path] = []
    for inp in args.inputs:
        if inp.is_dir():
            for w in sorted(inp.glob("*.wav")):
                if not any(fnmatch.fnmatch(w.name, g) for g in args.exclude):
                    wavs.append(w)
        elif inp.suffix.lower() == ".wav":
            wavs.append(inp)
    # de-dupe identical basenames loudly rather than silently overwrite
    seen: dict[str, Path] = {}
    for w in wavs:
        if w.name in seen and seen[w.name].resolve() != w.resolve():
            sys.exit(f"basename collision: {w} vs {seen[w.name]}")
        seen[w.name] = w
    if not wavs:
        sys.exit("no .wav inputs found")

    ff = ffmpeg_bin()
    args.out.mkdir(parents=True, exist_ok=True)
    tgt = {"i": args.target, "tp": args.tp}
    rows = []
    fails = 0
    for w in wavs:
        pre = measure(ff, w)
        dst = args.out / w.name
        gain = normalize(ff, w, dst, tgt, pre)
        post = measure(ff, dst)
        tp_limited = (tgt["tp"] - pre["tp"]) < (tgt["i"] - pre["i"]) - 0.01
        expected_i = pre["i"] + gain
        ok = (
            post["tp"] <= tgt["tp"] + 0.02
            and abs(post["i"] - expected_i) <= 0.3
            and (post["clips"] <= pre["clips"])
        )
        fails += 0 if ok else 1
        rows.append({
            "file": w.name, "src": str(w.parent.name),
            "in_i": pre["i"], "in_tp": pre["tp"], "in_lra": pre["lra"],
            "in_clips": pre["clips"],
            "gain": round(gain, 2),
            "out_i": post["i"], "out_tp": post["tp"], "out_clips": post["clips"],
            "verdict": "TP-limited" if tp_limited else ("OK" if ok else "FAIL"),
        })
        print(f"{w.name:28s} {pre['i']:+6.2f} LUFS -> {post['i']:+6.2f} "
              f"TP {post['tp']:+5.2f} clips {pre['clips']}->{post['clips']} "
              f"[{rows[-1]['verdict']}]")

    man = args.out / "manifest.tsv"
    with man.open("w") as f:
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\n{len(rows)} files -> {args.out}  manifest: {man}")
    if fails:
        sys.exit(f"{fails} file(s) FAILED verification")


if __name__ == "__main__":
    main()
