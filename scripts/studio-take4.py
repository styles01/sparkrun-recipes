#!/usr/bin/env python3
"""Take 4: 10 seconds, single continuous behind-the-back shot, small creeping blob."""
import json
import os
import urllib.request

KEY = open(os.path.expanduser("~/ImaginationStudio/access-key")).read().strip()
BASE = "http://127.0.0.1:7870"

PROMPT = (
    "integrated_multimodal_description: [Shot 1] A young woman in a black bikini walks to the edge of the "
    "turquoise sea on a tropical beach and jumps in, splashing. The camera follows her from behind, close, "
    "at her back the whole time. She surfaces and swims forward through clear water, unaware.\n\n"
    "[Shot 2] Still from behind, still following her as she swims. In the far distance behind her, a small "
    "dark shape drifts into frame and begins to follow. It is small, low and indistinct, no bigger than a "
    "basketball, its outline unclear and shifting, never resolving into a face or a body. It trails her "
    "quietly, keeping its distance, creeping closer a little at a time. She keeps swimming, never turns "
    "around, never sees it. Restrained and uneasy.\n\n"
    "overall_soundscape: Beach surf, one splash, then muffled underwater ambience and slow bubbles. "
    "Only ambient sound is heard.\n\n"
    "non_diegetic_music: N/A"
)

body = json.dumps({
    "kind": "video",
    "params": {
        "prompt": PROMPT,
        "image": None,
        "preset": "small",
        "quality": "fh4",
        "seed": 1,
        "seconds": 10,
    },
}).encode()

req = urllib.request.Request(
    BASE + "/api/jobs", data=body,
    headers={"Content-Type": "application/json", "X-Studio-Key": KEY},
)
out = json.loads(urllib.request.urlopen(req, timeout=60).read())
print("JOB:", out.get("id"), out.get("status"), "| estimate:", out.get("estimate"))
