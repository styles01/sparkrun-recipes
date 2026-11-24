#!/usr/bin/env python3
"""Take 5: three explicit shots. Shot 3 = camera holds back, blob at the lens, morphing."""
import json
import os
import urllib.request

KEY = open(os.path.expanduser("~/ImaginationStudio/access-key")).read().strip()
BASE = "http://127.0.0.1:7870"

PROMPT = (
    "integrated_multimodal_description: [Shot 1] Above water, on a tropical beach. The camera sits behind a "
    "young woman in a black bikini, close at her back. She walks to the edge of the turquoise sea and jumps "
    "in, splashing. The sea and the beach look normal and calm.\n\n"
    "[Shot 2] Underwater now, still behind her, still close. She swims forward through clear turquoise water "
    "with a pale sandy seabed below. Everything looks fine. Nothing else is in frame.\n\n"
    "[Shot 3] The camera stops following her and holds back, staying put in the water while she continues "
    "swimming away into the distance, getting smaller. Beside the camera, very close to the lens and out of "
    "focus, a small black glossy mass appears and begins to change shape. It is made of thick liquid black "
    "tendrils that pull in and unfold against each other, gathering, bulging, then collapsing small again, "
    "never holding one form, never resolving into a face. It drifts and creeps after her, ominous and quiet. "
    "She keeps swimming away and has no idea it is there.\n\n"
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
