#!/usr/bin/env python3
"""Fire take 2 of the underwater shot with the corrected prompt."""
import json
import os
import urllib.request

KEY = open(os.path.expanduser("~/ImaginationStudio/access-key")).read().strip()
BASE = "http://127.0.0.1:7870"

PROMPT = (
    "integrated_multimodal_description: [Shot 1] Single continuous underwater shot, camera behind her. "
    "A young woman in a black bikini swims down through clear turquoise water toward a pale sandy seabed, "
    "her body angled forward and away from the camera, hair drifting behind her. In the blurred distance "
    "behind her, a matte-black mass of thin writhing tendrils and hooked fingertips slowly gathers out of "
    "the murk, its shape never fully resolving, always partly hidden by haze and her own drifting hair. "
    "The tendrils unfurl and probe the water, then retract as the camera drifts forward. She never turns "
    "around, never sees it. Ambiguous, tense, restrained. The creature is only ever suggested, never shown "
    "clearly.\n\n"
    "overall_soundscape: Muffled underwater ambience, distant water movement, slow bubbles. "
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
        "seconds": 5,
    },
}).encode()

req = urllib.request.Request(
    BASE + "/api/jobs", data=body,
    headers={"Content-Type": "application/json", "X-Studio-Key": KEY},
)
out = json.loads(urllib.request.urlopen(req, timeout=60).read())
print("JOB:", json.dumps(out)[:300])
