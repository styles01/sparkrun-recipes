#!/usr/bin/env python3
"""Take 6: deep-water drop-off, camera enters with her, faceless fluid blob."""
import json
import os
import urllib.request

KEY = open(os.path.expanduser("~/ImaginationStudio/access-key")).read().strip()
BASE = "http://127.0.0.1:7870"

PROMPT = (
    "integrated_multimodal_description: [Shot 1] Above water, on a tropical beach, the camera close behind a "
    "young woman in a black bikini. The water in front of her is deep and dark blue, a sudden drop-off, "
    "with no sandy bottom visible. She runs the last two steps and leaps forward into the deep water, both "
    "feet leaving the ground, her arms out.\n\n"
    "[Shot 2] The camera goes into the water with her in one continuous move, following her down through a "
    "cloud of bubbles. The camera stays behind her at her back the whole time, close. She swims forward and "
    "down through clear deep blue water, kicking steadily, unaware. Everything looks fine. Nothing else is "
    "in frame.\n\n"
    "[Shot 3] The camera stops following her and holds its position in the water while she carries on "
    "swimming away into the distance, getting smaller. Right beside the camera, very close to the lens and "
    "softly out of focus, a low spreading pool of thick glossy black liquid appears. It is flat and "
    "horizontal, a shapeless blob of slow heavy tar that bulges, stretches into long thin strands, and "
    "pulls back into itself, changing outline constantly. There is no head, no limbs, no shoulders, nothing "
    "that reads as a creature. It creeps after her, ominous and quiet. She keeps swimming away and has no "
    "idea it is there.\n\n"
    "overall_soundscape: Beach surf, then one deep splash, then muffled underwater ambience and slow "
    "bubbles. Only ambient sound is heard.\n\n"
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
