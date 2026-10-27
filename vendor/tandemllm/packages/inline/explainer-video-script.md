# Explainer video: the script without the tool

This chat has no video tool, so no mp4 can be made here. Deliver the two lists the renderer needs, so the video can be rendered later:

1. `scenes`: a JSON list of scene objects in order. Each has `id`, `kind` (`title`, `number`, `schematic`, `compare`), `seconds`, `title` (3 to 5 words), `lines` (at most 3, each at most 60 characters), and for a number scene `numbers: [{"value": "...", "source": "..."}]`.
2. `cues`: `{"bed": {"chords": [...], "seconds_per_chord": 16}, "events": [{"at": "s2", "kind": "whoosh"}], "narration": []}`.

Rules: every number names its `source` (the prompt's data or a named file); a schematic scene says "schematic" in its lines; the total seconds match the asked length within 10 %.

Answer with the two JSON blocks and one line before them: the video itself needs the render tool; here is the script to render.
