# Sources

Inline numbers match the 2026-09-17 research pass; 10–14 are the 2026-09-20 still-factory pass.

1. [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) — official weights and `h3-prompt-writing` (`references/base-en.txt`).
2. [joeygambino/MiniMax-H3-Multishot-Workflow](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow) — last-frame → first-frame chain, `context_pin`, preview shot 1, texture ratchet on long extend-takes.
3. [Kling: character consistency guide](https://kling.ai/blog/ai-character-consistency-guide) — master description, identical keywords, negative/forbidden, reference images.
4. [StudioBinder shot list](https://www.studiobinder.com/blog/shot-list-template-free-download) — scene, shot, description, size, movement, subject.
5. [Pixel Valley: script continuity sheet](https://pixelvalleystudio.com/pmf-articles/script-continuity-sheet-and-other-important-film-production-notes) — what changed; stills of costumes/props; circle takes.
6. [Memento (arXiv 2606.14667)](https://arxiv.org/html/2606.14667v1) — identity grounding vs local continuation.
7. [StoryMem (arXiv 2512.19539)](https://arxiv.org/html/2512.19539v1) — visual memory for multi-shot story.
8. [GroundShot (arXiv 2606.20799)](https://arxiv.org/html/2606.20799v1) — entity-grounded shot scheduling.
9. [FL2VA first/last frame notes](https://minimax3.com/blog/minimax-h3-first-last-frame) — path between stills, not two descriptions; alignment line + one shot.
10. [ComfyUI MiniMax H3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) — `MiniMaxH3ImageToVideo` is FL2VA; I2V template is first/last-frame. Native 16:9 canvas 1344×768.
11. [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — FL2VA accepts 0 / 1 / 2 images (T2VA / first-or-last / first-and-last). Ref2VA is a different checkpoint.
12. [Qwen-Image-2.1 on one Spark](https://www.smfclearinghouse.com/blog/2026-09-20-qwen-image-21-one-spark) — SMF measured still factory: 1344×768 / 25 step **24 s**; edits ~**28 s**; `TextEncodeQwenImageEdit` + `VAEEncode` (not `image_1=` on `TextEncodeQwenImage21`).
13. [CANVAS (arXiv 2604.13452)](https://arxiv.org/html/2604.13452v1) — long-form pipelines use storyboard keyframes as the visual anchor of each clip.
14. [CineCrew (arXiv 2609.07720)](https://arxiv.org/html/2609.07720v1) — per-clip keyframe from character sheets / locked props before video synthesis.

SMF measured numbers (walls, peaks, 262.846 s join; Image-2.1 23/23) are from spark-56bc `sigils.jsonl` and spark-d369 `qwen21-full/report.json`, published in the Clearinghouse posts. They are not vendor claims. Comfy native I2VA hop-1 + Motion-Context SaveLatent from a Qwen plate is **prescribed, not yet a PSNR pin**.
