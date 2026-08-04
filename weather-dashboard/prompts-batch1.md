# MiniMax H3 Weather Video Prompts — First 3 Test Batch

**Resolution:** 800×480
**Duration:** 15 seconds
**FPS:** 24
**No audio**
**Loop:** Crossfade, last frame matches first frame

## Prompt Design Principles

- Wide landscape, slightly elevated angle (dashboard background context)
- Motion that can repeat seamlessly (cyclical, no hard transitions)
- Small moving elements add life without distracting from weather data overlay
- Rich color palette but not overwhelming on small screen
- No text, no UI elements — pure atmospheric background
- Last frame should visually match first frame for clean crossfade loop
- Avoid camera movement (static camera) — motion comes from weather elements only

---

## Video 1: Sunny Day

**Prompt:**
> A serene sunny day landscape viewed from a slightly elevated angle. A brilliant clear blue sky fills the upper two-thirds of the frame with a bright warm sun positioned upper-right. Below, a lush green meadow stretches to the horizon with wildflowers dotting the grass. Soft heat shimmer rises from the ground. A single orange butterfly drifts slowly from left to right across the frame, wings gently fluttering. Small insects float in the warm light near the meadow. The grass sways with a gentle breeze in a continuous rhythmic motion. The lighting is golden and warm, casting soft shadows. The scene is calm, peaceful, and continuous with no scene changes. The final frame returns to the same composition as the opening frame.

**Motion elements:** Butterfly drifting L→R, heat shimmer, grass swaying, insects floating
**Color palette:** Warm golden, vivid blue sky, green meadow
**Mood:** Peaceful, warm, bright

---

## Video 2: Rainy

**Prompt:**
> A rainy day landscape viewed from a slightly elevated angle. A flat grey overcast sky fills the upper portion of the frame. Steady rain falls in visible streaks across the entire scene. Below, a wet reflective ground with shallow puddles catches the diffused light. Raindrops create expanding ripples on the puddle surfaces in continuous overlapping patterns. A single autumn leaf floats down a stream of rainwater across the foreground, bobbing gently. Fine mist rises softly from the ground where rain meets warm earth. The overall lighting is cool, muted, and diffused through the cloud cover. The rain falls at a consistent rate with no change in intensity. The scene is continuous with no scene changes. The final frame returns to the same composition as the opening frame.

**Motion elements:** Rain streaks falling, puddle ripples expanding, leaf floating in water, mist rising
**Color palette:** Cool grey, muted greens, silver reflections on water
**Mood:** Melancholy, steady, cool

---

## Video 3: Nighttime

**Prompt:**
> A tranquil nighttime landscape viewed from a slightly elevated angle. A deep dark blue-black sky fills the upper two-thirds of the frame with hundreds of tiny stars twinkling gently. A bright full moon glows in the upper-left, casting silver light across the scene. Below, silhouetted rolling hills and bare trees frame the horizon. Soft golden fireflies blink on and off, drifting slowly through the lower third of the frame near the tree line. A single shooting star streaks briefly across the upper sky about midway through. The moonlight creates soft silver rim lighting on the tree branches. The atmosphere is still and quiet with a sense of depth and vastness. The stars twinkle in a continuous gentle pattern. The scene is continuous with no scene changes. The final frame returns to the same composition as the opening frame.

**Motion elements:** Stars twinkling, fireflies blinking and drifting, shooting star, moonlight shimmer
**Color palette:** Deep blue-black, silver moonlight, golden firefly glow
**Mood:** Peaceful, vast, mysterious

---

## Post-Processing Plan (per video)

1. Generate at H3 default resolution (768p short side)
2. Crop/resize to 800×480 using ffmpeg:
   ```bash
   ffmpeg -i input.mp4 -vf "scale=800:480:force_original_aspect_ratio=increase,crop=800:480" -c:v libx264 -crf 18 -an output_800x480.mp4
   ```
3. Create seamless crossfade loop:
   ```bash
   # Extract last 2 seconds and crossfade with first 2 seconds
   ffmpeg -i output_800x480.mp4 -t 2 -c copy last_2s.mp4
   ffmpeg -i output_800x480.mp4 -ss 13 -t 2 -c copy first_2s.mp4
   # Crossfade using xfade filter
   ffmpeg -i output_800x480.mp4 -i last_2s.mp4 -filter_complex "[1]format=yuv420p[loop];[0][loop]xfade=transition=fade:duration=1:offset=14" -c:v libx264 -crf 18 -an final_loop.mp4
   ```
4. Strip audio (`-an` flag already included)
5. Verify loop point is seamless