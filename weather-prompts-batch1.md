# MiniMax H3 Weather Video Prompts — First 3 Test Batch

**Resolution:** 800×480
**Duration:** 15 seconds
**FPS:** 24
**No audio**
**Loop:** Crossfade, last frame matches first frame
**Camera:** Pointed upward toward the sky — sky fills the frame

---

## Video 1: Sunny Day

> A serene sunny sky viewed from below, camera pointed upward. Brilliant clear blue sky fills the entire frame with a bright warm sun positioned upper-right. A few wispy cirrus clouds drift slowly across the upper portion. An orange butterfly drifts gently from left to right through the frame, wings softly fluttering. Small insects float in the warm sunlit air. Subtle heat shimmer near the edges. The lighting is golden and warm with lens flare from the sun. Calm, peaceful, continuous with no scene changes. The final frame returns to the same composition as the opening.

**Motion:** Butterfly drifting L→R, heat shimmer, clouds drifting, insects floating
**Colors:** Warm golden, vivid blue sky
**Mood:** Peaceful, warm, bright

---

## Video 2: Rainy

> A rainy sky viewed from below, camera pointed upward. A flat grey overcast sky fills the entire frame. Steady rain falls in visible streaks directly toward the camera, creating a sense of being in the rain. Raindrops vary in size and speed, some closer and larger, some distant and fine. Occasional water droplets hit an implied surface at the bottom edge, creating small splashes. Fine mist drifts through the frame. The lighting is cool, muted, and diffused through the cloud cover. Rain falls at a consistent rate with no change in intensity. Continuous, no scene changes. Final frame matches opening.

**Motion:** Rain streaks falling, splashes at bottom, mist drifting
**Colors:** Cool grey, muted, silver
**Mood:** Steady, cool, melancholy

---

## Video 3: Nighttime

> A tranquil night sky viewed from below, camera pointed upward. A deep blue-black sky fills the entire frame with hundreds of tiny stars twinkling gently. A bright full moon glows in the upper-left, casting soft silver light with a gentle halo. Wispy clouds drift slowly past the moon, partially obscuring it then clearing. Soft golden fireflies blink on and off, drifting through the lower portion of the frame. A single shooting star streaks briefly across the upper sky midway through. The stars twinkle in a continuous gentle pattern. Still, vast, mysterious. Continuous, no scene changes. Final frame matches opening.

**Motion:** Stars twinkling, fireflies blinking, clouds past moon, shooting star
**Colors:** Deep blue-black, silver moonlight, golden firefly glow
**Mood:** Peaceful, vast, mysterious

---

## Post-Processing (per video)

```bash
# 1. Crop/resize to 800x480
ffmpeg -i input.mp4 -vf "scale=800:480:force_original_aspect_ratio=increase,crop=800:480" -c:v libx264 -crf 18 -an output_800x480.mp4

# 2. Seamless crossfade loop (last 2s crossfaded with first 2s)
ffmpeg -i output_800x480.mp4 -i last_2s.mp4 -filter_complex "[1]format=yuv420p[loop];[0][loop]xfade=transition=fade:duration=1:offset=14" -c:v libx264 -crf 18 -an final_loop.mp4

# 3. Strip audio (already done with -an flag)
```
