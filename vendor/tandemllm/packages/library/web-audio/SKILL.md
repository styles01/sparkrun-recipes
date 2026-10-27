---
name: web-audio
description: Playing and synthesizing sound in the browser with the Web Audio API - AudioContext unlock on a gesture, oscillators, samples, gain envelopes, timing on the audio clock.
---
# Web Audio

Rules:
- One `AudioContext`, created or resumed inside a user gesture (click, key). Before that the context is suspended and nothing plays.
- Schedule with the audio clock (`ctx.currentTime + offset`), never with `setTimeout`, for anything rhythmic.
- Every sound goes through a `GainNode`; shape it with an envelope (`setValueAtTime`, `linearRampToValueAtTime`) to avoid clicks.
- Decode samples once (`decodeAudioData`), keep the buffer, create a new `AudioBufferSourceNode` per play.
- Master gain at 0.2 to 0.5. A mute toggle and a volume slider for anything with a loop.
- Stop and disconnect nodes you no longer need.

What goes wrong:
- Silence: the context is `suspended`; log `ctx.state` and call `resume()` in the handler.
- Clicks and pops: a gain jump from 0 to 1; ramp over 5 to 20 ms.
- Drift in sequences: `setInterval` used for beats; use lookahead scheduling.

Check: in the browser tool click the start control, read `ctx.state` (running), trigger a sound, read an analyser or a counter of scheduled notes, and confirm no console errors.

Done when:
- [ ] `ctx.state === 'running'` read after a real click through the browser tool.
- [ ] A counter of started sources increases after the triggering input.
- [ ] Console has 0 errors; a mute control exists and works (gain read back 0).

## Reference

Unlock and beep:
```js
let ctx, master, started = 0;
function ensure() { ctx ??= new AudioContext(); if (ctx.state !== 'running') ctx.resume(); master ??= new GainNode(ctx, { gain: 0.3 }); master.connect(ctx.destination); }
function beep(freq = 440, dur = 0.15, type = 'square') {
  ensure(); const t = ctx.currentTime;
  const osc = new OscillatorNode(ctx, { type, frequency: freq });
  const g = new GainNode(ctx, { gain: 0 });
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.6, t + 0.01); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  osc.connect(g).connect(master); osc.start(t); osc.stop(t + dur + 0.02); started++;
}
document.querySelector('#start').addEventListener('click', ensure);
window.__state = () => ({ tandem_state: 1, state: ctx?.state, started, gain: master?.gain.value });
```
Sample:
```js
const buf = await ctx.decodeAudioData(await (await fetch('hit.wav')).arrayBuffer());
function play(buf) { const s = new AudioBufferSourceNode(ctx, { buffer: buf }); s.connect(master); s.start(); }
```
Sequencer with lookahead (25 ms tick, 100 ms ahead):
```js
let next = ctx.currentTime, i = 0;
setInterval(() => { while (next < ctx.currentTime + 0.1) { beepAt(next, i++ % 4 === 0 ? 880 : 440); next += 60 / bpm / 2; } }, 25);
```

More pitfalls:
- A source reused twice: `start()` throws; make a new one.
- Volume at 1 on an oscillator; painful and clips.
