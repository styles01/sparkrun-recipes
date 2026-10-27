# WebAudio patterns

Read with `package_read("references/webaudio-patterns.md")`. Every pattern assumes the `audio()`
unlock of the skill's reference: one `AudioContext` named `ctx2` and a `master` gain at 0.3.

## Levels

- `master.gain` 0.3. A voice's own gain at most 0.2. Two voices at once stay under -12 dBFS peak.
- Never jump a gain from 0 to its level: ramp over 10 ms (`linearRampToValueAtTime`) or it clicks.
- End every voice with `exponentialRampToValueAtTime(0.001, t + decay)` and `stop(t + decay + 0.05)`.

## Ambient bed (pad)

```js
function bed(){
  const lp = new BiquadFilterNode(ctx2, { type: 'lowpass', frequency: 600, Q: 0.7 });
  lp.connect(master);
  const chord = [110, 138.6, 164.8, 207.7];                 // A minor 7 (A C E G), low
  for (const f of chord) for (const d of [-0.4, 0.4]) {    // detuned pairs for width
    const o = new OscillatorNode(ctx2, { type: 'triangle', frequency: f, detune: d * 10 });
    const g = new GainNode(ctx2, { gain: 0.03 }); o.connect(g).connect(lp); o.start();
  }
  const lfo = new OscillatorNode(ctx2, { frequency: 0.05 }), lg = new GainNode(ctx2, { gain: 300 });
  lfo.connect(lg).connect(lp.frequency); lfo.start();      // the filter breathes, 600 ± 300 Hz
}
```

## Wind and water (filtered noise)

```js
function noiseBuffer(){ const b = ctx2.createBuffer(1, ctx2.sampleRate * 2, ctx2.sampleRate), d = b.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = rand() * 2 - 1; return b; }
function wind(){ const s = new AudioBufferSourceNode(ctx2, { buffer: noiseBuffer(), loop: true }); const f = new BiquadFilterNode(ctx2, { type: 'lowpass', frequency: 400, Q: 0.6 }); const g = new GainNode(ctx2, { gain: 0.12 }); s.connect(f).connect(g).connect(master); s.start(); return { gain: g, filter: f }; }
function water(){ const s = new AudioBufferSourceNode(ctx2, { buffer: noiseBuffer(), loop: true }); const f = new BiquadFilterNode(ctx2, { type: 'bandpass', frequency: 900, Q: 0.8 }); const g = new GainNode(ctx2, { gain: 0.05 }); s.connect(f).connect(g).connect(master); s.start(); }
```
Modulate `wind().gain.gain` with the scene's wind strength each frame (0.05 to 0.2).

## Event sounds

```js
function pluck(freq){ const t = ctx2.currentTime, o = new OscillatorNode(ctx2, { type: 'triangle', frequency: freq }), g = new GainNode(ctx2, { gain: 0 }); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.18, t + 0.005); g.gain.exponentialRampToValueAtTime(0.001, t + 0.5); o.connect(g).connect(master); o.start(t); o.stop(t + 0.55); }
function thud(){ const t = ctx2.currentTime, o = new OscillatorNode(ctx2, { type: 'sine', frequency: 120 }), g = new GainNode(ctx2, { gain: 0 }); o.frequency.exponentialRampToValueAtTime(40, t + 0.15); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.2, t + 0.005); g.gain.exponentialRampToValueAtTime(0.001, t + 0.25); o.connect(g).connect(master); o.start(t); o.stop(t + 0.3); }
function chime(freq = 1320){ const t = ctx2.currentTime; for (const [m, a] of [[1, 0.12], [2.76, 0.04], [5.4, 0.02]]) { const o = new OscillatorNode(ctx2, { type: 'sine', frequency: freq * m }), g = new GainNode(ctx2, { gain: 0 }); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(a, t + 0.005); g.gain.exponentialRampToValueAtTime(0.001, t + 1.5 / m); o.connect(g).connect(master); o.start(t); o.stop(t + 1.6); } }
function whoosh(){ const t = ctx2.currentTime, s = new AudioBufferSourceNode(ctx2, { buffer: noiseBuffer() }), f = new BiquadFilterNode(ctx2, { type: 'bandpass', Q: 2 }), g = new GainNode(ctx2, { gain: 0 }); f.frequency.setValueAtTime(300, t); f.frequency.exponentialRampToValueAtTime(2000, t + 0.4); g.gain.linearRampToValueAtTime(0.15, t + 0.1); g.gain.exponentialRampToValueAtTime(0.001, t + 0.6); s.connect(f).connect(g).connect(master); s.start(t); s.stop(t + 0.7); }
```
Pitch from the scene: a bounce's `pluck(220 + speed * 2)`, a collision's `thud()`, a spark's
`chime(880 + rand() * 880)`.

## A sequencer (lookahead scheduling, never setTimeout for rhythm)

```js
let next = 0, step = 0; const bpm = 90, notes = [220, 262, 330, 392];
function tick(){ while (next < ctx2.currentTime + 0.1) { pluckAt(next, notes[step % 4]); step++; next += 60 / bpm / 2; } }
setInterval(tick, 25);
function pluckAt(t, freq){ const o = new OscillatorNode(ctx2, { type: 'triangle', frequency: freq }), g = new GainNode(ctx2, { gain: 0 }); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.15, t + 0.01); g.gain.exponentialRampToValueAtTime(0.001, t + 0.4); o.connect(g).connect(master); o.start(t); o.stop(t + 0.45); }
```
Start it inside `audio()` with `next = ctx2.currentTime + 0.05`.

## Metronome and heartbeat

- Metronome at N bpm: `pluckAt(next, step % 4 ? 880 : 1320); next += 60 / N`.
- Heartbeat: two thuds 0.18 s apart every 60/bpm s; the second at gain 0.6 of the first.

## Voices

`speechSynthesis.speak(Object.assign(new SpeechSynthesisUtterance(text), { pitch, rate }))`;
pitch 0.6 for a giant, 1.6 for a mouse, rate 0.8 to 1.1. Queue lines; `speechSynthesis.cancel()`
on restart. Speech plays outside the AudioContext (it has no gain node); keep it short.

## What goes wrong

- Silence: the context was created outside a gesture. Create it in the click handler.
- A click at the start of a sound: the gain jumped. Ramp 5 to 10 ms.
- Distortion: too many voices at full gain. Count voices; keep `master` at 0.3.
- Drift: `setInterval` timing used for notes. Schedule on `ctx2.currentTime`.
