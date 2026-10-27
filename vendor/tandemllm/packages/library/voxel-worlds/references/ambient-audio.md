# Ambient audio

Read with `package_read("references/ambient-audio.md")`. One context, one master gain, two noise
beds that follow the scene, two event sounds gated by the time of day.

```js
// js/creatures/audio.js
let ctx = null, master = null, windGain = null, started = false;
function noiseBuffer(c){ const b = c.createBuffer(1, c.sampleRate * 2, c.sampleRate), d = b.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = rand() * 2 - 1; return b; }
export function setupAudio(){                                   // called from the first pointerdown or keydown
  if (started) return; started = true;
  ctx = new AudioContext(); master = ctx.createGain(); master.gain.value = 0.3; master.connect(ctx.destination);
  const noise = noiseBuffer(ctx);
  const ws = ctx.createBufferSource(); ws.buffer = noise; ws.loop = true;                       // wind
  const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 420; lp.Q.value = 0.6;
  windGain = ctx.createGain(); windGain.gain.value = 0; ws.connect(lp).connect(windGain).connect(master); ws.start();
  const wsrc = ctx.createBufferSource(); wsrc.buffer = noise; wsrc.loop = true;                 // water
  const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 900; bp.Q.value = 0.7;
  const wg = ctx.createGain(); wg.gain.value = 0.045; wsrc.connect(bp).connect(wg).connect(master); wsrc.start();
}
export function setAudio(on){ if (ctx) (on ? ctx.resume() : ctx.suspend()); }      // the panel's audio toggle
function chirp(){
  if (!ctx || ctx.state !== 'running') return;
  const o = ctx.createOscillator(), g = ctx.createGain(), f = 1800 + rand() * 1600, t = ctx.currentTime;
  o.type = 'sine'; o.frequency.setValueAtTime(f, t); o.frequency.exponentialRampToValueAtTime(f * 1.4, t + 0.08);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.06, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
  o.connect(g).connect(master); o.start(); o.stop(t + 0.2);
}
function bell(){
  if (!ctx || ctx.state !== 'running') return;
  const t = ctx.currentTime;
  for (const [f, a, d] of [[520, 0.10, 2.5], [782, 0.04, 1.8]]) {                              // two partials
    const o = ctx.createOscillator(), g = ctx.createGain(); o.type = 'sine'; o.frequency.value = f;
    g.gain.setValueAtTime(a, t); g.gain.exponentialRampToValueAtTime(0.0001, t + d);
    o.connect(g).connect(master); o.start(); o.stop(t + d + 0.1);
  }
}
export function updateAudio(dt){
  if (!ctx || !windGain || ctx.state !== 'running') return;
  windGain.gain.value = Math.min(0.22, wind.strength * 0.16);                                  // the wind system's 0..1 strength
  if (dn.day > 0.5 && rand() < dt * 0.35) chirp();                                             // a bird about every 3 s by day
  if (dn.night > 0.5 && rand() < dt * 0.06) bell();                                            // a bell about every 17 s by night
}
```

Rules:
- `setupAudio` runs inside a user gesture (`renderer.domElement.addEventListener('pointerdown', setupAudio, { once: true })`), or the context stays suspended and the page is silent.
- The master at 0.3; the beds under 0.25 together; an event sound's own gain 0.04 to 0.12. The probe wants RMS above -40 dBFS and peak below -1 dBFS: these levels give about -28 dBFS RMS.
- A click sound (the shrine bell, a chime on a wish) is the same `bell()` shape at a higher gain, called from the click handler, not from the loop.
- A rain toggle adds a third noise bed: lowpass 1200 Hz, gain 0.08 while `state.rain`.
- Every `rand()` here is the seeded one, so the chirps fall at the same times for the same seed.
