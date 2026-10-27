# Day-night cycle

Read with `package_read("references/day-night.md")`. One function, called every frame, reads
`state.timeOfDay` (0 to 1, a day of `DAY_LEN` = 120 s at speed 1) and sets every colour and light.

```js
// js/scene/daynight.js
export const dn = { day: 1, night: 0, dusk: 0 };
export const sunDir = new THREE.Vector3();
const cTopNight = new THREE.Color(0x04070f), cTopDay = new THREE.Color(0x2a63c8), cTopDusk = new THREE.Color(0x2a4a86);
const cHorNight = new THREE.Color(0x0b1530), cHorDay = new THREE.Color(0xa9cdea), cHorDusk = new THREE.Color(0xffab6a);
const cSunLow = new THREE.Color(0xffa04d), cSunHigh = new THREE.Color(0xfff3e0);
const cFogDay = new THREE.Color(0x6b8fae), cFogNight = new THREE.Color(0x2b3a4e);
const skyTop = new THREE.Color(), skyHor = new THREE.Color();
export function updateDayNight(t){
  const ang = (state.timeOfDay - 0.25) * TAU;                               // 0.25 = sunrise at the east
  sunDir.set(Math.cos(ang) * 0.78, Math.sin(ang), Math.cos(ang) * 0.42).normalize();
  const elev = sunDir.y;
  dn.day = smoothstep(-0.06, 0.20, elev); dn.night = 1 - dn.day;
  dn.dusk = Math.pow(Math.min(1, Math.max(0, 1 - Math.abs(elev) / 0.30)), 1.4);   // a band around the horizon
  skyTop.copy(cTopNight).lerp(cTopDay, dn.day).lerp(cTopDusk, dn.dusk * 0.35);
  skyHor.copy(cHorNight).lerp(cHorDay, dn.day).lerp(cHorDusk, dn.dusk * 0.85);
  if (state.rain) { skyTop.lerp(tmp.set(0x39424b), 0.6); skyHor.lerp(tmp.set(0x59646e), 0.6); }
  sky.material.uniforms.topColor.value.copy(skyTop); sky.material.uniforms.horizonColor.value.copy(skyHor); sky.material.uniforms.sunDir.value.copy(sunDir);
  const sunCol = cSunLow.clone().lerp(cSunHigh, Math.min(1, Math.max(0, elev / 0.5)));
  sunLight.color.copy(sunCol); sunLight.intensity = dn.day * 2.6 * (state.rain ? 0.35 : 1);
  sunLight.position.copy(sunDir).multiplyScalar(220); sunLight.castShadow = state.shadows && dn.day > 0.02;
  moonLight.intensity = dn.night * 0.55; moonLight.position.copy(sunDir).multiplyScalar(-220);
  hemiLight.color.copy(skyHor); hemiLight.intensity = 0.18 + dn.day * 0.8;
  const fogCol = skyHor.clone().lerp(dn.day > 0.5 ? cFogDay : cFogNight, 0.34);
  scene.fog = new THREE.FogExp2(fogCol.getHex(), (0.0026 + dn.night * 0.0011) * state.fog);
  renderer.toneMappingExposure = state.exposure;
  for (let i = 0; i < lanternLights.length; i++) lanternLights[i].intensity = dn.night * 40 * (1 + 0.14 * Math.sin(11 * t + 2.3 * i) + 0.08 * Math.sin(23 * t + 4.1 * i));   // flicker
  bloomPass.strength = state.bloom * (1 + dn.night * 0.55);
  sunMesh.position.copy(sunDir).multiplyScalar(2000); moonMesh.position.copy(sunDir).multiplyScalar(-2000);
  sunMesh.visible = dn.day > 0.05; moonMesh.visible = dn.night > 0.05;
  if (stars) { stars.material.opacity = dn.night * 0.85; stars.rotation.y += 0.0002; }
  const ff = fireflyMesh(); if (ff) ff.visible = state.fireflies && dn.night > 0.25;    // night-gated effects read dn.night
}
```

The sky dome is a big sphere (`BackSide`) with a shader that mixes `topColor` and `horizonColor`
by the view direction's y and adds a sun glow by `dot(dir, sunDir)`. Stars are a `Points` cloud
with `transparent: true`; clouds are a few flat sprites tinted by the horizon colour.

Rules:
- Everything that is "at night" gates on `dn.night > 0.25` (lanterns, fireflies, stars, the
  aurora, fireworks). Everything "by day" on `dn.day > 0.5`. Nothing checks the hour.
- One `FogExp2` is replaced per frame (cheap); the colour follows the horizon so the island's
  edge melts into the sky.
- A time chip sets `state.timeOfDay` directly: sunrise 0.25, noon 0.5, sunset 0.75, midnight 0.
- A fast-forward key multiplies `state.timeScale` (1, 10, 60); the HUD clock shows `hh:mm` from
  `timeOfDay * 24`.
