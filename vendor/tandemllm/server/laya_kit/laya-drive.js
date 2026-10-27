// Laya plays this page: TandemLLM's in-page playtest driver (laya-drive.js, the session's scratch directory).
//
// For when only the browser tool can reach the page (no `laya_drive.py browser`, which gives real
// input over the DevTools protocol). Load it into the page (the browser tool's evaluate script:
// `eval(await (await fetch(URL + "/v1/laya/kit/laya-drive.js")).text())`, or a <script> tag), then
//
//   await tandem_drive.start({
//     actions: { run: "move right", jump: "jump straight up", run_jump: "jump forward", wait: "stand still" },
//     keys: { run: ["ArrowRight"], jump: [" "], run_jump: ["ArrowRight", " "], wait: [] },
//     everyMs: 250, episodes: 3, steps: 400,
//     logUrl: "http://127.0.0.1:8765/log",     // optional: `python3 laya_drive.py serve-log`
//   });
//   tandem_drive.summary()   // {episodes, steps, wins, deaths: [{ep, step, x, y, cause}], actions, ms_p50, ...}
//   tandem_drive.lines()     // the log as JSON lines (laya_drive.py report reads them)
//
// The page defines window.tandem_state() -> a small JSON object (what matters for the next move;
// dead/over, won, lives, x/y, cause) and, for more than one episode, window.tandem_reset().
// The keys are synthetic KeyboardEvents (isTrusted false) on window and document: a game that
// listens for keydown/keyup sees them; one that checks isTrusted needs laya_drive.py browser.
(() => {
  const QUESTION = "Which action should the player take now to stay alive and make progress?";
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const round = (v) => (typeof v === "number" && !Number.isInteger(v) ? Math.round(v * 10) / 10 : v);
  const compact = (s, limit = 1000) => {
    for (const cut of [64, 16, 8, 4, 2, 1]) {
      const walk = (v) => Array.isArray(v) ? v.slice(0, cut).map(walk)
        : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).filter(([, x]) => x != null).map(([k, x]) => [k, walk(x)]))
        : round(v);
      const t = JSON.stringify(walk(s));
      if (t.length <= limit) return t;
    }
    return JSON.stringify(s).slice(0, limit);
  };
  const pos = (s) => {
    const p = s && typeof s.player === "object" && s.player ? s.player : s;
    return p && typeof p.x === "number" ? [round(p.x), typeof p.y === "number" ? round(p.y) : null] : null;
  };
  const lost = (s) => !!s && ["dead", "over", "lost", "game_over"].some((k) => s[k] === true);
  const won = (s) => !!s && (s.won === true || s.win === true);
  const codeOf = (k) => k === " " ? "Space" : k.length === 1 ? (/\d/.test(k) ? "Digit" + k : "Key" + k.toUpperCase()) : k;
  const keyCodeOf = (k) => ({ ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40, " ": 32, Enter: 13, Escape: 27, Shift: 16 }[k] ?? k.toUpperCase().charCodeAt(0));
  const fire = (type, k) => {
    // at the focused element (a canvas with tabindex, the body): it bubbles to document and window
    const e = new KeyboardEvent(type, { key: k, code: codeOf(k), bubbles: true, cancelable: true });
    Object.defineProperty(e, "keyCode", { get: () => keyCodeOf(k) });
    Object.defineProperty(e, "which", { get: () => keyCodeOf(k) });
    (document.activeElement || document).dispatchEvent(e);
  };
  const D = (window.tandem_drive = { log: [], running: false, held: [] });
  D.stop = () => { D.running = false; };
  D.lines = () => D.log.map((r) => JSON.stringify(r)).join("\n") + "\n";
  D.download = (name = "drive-log.jsonl") => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([D.lines()], { type: "application/x-ndjson" }));
    a.download = name; a.click();
  };
  D.summary = () => {
    const steps = D.log.filter((r) => "a" in r), ev = D.log.filter((r) => r.event);
    const ms = steps.map((r) => r.ms).filter((x) => typeof x === "number").sort((a, b) => a - b);
    const hist = {};
    for (const r of steps) hist[r.a] = (hist[r.a] || 0) + 1;
    return {
      episodes: new Set(D.log.filter((r) => r.ep).map((r) => r.ep)).size, steps: steps.length,
      wins: ev.filter((e) => e.event === "win").length, timeouts: ev.filter((e) => e.event === "timeout").length,
      deaths: ev.filter((e) => e.event === "death").map((e) => ({ ep: e.ep, step: e.step, x: e.pos?.[0], y: e.pos?.[1], cause: e.cause })),
      stuck: ev.filter((e) => e.event === "stuck").map((e) => ({ ep: e.ep, step: e.step, x: e.pos?.[0], y: e.pos?.[1] })),
      actions: hist, ms_p50: ms[Math.floor(ms.length / 2)] ?? null, ms_p95: ms[Math.min(ms.length - 1, Math.floor(ms.length * 0.95))] ?? null,
      unsure: steps.filter((r) => r.sure === false).length, fallbacks: steps.filter((r) => r.why).length,
    };
  };
  D.start = async (o = {}) => {
    const url = (o.url || "http://127.0.0.1:8000").replace(/\/$/, "");
    const actions = o.actions || {}, keys = o.keys || {};
    const session = o.session || `drive-page-${Date.now()}`;
    const every = o.everyMs ?? 250, steps = o.steps ?? 400, episodes = o.episodes ?? 3, stuckSteps = o.stuck ?? 20;
    const fallback = o.fallback || ("wait" in actions ? "wait" : Object.keys(actions)[0]);
    const read = () => (window.tandem_state || window.__state)();
    const t0 = performance.now(), t = () => Math.round(performance.now() - t0);
    let sent = 0;
    const flush = async () => {
      if (!o.logUrl || sent >= D.log.length) return;
      const chunk = D.log.slice(sent).map((r) => JSON.stringify(r)).join("\n");
      sent = D.log.length;
      try { await fetch(o.logUrl, { method: "POST", body: chunk }); } catch (e) { /* the log stays in D.log */ }
    };
    const press = (a) => {
      for (const k of D.held) fire("keyup", k);
      D.held = keys[a] || [];
      for (const k of D.held) fire("keydown", k);
    };
    D.log.push({ run: session, at: new Date().toISOString(), label: o.label || "", actions, every_ms: every, steps, episodes });
    D.running = true;
    const count = { episodes: 0, steps: 0, deaths: 0, wins: 0, timeouts: 0, stuck: 0, busy: 0, fallbacks: 0 };
    try {
      for (let ep = 1; ep <= episodes && D.running; ep++) {
        if (ep > 1 && window.tandem_reset) { press("__none__"); await window.tandem_reset(); }
        count.episodes++;
        let s = read(), lives = s?.lives, spot = pos(s)?.join(","), same = 0, step = 0, lastPos = null;
        let end = lost(s) ? "death" : won(s) ? "win" : null;
        const flagged = new Set();
        while (!end && D.running && step < steps) {
          step++;
          const tick = performance.now();
          const ch = Array.isArray(s?.actions) && s.actions.length
            ? Object.fromEntries(s.actions.map((a) => [a, actions[a] || a])) : actions;
          let v;
          const q0 = performance.now();
          try {
            const r = await fetch(url + "/v1/laya/ask", { method: "POST", body: JSON.stringify({
              question: o.question || QUESTION, choices: ch, text: compact(s), session, deadline_ms: Math.max(250, 2 * every) }) });
            const j = await r.json();
            v = r.ok ? { label: j.label, p: j.p, sure: j.sure } : { error: j.error?.type || String(r.status) };
          } catch (e) { v = { error: "unreachable" }; }
          const ms = Math.round((performance.now() - q0) * 10) / 10;
          const a = v.error ? (fallback in ch ? fallback : Object.keys(ch)[0]) : v.label;
          if (v.error) { count.fallbacks++; if (v.error === "busy") count.busy++; }
          count.steps++;
          D.log.push({ ep, step, t: t(), s: compact(s, 600), a, p: v.p ?? null, sure: v.sure ?? null, ms, ...(v.error ? { why: v.error } : {}) });
          press(a);
          await sleep(Math.max(0, every - (performance.now() - tick)));
          lastPos = pos(s);                 // a death's place: the decision's before it
          s = read();
          if (lost(s)) { end = "death"; break; }
          if (won(s)) { end = "win"; break; }
          if (typeof lives === "number" && typeof s?.lives === "number" && s.lives < lives) {
            count.deaths++;
            D.log.push({ ep, step, t: t(), event: "death", s: compact(s, 600), pos: lastPos, cause: s.cause ?? null, lives: s.lives });
          }
          lives = s?.lives;
          const p = pos(s), key = p && p.join(",");
          if (key && key === spot) {
            if (++same === stuckSteps && !flagged.has(key)) {
              flagged.add(key); count.stuck++;
              D.log.push({ ep, step, t: t(), event: "stuck", pos: p, steps: same, s: compact(s, 600) });
            }
          } else { spot = key; same = 0; }
          if (step % 20 === 0) flush();
        }
        end = end || "timeout";
        count[{ death: "deaths", win: "wins", timeout: "timeouts" }[end]]++;
        D.log.push({ ep, step, t: t(), event: end, s: compact(s, 600), pos: end === "death" && lastPos ? lastPos : pos(s), cause: s?.cause ?? null });
      }
    } finally {
      press("__none__");
      D.running = false;
      let srv = null;
      try {
        const r = await fetch(url + "/v1/laya/drive/end", { method: "POST", body: JSON.stringify({ session, summary: count }) });
        if (r.ok) srv = await r.json();
      } catch (e) { /* the summary stays in the log */ }
      D.log.push({ end: session, t: t(), summary: count, actions: D.summary().actions,
        laya: srv && { ms_p50: srv.ms_p50, ms_p95: srv.ms_p95, unsure: srv.unsure, errors: srv.errors } });
      await flush();
    }
    return D.summary();
  };
})();
