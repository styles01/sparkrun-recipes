var models = [
  {"name":"DS4-Flash","score":50.7,"rank":2,"params":"159B","active":"11B","disk":227,"runtime":"vLLM MTP k=2","agent":45.0,"reasoning":54.7,"code":57.2,"math":91.9,"speed":15.0,"lanes":15.0,"context":15.0,"overhead":82.8,"tok_s":21,"lanes_n":2,"ctx_k":128,"free_gb":22,"claweval":57.8,"mmlu_pro":86.4,"gpqa":87.4,"hle":29.4,"hmmt":91.9,"scicode":null,"terminal_bench":56.6,"status":"TESTED","aa_index":40},
  {"name":"Qwen 122B","score":52.3,"rank":1,"params":"122B","active":"10B","disk":67,"runtime":"vLLM v26 DFlash n=7 fp8 KV","agent":39.7,"reasoning":50.6,"code":42.0,"math":91.4,"speed":28.0,"lanes":73.0,"context":30.8,"overhead":96.7,"tok_s":43.6,"lanes_n":5,"ctx_k":256,"free_gb":7,"claweval":null,"mmlu_pro":86.7,"gpqa":86.6,"hle":25.3,"hmmt":91.4,"scicode":42.0,"terminal_bench":49.4,"status":"PROD","aa_index":32},
  {"name":"Qwen 35B","score":48.7,"rank":3,"params":"35B","active":"3B","disk":22,"runtime":"vLLM MTP k=3","agent":20.2,"reasoning":44.7,"code":38.0,"math":89.0,"speed":100,"lanes":100,"context":84.1,"overhead":100,"tok_s":102.8,"lanes_n":6,"ctx_k":256,"free_gb":45,"claweval":null,"mmlu_pro":85.3,"gpqa":84.2,"hle":22.4,"hmmt":89.0,"scicode":null,"terminal_bench":40.5,"status":"PROD","aa_index":29},
  {"name":"Puzzle-75B","score":48.2,"rank":4,"params":"75B","active":"9.3B","disk":50,"runtime":"vLLM NGC 26.06 MTP k=3","agent":24.0,"reasoning":61.6,"code":40.6,"math":93.4,"speed":48.7,"lanes":43.9,"context":100,"overhead":95.3,"tok_s":39.7,"lanes_n":3,"ctx_k":300,"free_gb":37,"claweval":null,"mmlu_pro":82.4,"gpqa":78.6,"hle":16.5,"hmmt":93.4,"scicode":40.6,"terminal_bench":24.0,"status":"PROD","aa_index":null},
  {"name":"Step 3.7 Flash","score":40.1,"rank":5,"params":"198B","active":"11B","disk":105,"runtime":"llama.cpp IQ4_XS","agent":43.1,"reasoning":46.5,"code":36.0,"math":50,"speed":30.3,"lanes":15.0,"context":59.5,"overhead":15.0,"tok_s":28.1,"lanes_n":2,"ctx_k":200,"free_gb":0.5,"claweval":67.1,"mmlu_pro":null,"gpqa":null,"hle":null,"hmmt":null,"scicode":null,"terminal_bench":59.6,"status":"TESTED","aa_index":30},
  {"name":"Nemotron Super 120B","score":36.4,"rank":6,"params":"120B","active":"12B","disk":67,"runtime":"vLLM (untested)","agent":17.7,"reasoning":54.5,"code":40.0,"math":93.7,"speed":0,"lanes":0,"context":0,"overhead":0,"tok_s":null,"lanes_n":null,"ctx_k":null,"free_gb":null,"claweval":null,"mmlu_pro":83.73,"gpqa":79.23,"hle":18.26,"hmmt":93.67,"scicode":42.05,"terminal_bench":31.0,"status":"EST","aa_index":25}
];

var modelIcons = {"DS4-Flash":"🐉","Qwen 122B":"👑","Qwen 35B":"🧠","Puzzle-75B":"🧩","Step 3.7 Flash":"⚡","Nemotron Super 120B":"🟢"};
var pillMap = {"PROD":"pill-prod","TESTED":"pill-test","EST":"pill-est","DISMISSED":"pill-dismissed"};

// Leaderboard
var lb = document.getElementById("leaderboard");
models.forEach(function(m) {
  var rc = m.rank <= 3 ? "rank-" + m.rank : "";
  var icon = modelIcons[m.name] || "";
  lb.innerHTML += '<div class="model-card"><div class="rank ' + rc + '">' + icon + m.rank + '</div><div class="model-info"><h2>' + m.name + '</h2><div class="model-meta"><span>📦 ' + m.params + ' (' + m.active + ' active)</span><span>💾 ' + m.disk + 'GB</span><span>⚙️ ' + m.runtime + '</span><span>🚀 ' + (m.tok_s || "N/A") + ' tok/s</span><span>🛣️ ' + (m.lanes_n || "N/A") + '×' + (m.ctx_k || "N/A") + 'K</span></div><div class="score-bar"><div class="score-fill" style="width:' + m.score + '%"></div></div></div><div class="score-badge"><div class="score">' + m.score + '</div><div class="label">Score</div></div></div>';
});

// Table
var tb = document.getElementById("tableBody");
function f(v) { return v != null ? v : '<span class="na">N/A</span>'; }
models.forEach(function(m) {
  tb.innerHTML += '<tr><td>' + m.name + '</td><td><b>' + m.score + '</b></td><td>#' + m.rank + '</td><td>' + m.params + '</td><td>' + m.active + '</td><td>' + f(m.claweval) + '</td><td>' + f(m.mmlu_pro) + '</td><td>' + f(m.gpqa) + '</td><td>' + f(m.hle) + '</td><td>' + f(m.hmmt) + '</td><td>' + f(m.scicode) + '</td><td>' + f(m.terminal_bench) + '</td><td>' + f(m.tok_s) + '</td><td>' + f(m.lanes_n) + '</td><td>' + (m.ctx_k ? m.ctx_k + 'K' : f(null)) + '</td><td>' + f(m.free_gb) + '</td><td><span class="pill ' + pillMap[m.status] + '">' + m.status + '</span></td></tr>';
});

// Charts
var colors = ["rgba(63,185,80,","rgba(88,166,255,","rgba(210,153,34,","rgba(248,81,73,","rgba(163,113,247,","rgba(255,166,87,"];

// Radar chart
new Chart(document.getElementById("radarChart"), {
  type: "radar",
  data: {
    labels: ["Agent","Reasoning","Code","Math","Speed","Lanes","Context","Overhead"],
    datasets: models.map(function(m, i) {
      return {
        label: m.name,
        data: [m.agent, m.reasoning, m.code, m.math, m.speed || null, m.lanes || null, m.context || null, m.overhead || null],
        backgroundColor: colors[i] + "0.1)",
        borderColor: colors[i] + "0.8)",
        borderWidth: 2,
        pointRadius: 3,
        spanGaps: false
      };
    })
  },
  options: {
    responsive: true,
    scales: { r: { beginAtZero: true, max: 100, grid: { color: "#30363d" }, pointLabels: { color: "#8b949e", font: { size: 11 } }, ticks: { color: "#484f58", backdropColor: "transparent" } } },
    plugins: { legend: { labels: { color: "#e6edf3", font: { size: 11 } } } }
  }
});

// Intelligence bar chart
new Chart(document.getElementById("intelChart"), {
  type: "bar",
  data: {
    labels: ["ClawEval","MMLU-Pro","GPQA","HLE","HMMT","SciCode","Term-Bench"],
    datasets: models.map(function(m, i) {
      return {
        label: m.name,
        data: [m.claweval, m.mmlu_pro, m.gpqa, m.hle, m.hmmt, m.scicode, m.terminal_bench],
        backgroundColor: colors[i] + "0.7)"
      };
    })
  },
  options: {
    responsive: true,
    scales: {
      x: { ticks: { color: "#8b949e", font: { size: 10 } }, grid: { color: "#30363d" } },
      y: { beginAtZero: true, max: 100, ticks: { color: "#8b949e" }, grid: { color: "#30363d" } }
    },
    plugins: { legend: { labels: { color: "#e6edf3", font: { size: 11 } } } }
  }
});

// Infrastructure bar chart
new Chart(document.getElementById("infraChart"), {
  type: "bar",
  data: {
    labels: models.map(function(m) { return m.name; }),
    datasets: [
      { label: "tok/s", data: models.map(function(m) { return m.tok_s; }), backgroundColor: "rgba(63,185,80,0.7)" },
      { label: "Lanes", data: models.map(function(m) { return m.lanes_n; }), backgroundColor: "rgba(88,166,255,0.7)" },
      { label: "Context (K)", data: models.map(function(m) { return m.ctx_k; }), backgroundColor: "rgba(210,153,34,0.7)" },
      { label: "Free RAM (GB)", data: models.map(function(m) { return m.free_gb; }), backgroundColor: "rgba(248,81,73,0.7)" }
    ]
  },
  options: {
    responsive: true,
    scales: {
      x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
      y: { beginAtZero: true, ticks: { color: "#8b949e" }, grid: { color: "#30363d" } }
    },
    plugins: { legend: { labels: { color: "#e6edf3" } } }
  }
});