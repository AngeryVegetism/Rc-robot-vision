// ═══════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════
const WS_URL  = 'ws://localhost:8000/ws';
const API_URL = 'http://localhost:8000';
const MAX_PTS = 100;

const SIGNALS = [
  { key:'ir',        label:'IR Motion',     color:'#00d4ff', row:0 },
  { key:'acoustic',  label:'Acoustic',      color:'#b06aff', row:0 },
  { key:'radar',     label:'Radar/Doppler', color:'#00e87a', row:0 },
  { key:'co2',       label:'CO₂ (ppm)',     color:'#ff8c42', row:1 },
  { key:'thermal',   label:'Thermal (°C)',  color:'#ff5fa0', row:1 },
  { key:'vibration', label:'Vibration',     color:'#ffe03a', row:1 },
];

const MATRIX_DEFS = [
  { key:'ir_motion',         label:'IR Motion',   check: v => v === true },
  { key:'person_in_frame',   label:'Person',      check: v => v === true },
  { key:'breathing_rate_bpm',label:'Breathing',   check: v => v >= 8 && v <= 30,   fmt: v => v + ' bpm' },
  { key:'heart_rate_bpm',    label:'Heart Rate',  check: v => v >= 40 && v <= 180,  fmt: v => v + ' bpm' },
  { key:'acoustic_rms',      label:'Acoustic',    check: v => v > 0.05,             fmt: v => v.toFixed(3) },
  { key:'co2_ppm',           label:'CO₂',         check: v => v > 450,              fmt: v => v.toFixed(0) + ' ppm' },
  { key:'thermal_temp_c',    label:'Thermal',     check: v => v >= 30 && v <= 40,   fmt: v => v.toFixed(1) + '°C' },
  { key:'vibration_rms',     label:'Vibration',   check: v => v > 0.02,             fmt: v => v.toFixed(3) },
];

const PALETTE = {
  ALIVE:     { fg:'#00ffa3', dim:'rgba(0,255,163,0.12)', glow:'0 0 32px rgba(0,255,163,0.4)' },
  NOT_ALIVE: { fg:'#ff2d55', dim:'rgba(255,45,85,0.12)',  glow:'0 0 32px rgba(255,45,85,0.4)' },
  UNCERTAIN: { fg:'#ffb800', dim:'rgba(255,184,0,0.12)', glow:'0 0 32px rgba(255,184,0,0.4)' },
  UNKNOWN:   { fg:'#4a5a6a', dim:'rgba(74,90,106,0.08)', glow:'none' },
};

// ═══════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════
const state = {
  verdict:    'UNKNOWN',
  confidence: 0,
  signals:    {},
  detections: [],
  wsStatus:   'disconnected',
  demoRunning:false,
  history:    Object.fromEntries(SIGNALS.map(s => [s.key, []])),
  prevVerdict:'UNKNOWN',
  tick:       0,
  canvases:   {},
  ctxs:       {},
};

// ═══════════════════════════════════════════════════════
// BUILD UI
// ═══════════════════════════════════════════════════════
function buildCharts() {
  ['charts-top','charts-bot'].forEach((id,row) => {
    const wrap = document.getElementById(id);
    SIGNALS.filter(s => s.row === row).forEach(sig => {
      const card = document.createElement('div');
      card.className = 'chart-card';
      card.innerHTML = `
        <div class="chart-top">
          <span class="chart-label">${sig.label}</span>
          <span class="chart-val" id="cv-${sig.key}" style="color:${sig.color}">—</span>
        </div>
        <canvas class="signal-canvas" id="cc-${sig.key}"></canvas>
      `;
      wrap.appendChild(card);
    });
  });

  // Set up canvas contexts after DOM insert
  SIGNALS.forEach(sig => {
    const c = document.getElementById('cc-'+sig.key);
    state.canvases[sig.key] = c;
    state.ctxs[sig.key]     = c.getContext('2d');
  });
}

function buildMatrix() {
  const grid = document.getElementById('matrix-grid');
  MATRIX_DEFS.forEach(m => {
    const cell = document.createElement('div');
    cell.className = 'matrix-cell';
    cell.id = 'mc-'+m.key;
    cell.innerHTML = `
      <div class="mc-label">${m.label}</div>
      <div class="mc-val" id="mv-${m.key}">—</div>
      <div class="mc-dot" id="md-${m.key}"></div>
    `;
    grid.appendChild(cell);
  });
}

// ═══════════════════════════════════════════════════════
// RENDER
// ═══════════════════════════════════════════════════════
function applyVerdict(verdict, confidence) {
  const p = PALETTE[verdict] || PALETTE.UNKNOWN;
  const pct = Math.round((confidence || 0) * 100);

  document.getElementById('verdict-text').textContent = verdict;
  document.getElementById('conf-pct').textContent = pct;
  document.getElementById('conf-bar').style.width = pct + '%';

  // Update CSS vars for accent colour
  const root = document.documentElement;
  root.style.setProperty('--accent',     p.fg);
  root.style.setProperty('--accent-dim', p.dim);
  root.style.setProperty('--accent-glow',p.glow);

  const vc = document.getElementById('verdict-card');
  vc.style.borderColor = p.fg;
  vc.style.background  = p.dim;
  vc.style.boxShadow   = p.glow;

  document.getElementById('verdict-text').style.color = p.fg;
  document.getElementById('verdict-text').style.textShadow = p.glow;
  document.getElementById('conf-bar').style.background = p.fg;
  document.getElementById('conf-bar').style.boxShadow  = '0 0 10px ' + p.fg;
  document.getElementById('top-bar').style.background  =
    `linear-gradient(90deg,transparent,${p.fg},transparent)`;

  // Alert on change
  if (verdict !== state.prevVerdict) {
    const type = verdict === 'ALIVE' ? 'alive' : verdict === 'NOT_ALIVE' ? 'dead' : 'uncertain';
    pushAlert('Verdict → ' + verdict, type);
    state.prevVerdict = verdict;
  }
}

function applySignals(sigs) {
  const fmt = (v, d=2) => v != null ? parseFloat(v).toFixed(d) : '—';
  document.getElementById('v-breath').textContent = sigs.breathing_rate_bpm != null ? sigs.breathing_rate_bpm + ' bpm' : '—';
  document.getElementById('v-hr').textContent     = sigs.heart_rate_bpm != null ? sigs.heart_rate_bpm + ' bpm' : '—';
  document.getElementById('v-co2').textContent    = sigs.co2_ppm != null ? sigs.co2_ppm.toFixed(0) + ' ppm' : '—';
  document.getElementById('v-temp').textContent   = sigs.thermal_temp_c != null ? sigs.thermal_temp_c.toFixed(1) + '°C' : '—';

  const ps = document.getElementById('person-status');
  const pl = document.getElementById('person-label');
  if (sigs.person_in_frame) {
    ps.classList.add('active');
    pl.textContent = 'PERSON IN FRAME';
  } else {
    ps.classList.remove('active');
    pl.textContent = 'NO PERSON DETECTED';
  }

  // Matrix
  MATRIX_DEFS.forEach(m => {
    const raw = sigs[m.key];
    const on  = raw != null && m.check(raw);
    const cell = document.getElementById('mc-'+m.key);
    const val  = document.getElementById('mv-'+m.key);
    if (!cell) return;
    cell.classList.toggle('on', on);
    if (raw != null) {
      if (typeof raw === 'boolean') val.textContent = raw ? 'YES' : 'NO';
      else if (m.fmt) val.textContent = m.fmt(raw);
      else val.textContent = parseFloat(raw).toFixed(3);
    } else {
      val.textContent = '—';
    }
  });
}

function applyDetections(dets) {
  const wrap = document.getElementById('detections-wrap');
  wrap.innerHTML = '';
  (dets || []).forEach(d => {
    const tag = document.createElement('span');
    tag.className = 'det-tag';
    tag.textContent = d.label + ' ' + Math.round(d.confidence * 100) + '%';
    wrap.appendChild(tag);
  });
}

function applyVideo(b64) {
  const img = document.getElementById('video-img');
  const ph  = document.getElementById('no-video');
  const dot = document.getElementById('live-dot');
  if (b64) {
    img.src = 'data:image/jpeg;base64,' + b64;
    img.style.display = 'block';
    ph.style.display  = 'none';
    dot.classList.add('active');
  }
}

// ── Canvas signal chart ────────────────────────────────
function drawChart(key, color) {
  const canvas = state.canvases[key];
  const ctx    = state.ctxs[key];
  if (!canvas || !ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.clientWidth;
  const H   = canvas.clientHeight;
  if (canvas.width !== W*dpr || canvas.height !== H*dpr) {
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
  }

  const data = state.history[key];
  ctx.clearRect(0, 0, W, H);

  if (data.length < 2) return;

  // Find range
  let mn = Infinity, mx = -Infinity;
  data.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; });
  const range = mx - mn || 1;

  // Draw area fill
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (MAX_PTS - 1)) * W;
    const y = H - ((v - mn) / range) * (H - 8) - 4;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, color + '30');
  grad.addColorStop(1, color + '00');
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Draw line
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (MAX_PTS - 1)) * W;
    const y = H - ((v - mn) / range) * (H - 8) - 4;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.shadowColor = color;
  ctx.shadowBlur  = 6;
  ctx.stroke();
  ctx.shadowBlur  = 0;

  // Latest value label
  const last = data[data.length - 1];
  const isLargeUnit = key === 'co2' || key === 'thermal';
  document.getElementById('cv-'+key).textContent =
    last != null ? last.toFixed(isLargeUnit ? 1 : 3) : '—';
}

function renderAllCharts() {
  SIGNALS.forEach(s => drawChart(s.key, s.color));
}

// ── System log ─────────────────────────────────────────
function renderLog() {
  const p = PALETTE[state.verdict] || PALETTE.UNKNOWN;
  const logLines = [
    { key:'backend',    val:'ws://localhost:8000/ws',              cls:'' },
    { key:'yolo',       val:'yolov8n.pt [ready]',                  cls:'' },
    { key:'signal_proc',val:'scipy bandpass [active]',             cls:'' },
    { key:'connection', val:state.wsStatus,
      cls: (state.wsStatus==='connected'||state.wsStatus==='demo') ? 'ok' : 'err' },
    { key:'verdict',    val:state.verdict+' ('+Math.round(state.confidence*100)+'%)', cls:'highlight' },
    { key:'uptime',     val:state.tick + 's',                      cls:'' },
  ];

  document.getElementById('log-body').innerHTML = logLines.map(l => `
    <div class="log-line ${l.cls}">
      <span class="log-prompt">></span>
      <span class="log-key">${l.key} ::</span>
      <span class="log-val">${l.val}</span>
    </div>
  `).join('');
}

// ── Connection badge ───────────────────────────────────
function setConnStatus(s) {
  state.wsStatus = s;
  const badge = document.getElementById('conn-badge');
  const label = document.getElementById('conn-label');
  badge.className = 'badge ' + s;
  const MAP = { connected:'CONNECTED', connecting:'CONNECTING', disconnected:'DISCONNECTED', demo:'DEMO MODE' };
  label.textContent = MAP[s] || s.toUpperCase();
}

// ═══════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════
let ws, reconnTimer, pingInterval;

function connect() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  setConnStatus('connecting');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setConnStatus('connected');
    clearTimeout(reconnTimer);
    pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping');
    }, 10000);
  };

  ws.onmessage = e => {
    try {
      const msg = JSON.parse(e.data);

      if (msg.type === 'init') {
        const d = msg.diagnosis || {};
        state.verdict    = d.verdict    || 'UNKNOWN';
        state.confidence = d.confidence || 0;
        state.signals    = d.signals    || {};
        state.detections = d.detections || [];
        if (msg.signals) {
          Object.keys(msg.signals).forEach(k => {
            state.history[k] = (msg.signals[k] || []).slice(-MAX_PTS);
          });
        }
        refresh();
      }

      if (msg.type === 'diagnosis') {
        state.verdict    = msg.verdict;
        state.confidence = msg.confidence;
        state.signals    = msg.signals    || {};
        state.detections = msg.detections || [];
        refresh();
      }

      if (msg.type === 'signals' && msg.data) {
        Object.entries(msg.data).forEach(([k, arr]) => {
          if (!Array.isArray(arr) || arr.length === 0) return;
          const last = arr[arr.length-1];
          if (state.history[k]) {
            state.history[k].push(last);
            if (state.history[k].length > MAX_PTS) state.history[k].shift();
          }
        });
        renderAllCharts();
      }

      if (msg.type === 'video_frame') {
        applyVideo(msg.frame_b64);
        applyDetections(msg.detections);
      }
    } catch(_) {}
  };

  ws.onclose = () => {
    setConnStatus('disconnected');
    clearInterval(pingInterval);
    reconnTimer = setTimeout(connect, 3000);
  };

  ws.onerror = () => ws.close();
}

function refresh() {
  applyVerdict(state.verdict, state.confidence);
  applySignals(state.signals);
  applyDetections(state.detections);
  renderAllCharts();
  renderLog();
}

// ═══════════════════════════════════════════════════════
// DEMO & RESET
// ═══════════════════════════════════════════════════════
async function toggleDemo() {
  const btn = document.getElementById('demo-btn');
  if (state.demoRunning) {
    await fetch(API_URL + '/demo/stop').catch(()=>{});
    state.demoRunning = false;
    btn.textContent = '▶ DEMO MODE';
    btn.classList.remove('demo-active');
    setConnStatus('connected');
  } else {
    await fetch(API_URL + '/demo/start').catch(()=>{});
    state.demoRunning = true;
    btn.textContent = '◼ STOP DEMO';
    btn.classList.add('demo-active');
    setConnStatus('demo');
    pushAlert('Demo stream started', 'info');

    // Local simulation fallback if backend unreachable
    startLocalSim();
  }
}

async function resetAll() {
  await fetch(API_URL + '/signals/reset', { method:'DELETE' }).catch(()=>{});
  SIGNALS.forEach(s => { state.history[s.key] = []; });
  state.verdict = 'UNKNOWN';
  state.confidence = 0;
  state.signals = {};
  state.detections = [];
  refresh();
  pushAlert('All buffers reset', 'info');
}

// ── Local sim (if no backend) ──────────────────────────
let simInterval;
function startLocalSim() {
  clearInterval(simInterval);
  let t = 0;
  simInterval = setInterval(() => {
    t += 0.12;
    const alive = Math.sin(t * 0.08) > 0;
    const co2   = 450 + 30*Math.sin(t*0.1)  + (Math.random()-0.5)*8;
    const temp  = 36.5 +    Math.sin(t*0.05) + (Math.random()-0.5)*0.3;
    const breath= alive ? 15 + 3*Math.sin(t*0.07) : 0;
    const hr    = alive ? 72 + 8*Math.sin(t*0.15) : 0;

    // Push signal history
    const vals = {
      ir:        Math.abs(Math.sin(t*0.8) + (Math.random()-0.5)*0.08),
      acoustic:  Math.abs(Math.sin(t*0.25)*0.4 + (Math.random()-0.5)*0.05),
      radar:     Math.abs(Math.sin(t*1.4)*0.6  + (Math.random()-0.5)*0.07),
      co2,
      thermal:   temp,
      vibration: Math.abs((Math.random()-0.5)*0.06 + 0.01*Math.sin(t*2)),
    };

    SIGNALS.forEach(s => {
      state.history[s.key].push(vals[s.key]);
      if (state.history[s.key].length > MAX_PTS) state.history[s.key].shift();
    });

    const score = (alive?0.65:0.1) + Math.random()*0.1;
    state.verdict    = score > 0.55 ? 'ALIVE' : score > 0.25 ? 'UNCERTAIN' : 'NOT_ALIVE';
    state.confidence = Math.min(score, 1);
    state.signals = {
      ir_motion:          vals.ir > 0.3,
      acoustic_rms:       vals.acoustic,
      breathing_rate_bpm: parseFloat(breath.toFixed(1)),
      heart_rate_bpm:     parseFloat(hr.toFixed(1)),
      co2_ppm:            parseFloat(co2.toFixed(1)),
      thermal_temp_c:     parseFloat(temp.toFixed(2)),
      vibration_rms:      vals.vibration,
      person_in_frame:    alive,
    };

    refresh();
  }, 120);
}

// ═══════════════════════════════════════════════════════
// ALERTS
// ═══════════════════════════════════════════════════════
function pushAlert(msg, type='info') {
  const wrap = document.getElementById('alerts');
  const el   = document.createElement('div');
  el.className = 'alert ' + type;
  el.textContent = msg;
  wrap.appendChild(el);
  setTimeout(() => {
    el.style.animation = 'fadeOut 0.4s ease forwards';
    setTimeout(() => el.remove(), 400);
  }, 4500);
}

// ═══════════════════════════════════════════════════════
// TICK
// ═══════════════════════════════════════════════════════
setInterval(() => {
  state.tick++;
  renderLog();
}, 1000);

// ═══════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════
buildCharts();
buildMatrix();
renderLog();
connect();