/**
 * ROVER DASHBOARD — app.js
 * Handles: WebSocket, Audio Waveform, IR Heatmap,
 *          Manual Control, Detection/Bounding Boxes,
 *          DB Session Logging, Theme Toggle
 */

'use strict';

// ═══════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════

const State = {
  ws: null,
  wsConnected: false,

  operator: null,        // { id, name, initials }
  sessionId: null,
  sessionStart: null,
  sessionTimer: null,

  detectionCount: 0,
  humanCount: 0,
  objectCount: 0,
  alertCount: 0,

  manualMode: false,
  speed: 50,
  pressedKeys: new Set(),

  bboxColor: '#5c5cff',
  showConf: true,
  humanOnly: true,
  dbSave: true,
  dbApiUrl: 'http://localhost:3001/api',

  fpsSamples: [],
  lastFrameTime: 0,

  // Waveform
  audioCtx: null,
  analyser: null,
  micStream: null,
  waveformAnim: null,

  // Heatmap — simulated 8×8 IR grid
  irData: Array(64).fill(20),
  irAnim: null,
};

// ═══════════════════════════════════════════
//  DOM REFS
// ═══════════════════════════════════════════

const $ = id => document.getElementById(id);

const DOM = {
  mainContent:     $('mainContent'),
  tilesRow:        $('tilesRow'),
  cameraSection:   $('cameraSection'),
  cameraCard:      $('cameraCard'),
  logCard:         $('logCard'),
  // Connection
  connPill:        $('connectionPill'),
  connLabel:       $('connLabel'),
  // Tiles
  audioLevel:      $('audioLevel'),
  audioBadge:      $('audioBadge'),
  irMaxTemp:       $('irMaxTemp'),
  irBadge:         $('irBadge'),
  detectionCount:  $('detectionCount'),
  statusBadge:     $('statusBadge'),
  humanCount:      $('humanCount'),
  objectCount:     $('objectCount'),
  sessionTime:     $('sessionTime'),
  operatorName:    $('operatorName'),
  controlMode:     $('controlMode'),
  controlBadge:    $('controlBadge'),
  // Canvases
  waveformCanvas:  $('waveformCanvas'),
  heatmapCanvas:   $('heatmapCanvas'),
  detectionCanvas: $('detectionCanvas'),
  // Camera
  cameraPlaceholder: $('cameraPlaceholder'),
  recIndicator:    $('recIndicator'),
  hudFPS:          $('hudFPS'),
  hudRes:          $('hudRes'),
  hudTimestamp:    $('hudTimestamp'),
  cameraViewport:  $('cameraViewport'),
  // Log
  logList:         $('logList'),
  alertCount:      $('alertCount'),
  // User
  displayUserName: $('displayUserName'),
  userChip:        $('userChip'),
  userInitials:    $('userInitials'),
  operatorDisplay: $('operatorName'),
  settingsOperatorName: $('settingsOperatorName'),
  // Settings
  settingsPanel:   $('settingsPanel'),
  settingsOverlay: $('settingsOverlay'),
  loginStatus:     $('loginStatus'),
  authOverlay:     $('authOverlay'),
  authLoginStatus: $('authLoginStatus'),
  speedVal:        $('speedVal'),
  manualToggle:    $('manualToggle'),
};

// ═══════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════

function init() {
  setupSettings();
  setupNavigation();
  setupControlPad();
  setupKeyboard();
  initWaveform();
  initHeatmap();
  initDetectionCanvas();
  setupSpeedSlider();
  setupManualToggle();
  setupButtons();
  tickSessionTimer();
  updateHUDTimestamp();
  syncOperatorUI();
  setAuthGate(!isAuthenticated());
  setViewMode('dashboard');
  setInterval(updateHUDTimestamp, 1000);
}

// ═══════════════════════════════════════════
//  SETTINGS PANEL
// ═══════════════════════════════════════════

function setupSettings() {
  const open  = () => { DOM.settingsPanel.classList.add('open'); DOM.settingsOverlay.classList.add('open'); };
  const close = () => { DOM.settingsPanel.classList.remove('open'); DOM.settingsOverlay.classList.remove('open'); };

  $('settingsBtn').addEventListener('click', open);
  $('settingsClose').addEventListener('click', close);
  DOM.settingsOverlay.addEventListener('click', close);

  // Theme
  $('themeDark').addEventListener('click', () => setTheme('dark'));
  $('themeLight').addEventListener('click', () => setTheme('light'));

  // Connect / Disconnect
  $('connectBtn').addEventListener('click', connectWS);
  $('disconnectBtn').addEventListener('click', disconnectWS);

  // Login / Logout
  $('loginBtn').addEventListener('click', loginOperator);
  $('logoutBtn').addEventListener('click', logoutOperator);
  $('operatorInput').addEventListener('keydown', handleLoginKeydown);
  $('operatorPass').addEventListener('keydown', handleLoginKeydown);

  // Camera settings
  $('bboxColor').addEventListener('input', e => { State.bboxColor = e.target.value; });
  $('showConfToggle').addEventListener('change', e => { State.showConf = e.target.checked; });
  $('humanOnlyToggle').addEventListener('change', e => { State.humanOnly = e.target.checked; });
  $('dbSaveToggle').addEventListener('change', e => { State.dbSave = e.target.checked; });
  $('dbApiUrl').addEventListener('change', e => { State.dbApiUrl = e.target.value.trim(); });
}

function isAuthenticated() {
  return Boolean(State.operator);
}

function setAuthGate(locked) {
  document.body.classList.toggle('auth-locked', locked);
  DOM.authOverlay.classList.toggle('open', locked);
  if (locked) {
    DOM.settingsPanel.classList.remove('open');
    DOM.settingsOverlay.classList.remove('open');
    window.setTimeout(() => $('operatorInput').focus(), 50);
  }
}

function syncOperatorUI() {
  const name = State.operator?.name || 'Not logged in';
  const initials = State.operator?.initials || '--';

  DOM.displayUserName.textContent = name;
  DOM.userChip.textContent = State.operator?.initials || '?';
  DOM.userInitials.textContent = initials;
  DOM.operatorName.textContent = State.operator?.name || '—';
  DOM.settingsOperatorName.textContent = name;
}

function showStatus(target, msg, type) {
  target.textContent = msg;
  target.className = `login-status ${type}`;
  setTimeout(() => {
    target.textContent = '';
    target.className = 'login-status';
  }, 4000);
}

function handleLoginKeydown(event) {
  if (event.key === 'Enter') {
    event.preventDefault();
    loginOperator();
  }
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  $('themeDark').classList.toggle('active', theme === 'dark');
  $('themeLight').classList.toggle('active', theme === 'light');
}

// ═══════════════════════════════════════════
//  NAVIGATION
// ═══════════════════════════════════════════

function setupNavigation() {
  document.querySelectorAll('.nav-item').forEach(btn => {
    if (btn.id === 'settingsBtn' || btn.classList.contains('user-avatar')) return;
    btn.addEventListener('click', () => {
      setViewMode(btn.dataset.view || 'dashboard');
    });
  });
}

function setViewMode(view) {
  const allowedViews = new Set(['dashboard', 'camera', 'logs', 'analytics']);
  const nextView = allowedViews.has(view) ? view : 'dashboard';

  DOM.mainContent.dataset.view = nextView;
  document.querySelectorAll('.nav-item[data-view]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === nextView);
  });

  DOM.cameraViewport.classList.toggle('fullscreen', nextView === 'camera');

  const titleMap = {
    dashboard: 'Operations',
    camera: 'Camera Feed',
    logs: 'Detection Log',
    analytics: 'Analytics',
  };
  document.querySelector('.page-title').textContent = titleMap[nextView];
}

// ═══════════════════════════════════════════
//  WEBSOCKET
// ═══════════════════════════════════════════

/**
 * Expected WebSocket message format (JSON):
 * {
 *   type: "frame",            // "frame" | "detection" | "ir" | "audio_level"
 *   data: <base64 JPEG>,      // for type:"frame"
 *   detections: [             // for type:"detection" or included in "frame"
 *     { label:"person", confidence:0.92, bbox:[x,y,w,h] }
 *   ],
 *   ir_grid: [64 floats],     // for type:"ir"
 *   audio_db: -32.5           // for type:"audio_level"
 * }
 */

function connectWS() {
  if (!isAuthenticated()) {
    showLoginStatus('Login required before connecting', 'error');
    setAuthGate(true);
    return;
  }

  const host     = $('wsHost').value.trim()     || 'localhost';
  const port     = $('wsPort').value.trim()     || '8765';
  const path     = $('wsPath').value.trim()     || '/ws';
  const protocol = $('wsProtocol').value;
  const url      = `${protocol}://${host}:${port}${path}`;

  if (State.ws) State.ws.close();

  try {
    State.ws = new WebSocket(url);
  } catch (e) {
    setConnStatus('error', 'Invalid URL');
    return;
  }

  setConnStatus('connecting', 'Connecting…');

  State.ws.onopen = () => {
    State.wsConnected = true;
    setConnStatus('connected', 'Connected');
    DOM.recIndicator.classList.add('active');
    DOM.cameraPlaceholder.style.display = 'none';
    startSession();
  };

  State.ws.onclose = () => {
    State.wsConnected = false;
    setConnStatus('disconnected', 'Disconnected');
    DOM.recIndicator.classList.remove('active');
    DOM.cameraPlaceholder.style.display = 'flex';
    endSession();
  };

  State.ws.onerror = () => {
    setConnStatus('error', 'Error');
  };

  State.ws.onmessage = handleWSMessage;
}

function disconnectWS() {
  State.wsConnected = false;
  if (State.ws) { State.ws.close(); State.ws = null; }
}

function setConnStatus(status, label) {
  DOM.connPill.className = 'connection-pill';
  if (status === 'connected')   DOM.connPill.classList.add('connected');
  if (status === 'error')       DOM.connPill.classList.add('error');
  DOM.connLabel.textContent = label;
}

function handleWSMessage(event) {
  let msg;
  try { msg = JSON.parse(event.data); } catch { return; }

  if (msg.type === 'frame' || msg.data) {
    renderFrame(msg.data, msg.detections || []);
  }
  if (msg.type === 'detection' && msg.detections) {
    processDetections(msg.detections);
  }
  if (msg.type === 'ir' && msg.ir_grid) {
    State.irData = msg.ir_grid;
  }
  if (msg.type === 'audio_level' && msg.audio_db !== undefined) {
    DOM.audioLevel.textContent = `${msg.audio_db.toFixed(1)} dB`;
  }
}

// ═══════════════════════════════════════════
//  CAMERA + DETECTION CANVAS
// ═══════════════════════════════════════════

function initDetectionCanvas() {
  const canvas = DOM.detectionCanvas;
  const viewport = DOM.cameraViewport;
  const resizeObserver = new ResizeObserver(() => {
    canvas.width  = viewport.clientWidth;
    canvas.height = viewport.clientHeight;
  });
  resizeObserver.observe(viewport);
}

function renderFrame(base64jpeg, detections) {
  const canvas = DOM.detectionCanvas;
  const ctx    = canvas.getContext('2d');

  if (!base64jpeg) {
    drawDetections(ctx, detections, canvas.width, canvas.height);
    return;
  }

  const img = new Image();
  img.onload = () => {
    // Fit image to canvas preserving aspect ratio
    const scale  = Math.min(canvas.width / img.width, canvas.height / img.height);
    const dw     = img.width  * scale;
    const dh     = img.height * scale;
    const dx     = (canvas.width  - dw) / 2;
    const dy     = (canvas.height - dh) / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, dx, dy, dw, dh);

    // FPS
    const now = performance.now();
    if (State.lastFrameTime) {
      const fps = Math.round(1000 / (now - State.lastFrameTime));
      State.fpsSamples.push(fps);
      if (State.fpsSamples.length > 20) State.fpsSamples.shift();
      const avgFPS = Math.round(State.fpsSamples.reduce((a,b)=>a+b,0)/State.fpsSamples.length);
      DOM.hudFPS.textContent = `${avgFPS} FPS`;
    }
    State.lastFrameTime = now;
    DOM.hudRes.textContent = `${img.width}×${img.height}`;

    drawDetections(ctx, detections, canvas.width, canvas.height, dx, dy, dw, dh, img.width, img.height);
    processDetections(detections);
  };
  img.src = `data:image/jpeg;base64,${base64jpeg}`;
}

function drawDetections(ctx, detections, cw, ch, dx=0, dy=0, dw=cw, dh=ch, iw=cw, ih=ch) {
  if (!Array.isArray(detections)) return;
  const color = State.bboxColor;

  detections.forEach(det => {
    const { label, confidence, bbox } = det;
    if (!bbox) return;

    const [bx, by, bw, bh] = bbox;
    // Map normalized or pixel coords to canvas
    const norm = bx <= 1.0;
    const sx = norm ? bx * iw : bx;
    const sy = norm ? by * ih : by;
    const sw = norm ? bw * iw : bw;
    const sh = norm ? bh * ih : bh;

    const cx = dx + sx * (dw / iw);
    const cy = dy + sy * (dh / ih);
    const cBw = sw * (dw / iw);
    const cBh = sh * (dh / ih);

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(cx, cy, cBw, cBh);

    // Corner accents
    const corner = 10;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    [[cx, cy], [cx+cBw, cy], [cx, cy+cBh], [cx+cBw, cy+cBh]].forEach(([px, py], i) => {
      const dx2 = i % 2 === 0 ? 1 : -1;
      const dy2 = i < 2 ? 1 : -1;
      ctx.beginPath();
      ctx.moveTo(px + dx2 * corner, py);
      ctx.lineTo(px, py);
      ctx.lineTo(px, py + dy2 * corner);
      ctx.stroke();
    });

    // Label
    const conf = confidence !== undefined ? ` ${(confidence * 100).toFixed(0)}%` : '';
    const text = State.showConf ? `${label}${conf}` : label;

    ctx.font = '12px "JetBrains Mono", monospace';
    const tw = ctx.measureText(text).width;
    const th = 16;
    const tx = cx;
    const ty = cy - th - 4;

    ctx.fillStyle = color;
    ctx.fillRect(tx, ty, tw + 10, th);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, tx + 5, ty + 12);
  });
}

function processDetections(detections) {
  if (!Array.isArray(detections)) return;

  detections.forEach(det => {
    State.detectionCount++;
    if (det.label === 'person' || det.label === 'human') {
      State.humanCount++;
      addLogEntry(det);
      incrementAlert();
      logToDatabase(det);
    } else {
      State.objectCount++;
    }
  });

  DOM.detectionCount.textContent = State.detectionCount;
  DOM.humanCount.textContent     = State.humanCount;
  DOM.objectCount.textContent    = State.objectCount;
}

// ═══════════════════════════════════════════
//  DETECTION LOG
// ═══════════════════════════════════════════

function addLogEntry(det) {
  const empty = DOM.logList.querySelector('.log-empty');
  if (empty) empty.remove();

  const entry = document.createElement('div');
  entry.className = 'log-entry';

  const now = new Date();
  const ts  = now.toLocaleTimeString('en-GB', { hour12: false });

  entry.innerHTML = `
    <div class="log-entry-time">${ts}</div>
    <div class="log-entry-label">🚶 Human Detected</div>
    <div class="log-entry-conf">conf: ${((det.confidence||0)*100).toFixed(1)}% &nbsp;|&nbsp; bbox: [${(det.bbox||[]).map(v=>v.toFixed?v.toFixed(0):v).join(', ')}]</div>
  `;

  DOM.logList.insertBefore(entry, DOM.logList.firstChild);

  // Keep max 100 entries
  const entries = DOM.logList.querySelectorAll('.log-entry');
  if (entries.length > 100) entries[entries.length - 1].remove();
}

function incrementAlert() {
  State.alertCount++;
  DOM.alertCount.textContent = State.alertCount > 99 ? '99+' : State.alertCount;
  DOM.alertCount.classList.add('visible');
}

$('clearLogBtn').addEventListener('click', () => {
  DOM.logList.innerHTML = '<div class="log-empty">No human detections yet</div>';
  State.alertCount = 0;
  DOM.alertCount.classList.remove('visible');
});

$('searchInput').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  DOM.logList.querySelectorAll('.log-entry').forEach(el => {
    el.style.display = el.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
});

// ═══════════════════════════════════════════
//  AUDIO WAVEFORM
// ═══════════════════════════════════════════

async function initWaveform() {
  const canvas = DOM.waveformCanvas;
  // Set actual pixel resolution
  canvas.width  = canvas.offsetWidth  * window.devicePixelRatio || 300;
  canvas.height = canvas.offsetHeight * window.devicePixelRatio || 60;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    State.micStream = stream;

    State.audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
    State.analyser  = State.audioCtx.createAnalyser();
    State.analyser.fftSize = 256;

    const src = State.audioCtx.createMediaStreamSource(stream);
    src.connect(State.analyser);

    DOM.audioBadge.textContent = 'mic active';
    drawWaveform();
  } catch (e) {
    // Mic not available — draw flat demo
    DOM.audioBadge.textContent = 'no mic access';
    drawDemoWaveform();
  }
}

function drawWaveform() {
  const canvas  = DOM.waveformCanvas;
  const ctx     = canvas.getContext('2d');
  const buf     = new Uint8Array(State.analyser.frequencyBinCount);

  function draw() {
    State.waveformAnim = requestAnimationFrame(draw);
    State.analyser.getByteTimeDomainData(buf);

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(0, 0, w, h);

    // Compute RMS for dB meter
    let sum = 0;
    buf.forEach(v => { const n = (v - 128) / 128; sum += n * n; });
    const rms = Math.sqrt(sum / buf.length);
    const db  = rms > 0 ? 20 * Math.log10(rms) : -100;
    DOM.audioLevel.textContent = `${Math.max(-60, db).toFixed(1)} dB`;

    // Draw waveform line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(140,140,255,0.9)';
    ctx.lineWidth   = 1.5 * window.devicePixelRatio;
    ctx.lineJoin    = 'round';

    const sliceW = w / buf.length;
    let x = 0;
    buf.forEach((v, i) => {
      const y = (v / 255.0) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    });

    ctx.stroke();

    // Filled area
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(92,92,255,0.12)';
    ctx.fill();
  }

  draw();
}

// Demo waveform when no mic
function drawDemoWaveform() {
  const canvas = DOM.waveformCanvas;
  const ctx    = canvas.getContext('2d');
  let t = 0;

  function draw() {
    State.waveformAnim = requestAnimationFrame(draw);
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(0, 0, w, h);

    ctx.beginPath();
    ctx.strokeStyle = 'rgba(140,140,255,0.5)';
    ctx.lineWidth = 1.5;

    for (let x = 0; x < w; x++) {
      const y = h/2 + Math.sin((x / w) * Math.PI * 6 + t) * (h * 0.08)
                     + Math.sin((x / w) * Math.PI * 10 + t * 1.3) * (h * 0.05);
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    t += 0.05;
  }
  draw();
}

// ═══════════════════════════════════════════
//  IR HEATMAP
// ═══════════════════════════════════════════

function initHeatmap() {
  const canvas = DOM.heatmapCanvas;
  canvas.width  = 8;
  canvas.height = 8;

  // Start simulated IR data when no WS
  simulateIR();
  drawHeatmap();
}

function simulateIR() {
  // Slowly evolve simulated IR grid
  setInterval(() => {
    if (State.wsConnected) return; // Real data from WS
    for (let i = 0; i < 64; i++) {
      State.irData[i] += (Math.random() - 0.5) * 1.5;
      State.irData[i] = Math.max(18, Math.min(45, State.irData[i]));
    }
    // Occasionally spike a region (simulated body)
    if (Math.random() < 0.1) {
      const r = Math.floor(Math.random() * 7);
      const c = Math.floor(Math.random() * 7);
      State.irData[r * 8 + c]     = 36 + Math.random() * 2;
      State.irData[r * 8 + c + 1] = 35 + Math.random() * 2;
    }
  }, 300);
}

function drawHeatmap() {
  const canvas = DOM.heatmapCanvas;
  const ctx    = canvas.getContext('2d');

  function draw() {
    State.irAnim = requestAnimationFrame(draw);

    const min  = Math.min(...State.irData);
    const max  = Math.max(...State.irData);
    const range = max - min || 1;

    DOM.irMaxTemp.textContent = `${max.toFixed(1)}°C`;

    if (max > 34) {
      DOM.irBadge.textContent = 'heat detected';
      DOM.irBadge.className   = 'tile-badge status-warn';
    } else {
      DOM.irBadge.textContent = 'sensor active';
      DOM.irBadge.className   = 'tile-badge';
    }

    const img = ctx.createImageData(8, 8);
    State.irData.forEach((v, i) => {
      const t = (v - min) / range;
      const [r, g, b] = tempToRGB(t);
      img.data[i * 4]     = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    });

    ctx.putImageData(img, 0, 0);
  }

  draw();
}

function tempToRGB(t) {
  // Cool (blue) → medium (purple) → warm (red) → hot (white)
  if (t < 0.33) {
    const s = t / 0.33;
    return [Math.round(s * 80), Math.round(s * 30), Math.round(80 + s * 100)];
  } else if (t < 0.66) {
    const s = (t - 0.33) / 0.33;
    return [Math.round(80 + s * 170), Math.round(30 + s * 20), Math.round(180 - s * 120)];
  } else {
    const s = (t - 0.66) / 0.34;
    return [255, Math.round(50 + s * 205), Math.round(60 + s * 195)];
  }
}

// ═══════════════════════════════════════════
//  MANUAL CONTROL PAD
// ═══════════════════════════════════════════

const CMD_MAP = {
  'ArrowUp':    'forward',
  'ArrowDown':  'backward',
  'ArrowLeft':  'left',
  'ArrowRight': 'right',
  ' ':          'stop',
};

function setupControlPad() {
  document.querySelectorAll('.dpad-btn').forEach(btn => {
    const key = btn.dataset.key;

    btn.addEventListener('mousedown',   () => sendControl(key, true));
    btn.addEventListener('mouseup',     () => sendControl(key, false));
    btn.addEventListener('mouseleave',  () => sendControl(key, false));
    btn.addEventListener('touchstart',  e => { e.preventDefault(); sendControl(key, true); });
    btn.addEventListener('touchend',    e => { e.preventDefault(); sendControl(key, false); });
  });
}

function setupKeyboard() {
  document.addEventListener('keydown', e => {
    if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight',' '].includes(e.key)) {
      e.preventDefault();
      if (!State.pressedKeys.has(e.key)) {
        State.pressedKeys.add(e.key);
        sendControl(e.key, true);
        highlightDPadBtn(e.key, true);
      }
    }
  });

  document.addEventListener('keyup', e => {
    if (State.pressedKeys.has(e.key)) {
      State.pressedKeys.delete(e.key);
      sendControl(e.key, false);
      highlightDPadBtn(e.key, false);
    }
  });
}

function highlightDPadBtn(key, active) {
  const btn = document.querySelector(`.dpad-btn[data-key="${key}"]`);
  if (btn) btn.classList.toggle('active', active);
}

function sendControl(key, pressed) {
  if (!isAuthenticated() || !State.manualMode) return;

  const cmd = CMD_MAP[key];
  if (!cmd) return;

  const payload = {
    type:    'control',
    command: pressed ? cmd : 'stop',
    speed:   State.speed / 100,
  };

  if (State.ws && State.ws.readyState === WebSocket.OPEN) {
    State.ws.send(JSON.stringify(payload));
  }

  // Visual feedback on stop
  if (cmd === 'stop' || !pressed) {
    DOM.controlMode.textContent = 'MANUAL';
  } else {
    DOM.controlMode.textContent = cmd.toUpperCase();
  }
}

function setupManualToggle() {
  DOM.manualToggle.addEventListener('change', e => {
    if (!isAuthenticated()) {
      e.target.checked = false;
      State.manualMode = false;
      showLoginStatus('Login required before enabling manual control', 'error');
      setAuthGate(true);
      return;
    }

    State.manualMode = e.target.checked;
    DOM.controlMode.textContent = State.manualMode ? 'MANUAL' : 'AUTO';
    DOM.controlBadge.textContent = State.manualMode ? 'manual override' : 'autonomous';
    DOM.controlBadge.className = 'tile-badge' + (State.manualMode ? ' status-warn' : '');
    if (!State.manualMode && State.ws?.readyState === WebSocket.OPEN) {
      State.ws.send(JSON.stringify({ type: 'control', command: 'stop', speed: 0 }));
    }
  });
}

function setupSpeedSlider() {
  $('speedSlider').addEventListener('input', e => {
    State.speed = +e.target.value;
    DOM.speedVal.textContent = `${State.speed}%`;
  });
}

// ═══════════════════════════════════════════
//  SESSION TIMER
// ═══════════════════════════════════════════

function tickSessionTimer() {
  setInterval(() => {
    if (!State.sessionStart) return;
    const elapsed = Math.floor((Date.now() - State.sessionStart) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    DOM.sessionTime.textContent = `${m}:${s}`;
  }, 1000);
}

function updateHUDTimestamp() {
  const now = new Date();
  DOM.hudTimestamp.textContent = now.toLocaleTimeString('en-GB', { hour12: false });
}

// ═══════════════════════════════════════════
//  OPERATOR LOGIN
// ═══════════════════════════════════════════

async function loginOperator() {
  const name = $('operatorInput').value.trim();
  const pass = $('operatorPass').value;

  if (!name) {
    showLoginStatus('Please enter a username', 'error');
    return;
  }

  if (!pass) {
    showLoginStatus('Please enter a password', 'error');
    return;
  }

  if (!State.dbApiUrl) {
    showLoginStatus('Authentication server is not configured', 'error');
    return;
  }

  const initials = name.split(/\s+/).map(w => w[0]).join('').toUpperCase().slice(0,2);

  try {
    const res = await fetch(`${State.dbApiUrl}/login`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ username: name, password: pass }),
    });

    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }

    if (!res.ok) {
      State.operator = null;
      showLoginStatus(data.message || 'Login failed', 'error');
      return;
    }

    State.operator = { id: data.id, name: data.username, initials };
  } catch {
    State.operator = null;
    showLoginStatus('Authentication server unavailable', 'error');
    return;
  }

  showLoginStatus(`Welcome, ${State.operator.name}`, 'success');
  syncOperatorUI();
  setAuthGate(false);
  $('operatorPass').value = '';

}

function logoutOperator() {
  disconnectWS();
  if (State.dbSave) endDbSession();
  State.operator = null;
  State.manualMode = false;
  DOM.manualToggle.checked = false;
  DOM.controlMode.textContent = 'AUTO';
  DOM.controlBadge.textContent = 'autonomous';
  DOM.controlBadge.className = 'tile-badge';
  $('operatorPass').value = '';
  syncOperatorUI();
  setAuthGate(true);
  showLoginStatus('Logged out', 'success');
}

function showLoginStatus(msg, type) {
  showStatus(DOM.loginStatus, msg, type);
  showStatus(DOM.authLoginStatus, msg, type);
}

// ═══════════════════════════════════════════
//  DATABASE SESSION LOGGING
// ═══════════════════════════════════════════

/**
 * DB schema (MySQL):
 *
 * CREATE TABLE operators (
 *   id       INT AUTO_INCREMENT PRIMARY KEY,
 *   username VARCHAR(64) UNIQUE NOT NULL,
 *   password VARCHAR(255) NOT NULL,   -- bcrypt hash
 *   created  DATETIME DEFAULT NOW()
 * );
 *
 * CREATE TABLE sessions (
 *   id           INT AUTO_INCREMENT PRIMARY KEY,
 *   operator_id  INT REFERENCES operators(id),
 *   started_at   DATETIME NOT NULL,
 *   ended_at     DATETIME,
 *   ws_endpoint  VARCHAR(255)
 * );
 *
 * CREATE TABLE detections (
 *   id           INT AUTO_INCREMENT PRIMARY KEY,
 *   session_id   INT REFERENCES sessions(id),
 *   detected_at  DATETIME NOT NULL,
 *   label        VARCHAR(64),
 *   confidence   FLOAT,
 *   bbox_x       FLOAT, bbox_y FLOAT, bbox_w FLOAT, bbox_h FLOAT
 * );
 *
 * REST API (Node/Express proxy) endpoints used:
 *   POST /api/login             { username, password }
 *   POST /api/sessions          { operator_id, ws_endpoint }  → { session_id }
 *   PUT  /api/sessions/:id/end  { ended_at }
 *   POST /api/detections        { session_id, label, confidence, bbox }
 */

async function startDbSession() {
  if (!State.operator?.id || !State.dbApiUrl) return;
  const wsEndpoint = buildWsUrl();
  try {
    const res  = await fetch(`${State.dbApiUrl}/sessions`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ operator_id: State.operator.id, ws_endpoint: wsEndpoint }),
    });
    const data = await res.json();
    State.sessionId = data.session_id;
  } catch { /* offline */ }
}

async function endDbSession() {
  if (!State.sessionId || !State.dbApiUrl) return;
  try {
    await fetch(`${State.dbApiUrl}/sessions/${State.sessionId}/end`, {
      method:  'PUT',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ ended_at: new Date().toISOString() }),
    });
  } catch { /* offline */ }
  State.sessionId = null;
}

async function logToDatabase(det) {
  if (!State.dbSave || !State.sessionId || !State.dbApiUrl) return;
  const [bx=0, by=0, bw=0, bh=0] = det.bbox || [];
  try {
    await fetch(`${State.dbApiUrl}/detections`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        session_id:  State.sessionId,
        label:       det.label,
        confidence:  det.confidence || 0,
        bbox_x: bx, bbox_y: by, bbox_w: bw, bbox_h: bh,
        detected_at: new Date().toISOString(),
      }),
    });
  } catch { /* offline */ }
}

function startSession() {
  State.sessionStart = Date.now();
  if (State.dbSave && State.operator) startDbSession();
}

function endSession() {
  State.sessionStart = null;
  if (State.dbSave) endDbSession();
}

function buildWsUrl() {
  const proto = $('wsProtocol').value;
  const host  = $('wsHost').value.trim() || 'localhost';
  const port  = $('wsPort').value.trim() || '8765';
  const path  = $('wsPath').value.trim() || '/ws';
  return `${proto}://${host}:${port}${path}`;
}

// ═══════════════════════════════════════════
//  MISC BUTTONS
// ═══════════════════════════════════════════

function setupButtons() {
  // Fullscreen camera
  $('fullscreenBtn').addEventListener('click', () => {
    const isCameraView = DOM.mainContent.dataset.view === 'camera';
    setViewMode(isCameraView ? 'dashboard' : 'camera');
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && DOM.mainContent.dataset.view !== 'dashboard') {
      setViewMode('dashboard');
    }
  });

  // Snapshot
  $('snapshotBtn').addEventListener('click', () => {
    const canvas = DOM.detectionCanvas;
    const link   = document.createElement('a');
    link.href     = canvas.toDataURL('image/png');
    link.download = `rover_snapshot_${Date.now()}.png`;
    link.click();
  });
}

// ═══════════════════════════════════════════
//  CANVAS RESIZE
// ═══════════════════════════════════════════

window.addEventListener('resize', () => {
  const wc = DOM.waveformCanvas;
  wc.width  = wc.offsetWidth  * window.devicePixelRatio;
  wc.height = wc.offsetHeight * window.devicePixelRatio;

  const dc = DOM.detectionCanvas;
  dc.width  = DOM.cameraViewport.clientWidth;
  dc.height = DOM.cameraViewport.clientHeight;
});

// ═══════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', init);
