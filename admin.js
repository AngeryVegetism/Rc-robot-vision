'use strict';

const State = {
  operator: null,
  dbApiUrl: 'http://localhost:3001/api',
  theme: 'dark',
  colorScheme: 'blue',
};

const $ = id => document.getElementById(id);

const DOM = {
  authOverlay: $('authOverlay'),
  authLoginStatus: $('authLoginStatus'),
  displayUserName: $('displayUserName'),
  userChip: $('userChip'),
  userInitials: $('userInitials'),
  adminStatus: $('adminStatus'),
  adminTotalSessions: $('adminTotalSessions'),
  adminTotalDetections: $('adminTotalDetections'),
  adminAvgDuration: $('adminAvgDuration'),
  adminActiveSessions: $('adminActiveSessions'),
  adminSessionsBody: $('adminSessionsBody'),
  adminOperatorBars: $('adminOperatorBars'),
  adminHourBars: $('adminHourBars'),
  adminDetectionsList: $('adminDetectionsList'),
};

function init() {
  initAppearance();
  $('loginBtn').addEventListener('click', loginOperator);
  $('logoutBtn').addEventListener('click', logoutOperator);
  $('adminRefreshBtn').addEventListener('click', loadAdminData);
  $('operatorInput').addEventListener('keydown', handleLoginKeydown);
  $('operatorPass').addEventListener('keydown', handleLoginKeydown);
  syncOperatorUI();
  setAuthGate(true);
}

function initAppearance() {
  const savedTheme = localStorage.getItem('robotTheme') || State.theme;
  const savedScheme = localStorage.getItem('robotColorScheme') || State.colorScheme;
  document.documentElement.setAttribute('data-theme', savedTheme);
  document.documentElement.setAttribute('data-scheme', savedScheme);
}

function handleLoginKeydown(event) {
  if (event.key === 'Enter') {
    event.preventDefault();
    loginOperator();
  }
}

async function loginOperator() {
  const name = $('operatorInput').value.trim();
  const pass = $('operatorPass').value;

  if (!name) {
    showStatus(DOM.authLoginStatus, 'Please enter a username', 'error');
    return;
  }

  if (!pass) {
    showStatus(DOM.authLoginStatus, 'Please enter a password', 'error');
    return;
  }

  try {
    const res = await fetch(`${State.dbApiUrl}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: name, password: pass }),
    });

    let data = {};
    try {
      data = await res.json();
    } catch {
      data = {};
    }

    if (!res.ok) {
      State.operator = null;
      showStatus(DOM.authLoginStatus, data.message || 'Login failed', 'error');
      return;
    }

    State.operator = {
      id: data.id,
      name: data.username,
      displayName: data.display_name || data.username,
      initials: getInitials(data.display_name || data.username),
    };
  } catch {
    State.operator = null;
    showStatus(DOM.authLoginStatus, 'Authentication server unavailable', 'error');
    return;
  }

  $('operatorPass').value = '';
  syncOperatorUI();
  setAuthGate(false);
  showStatus(DOM.adminStatus, `Welcome, ${State.operator.name}`, 'success');
  loadAdminData();
}

function logoutOperator() {
  State.operator = null;
  syncOperatorUI();
  setAuthGate(true);
  clearAdminData();
  $('operatorPass').value = '';
  showStatus(DOM.authLoginStatus, 'Logged out', 'success');
}

function setAuthGate(locked) {
  document.body.classList.toggle('auth-locked', locked);
  DOM.authOverlay.classList.toggle('open', locked);
  if (locked) window.setTimeout(() => $('operatorInput').focus(), 50);
}

function syncOperatorUI() {
  const name = State.operator?.displayName || State.operator?.name || 'Not logged in';
  const initials = State.operator?.initials || '--';

  DOM.displayUserName.textContent = name;
  DOM.userChip.textContent = State.operator?.initials || '?';
  DOM.userInitials.textContent = initials;
}

function showStatus(target, msg, type) {
  target.textContent = msg;
  target.className = `login-status ${type || ''}`;
  if (!msg) return;

  setTimeout(() => {
    target.textContent = '';
    target.className = 'login-status';
  }, 4000);
}

async function loadAdminData() {
  if (!State.operator) {
    setAuthGate(true);
    return;
  }

  showStatus(DOM.adminStatus, 'Loading database data...', '');
  try {
    const [sessionsRes, detectionsRes] = await Promise.all([
      fetch(`${State.dbApiUrl}/sessions`),
      fetch(`${State.dbApiUrl}/detections?limit=500`),
    ]);

    if (!sessionsRes.ok || !detectionsRes.ok) throw new Error();

    const sessions = await sessionsRes.json();
    const detections = await detectionsRes.json();
    renderAdminSummary(sessions, detections);
    renderAdminSessions(sessions);
    renderAdminOperatorBars(sessions);
    renderAdminHourBars(detections);
    renderAdminDetections(detections);
    showStatus(DOM.adminStatus, 'Database data updated', 'success');
  } catch {
    showStatus(DOM.adminStatus, 'Could not load admin data. Check that server.js is running.', 'error');
  }
}

function clearAdminData() {
  DOM.adminTotalSessions.textContent = '0';
  DOM.adminTotalDetections.textContent = '0';
  DOM.adminAvgDuration.textContent = '00:00';
  DOM.adminActiveSessions.textContent = '0';
  DOM.adminSessionsBody.innerHTML = '<tr><td colspan="5">No session data yet</td></tr>';
  DOM.adminOperatorBars.innerHTML = '';
  DOM.adminHourBars.innerHTML = '';
  DOM.adminDetectionsList.innerHTML = '<div class="admin-empty">No detections found</div>';
  showStatus(DOM.adminStatus, '', '');
}

function renderAdminSummary(sessions, detections) {
  const totalDuration = sessions.reduce((sum, session) => sum + getSessionDuration(session), 0);
  const avgDuration = sessions.length ? Math.round(totalDuration / sessions.length) : 0;
  const activeSessions = sessions.filter(session => !session.ended_at).length;

  DOM.adminTotalSessions.textContent = sessions.length;
  DOM.adminTotalDetections.textContent = detections.length;
  DOM.adminAvgDuration.textContent = formatDuration(avgDuration);
  DOM.adminActiveSessions.textContent = activeSessions;
}

function renderAdminSessions(sessions) {
  if (!sessions.length) {
    DOM.adminSessionsBody.innerHTML = '<tr><td colspan="5">No session data yet</td></tr>';
    return;
  }

  DOM.adminSessionsBody.innerHTML = sessions.map(session => `
    <tr>
      <td>${escapeHtml(session.username || 'Unknown')}</td>
      <td>${formatDateTime(session.started_at)}</td>
      <td>${session.ended_at ? formatDateTime(session.ended_at) : 'Active'}</td>
      <td>${formatDuration(getSessionDuration(session))}</td>
      <td>${Number(session.human_detections || 0)}</td>
    </tr>
  `).join('');
}

function renderAdminOperatorBars(sessions) {
  const counts = new Map();
  sessions.forEach(session => {
    const name = session.username || 'Unknown';
    counts.set(name, (counts.get(name) || 0) + 1);
  });
  renderBars(DOM.adminOperatorBars, [...counts.entries()], 'sessions');
}

function renderAdminHourBars(detections) {
  const counts = new Map();
  detections.forEach(det => {
    const hour = parseDbDate(det.detected_at).getHours();
    const label = Number.isNaN(hour) ? 'Unknown' : `${String(hour).padStart(2, '0')}:00`;
    counts.set(label, (counts.get(label) || 0) + 1);
  });
  renderBars(DOM.adminHourBars, [...counts.entries()].sort(([a], [b]) => a.localeCompare(b)), 'detections');
}

function renderBars(container, entries, unit) {
  if (!entries.length) {
    container.innerHTML = '<div class="admin-empty">No data yet</div>';
    return;
  }

  const max = Math.max(...entries.map(([, value]) => value), 1);
  container.innerHTML = entries.map(([label, value]) => `
    <div class="admin-bar-row">
      <span>${escapeHtml(label)}</span>
      <div class="admin-bar-track"><div class="admin-bar-fill" style="width:${Math.max(8, (value / max) * 100)}%"></div></div>
      <strong>${value} ${unit}</strong>
    </div>
  `).join('');
}

function renderAdminDetections(detections) {
  if (!detections.length) {
    DOM.adminDetectionsList.innerHTML = '<div class="admin-empty">No detections found</div>';
    return;
  }

  DOM.adminDetectionsList.innerHTML = detections.slice(0, 12).map(det => `
    <div class="admin-detection-item">
      <span>${formatDateTime(det.detected_at)}</span>
      <strong>${escapeHtml(det.label || 'person')} - ${((Number(det.confidence) || 0) * 100).toFixed(1)}%</strong>
    </div>
  `).join('');
}

function getSessionDuration(session) {
  if (Number(session.duration_s)) return Number(session.duration_s);
  if (!session.started_at) return 0;
  const start = parseDbDate(session.started_at).getTime();
  const end = session.ended_at ? parseDbDate(session.ended_at).getTime() : Date.now();
  if (Number.isNaN(start) || Number.isNaN(end)) return 0;
  return Math.max(0, Math.floor((end - start) / 1000));
}

function formatDuration(seconds) {
  const safeSeconds = Math.max(0, Number(seconds) || 0);
  const h = Math.floor(safeSeconds / 3600);
  const m = Math.floor((safeSeconds % 3600) / 60);
  const s = Math.floor(safeSeconds % 60);
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatDateTime(value) {
  if (!value) return '--';
  const date = parseDbDate(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString('en-GB', { hour12: false });
}

function parseDbDate(value) {
  if (value instanceof Date) return value;
  return new Date(String(value || '').replace(' ', 'T'));
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function getInitials(name) {
  return String(name || '')
    .trim()
    .split(/\s+/)
    .map(w => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2) || '--';
}

document.addEventListener('DOMContentLoaded', init);
