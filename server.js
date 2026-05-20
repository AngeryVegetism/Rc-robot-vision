/**
 * Robot Framework — server.js
 * Lightweight Express API proxy between the dashboard frontend
 * and MySQL database. Run with: node server.js
 *
 * Install deps:  npm install express mysql2 bcrypt cors dotenv
 */

require('dotenv').config();
const express = require('express');
const mysql   = require('mysql2/promise');
const bcrypt  = require('bcrypt');
const cors    = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app  = express();
const PORT = process.env.PORT || 3001;

app.use(cors({ origin: '*' }));   
app.use(express.json());

function toMysqlDateTime(value = new Date()) {
  const date = value instanceof Date ? value : new Date(value);
  const validDate = Number.isNaN(date.getTime()) ? new Date() : date;
  return validDate.toISOString().slice(0, 19).replace('T', ' ');
}

// ── DB Pool ──────────────────────────────────────────────────

const pool = mysql.createPool({
  host:     process.env.DB_HOST     || '',
  user:     process.env.DB_USER     || '',
  password: process.env.DB_PASS     || '',
  database: process.env.DB_NAME     || '',
  port:     process.env.DB_PORT     || '',
  waitForConnections: true,
  connectionLimit:    10,
});

// ── Routes ────────────────────────────────────────────────────

/**
 * POST /api/login
 * Body: { username, password }
 * Returns: { id, username }
 */
app.post('/api/login', async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) return res.status(400).json({ message: 'Missing fields' });

  try {
    const [rows] = await pool.query('SELECT * FROM operators WHERE username = ?', [username]);
    if (!rows.length) return res.status(401).json({ message: 'Invalid credentials' });

    const operator = rows[0];
    let ok = false;

    if (typeof operator.password === 'string' && operator.password.startsWith('$2')) {
      ok = await bcrypt.compare(password, operator.password);
    } else {
      ok = password === operator.password;

      if (ok) {
        const hash = await bcrypt.hash(password, 12);
        await pool.query('UPDATE operators SET password = ? WHERE id = ?', [hash, operator.id]);
      }
    }

    if (!ok) return res.status(401).json({ message: 'Invalid credentials' });

    res.json({ id: operator.id, username: operator.username, display_name: operator.display_name || operator.username });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/operators  (registration / seeding — protect in prod)
 * Body: { username, password, display_name? }
 */
app.post('/api/operators', async (req, res) => {
  const { username, password, display_name } = req.body;
  if (!username || !password) return res.status(400).json({ message: 'Missing fields' });

  try {
    const hash = await bcrypt.hash(password, 12);
    const [result] = await pool.query(
      'INSERT INTO operators (username, password, display_name) VALUES (?, ?, ?)',
      [username, hash, display_name || username]
    );
    res.status(201).json({ id: result.insertId, username });
  } catch (err) {
    if (err.code === 'ER_DUP_ENTRY') return res.status(409).json({ message: 'Username taken' });
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * PUT /api/operators/:id
 * Body: { username, display_name, current_password?, new_password? }
 */
app.put('/api/operators/:id', async (req, res) => {
  const { id } = req.params;
  const { username, display_name, current_password, new_password } = req.body;

  if (!username) return res.status(400).json({ message: 'Username required' });

  try {
    const [rows] = await pool.query('SELECT * FROM operators WHERE id = ?', [id]);
    if (!rows.length) return res.status(404).json({ message: 'Operator not found' });

    const operator = rows[0];
    const updates = {
      username,
      display_name: display_name || username,
      password: operator.password,
    };

    if (new_password) {
      let ok = false;
      if (typeof operator.password === 'string' && operator.password.startsWith('$2')) {
        ok = await bcrypt.compare(current_password || '', operator.password);
      } else {
        ok = current_password === operator.password;
      }

      if (!ok) return res.status(401).json({ message: 'Current password is incorrect' });
      updates.password = await bcrypt.hash(new_password, 12);
    }

    await pool.query(
      'UPDATE operators SET username = ?, display_name = ?, password = ? WHERE id = ?',
      [updates.username, updates.display_name, updates.password, id]
    );

    res.json({ id: Number(id), username: updates.username, display_name: updates.display_name });
  } catch (err) {
    if (err.code === 'ER_DUP_ENTRY') return res.status(409).json({ message: 'Username taken' });
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/sessions
 * Body: { operator_id, ws_endpoint }
 * Returns: { session_id }
 */
app.post('/api/sessions', async (req, res) => {
  const { operator_id, ws_endpoint } = req.body;
  if (!operator_id) return res.status(400).json({ message: 'operator_id required' });

  try {
    const [result] = await pool.query(
      'INSERT INTO sessions (operator_id, ws_endpoint, started_at) VALUES (?, ?, NOW())',
      [operator_id, ws_endpoint || null]
    );
    res.status(201).json({ session_id: result.insertId });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * PUT /api/sessions/:id/end
 * Body: { ended_at, duration_s? }
 */
app.put('/api/sessions/:id/end', async (req, res) => {
  const { id } = req.params;
  const ended_at = toMysqlDateTime(req.body.ended_at);
  const duration_s = Number.isFinite(Number(req.body.duration_s))
    ? Math.max(0, Math.floor(Number(req.body.duration_s)))
    : null;

  try {
    if (duration_s !== null) {
      try {
        await pool.query(
          'UPDATE sessions SET ended_at = ?, duration_s = ? WHERE id = ?',
          [ended_at, duration_s, id]
        );
        return res.json({ ok: true, ended_at, duration_s });
      } catch (err) {
        const generatedOrMissingDuration =
          err.code === 'ER_NON_DEFAULT_VALUE_FOR_GENERATED_COLUMN'
          || err.code === 'ER_BAD_FIELD_ERROR';
        if (!generatedOrMissingDuration) throw err;
      }
    }

    await pool.query('UPDATE sessions SET ended_at = ? WHERE id = ?', [ended_at, id]);
    res.json({ ok: true, ended_at, duration_s });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * POST /api/detections
 * Body: { session_id, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, detected_at? }
 */
app.post('/api/detections', async (req, res) => {
  const { session_id, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, detected_at } = req.body;
  if (!session_id) return res.status(400).json({ message: 'session_id required' });
  const detectedAt = toMysqlDateTime(detected_at);

  try {
    const [result] = await pool.query(
      `INSERT INTO detections
         (session_id, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, detected_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [session_id, label || 'person', confidence || 0,
       bbox_x || 0, bbox_y || 0, bbox_w || 0, bbox_h || 0,
       detectedAt]
    );
    res.status(201).json({ detection_id: result.insertId });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/sessions  — list recent sessions with operator info
 */
app.get('/api/sessions', async (req, res) => {
  try {
    const [rows] = await pool.query(`
      SELECT s.id, o.username, s.started_at, s.ended_at, s.duration_s,
             s.ws_endpoint, COUNT(d.id) AS human_detections
      FROM sessions s
      JOIN operators o ON o.id = s.operator_id
      LEFT JOIN detections d ON d.session_id = s.id
      GROUP BY s.id
      ORDER BY s.started_at DESC
      LIMIT 100
    `);
    res.json(rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * GET /api/detections?session_id=&limit=
 */
app.get('/api/detections', async (req, res) => {
  const { session_id, limit = 200 } = req.query;
  try {
    let sql   = 'SELECT * FROM detections';
    const params = [];
    if (session_id) { sql += ' WHERE session_id = ?'; params.push(session_id); }
    sql += ' ORDER BY detected_at DESC LIMIT ?';
    params.push(Number(limit));

    const [rows] = await pool.query(sql, params);
    res.json(rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error' });
  }
});

app.get('/api/rtsp-health', (req, res) => {
  res.json({ ok: true, ffmpeg: 'required' });
});

/**
 * GET /api/rtsp-proxy?url=rtsp://...
 * Converts an RTSP camera feed to browser-displayable multipart MJPEG.
 * Requires ffmpeg on the machine running this server.
 */
app.get('/api/rtsp-proxy', (req, res) => {
  const streamUrl = String(req.query.url || '').trim();
  let hasFrame = false;
  let closed = false;

  let parsed;
  try {
    parsed = new URL(streamUrl);
  } catch {
    return res.status(400).json({ message: 'Invalid RTSP URL' });
  }

  if (parsed.protocol !== 'rtsp:') {
    return res.status(400).json({ message: 'Only rtsp:// streams are supported by this proxy' });
  }

  res.writeHead(200, {
    'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Connection': 'close',
  });

  const ffmpeg = spawn('ffmpeg', [
    '-hide_banner',
    '-loglevel', 'error',
    '-rtsp_transport', 'tcp',
    '-i', streamUrl,
    '-an',
    '-vf', 'fps=15',
    '-q:v', '5',
    '-f', 'mjpeg',
    'pipe:1',
  ]);

  let buffer = Buffer.alloc(0);

  ffmpeg.stderr.on('data', chunk => {
    console.error(`rtsp proxy ffmpeg: ${chunk.toString().trim()}`);
  });

  ffmpeg.stdout.on('data', chunk => {
    buffer = Buffer.concat([buffer, chunk]);

    while (true) {
      const start = buffer.indexOf(Buffer.from([0xff, 0xd8]));
      const end = buffer.indexOf(Buffer.from([0xff, 0xd9]), start + 2);

      if (start === -1 || end === -1) {
        if (start > 0) buffer = buffer.slice(start);
        break;
      }

      const frame = buffer.slice(start, end + 2);
      buffer = buffer.slice(end + 2);
      hasFrame = true;

      res.write(`--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${frame.length}\r\n\r\n`);
      res.write(frame);
      res.write('\r\n');
    }
  });

  ffmpeg.on('error', err => {
    console.error('ffmpeg failed:', err.message);
    if (!closed) res.end();
  });

  ffmpeg.on('close', code => {
    if (!hasFrame) {
      console.error(`rtsp proxy ended before any frames were received. ffmpeg exit code: ${code}`);
    }
    if (!closed) res.end();
  });

  res.on('close', () => {
    closed = true;
    if (!ffmpeg.killed) ffmpeg.kill('SIGTERM');
  });
});

app.get('/admin', (req, res) => {
  res.sendFile(path.join(__dirname, 'admin.html'));
});

app.use(express.static(__dirname));

// ── Start ─────────────────────────────────────────────────────

app.listen(PORT, () => console.log(`Framework API listening on port ${PORT}`));
