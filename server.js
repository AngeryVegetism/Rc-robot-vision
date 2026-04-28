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

const app  = express();
const PORT = process.env.PORT || 3001;

app.use(cors({ origin: '*' }));   
app.use(express.json());

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

    res.json({ id: operator.id, username: operator.username });
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
 * Body: { ended_at }
 */
app.put('/api/sessions/:id/end', async (req, res) => {
  const { id } = req.params;
  const ended_at = req.body.ended_at || new Date().toISOString();

  try {
    await pool.query('UPDATE sessions SET ended_at = ? WHERE id = ?', [ended_at, id]);
    res.json({ ok: true });
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

  try {
    const [result] = await pool.query(
      `INSERT INTO detections
         (session_id, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, detected_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [session_id, label || 'person', confidence || 0,
       bbox_x || 0, bbox_y || 0, bbox_w || 0, bbox_h || 0,
       detected_at || new Date().toISOString()]
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

// ── Start ─────────────────────────────────────────────────────

app.listen(PORT, () => console.log(`Framework API listening on port ${PORT}`));
