"""
layer6_database.py — Layer 6: Database (PostgreSQL + SQLite fallback)
Uses PostgreSQL (Supabase) when DATABASE_URL is set, SQLite locally.
"""

import json
import os

DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_POSTGRES  = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "resume_screening.db")

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id                 SERIAL PRIMARY KEY,
    title              TEXT NOT NULL,
    description        TEXT NOT NULL,
    requirements       TEXT,
    screening_status   TEXT DEFAULT 'pending',
    screening_message  TEXT DEFAULT '',
    created_at         TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS candidates (
    id               SERIAL PRIMARY KEY,
    job_id           INTEGER REFERENCES jobs(id),
    filename         TEXT,
    candidate_name   TEXT,
    candidate_email  TEXT,
    final_score      REAL,
    ranking_label    TEXT,
    breakdown        TEXT,
    matched_skills   TEXT,
    missing_skills   TEXT,
    experience_years REAL,
    experience_level TEXT,
    highest_degree   TEXT,
    certifications   TEXT,
    email_sent       INTEGER DEFAULT 0,
    screened_at      TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS email_log (
    id           SERIAL PRIMARY KEY,
    candidate_id INTEGER REFERENCES candidates(id),
    email_type   TEXT,
    recipient    TEXT,
    status       TEXT,
    sent_at      TIMESTAMP DEFAULT NOW()
);
"""

CREATE_TABLES_SQLITE = """
CREATE TABLE IF NOT EXISTS jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    title             TEXT NOT NULL,
    description       TEXT NOT NULL,
    requirements      TEXT,
    screening_status  TEXT DEFAULT 'pending',
    screening_message TEXT DEFAULT '',
    created_at        TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS candidates (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           INTEGER REFERENCES jobs(id),
    filename         TEXT,
    candidate_name   TEXT,
    candidate_email  TEXT,
    final_score      REAL,
    ranking_label    TEXT,
    breakdown        TEXT,
    matched_skills   TEXT,
    missing_skills   TEXT,
    experience_years REAL,
    experience_level TEXT,
    highest_degree   TEXT,
    certifications   TEXT,
    email_sent       INTEGER DEFAULT 0,
    screened_at      TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS email_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER REFERENCES candidates(id),
    email_type   TEXT,
    recipient    TEXT,
    status       TEXT,
    sent_at      TEXT DEFAULT (datetime('now'))
);
"""


class Database:
    def __init__(self):
        self._init_db()

    def _connect(self):
        if USE_POSTGRES:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = False
            return conn
        else:
            conn = sqlite3.connect(SQLITE_PATH)
            conn.row_factory = sqlite3.Row
            return conn

    def _init_db(self):
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                for stmt in CREATE_TABLES_SQL.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
                # Add new columns to existing tables if they don't exist
                migrations = [
                    "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS screening_status TEXT DEFAULT 'pending'",
                    "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS screening_message TEXT DEFAULT ''",
                ]
                for migration in migrations:
                    try:
                        cur.execute(migration)
                    except Exception:
                        pass
            else:
                cur.executescript(CREATE_TABLES_SQLITE)
            conn.commit()
        finally:
            conn.close()

    def _rows_to_dicts(self, rows, cursor=None):
        if USE_POSTGRES:
            cols = [desc[0] for desc in cursor.description]
            return [dict(zip(cols, row)) for row in rows]
        else:
            return [dict(row) for row in rows]

    def save_job(self, title: str, description: str, requirements: dict = None) -> int:
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute(
                    "INSERT INTO jobs (title, description, requirements) VALUES (%s,%s,%s) RETURNING id",
                    (title, description, json.dumps(requirements or {}))
                )
                job_id = cur.fetchone()[0]
            else:
                cur.execute(
                    "INSERT INTO jobs (title, description, requirements) VALUES (?,?,?)",
                    (title, description, json.dumps(requirements or {}))
                )
                job_id = cur.lastrowid
            conn.commit()
            return job_id
        finally:
            conn.close()

    def get_jobs(self) -> list[dict]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            return self._rows_to_dicts(cur.fetchall(), cur)
        finally:
            conn.close()

    def get_job(self, job_id: int) -> dict | None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            else:
                cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._rows_to_dicts([row], cur)[0]
        finally:
            conn.close()

    def save_candidate(self, job_id: int, score_result: dict, filename: str) -> int:
        conn = self._connect()
        try:
            cur = conn.cursor()
            params = (
                job_id, filename,
                score_result.get("candidate_name"),
                score_result.get("candidate_email"),
                score_result.get("final_score"),
                score_result.get("ranking_label"),
                json.dumps(score_result.get("breakdown", {})),
                json.dumps(score_result.get("matched_skills", [])),
                json.dumps(score_result.get("missing_skills", [])),
                score_result.get("experience_years"),
                score_result.get("experience_level"),
                score_result.get("highest_degree"),
                json.dumps(score_result.get("certifications", [])),
            )
            if USE_POSTGRES:
                cur.execute("""
                    INSERT INTO candidates
                    (job_id,filename,candidate_name,candidate_email,final_score,ranking_label,
                     breakdown,matched_skills,missing_skills,experience_years,experience_level,
                     highest_degree,certifications)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id
                """, params)
                cid = cur.fetchone()[0]
            else:
                cur.execute("""
                    INSERT INTO candidates
                    (job_id,filename,candidate_name,candidate_email,final_score,ranking_label,
                     breakdown,matched_skills,missing_skills,experience_years,experience_level,
                     highest_degree,certifications)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, params)
                cid = cur.lastrowid
            conn.commit()
            return cid
        finally:
            conn.close()

    def get_candidates(self, job_id: int) -> list[dict]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute("SELECT * FROM candidates WHERE job_id=%s ORDER BY final_score DESC", (job_id,))
            else:
                cur.execute("SELECT * FROM candidates WHERE job_id=? ORDER BY final_score DESC", (job_id,))
            rows = self._rows_to_dicts(cur.fetchall(), cur)
            for r in rows:
                r["breakdown"]      = json.loads(r["breakdown"] or "{}")
                r["matched_skills"] = json.loads(r["matched_skills"] or "[]")
                r["missing_skills"] = json.loads(r["missing_skills"] or "[]")
                r["certifications"] = json.loads(r["certifications"] or "[]")
            return rows
        finally:
            conn.close()
    
        # ── Usage tracking ────────────────────────────────────────────────────────

    def get_usage_today(self, user_id: str, tool: str) -> int:
        """Count screenings by this user today (UTC)."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute("""
                    SELECT COUNT(*) FROM usage_log
                    WHERE user_id = %s
                      AND tool = %s
                      AND created_at >= NOW()::date
                """, (user_id, tool))
            else:
                cur.execute("""
                    SELECT COUNT(*) FROM usage_log
                    WHERE user_id = ?
                      AND tool = ?
                      AND date(created_at) = date('now')
                """, (user_id, tool))
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()

    def log_usage(self, user_id: str, tool: str, action: str):
        """Record one usage action."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute(
                    "INSERT INTO usage_log (user_id, tool, action) VALUES (%s, %s, %s)",
                    (user_id, tool, action)
                )
            else:
                cur.execute(
                    "INSERT INTO usage_log (user_id, tool, action) VALUES (?, ?, ?)",
                    (user_id, tool, action)
                )
            conn.commit()
        finally:
            conn.close()

    def mark_email_sent(self, candidate_id: int, email_type: str, recipient: str, status: str):
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute("UPDATE candidates SET email_sent=1 WHERE id=%s", (candidate_id,))
                cur.execute(
                    "INSERT INTO email_log (candidate_id,email_type,recipient,status) VALUES (%s,%s,%s,%s)",
                    (candidate_id, email_type, recipient, status)
                )
            else:
                cur.execute("UPDATE candidates SET email_sent=1 WHERE id=?", (candidate_id,))
                cur.execute(
                    "INSERT INTO email_log (candidate_id,email_type,recipient,status) VALUES (?,?,?,?)",
                    (candidate_id, email_type, recipient, status)
                )
            conn.commit()
        finally:
            conn.close()

    def get_analytics(self, job_id: int) -> dict:
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute(
                    "SELECT ranking_label,final_score,matched_skills FROM candidates WHERE job_id=%s",
                    (job_id,)
                )
            else:
                cur.execute(
                    "SELECT ranking_label,final_score,matched_skills FROM candidates WHERE job_id=?",
                    (job_id,)
                )
            rows = self._rows_to_dicts(cur.fetchall(), cur)
        finally:
            conn.close()

        if not rows:
            return {}

        scores     = [r["final_score"] for r in rows]
        labels     = [r["ranking_label"] for r in rows]
        all_skills = []
        for r in rows:
            all_skills.extend(json.loads(r["matched_skills"] or "[]"))

        skill_counts = {}
        for s in all_skills:
            skill_counts[s] = skill_counts.get(s, 0) + 1

        total = len(rows)
        return {
            "total_screened":  total,
            "fit_count":       labels.count("Fit"),
            "maybe_count":     labels.count("Maybe"),
            "not_fit_count":   labels.count("Not Fit"),
            "shortlist_ratio": round(labels.count("Fit") / total * 100, 1),
            "avg_score":       round(sum(scores) / total, 1),
            "top_skills":      sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        }

    def set_job_status(self, job_id: int, status: str, message: str = ""):
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute(
                    "UPDATE jobs SET screening_status=%s, screening_message=%s WHERE id=%s",
                    (status, message, job_id)
                )
            else:
                cur.execute(
                    "UPDATE jobs SET screening_status=?, screening_message=? WHERE id=?",
                    (status, message, job_id)
                )
            conn.commit()
        finally:
            conn.close()

    def get_job_status(self, job_id: int) -> dict | None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute(
                    "SELECT screening_status, screening_message FROM jobs WHERE id=%s",
                    (job_id,)
                )
            else:
                cur.execute(
                    "SELECT screening_status, screening_message FROM jobs WHERE id=?",
                    (job_id,)
                )
            row = cur.fetchone()
            if not row:
                return None
            if USE_POSTGRES:
                return {"screening_status": row[0], "screening_message": row[1]}
            else:
                return {"screening_status": row["screening_status"],
                        "screening_message": row["screening_message"]}
        finally:
            conn.close()

    def clear_job_candidates(self, job_id: int):
        conn = self._connect()
        try:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute("DELETE FROM candidates WHERE job_id=%s", (job_id,))
            else:
                cur.execute("DELETE FROM candidates WHERE job_id=?", (job_id,))
            conn.commit()
        finally:
            conn.close()