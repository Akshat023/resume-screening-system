"""
layer6_database.py — Layer 6: SQLite Database
Persists jobs, candidates, and email logs across sessions.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "resume_screening.db")


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    title        TEXT NOT NULL,
                    description  TEXT NOT NULL,
                    requirements TEXT,
                    created_at   TEXT DEFAULT (datetime('now'))
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
            """)

    def save_job(self, title: str, description: str, requirements: dict = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs (title, description, requirements) VALUES (?, ?, ?)",
                (title, description, json.dumps(requirements or {}))
            )
            return cur.lastrowid

    def get_jobs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_job(self, job_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def save_candidate(self, job_id: int, score_result: dict, filename: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO candidates (
                    job_id, filename, candidate_name, candidate_email,
                    final_score, ranking_label, breakdown,
                    matched_skills, missing_skills,
                    experience_years, experience_level,
                    highest_degree, certifications
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                filename,
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
            ))
            return cur.lastrowid

    def get_candidates(self, job_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE job_id = ? ORDER BY final_score DESC",
                (job_id,)
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["breakdown"]      = json.loads(r["breakdown"] or "{}")
                r["matched_skills"] = json.loads(r["matched_skills"] or "[]")
                r["missing_skills"] = json.loads(r["missing_skills"] or "[]")
                r["certifications"] = json.loads(r["certifications"] or "[]")
                results.append(r)
            return results

    def mark_email_sent(self, candidate_id: int, email_type: str, recipient: str, status: str):
        with self._connect() as conn:
            conn.execute("UPDATE candidates SET email_sent = 1 WHERE id = ?", (candidate_id,))
            conn.execute(
                "INSERT INTO email_log (candidate_id, email_type, recipient, status) VALUES (?, ?, ?, ?)",
                (candidate_id, email_type, recipient, status)
            )

    def get_analytics(self, job_id: int) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ranking_label, final_score, matched_skills FROM candidates WHERE job_id = ?",
                (job_id,)
            ).fetchall()

        if not rows:
            return {}

        scores = [r["final_score"] for r in rows]
        labels = [r["ranking_label"] for r in rows]
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

    def clear_job_candidates(self, job_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))