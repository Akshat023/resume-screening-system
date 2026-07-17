"""
app.py — Flask REST API (Docker deployment)
Pure JSON API — no render_template. Next.js handles all UI.
CORS enabled so Vercel frontend can communicate with this container.
"""

import os
import json
import tempfile
import io
import csv
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from layer1_ingestion import load_resume
from layer2_preprocessing import preprocess_resume
from layer4_extraction import extract_resume_info
from layer5_model import ResumeScorer, generate_ai_feedback
from layer6_database import Database
from layer8_email import EmailSender

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "resume-screening-secret-2025")

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow requests from your Vercel frontend and localhost for development
ALLOWED_ORIGINS = [
    "http://localhost:3000",                          # Next.js dev server
    "http://localhost:3001",
    os.getenv("FRONTEND_URL", ""),                    # set to your Vercel URL in .env
]
CORS(app, resources={r"/api/*": {"origins": [o for o in ALLOWED_ORIGINS if o]}})

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}

db     = Database()
scorer = ResumeScorer()
mailer = EmailSender()

# Preload BERT at startup so first request isn't slow
def _preload_models():
    try:
        from layer3_features import _get_bert_model
        _get_bert_model()
        print("[Startup] BERT model loaded.")
    except Exception as e:
        print(f"[Startup] BERT preload failed (non-fatal): {e}")

threading.Thread(target=_preload_models, daemon=True).start()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Health check — used by Docker healthcheck + frontend to verify connection
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status":  "ok",
        "service": "resume-screening-api",
        "version": "2.0.0",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/jobs")
def get_jobs():
    return jsonify(db.get_jobs())


@app.route("/api/jobs/<int:job_id>")
def get_job(job_id):
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ─────────────────────────────────────────────────────────────────────────────
# Screening
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/screen", methods=["POST"])
def screen():
    job_title       = request.form.get("job_title", "").strip()
    job_description = request.form.get("job_description", "").strip()
    min_exp         = int(request.form.get("min_exp", 0))
    min_edu         = int(request.form.get("min_edu", 3))
    w_skill         = float(request.form.get("w_skill", 0.40))
    w_exp           = float(request.form.get("w_exp",   0.30))
    w_tfidf         = float(request.form.get("w_tfidf", 0.20))
    w_edu           = float(request.form.get("w_edu",   0.10))

    if not job_title or not job_description:
        return jsonify({"error": "Job title and description are required"}), 400

    files = request.files.getlist("resumes")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "At least one resume file is required"}), 400

    total = round(w_skill + w_exp + w_tfidf + w_edu, 2)
    if abs(total - 1.0) > 0.01:
        return jsonify({"error": f"Weights must sum to 1.0 (currently {total})"}), 400

    # Save uploaded files to temp paths before thread starts
    saved_files = []
    for f in files:
        if not f or not allowed_file(f.filename):
            continue
        filename = secure_filename(f.filename)
        suffix   = Path(filename).suffix
        tmp      = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.save(tmp.name)
        tmp.close()
        saved_files.append({"filename": filename, "tmp_path": tmp.name})

    if not saved_files:
        return jsonify({"error": "No valid resume files found"}), 400

    weights = {"skill_match": w_skill, "experience": w_exp,
               "tfidf_sim": w_tfidf, "education": w_edu}
    job_requirements = {
        "min_years_experience":  min_exp,
        "required_degree_level": min_edu,
        "weights":               weights,
    }

    # Save job + set initial status in DB
    job_id = db.save_job(job_title, job_description, job_requirements)
    db.clear_job_candidates(job_id)
    db.set_job_status(job_id, "running", "Starting...")

    # Launch background thread
    threading.Thread(
        target=_run_screening,
        args=(job_id, job_description, job_requirements, saved_files),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "task_id": str(job_id), "status": "started"})


def _run_screening(job_id: int, job_description: str,
                   job_requirements: dict, saved_files: list):
    """Background thread — stores progress in DB so any worker can read it."""
    try:
        candidates = []
        errors     = []
        total      = len(saved_files)

        for i, file_info in enumerate(saved_files):
            filename = file_info["filename"]
            tmp_path = file_info["tmp_path"]
            db.set_job_status(job_id, "running",
                              f"Processing {filename} ({i+1}/{total})...")
            try:
                raw_text     = load_resume(tmp_path)
                preprocessed = preprocess_resume(raw_text)
                extracted    = extract_resume_info(preprocessed)
                candidates.append({
                    "filename":     filename,
                    "preprocessed": preprocessed,
                    "extracted":    extracted,
                })
            except Exception as e:
                errors.append(f"{filename}: {e}")
            finally:
                try: os.unlink(tmp_path)
                except: pass

        if not candidates:
            db.set_job_status(job_id, "error",
                              "No resumes could be processed. " + "; ".join(errors))
            return

        db.set_job_status(job_id, "running",
                          f"Scoring {len(candidates)} candidate(s) with BERT...")
        results = scorer.score_batch(candidates, job_description, job_requirements)

        db.set_job_status(job_id, "running", "Saving results...")
        for result in results:
            db.save_candidate(job_id, result, result.get("filename", ""))

        msg = f"Screened {len(results)} candidate(s)."
        if errors:
            msg += f" Skipped: {'; '.join(errors)}"
        db.set_job_status(job_id, "done", msg)

    except Exception as e:
        db.set_job_status(job_id, "error", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Status polling
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/status/<int:job_id>")
def screening_status(job_id):
    status = db.get_job_status(job_id)
    if not status:
        return jsonify({"status": "error", "message": "Job not found",
                        "job_id": job_id, "progress": 0, "total": 1})
    return jsonify({
        "status":   status["screening_status"],
        "message":  status["screening_message"],
        "job_id":   job_id,
        "progress": 1 if status["screening_status"] == "done" else 0,
        "total":    1,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Candidates
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/candidates/<int:job_id>")
def get_candidates(job_id):
    candidates_data = db.get_candidates(job_id)
    filter_label    = request.args.get("filter", "All")

    if filter_label != "All":
        candidates_data = [c for c in candidates_data
                           if c["ranking_label"] == filter_label]

    job_info = db.get_job(job_id)
    job_req  = json.loads(job_info.get("requirements", "{}")) if job_info else {}
    job_desc = job_info.get("description", "") if job_info else ""
    role     = job_info.get("title", "this role") if job_info else "this role"

    for candidate in candidates_data:
        score_result = _candidate_to_score_result(candidate)
        try:
            feedback = scorer.generate_feedback(score_result, job_desc, job_req, role)
        except Exception:
            feedback = {"strengths": [], "gaps": [], "suggestions": [],
                        "summary": "Feedback unavailable.", "fit_for_role": False}
        candidate["feedback"] = feedback

    return jsonify(candidates_data)


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/analytics/<int:job_id>")
def get_analytics(job_id):
    data = db.get_analytics(job_id)
    if not data:
        return jsonify({"error": "No analytics data found"}), 404
    return jsonify(data)


# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/send_email/<int:candidate_id>/<email_type>", methods=["POST"])
def send_email(candidate_id, email_type):
    data         = request.get_json()
    job_id       = data.get("job_id")
    company_name = data.get("company_name", "Our Company")
    job_info     = db.get_job(job_id)
    candidates_data = db.get_candidates(job_id)
    candidate    = next((c for c in candidates_data if c["id"] == candidate_id), None)

    if not candidate:
        return jsonify({"status": "error", "message": "Candidate not found"})

    email = candidate.get("candidate_email")
    name  = candidate.get("candidate_name") or "Candidate"

    if not email:
        return jsonify({"status": "error", "message": "No email address found"})

    if email_type == "rejection":
        result = mailer.send_rejection(name, email, job_info["title"], company_name)
    else:
        result = mailer.send_shortlist(name, email, job_info["title"], company_name)

    if result["status"] == "sent":
        db.mark_email_sent(candidate_id, email_type, email, "sent")

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# AI feedback
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/ai_feedback/<int:candidate_id>", methods=["POST"])
def ai_feedback(candidate_id):
    data     = request.get_json()
    job_id   = data.get("job_id")
    job_info = db.get_job(job_id)
    candidates_data = db.get_candidates(job_id)
    candidate = next((c for c in candidates_data if c["id"] == candidate_id), None)

    if not candidate:
        return jsonify({"status": "error", "text": "Candidate not found"})

    score_result = _candidate_to_score_result(candidate)
    job_req  = json.loads(job_info.get("requirements", "{}")) if job_info else {}
    job_desc = job_info.get("description", "") if job_info else ""
    role     = job_info.get("title", "this role") if job_info else "this role"

    feedback = scorer.generate_feedback(score_result, job_desc, job_req, role)
    ai_text  = generate_ai_feedback(score_result, feedback, job_desc, role)
    return jsonify({"status": "ok", "text": ai_text})


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/export_csv/<int:job_id>")
def export_csv(job_id):
    job_info        = db.get_job(job_id)
    candidates_data = db.get_candidates(job_id)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Email", "Score", "Label", "Experience (yrs)",
                     "Experience Level", "Education", "Filename"])
    for c in candidates_data:
        writer.writerow([
            c.get("candidate_name", ""), c.get("candidate_email", ""),
            c.get("final_score", ""),    c.get("ranking_label", ""),
            c.get("experience_years", ""), c.get("experience_level", ""),
            c.get("highest_degree", ""),   c.get("filename", ""),
        ])

    output.seek(0)
    fname = (f"candidates_{job_info['title'].replace(' ', '_')}.csv"
             if job_info else "candidates.csv")
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv", as_attachment=True, download_name=fname,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_to_score_result(candidate: dict) -> dict:
    return {
        "final_score":       candidate.get("final_score", 0),
        "ranking_label":     candidate.get("ranking_label", "Not Fit"),
        "breakdown":         candidate.get("breakdown", {}),
        "matched_skills":    candidate.get("matched_skills", []),
        "missing_skills":    candidate.get("missing_skills", []),
        "experience_years":  candidate.get("experience_years", 0),
        "experience_level":  candidate.get("experience_level", "Unknown"),
        "highest_degree":    candidate.get("highest_degree", "Unknown"),
        "certifications":    candidate.get("certifications", []),
        "candidate_name":    candidate.get("candidate_name"),
        "candidate_email":   candidate.get("candidate_email"),
        "jd_skills":         [], "resume_skills":     [],
        "_degree_level":     0,  "svm_label":         None,
        "svm_confidence":    None, "skill_gate_status": "pass",
        "skill_gate_reason": None,
    }


if __name__ == "__main__":
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)