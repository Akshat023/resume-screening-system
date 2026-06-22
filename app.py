"""
app.py — Flask Web Application
Screening runs in a background thread to avoid request timeouts.
BERT model loading + scoring can take 30-60s — this returns immediately
with a job_id and the frontend polls /status/<job_id> until done.
"""

import os
import json
import tempfile
import io
import csv
import uuid
import threading
from pathlib import Path
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, send_file, flash)
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
app.secret_key = os.getenv("SECRET_KEY", "resume-screening-secret-key-2025")

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}

# Module-level singletons
db     = Database()
scorer = ResumeScorer()
mailer = EmailSender()

# In-memory job status store
# { task_id: { "status": "pending"|"running"|"done"|"error",
#              "job_id": int, "progress": int, "total": int, "message": str } }
screening_tasks: dict[str, dict] = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Background screening worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_screening(task_id: str, job_id: int, file_data_list: list[dict],
                   job_description: str, job_requirements: dict):
    """
    Runs in a background thread.
    file_data_list: list of {"filename": str, "bytes": bytes, "suffix": str}
    """
    try:
        screening_tasks[task_id]["status"]  = "running"
        screening_tasks[task_id]["total"]   = len(file_data_list)
        screening_tasks[task_id]["progress"] = 0

        candidates = []
        errors     = []

        for i, file_data in enumerate(file_data_list):
            screening_tasks[task_id]["progress"] = i
            screening_tasks[task_id]["message"]  = f"Processing {file_data['filename']}..."

            try:
                # Write bytes to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_data["suffix"]
                ) as tmp:
                    tmp.write(file_data["bytes"])
                    tmp_path = tmp.name

                raw_text     = load_resume(tmp_path)
                os.unlink(tmp_path)
                preprocessed = preprocess_resume(raw_text)
                extracted    = extract_resume_info(preprocessed)
                candidates.append({
                    "filename":     file_data["filename"],
                    "preprocessed": preprocessed,
                    "extracted":    extracted,
                })
            except Exception as e:
                errors.append(f"{file_data['filename']}: {e}")

        if not candidates:
            screening_tasks[task_id]["status"]  = "error"
            screening_tasks[task_id]["message"] = "No resumes could be processed. " + "; ".join(errors)
            return

        screening_tasks[task_id]["message"] = "Scoring candidates..."
        results = scorer.score_batch(candidates, job_description, job_requirements)

        for result in results:
            db.save_candidate(job_id, result, result.get("filename", ""))

        screening_tasks[task_id]["status"]   = "done"
        screening_tasks[task_id]["progress"] = len(file_data_list)
        screening_tasks[task_id]["message"]  = (
            f"Done. {len(results)} candidate(s) scored."
            + (f" Skipped: {'; '.join(errors)}" if errors else "")
        )

    except Exception as e:
        screening_tasks[task_id]["status"]  = "error"
        screening_tasks[task_id]["message"] = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    jobs = db.get_jobs()
    return render_template("index.html", jobs=jobs)


@app.route("/screen", methods=["POST"])
def screen():
    """
    Reads form data + files, saves everything to memory,
    starts background thread, returns immediately with task_id.
    """
    job_title       = request.form.get("job_title", "").strip()
    job_description = request.form.get("job_description", "").strip()
    min_exp         = int(request.form.get("min_exp", 0))
    min_edu         = int(request.form.get("min_edu", 3))
    w_skill         = float(request.form.get("w_skill", 0.40))
    w_exp           = float(request.form.get("w_exp",   0.30))
    w_tfidf         = float(request.form.get("w_tfidf", 0.20))
    w_edu           = float(request.form.get("w_edu",   0.10))

    if not job_title or not job_description:
        flash("Please provide job title and description.", "error")
        return redirect(url_for("index"))

    files = request.files.getlist("resumes")
    if not files or all(f.filename == "" for f in files):
        flash("Please upload at least one resume.", "error")
        return redirect(url_for("index"))

    total = round(w_skill + w_exp + w_tfidf + w_edu, 2)
    if abs(total - 1.0) > 0.01:
        flash(f"Weights must sum to 1.0 (currently {total}).", "error")
        return redirect(url_for("index"))

    weights = {
        "skill_match": w_skill,
        "experience":  w_exp,
        "tfidf_sim":   w_tfidf,
        "education":   w_edu,
    }
    job_requirements = {
        "min_years_experience":  min_exp,
        "required_degree_level": min_edu,
        "weights":               weights,
    }

    # Read ALL file bytes NOW (before the request context closes)
    file_data_list = []
    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            suffix   = Path(filename).suffix
            data     = f.read()          # read into memory while request is open
            file_data_list.append({
                "filename": filename,
                "bytes":    data,
                "suffix":   suffix,
            })

    if not file_data_list:
        flash("No valid resume files uploaded.", "error")
        return redirect(url_for("index"))

    # Save job to DB
    job_id = db.save_job(job_title, job_description, job_requirements)
    db.clear_job_candidates(job_id)

    # Create task entry
    task_id = str(uuid.uuid4())
    screening_tasks[task_id] = {
        "status":   "pending",
        "job_id":   job_id,
        "progress": 0,
        "total":    len(file_data_list),
        "message":  "Starting...",
    }

    # Launch background thread — does NOT block the response
    thread = threading.Thread(
        target=_run_screening,
        args=(task_id, job_id, file_data_list, job_description, job_requirements),
        daemon=True,
    )
    thread.start()

    # Return progress page immediately
    return render_template("progress.html",
                           task_id=task_id,
                           job_id=job_id,
                           job_title=job_title,
                           total=len(file_data_list))


@app.route("/status/<task_id>")
def status(task_id):
    """Polled by the progress page every 2 seconds."""
    task = screening_tasks.get(task_id)
    if not task:
        return jsonify({"status": "error", "message": "Task not found"})
    return jsonify(task)


@app.route("/candidates/<int:job_id>")
def candidates(job_id):
    job_info        = db.get_job(job_id)
    candidates_data = db.get_candidates(job_id)
    jobs            = db.get_jobs()
    filter_label    = request.args.get("filter", "All")

    if filter_label != "All":
        candidates_data = [
            c for c in candidates_data if c["ranking_label"] == filter_label
        ]

    # Generate rule-based feedback for each candidate
    for candidate in candidates_data:
        score_result = _candidate_to_score_result(candidate)
        try:
            feedback = scorer.generate_feedback(
                score_result,
                job_info.get("description", "") if job_info else "",
                json.loads(job_info.get("requirements", "{}")) if job_info else {},
                job_info.get("title", "this role") if job_info else "this role",
            )
        except Exception:
            feedback = {
                "strengths": [], "gaps": [], "suggestions": [],
                "summary": "Feedback unavailable.", "fit_for_role": False,
            }
        candidate["feedback"] = feedback

    return render_template("candidates.html",
                           job_info=job_info,
                           candidates=candidates_data,
                           jobs=jobs,
                           filter_label=filter_label,
                           job_id=job_id)


@app.route("/analytics/<int:job_id>")
def analytics(job_id):
    job_info      = db.get_job(job_id)
    analytics_data = db.get_analytics(job_id)
    jobs          = db.get_jobs()
    return render_template("analytics.html",
                           job_info=job_info,
                           analytics=analytics_data,
                           jobs=jobs,
                           job_id=job_id)


@app.route("/send_email/<int:candidate_id>/<email_type>", methods=["POST"])
def send_email(candidate_id, email_type):
    data         = request.get_json()
    job_id       = data.get("job_id")
    company_name = data.get("company_name", "Our Company")

    job_info        = db.get_job(job_id)
    candidates_data = db.get_candidates(job_id)
    candidate       = next(
        (c for c in candidates_data if c["id"] == candidate_id), None
    )

    if not candidate:
        return jsonify({"status": "error", "message": "Candidate not found"})

    email = candidate.get("candidate_email")
    name  = candidate.get("candidate_name") or "Candidate"

    if not email:
        return jsonify({"status": "error", "message": "No email address found in resume"})

    if email_type == "rejection":
        result = mailer.send_rejection(name, email, job_info["title"], company_name)
    else:
        result = mailer.send_shortlist(name, email, job_info["title"], company_name)

    if result["status"] == "sent":
        db.mark_email_sent(candidate_id, email_type, email, "sent")

    return jsonify(result)


@app.route("/ai_feedback/<int:candidate_id>", methods=["POST"])
def ai_feedback(candidate_id):
    data     = request.get_json()
    job_id   = data.get("job_id")
    job_info = db.get_job(job_id)

    candidates_data = db.get_candidates(job_id)
    candidate = next(
        (c for c in candidates_data if c["id"] == candidate_id), None
    )

    if not candidate:
        return jsonify({"status": "error", "text": "Candidate not found"})

    score_result = _candidate_to_score_result(candidate)
    job_req  = json.loads(job_info.get("requirements", "{}")) if job_info else {}
    job_desc = job_info.get("description", "") if job_info else ""
    role     = job_info.get("title", "this role") if job_info else "this role"

    feedback = scorer.generate_feedback(score_result, job_desc, job_req, role)
    ai_text  = generate_ai_feedback(score_result, feedback, job_desc, role)

    return jsonify({"status": "ok", "text": ai_text})


@app.route("/export_csv/<int:job_id>")
def export_csv(job_id):
    job_info        = db.get_job(job_id)
    candidates_data = db.get_candidates(job_id)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Name", "Email", "Score", "Label",
        "Experience (yrs)", "Experience Level", "Education", "Filename"
    ])
    for c in candidates_data:
        writer.writerow([
            c.get("candidate_name", ""),
            c.get("candidate_email", ""),
            c.get("final_score", ""),
            c.get("ranking_label", ""),
            c.get("experience_years", ""),
            c.get("experience_level", ""),
            c.get("highest_degree", ""),
            c.get("filename", ""),
        ])

    output.seek(0)
    fname = (
        f"candidates_{job_info['title'].replace(' ', '_')}.csv"
        if job_info else "candidates.csv"
    )
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=fname,
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
        "jd_skills":         [],
        "resume_skills":     [],
        "_degree_level":     0,
        "svm_label":         None,
        "svm_confidence":    None,
        "skill_gate_status": "pass",
        "skill_gate_reason": None,
    }


if __name__ == "__main__":
    import os
    os.environ["WATCHDOG_EXTRA_DIRS"] = ""
    app.run(
        debug=True,
        threaded=True,
        use_reloader=False,   # disables watchdog completely
    )