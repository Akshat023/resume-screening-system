"""
layer7_app.py — Layer 7: Streamlit Web App
HR-facing interface for resume screening, candidate review, and analytics.

Run: streamlit run layer7_app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import tempfile
from pathlib import Path

from layer1_ingestion import load_resume
from layer2_preprocessing import preprocess_resume
from layer4_extraction import extract_resume_info
from layer5_model import ResumeScorer, generate_ai_feedback
from layer6_database import Database
from layer8_email import EmailSender

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("db",                   None),
    ("scorer",               None),
    ("mailer",               None),
    ("current_job_id",       None),
    ("screening_done",       False),
    ("feedback_cache",       {}),
    ("score_results",        {}),
    ("last_job_requirements",{}),
    ("last_job_description", ""),
    ("last_job_title",       ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.db     is None: st.session_state.db     = Database()
if st.session_state.scorer is None: st.session_state.scorer = ResumeScorer()
if st.session_state.mailer is None: st.session_state.mailer = EmailSender()

db     = st.session_state.db
scorer = st.session_state.scorer
mailer = st.session_state.mailer


# ── Helper ────────────────────────────────────────────────────────────────────
def _reconstruct_score_result(candidate: dict) -> dict:
    """Rebuild a score_result from DB row for feedback generation on previous sessions."""
    return {
        "final_score":      candidate.get("final_score", 0),
        "ranking_label":    candidate.get("ranking_label", "Not Fit"),
        "breakdown":        candidate.get("breakdown", {}),
        "matched_skills":   candidate.get("matched_skills", []),
        "missing_skills":   candidate.get("missing_skills", []),
        "experience_years": candidate.get("experience_years", 0),
        "experience_level": candidate.get("experience_level", "Unknown"),
        "highest_degree":   candidate.get("highest_degree", "Unknown"),
        "certifications":   candidate.get("certifications", []),
        "candidate_name":   candidate.get("candidate_name"),
        "candidate_email":  candidate.get("candidate_email"),
        "jd_skills":        [],
        "resume_skills":    [],
        "_degree_level":    0,
        "svm_label":        None,
        "svm_confidence":   None,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Resume Screener")
    st.caption("AI-powered ATS")
    st.divider()

    st.subheader("Company Settings")
    company_name = st.text_input("Company Name", value="Acme Corp")

    st.divider()
    st.subheader("Ranking Weights")
    st.caption("Must sum to 1.0")

    w_skill = st.slider("Skill Match",       0.0, 1.0, 0.40, 0.05)
    w_exp   = st.slider("Experience",        0.0, 1.0, 0.30, 0.05)
    w_tfidf = st.slider("BERT Similarity",   0.0, 1.0, 0.20, 0.05)
    w_edu   = st.slider("Education",         0.0, 1.0, 0.10, 0.05)

    total_weight = round(w_skill + w_exp + w_tfidf + w_edu, 2)
    if total_weight != 1.0:
        st.warning(f"Weights sum to {total_weight}. Adjust to 1.0")
    else:
        st.success("Weights OK")

    custom_weights = {
        "skill_match": w_skill,
        "experience":  w_exp,
        "tfidf_sim":   w_tfidf,
        "education":   w_edu,
    }

    st.divider()
    jobs = db.get_jobs()
    if jobs:
        st.subheader("Previous Jobs")
        for job in jobs[:5]:
            if st.button(f"📋 {job['title']}", key=f"job_{job['id']}"):
                st.session_state.current_job_id = job["id"]
                st.session_state.screening_done = True
                st.session_state.feedback_cache = {}
                st.session_state.score_results  = {}


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Screen Resumes", "👥 Candidates", "📊 Analytics"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Screen Resumes
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Screen Resumes")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Details")
        job_title       = st.text_input("Job Title", placeholder="e.g. Machine Learning Engineer")
        job_description = st.text_area("Job Description", height=200,
                                       placeholder="Paste the full job description here...")
        st.subheader("Requirements")
        min_exp = st.number_input("Minimum Years Experience", min_value=0, max_value=20, value=2)
        min_edu = st.selectbox("Minimum Education", options=[0,1,2,3,4,5],
                               format_func=lambda x: {0:"No requirement",1:"High School",
                               2:"Diploma",3:"Bachelor's",4:"Master's",5:"PhD"}[x], index=3)

    with col2:
        st.subheader("Upload Resumes")
        uploaded_files = st.file_uploader("Drop resume files here",
                                          type=["pdf","docx","txt","md"],
                                          accept_multiple_files=True)
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded")
            for f in uploaded_files:
                st.caption(f"📄 {f.name}")

    st.divider()
    can_screen = job_title and job_description and uploaded_files and total_weight == 1.0

    if st.button("🚀 Screen Resumes", type="primary", disabled=not can_screen):
        job_requirements = {
            "min_years_experience":  min_exp,
            "required_degree_level": min_edu,
            "weights":               custom_weights,
        }
        job_id = db.save_job(job_title, job_description, job_requirements)
        st.session_state.current_job_id       = job_id
        st.session_state.feedback_cache       = {}
        st.session_state.score_results        = {}
        st.session_state.last_job_requirements = job_requirements
        st.session_state.last_job_description  = job_description
        st.session_state.last_job_title        = job_title
        db.clear_job_candidates(job_id)

        progress   = st.progress(0, text="Starting screening...")
        candidates = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress((i+1)/len(uploaded_files), text=f"Processing {uploaded_file.name}...")
            try:
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                raw_text    = load_resume(tmp_path)
                os.unlink(tmp_path)
                preprocessed = preprocess_resume(raw_text)
                extracted    = extract_resume_info(preprocessed)
                candidates.append({"filename": uploaded_file.name,
                                   "preprocessed": preprocessed,
                                   "extracted": extracted})
            except Exception as e:
                st.warning(f"Could not process {uploaded_file.name}: {e}")

        if candidates:
            with st.spinner("Scoring candidates with BERT..."):
                results = scorer.score_batch(candidates, job_description, job_requirements)

            for result in results:
                db.save_candidate(job_id, result, result.get("filename",""))
                st.session_state.score_results[result.get("filename","")] = result

            st.session_state.screening_done = True
            progress.progress(1.0, text="Done!")
            st.success(f"Screened {len(results)} candidates. Switch to the **Candidates** tab.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Candidates
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Candidate Results")

    if not st.session_state.current_job_id:
        st.info("Screen some resumes first, or select a previous job from the sidebar.")
    else:
        job_id          = st.session_state.current_job_id
        job_info        = db.get_job(job_id)
        candidates_data = db.get_candidates(job_id)

        if job_info:
            st.subheader(f"Role: {job_info['title']}")

        if not candidates_data:
            st.info("No candidates found for this job.")
        else:
            col1, _ = st.columns([1, 3])
            with col1:
                filter_label = st.selectbox("Filter by status", ["All","Fit","Maybe","Not Fit"])

            filtered = candidates_data if filter_label == "All" else [
                c for c in candidates_data if c["ranking_label"] == filter_label
            ]
            st.caption(f"Showing {len(filtered)} of {len(candidates_data)} candidates")

            for candidate in filtered:
                score = candidate["final_score"]
                label = candidate["ranking_label"]
                name  = candidate["candidate_name"] or candidate["filename"]
                icon  = {"Fit":"🟢","Maybe":"🟡","Not Fit":"🔴"}.get(label,"⚪")

                with st.expander(f"{icon} {name}  —  {score}/100  [{label}]"):
                    # Skill gate warning
                    sr = st.session_state.score_results.get(candidate.get("filename",""))
                    if sr and sr.get("skill_gate_reason"):
                        st.warning(sr["skill_gate_reason"])

                    # Row 1: Core info + breakdown + skills
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Fit Score", f"{score}/100")
                        st.write(f"**Experience:** {candidate['experience_level']}")
                        st.write(f"**Education:** {candidate['highest_degree']}")
                        st.write(f"**Email:** {candidate['candidate_email'] or 'Not found'}")

                    with c2:
                        st.write("**Score Breakdown:**")
                        dim_labels = {"skill_match":"Skill Match","experience":"Experience",
                                      "tfidf_sim":"BERT Similarity","education":"Education"}
                        for dim, val in (candidate["breakdown"] or {}).items():
                            st.progress(int(val), text=f"{dim_labels.get(dim,dim)}: {val}%")

                    with c3:
                        if candidate["matched_skills"]:
                            st.write("**Matched Skills:**")
                            st.success(", ".join(candidate["matched_skills"]))
                        if candidate["missing_skills"]:
                            st.write("**Missing Skills:**")
                            st.error(", ".join(candidate["missing_skills"]))

                    # Row 2: Feedback
                    st.divider()
                    st.subheader("💡 Feedback & Improvement Areas")

                    candidate_id    = candidate["id"]
                    cached_feedback = st.session_state.feedback_cache.get(candidate_id)

                    if cached_feedback is None:
                        score_result = st.session_state.score_results.get(candidate["filename"])
                        if score_result is None:
                            score_result = _reconstruct_score_result(candidate)

                        job_req  = st.session_state.last_job_requirements or {}
                        job_desc = st.session_state.last_job_description or job_info.get("description","")
                        role     = st.session_state.last_job_title or job_info.get("title","this role")

                        try:
                            feedback = scorer.generate_feedback(score_result, job_desc, job_req, role)
                        except Exception as e:
                            feedback = {"strengths":["Could not generate feedback."],
                                        "gaps":[str(e)],"suggestions":[],
                                        "summary":"Feedback generation failed.",
                                        "fit_for_role": score >= 70}

                        st.session_state.feedback_cache[candidate_id] = feedback
                        cached_feedback = feedback

                    feedback = cached_feedback

                    if feedback["fit_for_role"]:
                        st.success(feedback["summary"])
                    elif score >= 45:
                        st.warning(feedback["summary"])
                    else:
                        st.error(feedback["summary"])

                    fb1, fb2, fb3 = st.columns(3)
                    with fb1:
                        st.markdown("**✅ Strengths**")
                        for s in feedback["strengths"]:
                            st.markdown(f"- {s}")
                    with fb2:
                        st.markdown("**❌ Gaps**")
                        for g in feedback["gaps"]:
                            st.markdown(f"- {g}")
                    with fb3:
                        st.markdown("**📌 Suggestions**")
                        for sg in feedback["suggestions"]:
                            st.markdown(f"- {sg}")

                    # AI Commentary
                    st.divider()
                    st.markdown("**🤖 AI Commentary**")
                    ai_key     = f"ai_{candidate_id}"
                    ai_summary = st.session_state.feedback_cache.get(ai_key)

                    if ai_summary is None:
                        if st.button("Generate AI Commentary", key=f"ai_{candidate_id}"):
                            sr       = st.session_state.score_results.get(candidate["filename"],
                                           _reconstruct_score_result(candidate))
                            job_desc = st.session_state.last_job_description or job_info.get("description","")
                            role     = st.session_state.last_job_title or job_info.get("title","this role")
                            with st.spinner("Generating AI commentary (20-40s on CPU)..."):
                                ai_summary = generate_ai_feedback(sr, feedback, job_desc, role)
                            st.session_state.feedback_cache[ai_key] = ai_summary
                            st.rerun()
                        else:
                            st.caption("Click to generate AI-powered commentary via local LLM (requires Ollama).")
                    else:
                        st.info(ai_summary)
                        if st.button("Regenerate", key=f"regen_{candidate_id}"):
                            del st.session_state.feedback_cache[ai_key]
                            st.rerun()

                    # Email actions
                    st.divider()
                    email        = candidate["candidate_email"]
                    already_sent = candidate["email_sent"]

                    if already_sent:
                        st.caption("✅ Email already sent")
                    elif not email:
                        st.caption("⚠️ No email address found in resume")
                    else:
                        e1, e2, _ = st.columns([1,1,2])
                        with e1:
                            if st.button("Send Rejection", key=f"rej_{candidate_id}"):
                                result = mailer.send_rejection(
                                    candidate["candidate_name"] or "Candidate",
                                    email, job_info["title"], company_name)
                                if result["status"] == "sent":
                                    db.mark_email_sent(candidate_id, "rejection", email, "sent")
                                    st.success(f"Rejection sent to {email}")
                                elif result["status"] == "not_configured":
                                    st.warning("Email not configured. Add credentials to .env")
                                else:
                                    st.error(result["message"])
                        with e2:
                            if st.button("Send Shortlist", key=f"sl_{candidate_id}"):
                                result = mailer.send_shortlist(
                                    candidate["candidate_name"] or "Candidate",
                                    email, job_info["title"], company_name)
                                if result["status"] == "sent":
                                    db.mark_email_sent(candidate_id, "shortlist", email, "sent")
                                    st.success(f"Shortlist email sent to {email}")
                                elif result["status"] == "not_configured":
                                    st.warning("Email not configured. Add credentials to .env")
                                else:
                                    st.error(result["message"])

            # CSV export
            st.divider()
            if st.button("Export to CSV"):
                df         = pd.DataFrame(candidates_data)
                export_cols = ["candidate_name","candidate_email","final_score",
                               "ranking_label","experience_years","experience_level",
                               "highest_degree","filename"]
                df_export  = df[[c for c in export_cols if c in df.columns]]
                st.download_button("Download CSV", data=df_export.to_csv(index=False),
                                   file_name=f"candidates_{job_info['title'].replace(' ','_')}.csv",
                                   mime="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Analytics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Analytics")

    if not st.session_state.current_job_id:
        st.info("Screen some resumes first to see analytics.")
    else:
        analytics = db.get_analytics(st.session_state.current_job_id)

        if not analytics:
            st.info("No data yet.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Screened",  analytics["total_screened"])
            c2.metric("Shortlisted",     analytics["fit_count"])
            c3.metric("Shortlist Ratio", f"{analytics['shortlist_ratio']}%")
            c4.metric("Avg Fit Score",   f"{analytics['avg_score']}/100")

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Hiring Funnel")
                st.bar_chart(pd.DataFrame({
                    "Status": ["Fit","Maybe","Not Fit"],
                    "Count":  [analytics["fit_count"],analytics["maybe_count"],analytics["not_fit_count"]]
                }).set_index("Status"))

            with col2:
                st.subheader("Top Skills Found")
                if analytics["top_skills"]:
                    st.bar_chart(pd.DataFrame(analytics["top_skills"],
                                              columns=["Skill","Count"]).set_index("Skill"))
                else:
                    st.info("No skill data yet.")

            st.subheader("Score Distribution")
            candidates_data = db.get_candidates(st.session_state.current_job_id)
            if candidates_data:
                st.bar_chart(pd.DataFrame([
                    {"Candidate": c["candidate_name"] or c["filename"], "Score": c["final_score"]}
                    for c in candidates_data
                ]).set_index("Candidate"))